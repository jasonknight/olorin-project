//! Application state and logic

use crate::api;
use crate::settings::{create_tab_definitions, DynamicSource, InputType, SettingDef, TabDef};
use crate::validation::{validate_and_convert, ValidationResult};
use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

/// Which element has focus
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Focus {
    FormField(usize),
    SearchInput,
}

/// A search result pointing to a setting in a specific tab
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub tab_idx: usize,
    pub field_idx: usize,
    pub tab_name: String,
}

/// Runtime value holder for a setting
#[derive(Debug, Clone)]
pub struct SettingValue {
    pub def: SettingDef,
    pub current_value: Value,
    pub input_buffer: String,
    pub cursor_pos: usize,
    pub select_index: usize,
    pub dynamic_options: Vec<String>,
    pub is_editing: bool,
    pub validation_error: Option<String>,
}

impl SettingValue {
    pub fn new(def: SettingDef, value: Value) -> Self {
        let input_buffer = value_to_string(&value, &def.input_type);
        let select_index = match &def.input_type {
            InputType::Select(options) => {
                let val_str = value.as_str().unwrap_or("");
                options.iter().position(|o| o == val_str).unwrap_or(0)
            }
            InputType::Toggle => {
                if value.as_bool().unwrap_or(false) {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        };

        // cursor_pos is now a CHARACTER index, not byte index
        let cursor_char_count = input_buffer.chars().count();

        Self {
            def,
            current_value: value,
            cursor_pos: cursor_char_count,
            input_buffer,
            select_index,
            dynamic_options: Vec::new(),
            is_editing: false,
            validation_error: None,
        }
    }

    pub fn get_options(&self) -> Vec<String> {
        match &self.def.input_type {
            InputType::Select(opts) => opts.clone(),
            InputType::DynamicSelect(_) => {
                if self.dynamic_options.is_empty() {
                    vec![self.input_buffer.clone()]
                } else {
                    self.dynamic_options.clone()
                }
            }
            InputType::Toggle => vec!["true".into(), "false".into()],
            _ => vec![],
        }
    }

    /// Get the character count of the input buffer
    pub fn char_count(&self) -> usize {
        self.input_buffer.chars().count()
    }

    /// Convert character index to byte index for slicing
    fn char_to_byte_idx(&self, char_idx: usize) -> usize {
        self.input_buffer
            .char_indices()
            .nth(char_idx)
            .map(|(byte_idx, _)| byte_idx)
            .unwrap_or(self.input_buffer.len())
    }

    /// Split text at cursor for rendering
    /// Returns (before_cursor, cursor_char, after_cursor)
    pub fn split_at_cursor(&self) -> (&str, char, &str) {
        let char_count = self.char_count();
        let cursor_pos = self.cursor_pos.min(char_count);
        let byte_idx = self.char_to_byte_idx(cursor_pos);

        let before = &self.input_buffer[..byte_idx];

        if cursor_pos < char_count {
            let cursor_char = self.input_buffer[byte_idx..].chars().next().unwrap();
            let after_byte_idx = byte_idx + cursor_char.len_utf8();
            let after = &self.input_buffer[after_byte_idx..];
            (before, cursor_char, after)
        } else {
            (before, ' ', "")
        }
    }

    /// Insert a character at the cursor position (UTF-8 safe)
    pub fn insert_char_at_cursor(&mut self, c: char) {
        let byte_idx = self.char_to_byte_idx(self.cursor_pos);
        self.input_buffer.insert(byte_idx, c);
        self.cursor_pos += 1;
    }

    /// Delete character before cursor (backspace) - UTF-8 safe
    /// Returns true if a character was deleted
    pub fn delete_char_before_cursor(&mut self) -> bool {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            let byte_idx = self.char_to_byte_idx(self.cursor_pos);
            if let Some(c) = self.input_buffer[byte_idx..].chars().next() {
                self.input_buffer
                    .replace_range(byte_idx..byte_idx + c.len_utf8(), "");
                return true;
            }
        }
        false
    }

    /// Delete character at cursor (delete forward) - UTF-8 safe
    /// Returns true if a character was deleted
    pub fn delete_char_at_cursor(&mut self) -> bool {
        let char_count = self.char_count();
        if self.cursor_pos < char_count {
            let byte_idx = self.char_to_byte_idx(self.cursor_pos);
            if let Some(c) = self.input_buffer[byte_idx..].chars().next() {
                self.input_buffer
                    .replace_range(byte_idx..byte_idx + c.len_utf8(), "");
                return true;
            }
        }
        false
    }

    /// Move cursor left by one character
    pub fn move_cursor_left(&mut self) {
        self.cursor_pos = self.cursor_pos.saturating_sub(1);
    }

    /// Move cursor right by one character
    pub fn move_cursor_right(&mut self) {
        let char_count = self.char_count();
        if self.cursor_pos < char_count {
            self.cursor_pos += 1;
        }
    }

    /// Move cursor to start of text
    pub fn cursor_to_start(&mut self) {
        self.cursor_pos = 0;
    }

    /// Move cursor to end of text
    pub fn cursor_to_end(&mut self) {
        self.cursor_pos = self.char_count();
    }

    /// Check if a character is valid for numeric input at current position
    pub fn is_valid_numeric_char(&self, c: char, allow_float: bool) -> bool {
        match c {
            '0'..='9' => true,
            '-' => self.cursor_pos == 0 && !self.input_buffer.contains('-'),
            '.' if allow_float => !self.input_buffer.contains('.'),
            _ => false,
        }
    }
}

fn value_to_string(value: &Value, input_type: &InputType) -> String {
    match input_type {
        InputType::Textarea => {
            if let Some(arr) = value.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                value.as_str().unwrap_or("").to_string()
            }
        }
        InputType::Toggle => {
            if value.as_bool().unwrap_or(false) {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        InputType::NullableText | InputType::NullableInt { .. } => {
            if value.is_null() {
                String::new()
            } else if let Some(s) = value.as_str() {
                s.to_string()
            } else if let Some(n) = value.as_i64() {
                n.to_string()
            } else if let Some(n) = value.as_f64() {
                n.to_string()
            } else {
                value.to_string()
            }
        }
        _ => {
            if let Some(s) = value.as_str() {
                s.to_string()
            } else if let Some(n) = value.as_i64() {
                n.to_string()
            } else if let Some(n) = value.as_f64() {
                format!("{:.2}", n)
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string()
            } else if value.is_null() {
                String::new()
            } else {
                value.to_string()
            }
        }
    }
}

/// Main application state
pub struct App {
    pub tabs: Vec<TabDef>,
    pub current_tab: usize,
    pub focus: Focus,
    pub values: Vec<Vec<SettingValue>>,
    pub form_scroll: usize,
    pub status: Option<String>,
    pub config_path: PathBuf,
    pub should_quit: bool,
    pub ollama_base_url: String,
    pub visible_height: usize,
    // Search state
    pub search_query: String,
    pub search_cursor_pos: usize,
    pub search_results: Vec<SearchResult>,
    pub search_selected: usize,
}

impl App {
    pub fn new(config_path: PathBuf) -> Result<Self> {
        let tabs = create_tab_definitions();
        let json = load_json(&config_path)?;

        let mut values = Vec::new();
        let mut ollama_base_url = "http://localhost:11434".to_string();

        for tab in &tabs {
            let mut tab_values = Vec::new();
            for setting in &tab.settings {
                let value = get_nested_value(&json, setting.key)
                    .unwrap_or_else(|| setting.default_value.clone().unwrap_or(Value::Null));

                // Extract ollama base URL for API calls
                if setting.key == "ollama.base_url" {
                    if let Some(url) = value.as_str() {
                        ollama_base_url = url.to_string();
                    }
                }

                tab_values.push(SettingValue::new(setting.clone(), value));
            }
            values.push(tab_values);
        }

        Ok(Self {
            tabs,
            current_tab: 0,
            focus: Focus::SearchInput, // Start with search input focused
            values,
            form_scroll: 0,
            status: Some("Type to search settings, Tab to navigate results, Esc to quit".into()),
            config_path,
            should_quit: false,
            ollama_base_url,
            visible_height: 20,
            search_query: String::new(),
            search_cursor_pos: 0,
            search_results: Vec::new(),
            search_selected: 0,
        })
    }

    /// Check if we're on the Search tab
    pub fn is_search_tab(&self) -> bool {
        self.current_tab == 0
    }

    /// Perform search across all tabs (excluding Search tab itself)
    pub fn perform_search(&mut self) {
        self.search_results.clear();
        self.search_selected = 0;
        self.form_scroll = 0;

        if self.search_query.is_empty() {
            return;
        }

        let query = self.search_query.to_lowercase();

        // Search through all tabs except the Search tab (index 0)
        for (tab_idx, tab) in self.tabs.iter().enumerate().skip(1) {
            for (field_idx, setting) in tab.settings.iter().enumerate() {
                // Search in key, label, and description
                let matches = setting.key.to_lowercase().contains(&query)
                    || setting.label.to_lowercase().contains(&query)
                    || setting.description.to_lowercase().contains(&query);

                // Also search in current value
                let value_matches =
                    if let Some(sv) = self.values.get(tab_idx).and_then(|v| v.get(field_idx)) {
                        sv.input_buffer.to_lowercase().contains(&query)
                    } else {
                        false
                    };

                if matches || value_matches {
                    self.search_results.push(SearchResult {
                        tab_idx,
                        field_idx,
                        tab_name: tab.name.to_string(),
                    });
                }
            }
        }

        self.status = Some(format!("Found {} results", self.search_results.len()));
    }

    /// Get the SettingValue for a search result
    pub fn get_search_result_value(&self, result_idx: usize) -> Option<&SettingValue> {
        self.search_results.get(result_idx).and_then(|r| {
            self.values
                .get(r.tab_idx)
                .and_then(|tab| tab.get(r.field_idx))
        })
    }

    /// Convert character index to byte index for search query
    fn search_char_to_byte_idx(&self, char_idx: usize) -> usize {
        self.search_query
            .char_indices()
            .nth(char_idx)
            .map(|(byte_idx, _)| byte_idx)
            .unwrap_or(self.search_query.len())
    }

    /// Get character count of search query
    fn search_char_count(&self) -> usize {
        self.search_query.chars().count()
    }

    /// Split search query at cursor for rendering (UTF-8 safe)
    pub fn split_search_at_cursor(&self) -> (&str, char, &str) {
        let char_count = self.search_char_count();
        let cursor_pos = self.search_cursor_pos.min(char_count);
        let byte_idx = self.search_char_to_byte_idx(cursor_pos);

        let before = &self.search_query[..byte_idx];

        if cursor_pos < char_count {
            let cursor_char = self.search_query[byte_idx..].chars().next().unwrap();
            let after_byte_idx = byte_idx + cursor_char.len_utf8();
            let after = &self.search_query[after_byte_idx..];
            (before, cursor_char, after)
        } else {
            (before, ' ', "")
        }
    }

    /// Insert character into search query (UTF-8 safe)
    pub fn search_insert_char(&mut self, c: char) {
        let byte_idx = self.search_char_to_byte_idx(self.search_cursor_pos);
        self.search_query.insert(byte_idx, c);
        self.search_cursor_pos += 1;
        self.perform_search();
    }

    /// Delete character from search query (UTF-8 safe)
    pub fn search_delete_char(&mut self) {
        if self.search_cursor_pos > 0 {
            self.search_cursor_pos -= 1;
            let byte_idx = self.search_char_to_byte_idx(self.search_cursor_pos);
            if let Some(c) = self.search_query[byte_idx..].chars().next() {
                self.search_query
                    .replace_range(byte_idx..byte_idx + c.len_utf8(), "");
            }
            self.perform_search();
        }
    }

    /// Move search cursor left
    pub fn search_cursor_left(&mut self) {
        self.search_cursor_pos = self.search_cursor_pos.saturating_sub(1);
    }

    /// Move search cursor right
    pub fn search_cursor_right(&mut self) {
        let char_count = self.search_char_count();
        if self.search_cursor_pos < char_count {
            self.search_cursor_pos += 1;
        }
    }

    /// Move search cursor to start
    pub fn search_cursor_to_start(&mut self) {
        self.search_cursor_pos = 0;
    }

    /// Move search cursor to end
    pub fn search_cursor_to_end(&mut self) {
        self.search_cursor_pos = self.search_char_count();
    }

    /// Ensure selected search result is visible
    pub fn ensure_search_result_visible(&mut self) {
        let lines_per_result = 3;
        let result_line = self.search_selected * lines_per_result;

        if result_line < self.form_scroll {
            self.form_scroll = result_line;
        } else if result_line >= self.form_scroll + self.visible_height {
            self.form_scroll = result_line.saturating_sub(self.visible_height) + lines_per_result;
        }
    }

    /// Get the input type of the currently selected search result
    pub fn get_search_result_input_type(&self) -> Option<InputType> {
        let result = self.search_results.get(self.search_selected)?;
        self.values
            .get(result.tab_idx)
            .and_then(|tab| tab.get(result.field_idx))
            .map(|sv| sv.def.input_type.clone())
    }

    /// Toggle a boolean search result
    pub fn toggle_search_result(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            if matches!(setting.def.input_type, InputType::Toggle) {
                let current = setting.input_buffer == "true";
                setting.input_buffer = if current { "false" } else { "true" }.to_string();
                setting.select_index = if current { 1 } else { 0 };
            }
        }

        self.save_search_result_field();
    }

    /// Select previous option for a search result
    pub fn search_result_select_prev(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            let options = setting.get_options();
            if !options.is_empty() {
                setting.select_index = if setting.select_index == 0 {
                    options.len() - 1
                } else {
                    setting.select_index - 1
                };
                setting.input_buffer = options[setting.select_index].clone();
            }
        }

        self.save_search_result_field();
    }

    /// Select next option for a search result
    pub fn search_result_select_next(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            let options = setting.get_options();
            if !options.is_empty() {
                setting.select_index = (setting.select_index + 1) % options.len();
                setting.input_buffer = options[setting.select_index].clone();
            }
        }

        self.save_search_result_field();
    }

    /// Insert character into search result text input (UTF-8 safe)
    pub fn search_result_insert_char(&mut self, c: char) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            // Validate character for numeric inputs
            let valid = match &setting.def.input_type {
                InputType::IntNumber { .. } | InputType::NullableInt { .. } => {
                    setting.is_valid_numeric_char(c, false)
                }
                InputType::FloatNumber { .. } => setting.is_valid_numeric_char(c, true),
                _ => true,
            };

            if valid {
                setting.insert_char_at_cursor(c);
                setting.is_editing = true;
            }
        }
    }

    /// Delete character from search result (backspace) - UTF-8 safe
    pub fn search_result_delete_char(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            if setting.delete_char_before_cursor() {
                setting.is_editing = true;
            }
        }
    }

    /// Delete character forward from search result - UTF-8 safe
    pub fn search_result_delete_char_forward(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            if setting.delete_char_at_cursor() {
                setting.is_editing = true;
            }
        }
    }

    /// Move cursor left in search result text input
    pub fn search_result_cursor_left(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            setting.move_cursor_left();
        }
    }

    /// Move cursor right in search result text input
    pub fn search_result_cursor_right(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            setting.move_cursor_right();
        }
    }

    /// Move cursor to start in search result text input
    pub fn search_result_cursor_to_start(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            setting.cursor_to_start();
        }
    }

    /// Move cursor to end in search result text input
    pub fn search_result_cursor_to_end(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            setting.cursor_to_end();
        }
    }

    /// Commit the current search result edit and save to file
    pub fn commit_search_result_edit(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        // Mark as no longer editing
        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            setting.is_editing = false;
        }

        self.save_search_result_field();
    }

    /// Save the currently selected search result field to file
    fn save_search_result_field(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        // Extract info needed for saving before mutable borrow
        let key: String;
        let label: String;
        let new_value: Value;

        {
            let Some(setting) = self
                .values
                .get_mut(result.tab_idx)
                .and_then(|tab| tab.get_mut(result.field_idx))
            else {
                return;
            };

            // Use shared validation logic
            match validate_and_convert(&setting.input_buffer, &setting.def.input_type) {
                ValidationResult::Valid(value) => {
                    setting.validation_error = None;
                    key = setting.def.key.to_string();
                    label = setting.def.label.to_string();
                    new_value = value;
                    setting.current_value = new_value.clone();
                }
                ValidationResult::Invalid(err) => {
                    setting.validation_error = Some(err);
                    return;
                }
            }
        }

        // Save to file
        if let Err(e) = self.save_to_file(&key, &new_value) {
            self.status = Some(format!("Save error: {}", e));
        } else {
            self.status = Some(format!("Saved: {}", label));

            // Update ollama_base_url if that's what changed
            if key == "ollama.base_url" {
                if let Some(url) = new_value.as_str() {
                    self.ollama_base_url = url.to_string();
                }
            }
        }
    }

    pub fn current_tab_values(&self) -> &Vec<SettingValue> {
        &self.values[self.current_tab]
    }

    pub fn current_field(&self) -> Option<&SettingValue> {
        if let Focus::FormField(idx) = self.focus {
            self.values[self.current_tab].get(idx)
        } else {
            None
        }
    }

    pub fn prev_tab(&mut self) {
        self.current_tab = if self.current_tab == 0 {
            self.tabs.len() - 1
        } else {
            self.current_tab - 1
        };
        self.form_scroll = 0;
        self.clear_editing();
        // Set appropriate focus based on tab
        if self.is_search_tab() {
            self.focus = Focus::SearchInput;
        } else {
            self.focus = Focus::FormField(0);
        }
    }

    pub fn next_field(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let max_idx = self.values[self.current_tab].len().saturating_sub(1);
            if idx < max_idx {
                self.commit_current_edit();
                self.focus = Focus::FormField(idx + 1);
                self.ensure_field_visible(idx + 1);
            }
        }
    }

    pub fn ensure_field_visible(&mut self, field_idx: usize) {
        let lines_per_field = 3;
        let field_line = field_idx * lines_per_field;

        if field_line < self.form_scroll {
            self.form_scroll = field_line;
        } else if field_line >= self.form_scroll + self.visible_height {
            self.form_scroll = field_line.saturating_sub(self.visible_height) + lines_per_field;
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.form_scroll = self.form_scroll.saturating_sub(amount);
    }

    pub fn scroll_down(&mut self, amount: usize) {
        let max_scroll = self.values[self.current_tab].len() * 3;
        self.form_scroll = (self.form_scroll + amount).min(max_scroll);
    }

    fn clear_editing(&mut self) {
        for tab_values in &mut self.values {
            for setting in tab_values {
                setting.is_editing = false;
            }
        }
    }

    pub fn start_editing(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.is_editing = true;
            setting.cursor_to_end();
        }
    }

    pub fn commit_current_edit(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            if setting.is_editing {
                setting.is_editing = false;
                self.save_field(idx);
            }
        }
    }

    pub fn insert_char(&mut self, c: char) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];

            // Validate character for numeric inputs
            let valid = match &setting.def.input_type {
                InputType::IntNumber { .. } | InputType::NullableInt { .. } => {
                    setting.is_valid_numeric_char(c, false)
                }
                InputType::FloatNumber { .. } => setting.is_valid_numeric_char(c, true),
                _ => true,
            };

            if valid {
                setting.insert_char_at_cursor(c);
                setting.is_editing = true;
            }
        }
    }

    pub fn delete_char(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            if setting.delete_char_before_cursor() {
                setting.is_editing = true;
            }
        }
    }

    pub fn delete_char_forward(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            if setting.delete_char_at_cursor() {
                setting.is_editing = true;
            }
        }
    }

    pub fn move_cursor_left(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.move_cursor_left();
        }
    }

    pub fn move_cursor_right(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.move_cursor_right();
        }
    }

    pub fn move_cursor_to_start(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.cursor_to_start();
        }
    }

    pub fn move_cursor_to_end(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.cursor_to_end();
        }
    }

    pub fn select_next(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            let options = setting.get_options();
            if !options.is_empty() {
                setting.select_index = (setting.select_index + 1) % options.len();
                setting.input_buffer = options[setting.select_index].clone();
                self.save_field(idx);
            }
        }
    }

    pub fn select_prev(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            let options = setting.get_options();
            if !options.is_empty() {
                setting.select_index = if setting.select_index == 0 {
                    options.len() - 1
                } else {
                    setting.select_index - 1
                };
                setting.input_buffer = options[setting.select_index].clone();
                self.save_field(idx);
            }
        }
    }

    pub fn toggle(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            if matches!(setting.def.input_type, InputType::Toggle) {
                let current = setting.input_buffer == "true";
                setting.input_buffer = if current { "false" } else { "true" }.to_string();
                setting.select_index = if current { 1 } else { 0 };
                self.save_field(idx);
            }
        }
    }

    pub fn refresh_dynamic_options(&mut self) {
        // Handle search mode: refresh the selected search result's dynamic options
        if self.is_search_tab() {
            if let Some(result) = self.search_results.get(self.search_selected).cloned() {
                if let Some(setting) = self
                    .values
                    .get(result.tab_idx)
                    .and_then(|tab| tab.get(result.field_idx))
                {
                    if let InputType::DynamicSelect(source) = &setting.def.input_type {
                        self.status = Some("Fetching options...".into());

                        let fetch_result = match source {
                            DynamicSource::OllamaModels => {
                                api::fetch_ollama_models(&self.ollama_base_url)
                            }
                            DynamicSource::TTSModels => api::fetch_tts_models(),
                            DynamicSource::TTSSpeakers => api::fetch_tts_speakers(),
                        };

                        match fetch_result {
                            Ok(options) => {
                                let count = options.len();
                                if let Some(setting) = self
                                    .values
                                    .get_mut(result.tab_idx)
                                    .and_then(|tab| tab.get_mut(result.field_idx))
                                {
                                    // Update select index to match current value
                                    if let Some(pos) =
                                        options.iter().position(|o| o == &setting.input_buffer)
                                    {
                                        setting.select_index = pos;
                                    } else if !options.is_empty() {
                                        setting.select_index = 0;
                                    }
                                    setting.dynamic_options = options;
                                }
                                self.status = Some(format!("Found {} models", count));
                            }
                            Err(e) => {
                                self.status = Some(format!("Error fetching: {}", e));
                            }
                        }
                        return;
                    }
                }
            }
            self.status = Some("No dynamic field selected".into());
            return;
        }

        // Handle regular tab mode
        if let Focus::FormField(idx) = self.focus {
            let setting = &self.values[self.current_tab][idx];

            if let InputType::DynamicSelect(source) = &setting.def.input_type {
                self.status = Some("Fetching options...".into());

                let result = match source {
                    DynamicSource::OllamaModels => api::fetch_ollama_models(&self.ollama_base_url),
                    DynamicSource::TTSModels => api::fetch_tts_models(),
                    DynamicSource::TTSSpeakers => api::fetch_tts_speakers(),
                };

                match result {
                    Ok(options) => {
                        let setting = &mut self.values[self.current_tab][idx];
                        setting.dynamic_options = options;
                        // Update select index to match current value
                        if let Some(pos) = setting
                            .dynamic_options
                            .iter()
                            .position(|o| o == &setting.input_buffer)
                        {
                            setting.select_index = pos;
                        } else if !setting.dynamic_options.is_empty() {
                            setting.select_index = 0;
                        }
                        self.status = Some(format!(
                            "Found {} models",
                            self.values[self.current_tab][idx].dynamic_options.len()
                        ));
                    }
                    Err(e) => {
                        self.status = Some(format!("Error fetching: {}", e));
                    }
                }
            }
        }
    }

    /// Refresh all dynamic options across all tabs (called on startup)
    pub fn refresh_all_dynamic_options(&mut self) {
        for tab_idx in 0..self.values.len() {
            for field_idx in 0..self.values[tab_idx].len() {
                let setting = &self.values[tab_idx][field_idx];

                if let InputType::DynamicSelect(source) = &setting.def.input_type {
                    let result = match source {
                        DynamicSource::OllamaModels => {
                            api::fetch_ollama_models(&self.ollama_base_url)
                        }
                        DynamicSource::TTSModels => api::fetch_tts_models(),
                        DynamicSource::TTSSpeakers => api::fetch_tts_speakers(),
                    };

                    if let Ok(options) = result {
                        let setting = &mut self.values[tab_idx][field_idx];
                        // Update select index to match current value
                        if let Some(pos) = options.iter().position(|o| o == &setting.input_buffer) {
                            setting.select_index = pos;
                        } else if !options.is_empty() {
                            setting.select_index = 0;
                        }
                        setting.dynamic_options = options;
                    }
                }
            }
        }
    }

    pub fn save_field(&mut self, field_idx: usize) {
        // Extract info needed for saving before mutable borrow
        let key: String;
        let label: String;
        let new_value: Value;

        {
            let setting = &mut self.values[self.current_tab][field_idx];

            // Use shared validation logic
            match validate_and_convert(&setting.input_buffer, &setting.def.input_type) {
                ValidationResult::Valid(value) => {
                    setting.validation_error = None;
                    key = setting.def.key.to_string();
                    label = setting.def.label.to_string();
                    new_value = value;
                    setting.current_value = new_value.clone();
                }
                ValidationResult::Invalid(err) => {
                    setting.validation_error = Some(err);
                    return;
                }
            }
        }

        // Save to file (outside the borrow scope)
        if let Err(e) = self.save_to_file(&key, &new_value) {
            self.status = Some(format!("Save error: {}", e));
        } else {
            self.status = Some(format!("Saved: {}", label));

            // Update ollama_base_url if that's what changed
            if key == "ollama.base_url" {
                if let Some(url) = new_value.as_str() {
                    self.ollama_base_url = url.to_string();
                }
            }
        }
    }

    fn save_to_file(&self, key: &str, value: &Value) -> Result<()> {
        let content = fs::read_to_string(&self.config_path)?;
        let mut json: Value = serde_json::from_str(&content)?;

        set_nested_value(&mut json, key, value.clone());

        let formatted = serde_json::to_string_pretty(&json)?;
        fs::write(&self.config_path, formatted)?;

        Ok(())
    }
}

fn load_json(path: &PathBuf) -> Result<Value> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

fn get_nested_value(json: &Value, path: &str) -> Option<Value> {
    let keys: Vec<&str> = path.split('.').collect();
    let mut current = json;

    for key in keys {
        current = current.get(key)?;
    }

    Some(current.clone())
}

fn set_nested_value(json: &mut Value, path: &str, value: Value) {
    let keys: Vec<&str> = path.split('.').collect();
    let mut current = json;

    for (i, key) in keys.iter().enumerate() {
        if i == keys.len() - 1 {
            if let Value::Object(obj) = current {
                obj.insert((*key).to_string(), value);
                return;
            }
        } else if let Value::Object(obj) = current {
            if !obj.contains_key(*key) {
                obj.insert((*key).to_string(), Value::Object(serde_json::Map::new()));
            }
            current = obj.get_mut(*key).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_setting(input_buffer: &str) -> SettingValue {
        let def = SettingDef {
            key: "test.key",
            label: "Test",
            description: "Test setting",
            input_type: InputType::Text,
            default_value: None,
        };
        let mut sv = SettingValue::new(def, json!(""));
        sv.input_buffer = input_buffer.to_string();
        sv.cursor_pos = sv.input_buffer.chars().count();
        sv
    }

    #[test]
    fn test_setting_value_char_count() {
        let sv = create_test_setting("hello");
        assert_eq!(sv.char_count(), 5);

        let sv = create_test_setting("héllo"); // é is single char
        assert_eq!(sv.char_count(), 5);

        let sv = create_test_setting("日本語"); // 3 Japanese characters
        assert_eq!(sv.char_count(), 3);
    }

    #[test]
    fn test_setting_value_split_at_cursor_ascii() {
        let mut sv = create_test_setting("hello");
        sv.cursor_pos = 2;
        let (before, cursor, after) = sv.split_at_cursor();
        assert_eq!(before, "he");
        assert_eq!(cursor, 'l');
        assert_eq!(after, "lo");
    }

    #[test]
    fn test_setting_value_split_at_cursor_unicode() {
        let mut sv = create_test_setting("héllo");
        sv.cursor_pos = 1;
        let (before, cursor, after) = sv.split_at_cursor();
        assert_eq!(before, "h");
        assert_eq!(cursor, 'é');
        assert_eq!(after, "llo");
    }

    #[test]
    fn test_setting_value_split_at_cursor_japanese() {
        let mut sv = create_test_setting("日本語");
        sv.cursor_pos = 1;
        let (before, cursor, after) = sv.split_at_cursor();
        assert_eq!(before, "日");
        assert_eq!(cursor, '本');
        assert_eq!(after, "語");
    }

    #[test]
    fn test_setting_value_split_at_cursor_end() {
        let mut sv = create_test_setting("hello");
        sv.cursor_pos = 5;
        let (before, cursor, after) = sv.split_at_cursor();
        assert_eq!(before, "hello");
        assert_eq!(cursor, ' '); // Space when at end
        assert_eq!(after, "");
    }

    #[test]
    fn test_setting_value_insert_char_at_cursor() {
        let mut sv = create_test_setting("helo");
        sv.cursor_pos = 3;
        sv.insert_char_at_cursor('l');
        assert_eq!(sv.input_buffer, "hello");
        assert_eq!(sv.cursor_pos, 4);
    }

    #[test]
    fn test_setting_value_insert_unicode_char() {
        let mut sv = create_test_setting("hello");
        sv.cursor_pos = 1;
        sv.insert_char_at_cursor('é');
        assert_eq!(sv.input_buffer, "héello");
        assert_eq!(sv.cursor_pos, 2);
    }

    #[test]
    fn test_setting_value_delete_char_before_cursor() {
        let mut sv = create_test_setting("hello");
        sv.cursor_pos = 3;
        assert!(sv.delete_char_before_cursor());
        assert_eq!(sv.input_buffer, "helo");
        assert_eq!(sv.cursor_pos, 2);
    }

    #[test]
    fn test_setting_value_delete_unicode_char_before_cursor() {
        let mut sv = create_test_setting("héllo");
        sv.cursor_pos = 2;
        assert!(sv.delete_char_before_cursor());
        assert_eq!(sv.input_buffer, "hllo");
        assert_eq!(sv.cursor_pos, 1);
    }

    #[test]
    fn test_setting_value_delete_char_at_cursor() {
        let mut sv = create_test_setting("hello");
        sv.cursor_pos = 2;
        assert!(sv.delete_char_at_cursor());
        assert_eq!(sv.input_buffer, "helo");
        assert_eq!(sv.cursor_pos, 2); // Cursor stays
    }

    #[test]
    fn test_setting_value_delete_unicode_char_at_cursor() {
        let mut sv = create_test_setting("héllo");
        sv.cursor_pos = 1;
        assert!(sv.delete_char_at_cursor());
        assert_eq!(sv.input_buffer, "hllo");
        assert_eq!(sv.cursor_pos, 1);
    }

    #[test]
    fn test_setting_value_cursor_movement() {
        let mut sv = create_test_setting("hello");
        sv.cursor_pos = 2;

        sv.move_cursor_left();
        assert_eq!(sv.cursor_pos, 1);

        sv.move_cursor_left();
        assert_eq!(sv.cursor_pos, 0);

        sv.move_cursor_left(); // Should not go negative
        assert_eq!(sv.cursor_pos, 0);

        sv.move_cursor_right();
        assert_eq!(sv.cursor_pos, 1);

        sv.cursor_to_end();
        assert_eq!(sv.cursor_pos, 5);

        sv.cursor_to_start();
        assert_eq!(sv.cursor_pos, 0);
    }

    #[test]
    fn test_setting_value_cursor_movement_unicode() {
        let mut sv = create_test_setting("日本語");
        sv.cursor_pos = 1;

        sv.move_cursor_right();
        assert_eq!(sv.cursor_pos, 2);

        sv.move_cursor_left();
        assert_eq!(sv.cursor_pos, 1);

        sv.cursor_to_end();
        assert_eq!(sv.cursor_pos, 3);
    }

    #[test]
    fn test_setting_value_numeric_validation() {
        let def = SettingDef {
            key: "test.num",
            label: "Number",
            description: "A number",
            input_type: InputType::IntNumber {
                min: Some(0),
                max: Some(100),
            },
            default_value: None,
        };
        let mut sv = SettingValue::new(def, json!(0));
        sv.input_buffer = "".to_string();
        sv.cursor_pos = 0;

        // Digits should be valid
        assert!(sv.is_valid_numeric_char('5', false));

        // Minus at start should be valid
        assert!(sv.is_valid_numeric_char('-', false));

        // Letters should be invalid
        assert!(!sv.is_valid_numeric_char('a', false));

        // Dot should be invalid for int
        assert!(!sv.is_valid_numeric_char('.', false));

        // Dot should be valid for float
        assert!(sv.is_valid_numeric_char('.', true));
    }

    #[test]
    fn test_get_nested_value() {
        let json = json!({
            "level1": {
                "level2": {
                    "value": "test"
                }
            }
        });

        let result = get_nested_value(&json, "level1.level2.value");
        assert_eq!(result, Some(json!("test")));

        let result = get_nested_value(&json, "nonexistent");
        assert_eq!(result, None);
    }

    #[test]
    fn test_set_nested_value() {
        let mut json = json!({
            "level1": {}
        });

        set_nested_value(&mut json, "level1.level2.value", json!("test"));

        let result = get_nested_value(&json, "level1.level2.value");
        assert_eq!(result, Some(json!("test")));
    }
}
