//! Application state and logic

use crate::api;
use crate::settings::{DynamicSource, InputType, SettingDef, TabDef, create_tab_definitions};
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

        Self {
            def,
            current_value: value,
            cursor_pos: input_buffer.len(),
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

    /// Insert character into search query
    pub fn search_insert_char(&mut self, c: char) {
        self.search_query.insert(self.search_cursor_pos, c);
        self.search_cursor_pos += 1;
        self.perform_search();
    }

    /// Delete character from search query
    pub fn search_delete_char(&mut self) {
        if self.search_cursor_pos > 0 {
            self.search_cursor_pos -= 1;
            self.search_query.remove(self.search_cursor_pos);
            self.perform_search();
        }
    }

    /// Move search cursor left
    pub fn search_cursor_left(&mut self) {
        self.search_cursor_pos = self.search_cursor_pos.saturating_sub(1);
    }

    /// Move search cursor right
    pub fn search_cursor_right(&mut self) {
        self.search_cursor_pos = (self.search_cursor_pos + 1).min(self.search_query.len());
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

    /// Insert character into search result text input
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
                    c.is_ascii_digit()
                        || (c == '-'
                            && setting.cursor_pos == 0
                            && !setting.input_buffer.contains('-'))
                }
                InputType::FloatNumber { .. } => {
                    c.is_ascii_digit()
                        || (c == '-'
                            && setting.cursor_pos == 0
                            && !setting.input_buffer.contains('-'))
                        || (c == '.' && !setting.input_buffer.contains('.'))
                }
                _ => true,
            };

            if valid {
                setting.input_buffer.insert(setting.cursor_pos, c);
                setting.cursor_pos += 1;
                setting.is_editing = true;
            }
        }
    }

    /// Delete character from search result (backspace)
    pub fn search_result_delete_char(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            if setting.cursor_pos > 0 {
                setting.cursor_pos -= 1;
                setting.input_buffer.remove(setting.cursor_pos);
                setting.is_editing = true;
            }
        }
    }

    /// Delete character forward from search result
    pub fn search_result_delete_char_forward(&mut self) {
        let Some(result) = self.search_results.get(self.search_selected).cloned() else {
            return;
        };

        if let Some(setting) = self
            .values
            .get_mut(result.tab_idx)
            .and_then(|tab| tab.get_mut(result.field_idx))
        {
            if setting.cursor_pos < setting.input_buffer.len() {
                setting.input_buffer.remove(setting.cursor_pos);
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
            setting.cursor_pos = setting.cursor_pos.saturating_sub(1);
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
            setting.cursor_pos = (setting.cursor_pos + 1).min(setting.input_buffer.len());
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

            // Convert input buffer to proper JSON value
            let result = match &setting.def.input_type {
                InputType::Text => Some(Value::String(setting.input_buffer.clone())),
                InputType::Textarea => {
                    let lines: Vec<Value> = setting
                        .input_buffer
                        .lines()
                        .map(|l| Value::String(l.to_string()))
                        .collect();
                    Some(Value::Array(lines))
                }
                InputType::Select(_) | InputType::DynamicSelect(_) => {
                    Some(Value::String(setting.input_buffer.clone()))
                }
                InputType::IntNumber { min, max } => {
                    if let Ok(n) = setting.input_buffer.parse::<i64>() {
                        if let Some(min_val) = min {
                            if n < *min_val {
                                setting.validation_error = Some(format!("Min: {}", min_val));
                                None
                            } else if let Some(max_val) = max {
                                if n > *max_val {
                                    setting.validation_error = Some(format!("Max: {}", max_val));
                                    None
                                } else {
                                    setting.validation_error = None;
                                    Some(Value::Number(n.into()))
                                }
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else if let Some(max_val) = max {
                            if n > *max_val {
                                setting.validation_error = Some(format!("Max: {}", max_val));
                                None
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else {
                            setting.validation_error = None;
                            Some(Value::Number(n.into()))
                        }
                    } else {
                        setting.validation_error = Some("Invalid number".into());
                        None
                    }
                }
                InputType::FloatNumber { min, max } => {
                    if let Ok(n) = setting.input_buffer.parse::<f64>() {
                        if let Some(min_val) = min {
                            if n < *min_val {
                                setting.validation_error = Some(format!("Min: {:.1}", min_val));
                                None
                            } else if let Some(max_val) = max {
                                if n > *max_val {
                                    setting.validation_error = Some(format!("Max: {:.1}", max_val));
                                    None
                                } else {
                                    setting.validation_error = None;
                                    serde_json::Number::from_f64(n).map(Value::Number)
                                }
                            } else {
                                setting.validation_error = None;
                                serde_json::Number::from_f64(n).map(Value::Number)
                            }
                        } else if let Some(max_val) = max {
                            if n > *max_val {
                                setting.validation_error = Some(format!("Max: {:.1}", max_val));
                                None
                            } else {
                                setting.validation_error = None;
                                serde_json::Number::from_f64(n).map(Value::Number)
                            }
                        } else {
                            setting.validation_error = None;
                            serde_json::Number::from_f64(n).map(Value::Number)
                        }
                    } else {
                        setting.validation_error = Some("Invalid number".into());
                        None
                    }
                }
                InputType::Toggle => Some(Value::Bool(setting.input_buffer == "true")),
                InputType::NullableText => {
                    if setting.input_buffer.is_empty() {
                        Some(Value::Null)
                    } else {
                        Some(Value::String(setting.input_buffer.clone()))
                    }
                }
                InputType::NullableInt { min, max } => {
                    if setting.input_buffer.is_empty() {
                        setting.validation_error = None;
                        Some(Value::Null)
                    } else if let Ok(n) = setting.input_buffer.parse::<i64>() {
                        if let Some(min_val) = min {
                            if n < *min_val {
                                setting.validation_error = Some(format!("Min: {}", min_val));
                                None
                            } else if let Some(max_val) = max {
                                if n > *max_val {
                                    setting.validation_error = Some(format!("Max: {}", max_val));
                                    None
                                } else {
                                    setting.validation_error = None;
                                    Some(Value::Number(n.into()))
                                }
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else if let Some(max_val) = max {
                            if n > *max_val {
                                setting.validation_error = Some(format!("Max: {}", max_val));
                                None
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else {
                            setting.validation_error = None;
                            Some(Value::Number(n.into()))
                        }
                    } else {
                        setting.validation_error = Some("Invalid number".into());
                        None
                    }
                }
            };

            // If validation failed, return early
            let Some(value) = result else {
                return;
            };

            key = setting.def.key.to_string();
            label = setting.def.label.to_string();
            new_value = value;
            setting.current_value = new_value.clone();
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
            setting.cursor_pos = setting.input_buffer.len();
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
                    c.is_ascii_digit()
                        || (c == '-'
                            && setting.cursor_pos == 0
                            && !setting.input_buffer.contains('-'))
                }
                InputType::FloatNumber { .. } => {
                    c.is_ascii_digit()
                        || (c == '-'
                            && setting.cursor_pos == 0
                            && !setting.input_buffer.contains('-'))
                        || (c == '.' && !setting.input_buffer.contains('.'))
                }
                _ => true,
            };

            if valid {
                setting.input_buffer.insert(setting.cursor_pos, c);
                setting.cursor_pos += 1;
                setting.is_editing = true;
            }
        }
    }

    pub fn delete_char(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            if setting.cursor_pos > 0 {
                setting.cursor_pos -= 1;
                setting.input_buffer.remove(setting.cursor_pos);
                setting.is_editing = true;
            }
        }
    }

    pub fn delete_char_forward(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            if setting.cursor_pos < setting.input_buffer.len() {
                setting.input_buffer.remove(setting.cursor_pos);
                setting.is_editing = true;
            }
        }
    }

    pub fn move_cursor_left(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.cursor_pos = setting.cursor_pos.saturating_sub(1);
        }
    }

    pub fn move_cursor_right(&mut self) {
        if let Focus::FormField(idx) = self.focus {
            let setting = &mut self.values[self.current_tab][idx];
            setting.cursor_pos = (setting.cursor_pos + 1).min(setting.input_buffer.len());
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

            // Convert input buffer to proper JSON value
            let result = match &setting.def.input_type {
                InputType::Text => Some(Value::String(setting.input_buffer.clone())),
                InputType::Textarea => {
                    let lines: Vec<Value> = setting
                        .input_buffer
                        .lines()
                        .map(|l| Value::String(l.to_string()))
                        .collect();
                    Some(Value::Array(lines))
                }
                InputType::Select(_) | InputType::DynamicSelect(_) => {
                    Some(Value::String(setting.input_buffer.clone()))
                }
                InputType::IntNumber { min, max } => {
                    if let Ok(n) = setting.input_buffer.parse::<i64>() {
                        if let Some(min_val) = min {
                            if n < *min_val {
                                setting.validation_error = Some(format!("Min: {}", min_val));
                                None
                            } else if let Some(max_val) = max {
                                if n > *max_val {
                                    setting.validation_error = Some(format!("Max: {}", max_val));
                                    None
                                } else {
                                    setting.validation_error = None;
                                    Some(Value::Number(n.into()))
                                }
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else if let Some(max_val) = max {
                            if n > *max_val {
                                setting.validation_error = Some(format!("Max: {}", max_val));
                                None
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else {
                            setting.validation_error = None;
                            Some(Value::Number(n.into()))
                        }
                    } else {
                        setting.validation_error = Some("Invalid number".into());
                        None
                    }
                }
                InputType::FloatNumber { min, max } => {
                    if let Ok(n) = setting.input_buffer.parse::<f64>() {
                        if let Some(min_val) = min {
                            if n < *min_val {
                                setting.validation_error = Some(format!("Min: {:.1}", min_val));
                                None
                            } else if let Some(max_val) = max {
                                if n > *max_val {
                                    setting.validation_error = Some(format!("Max: {:.1}", max_val));
                                    None
                                } else {
                                    setting.validation_error = None;
                                    serde_json::Number::from_f64(n).map(Value::Number)
                                }
                            } else {
                                setting.validation_error = None;
                                serde_json::Number::from_f64(n).map(Value::Number)
                            }
                        } else if let Some(max_val) = max {
                            if n > *max_val {
                                setting.validation_error = Some(format!("Max: {:.1}", max_val));
                                None
                            } else {
                                setting.validation_error = None;
                                serde_json::Number::from_f64(n).map(Value::Number)
                            }
                        } else {
                            setting.validation_error = None;
                            serde_json::Number::from_f64(n).map(Value::Number)
                        }
                    } else {
                        setting.validation_error = Some("Invalid number".into());
                        None
                    }
                }
                InputType::Toggle => Some(Value::Bool(setting.input_buffer == "true")),
                InputType::NullableText => {
                    if setting.input_buffer.is_empty() {
                        Some(Value::Null)
                    } else {
                        Some(Value::String(setting.input_buffer.clone()))
                    }
                }
                InputType::NullableInt { min, max } => {
                    if setting.input_buffer.is_empty() {
                        setting.validation_error = None;
                        Some(Value::Null)
                    } else if let Ok(n) = setting.input_buffer.parse::<i64>() {
                        if let Some(min_val) = min {
                            if n < *min_val {
                                setting.validation_error = Some(format!("Min: {}", min_val));
                                None
                            } else if let Some(max_val) = max {
                                if n > *max_val {
                                    setting.validation_error = Some(format!("Max: {}", max_val));
                                    None
                                } else {
                                    setting.validation_error = None;
                                    Some(Value::Number(n.into()))
                                }
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else if let Some(max_val) = max {
                            if n > *max_val {
                                setting.validation_error = Some(format!("Max: {}", max_val));
                                None
                            } else {
                                setting.validation_error = None;
                                Some(Value::Number(n.into()))
                            }
                        } else {
                            setting.validation_error = None;
                            Some(Value::Number(n.into()))
                        }
                    } else {
                        setting.validation_error = Some("Invalid number".into());
                        None
                    }
                }
            };

            // If validation failed, return early
            let Some(value) = result else {
                return;
            };

            key = setting.def.key.to_string();
            label = setting.def.label.to_string();
            new_value = value;
            setting.current_value = new_value.clone();
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
