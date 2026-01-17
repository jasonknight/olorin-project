//! Input handling for keyboard events
//!
//! This module separates keystroke handling from the main event loop,
//! making it easier to test and maintain.

use crate::app::{App, Focus};
use crate::settings::InputType;
use crossterm::event::{KeyCode, KeyModifiers};

/// Result of handling a key event
#[derive(Debug, Clone, PartialEq)]
pub enum KeyAction {
    /// No action needed
    None,
    /// Application should quit
    Quit,
    /// UI needs to be redrawn (implied for most actions)
    Redraw,
}

/// Handle a key event and return the action to take
pub fn handle_key_event(app: &mut App, code: KeyCode, modifiers: KeyModifiers) -> KeyAction {
    // Global keys that work regardless of focus
    if let Some(action) = handle_global_keys(app, code) {
        return action;
    }

    // Shift+Tab always changes tabs
    if code == KeyCode::BackTab {
        app.prev_tab();
        return KeyAction::Redraw;
    }

    // Tab navigates fields/results
    if code == KeyCode::Tab && modifiers == KeyModifiers::NONE {
        handle_tab_key(app);
        return KeyAction::Redraw;
    }

    // Focus-specific handling
    match app.focus {
        Focus::SearchInput => {
            handle_search_input_keys(app, code);
        }
        Focus::FormField(_) => {
            if app.is_search_tab() {
                handle_search_result_keys(app, code, modifiers);
            } else {
                handle_field_keys(app, code, modifiers);
            }
        }
    }

    KeyAction::Redraw
}

/// Handle global keys that work regardless of focus
fn handle_global_keys(app: &mut App, code: KeyCode) -> Option<KeyAction> {
    match code {
        KeyCode::Esc => {
            app.should_quit = true;
            Some(KeyAction::Quit)
        }
        KeyCode::F(5) => {
            app.refresh_dynamic_options();
            Some(KeyAction::Redraw)
        }
        KeyCode::PageUp => {
            app.scroll_up(10);
            Some(KeyAction::Redraw)
        }
        KeyCode::PageDown => {
            app.scroll_down(10);
            Some(KeyAction::Redraw)
        }
        // Explicitly ignore Insert key (no insert/overwrite mode in this app)
        KeyCode::Insert => Some(KeyAction::None),
        _ => None,
    }
}

/// Handle Tab key navigation
fn handle_tab_key(app: &mut App) {
    match app.focus {
        Focus::SearchInput => {
            // From search input, go to first search result (if any)
            if !app.search_results.is_empty() {
                app.focus = Focus::FormField(0);
                app.form_scroll = 0;
            }
        }
        Focus::FormField(idx) => {
            // Move to next field (with cycling)
            let field_count = if app.is_search_tab() {
                app.search_results.len()
            } else {
                app.values[app.current_tab].len()
            };

            if field_count > 0 {
                // Commit edit using appropriate method for tab type
                if app.is_search_tab() {
                    app.commit_search_result_edit();
                    if idx + 1 < field_count {
                        app.search_selected = idx + 1;
                        app.focus = Focus::FormField(idx + 1);
                    } else {
                        // Cycle back to search input
                        app.focus = Focus::SearchInput;
                    }
                    app.ensure_search_result_visible();
                } else {
                    app.commit_current_edit();
                    if idx + 1 < field_count {
                        app.focus = Focus::FormField(idx + 1);
                        app.ensure_field_visible(idx + 1);
                    } else {
                        // Cycle back to first field
                        app.focus = Focus::FormField(0);
                        app.form_scroll = 0;
                    }
                }
            }
        }
    }
}

/// Handle keys when search input is focused
fn handle_search_input_keys(app: &mut App, code: KeyCode) {
    match code {
        // Text input
        KeyCode::Char(c) => {
            app.search_insert_char(c);
        }
        KeyCode::Backspace => {
            app.search_delete_char();
        }
        KeyCode::Left => {
            app.search_cursor_left();
        }
        KeyCode::Right => {
            app.search_cursor_right();
        }
        KeyCode::Home => {
            app.search_cursor_to_start();
        }
        KeyCode::End => {
            app.search_cursor_to_end();
        }
        // Enter/Down: go to first result
        KeyCode::Enter | KeyCode::Down => {
            if !app.search_results.is_empty() {
                app.focus = Focus::FormField(0);
                app.search_selected = 0;
                app.form_scroll = 0;
            }
        }
        _ => {}
    }
}

/// Handle keys when focused on a search result (editable in-place)
fn handle_search_result_keys(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
    // Get the input type of the currently selected search result
    let input_type = app.get_search_result_input_type();

    let is_select = matches!(
        input_type,
        Some(InputType::Select(_)) | Some(InputType::DynamicSelect(_)) | Some(InputType::Toggle)
    );

    match code {
        // Up/Down on selects: change value
        KeyCode::Up if is_select => {
            if matches!(input_type, Some(InputType::Toggle)) {
                app.toggle_search_result();
            } else {
                app.search_result_select_prev();
            }
        }
        KeyCode::Down if is_select => {
            if matches!(input_type, Some(InputType::Toggle)) {
                app.toggle_search_result();
            } else {
                app.search_result_select_next();
            }
        }

        // Space toggles boolean
        KeyCode::Char(' ') if matches!(input_type, Some(InputType::Toggle)) => {
            app.toggle_search_result();
        }

        // Text input handling (for non-select types)
        KeyCode::Char(c) if !is_select => {
            app.search_result_insert_char(c);
        }

        KeyCode::Backspace if !is_select => {
            app.search_result_delete_char();
        }

        KeyCode::Delete if !is_select => {
            app.search_result_delete_char_forward();
        }

        KeyCode::Left if !is_select => {
            app.search_result_cursor_left();
        }

        KeyCode::Right if !is_select => {
            app.search_result_cursor_right();
        }

        KeyCode::Home if !is_select => {
            app.search_result_cursor_to_start();
        }

        KeyCode::End if !is_select => {
            app.search_result_cursor_to_end();
        }

        // Shift+Enter for newline in textarea (must come before plain Enter)
        KeyCode::Enter if modifiers.contains(KeyModifiers::SHIFT) => {
            if matches!(input_type, Some(InputType::Textarea)) {
                app.search_result_insert_char('\n');
            }
        }

        // Enter: save and stay (or move to next for selects)
        KeyCode::Enter => {
            app.commit_search_result_edit();
        }

        // Up without select: go back to search input
        KeyCode::Up if !is_select => {
            if let Focus::FormField(0) = app.focus {
                app.focus = Focus::SearchInput;
            }
        }

        _ => {}
    }
}

/// Handle keys when focused on a regular form field
fn handle_field_keys(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
    let Some(setting) = app.current_field() else {
        return;
    };

    let input_type = setting.def.input_type.clone();
    let is_select = matches!(
        input_type,
        InputType::Select(_) | InputType::DynamicSelect(_) | InputType::Toggle
    );

    match code {
        // Up/Down on selects: change value (not navigate)
        KeyCode::Up if is_select => {
            if matches!(input_type, InputType::Toggle) {
                app.toggle();
            } else {
                app.select_prev();
            }
        }
        KeyCode::Down if is_select => {
            if matches!(input_type, InputType::Toggle) {
                app.toggle();
            } else {
                app.select_next();
            }
        }

        // Toggle with space
        KeyCode::Char(' ') if matches!(input_type, InputType::Toggle) => {
            app.toggle();
        }

        // Enter for selects: confirm and move to next
        KeyCode::Enter if is_select => {
            app.next_field();
        }

        // Text input handling
        KeyCode::Char(c) if !is_select => {
            // Start editing on first character
            if !app.current_field().map(|f| f.is_editing).unwrap_or(false) {
                app.start_editing();
            }
            app.insert_char(c);
        }

        KeyCode::Backspace if !is_select => {
            app.delete_char();
        }

        KeyCode::Delete if !is_select => {
            app.delete_char_forward();
        }

        KeyCode::Left if !is_select => {
            app.move_cursor_left();
        }

        KeyCode::Right if !is_select => {
            app.move_cursor_right();
        }

        KeyCode::Home if !is_select => {
            app.move_cursor_to_start();
        }

        KeyCode::End if !is_select => {
            app.move_cursor_to_end();
        }

        // Shift+Enter for newline in textarea (must come before plain Enter)
        KeyCode::Enter if modifiers.contains(KeyModifiers::SHIFT) => {
            if matches!(input_type, InputType::Textarea) {
                app.insert_char('\n');
            }
        }

        // Enter to confirm text input
        KeyCode::Enter if !is_select => {
            app.commit_current_edit();
            app.next_field();
        }

        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require mocking the App struct
    // These tests verify the KeyAction enum and basic logic

    #[test]
    fn test_key_action_equality() {
        assert_eq!(KeyAction::None, KeyAction::None);
        assert_eq!(KeyAction::Quit, KeyAction::Quit);
        assert_eq!(KeyAction::Redraw, KeyAction::Redraw);
        assert_ne!(KeyAction::None, KeyAction::Quit);
    }
}
