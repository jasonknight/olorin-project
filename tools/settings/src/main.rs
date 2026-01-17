//! Olorin Settings TUI
//!
//! A terminal-based settings editor for the Olorin project.

mod api;
mod app;
mod settings;
mod ui;

use anyhow::Result;
use app::{App, Focus};
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use olorin_config::Config;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use settings::InputType;
use std::io;
use std::time::Duration;

fn main() -> Result<()> {
    // Get config path
    let config = Config::new(None, false)?;
    let config_path = config.config_path().clone();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = App::new(config_path)?;

    // Fetch dynamic options on startup (e.g., Ollama models)
    app.refresh_all_dynamic_options();

    // Run event loop
    let result = run_event_loop(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        // Draw UI
        terminal.draw(|f| ui::render(f, app))?;

        // Poll for events
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                handle_key_event(app, key.code, key.modifiers);
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

fn handle_key_event(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
    // Global keys
    match code {
        KeyCode::Esc => {
            app.should_quit = true;
            return;
        }
        KeyCode::F(5) => {
            app.refresh_dynamic_options();
            return;
        }
        KeyCode::PageUp => {
            app.scroll_up(10);
            return;
        }
        KeyCode::PageDown => {
            app.scroll_down(10);
            return;
        }
        _ => {}
    }

    // Shift+Tab ALWAYS changes tabs
    if code == KeyCode::BackTab {
        app.prev_tab();
        return;
    }

    // Tab navigates fields/results
    if code == KeyCode::Tab && modifiers == KeyModifiers::NONE {
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
        return;
    }

    // Focus-specific handling
    match app.focus {
        Focus::SearchInput => handle_search_input_keys(app, code),
        Focus::FormField(_) => {
            if app.is_search_tab() {
                handle_search_result_keys(app, code, modifiers);
            } else {
                handle_field_keys(app, code, modifiers);
            }
        }
    }
}

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
