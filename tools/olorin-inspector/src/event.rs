//! Event handling for olorin-inspector

use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, FocusedPanel};

/// Application events
pub enum AppEvent {
    /// Key press event
    Key(KeyEvent),
    /// Tick event for periodic updates
    Tick,
    /// Terminal resize event
    Resize(u16, u16),
}

/// Event handler with configurable tick rate
pub struct EventHandler {
    tick_rate: Duration,
}

impl EventHandler {
    /// Create a new event handler
    ///
    /// # Arguments
    /// * `tick_rate` - Duration between tick events
    pub fn new(tick_rate: Duration) -> Self {
        Self { tick_rate }
    }

    /// Wait for and return the next event
    pub fn next(&self) -> std::io::Result<AppEvent> {
        if event::poll(self.tick_rate)? {
            match event::read()? {
                Event::Key(key) => return Ok(AppEvent::Key(key)),
                Event::Resize(w, h) => return Ok(AppEvent::Resize(w, h)),
                _ => {}
            }
        }
        Ok(AppEvent::Tick)
    }
}

/// Handle an application event
pub fn handle_event(app: &mut App, event: AppEvent) {
    match event {
        AppEvent::Key(key) => handle_key_event(app, key),
        AppEvent::Tick => {
            // Handle periodic tasks (config reload, health checks)
            app.on_tick();
        }
        AppEvent::Resize(_, _) => {
            // TUI will auto-resize, nothing to do
        }
    }
}

/// Handle a key event
fn handle_key_event(app: &mut App, key: KeyEvent) {
    // Handle detail modal keys first if open
    if app.detail_modal.is_some() {
        handle_detail_modal_keys(app, key);
        return;
    }

    // Handle clear modal keys if open
    if app.clear_modal.is_some() {
        handle_modal_keys(app, key);
        return;
    }

    // Global key bindings
    match key.code {
        KeyCode::Esc => {
            app.should_quit = true;
            return;
        }
        KeyCode::Tab => {
            app.next_focus();
            return;
        }
        KeyCode::Char('R') => {
            // Retry ChromaDB connection (Shift+R for global retry)
            app.retry_chromadb_connection();
            return;
        }
        _ => {}
    }

    // Panel-specific key bindings
    match app.focus {
        FocusedPanel::DatabaseList => handle_db_list_keys(app, key),
        FocusedPanel::RecordsTable => handle_records_keys(app, key),
        FocusedPanel::QueryInput => handle_input_keys(app, key),
    }
}

/// Handle keys when record detail modal is open
fn handle_detail_modal_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => {
            app.hide_detail_modal();
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.detail_modal_scroll_up();
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.detail_modal_scroll_down();
        }
        KeyCode::PageUp => {
            app.detail_modal_page_up();
        }
        KeyCode::PageDown => {
            app.detail_modal_page_down();
        }
        KeyCode::Home | KeyCode::Char('g') => {
            if let Some(ref mut modal) = app.detail_modal {
                modal.scroll = 0;
            }
        }
        KeyCode::End | KeyCode::Char('G') => {
            if let Some(ref mut modal) = app.detail_modal {
                modal.scroll = 1000; // Large number, will be clamped by render
            }
        }
        _ => {}
    }
}

/// Handle keys when clear database modal is open
fn handle_modal_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Tab | KeyCode::Left | KeyCode::Right => {
            app.toggle_clear_modal_selection();
        }
        KeyCode::Char('y') | KeyCode::Char('Y') => {
            app.set_clear_modal_selection(true);
            app.clear_current_database();
        }
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
            app.hide_clear_modal();
        }
        KeyCode::Enter => {
            if let Some(modal) = app.clear_modal {
                if modal.yes_selected {
                    app.clear_current_database();
                } else {
                    app.hide_clear_modal();
                }
            }
        }
        _ => {}
    }
}

/// Handle keys when database list is focused
fn handle_db_list_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.select_prev_db(),
        KeyCode::Down | KeyCode::Char('j') => app.select_next_db(),
        KeyCode::Enter => {
            app.refresh_records();
            app.next_focus();
        }
        KeyCode::Char('r') => {
            // Refresh current database
            app.refresh_records();
        }
        KeyCode::Char('z') => {
            // Show clear database confirmation modal
            app.show_clear_modal();
        }
        _ => {}
    }
}

/// Handle keys when records table is focused
fn handle_records_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Enter => {
            // Open detail modal for selected record
            app.show_detail_modal();
        }
        KeyCode::Up | KeyCode::Char('k') => app.scroll_records_up(),
        KeyCode::Down | KeyCode::Char('j') => app.scroll_records_down(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Home => {
            app.record_scroll = 0;
        }
        KeyCode::End => {
            app.record_scroll = app.records.len().saturating_sub(1);
        }
        KeyCode::Char('r') => {
            app.refresh_records();
        }
        KeyCode::Char('g') => {
            // Go to top
            app.record_scroll = 0;
        }
        KeyCode::Char('G') => {
            // Go to bottom (Shift+G)
            app.record_scroll = app.records.len().saturating_sub(1);
            app.load_more_records();
        }
        _ => {}
    }
}

/// Handle keys when query input is focused
fn handle_input_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Enter => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                // Shift+Enter inserts newline
                app.input_char('\n');
            } else {
                // Enter executes query
                app.execute_query();
            }
        }
        KeyCode::Char(c) => {
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                match c {
                    'c' | 'u' => app.clear_query(),
                    'a' => app.cursor_home(),
                    'e' => app.cursor_end(),
                    _ => {}
                }
            } else {
                app.input_char(c);
            }
        }
        KeyCode::Backspace => app.input_backspace(),
        KeyCode::Delete => app.input_delete(),
        KeyCode::Left => app.cursor_left(),
        KeyCode::Right => app.cursor_right(),
        KeyCode::Home => app.cursor_home(),
        KeyCode::End => app.cursor_end(),
        _ => {}
    }
}
