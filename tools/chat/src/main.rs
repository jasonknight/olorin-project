//! Olorin Chat TUI
//!
//! A terminal user interface for monitoring and interacting with the Olorin AI pipeline.
//! Displays chat messages from the database and allows sending new prompts via Kafka.
//! Supports slash commands via the control API with inline autocomplete.

mod api;
mod app;
mod db;
mod kafka;
mod message;
mod ui;

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use olorin_config::Config;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io;
use std::path::PathBuf;
use std::time::Duration;
use tui_textarea::Input;

use crate::app::{ActiveTab, App};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::new(None, false)?;

    // Get paths from config
    let chat_db_path = config
        .get_path("CHAT_DB_PATH", Some("./cortex/data/chat.db"))
        .unwrap_or_else(|| PathBuf::from("./cortex/data/chat.db"));

    let bootstrap_servers = config
        .get("KAFKA_BOOTSTRAP_SERVERS", Some("localhost:9092"))
        .unwrap_or_else(|| "localhost:9092".to_string());

    // Get control API URL from config
    let control_api_host = config
        .get("CONTROL_API_HOST", Some("localhost"))
        .unwrap_or_else(|| "localhost".to_string());
    let control_api_port = config
        .get_int("CONTROL_API_PORT", Some(8765))
        .unwrap_or(8765);
    let control_api_url = format!("http://{}:{}", control_api_host, control_api_port);

    // Get search tool URL from config
    let search_tool_host = config
        .get("SEARCH_TOOL_HOST", Some("localhost"))
        .unwrap_or_else(|| "localhost".to_string());
    let search_tool_port = config
        .get_int("SEARCH_TOOL_PORT", Some(8772))
        .unwrap_or(8772);
    let search_tool_url = format!("http://{}:{}", search_tool_host, search_tool_port);

    // Get context DB path from config
    let context_db_path = config
        .get_path("CONTEXT_DB_PATH", Some("./hippocampus/data/context.db"))
        .unwrap_or_else(|| PathBuf::from("./hippocampus/data/context.db"));

    // Fetch slash commands BEFORE entering raw mode
    let slash_commands = crate::api::fetch_commands(&control_api_url).await;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = App::new(
        chat_db_path,
        &bootstrap_servers,
        &control_api_url,
        slash_commands,
        &search_tool_url,
        context_db_path,
    )?;

    // Main event loop
    let result = run_event_loop(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App<'_>,
) -> Result<()> {
    loop {
        // Process background events (new messages, etc.)
        app.process_events();

        // Draw UI
        terminal.draw(|f| ui::render(f, app))?;

        // Check for user input with timeout
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Handle quit confirmation modal first
                if app.showing_quit_modal {
                    match key.code {
                        KeyCode::Char('y') | KeyCode::Char('Y') => {
                            app.confirm_quit();
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                            app.hide_quit_modal();
                        }
                        _ => {}
                    }
                    continue;
                }

                match (key.code, key.modifiers) {
                    // Ctrl+C always quits immediately
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        app.should_quit = true;
                    }

                    // 'q' shows quit confirmation modal (global, except when typing in input)
                    (KeyCode::Char('q'), KeyModifiers::NONE) => {
                        // Only show quit modal if not in a text input context
                        let in_text_input = app.active_tab == ActiveTab::Chat
                            || (app.active_tab == ActiveTab::Search
                                && app.search_state.focus == crate::app::SearchFocus::Input
                                && !app.search_state.showing_modal);
                        if in_text_input {
                            // Pass through to text input
                            if app.active_tab == ActiveTab::Chat {
                                let input = Input::from(key);
                                app.input.input(input);
                                app.update_completion();
                            } else {
                                app.search_input_char('q');
                            }
                        } else {
                            app.show_quit_modal();
                        }
                        continue;
                    }

                    // Esc is context-sensitive
                    (KeyCode::Esc, _) => {
                        match app.active_tab {
                            ActiveTab::Chat | ActiveTab::State => {
                                app.show_quit_modal();
                            }
                            ActiveTab::Search => {
                                if app.search_state.showing_help {
                                    app.close_search_help();
                                } else if app.search_state.showing_modal {
                                    app.close_search_modal();
                                } else {
                                    app.show_quit_modal();
                                }
                            }
                        }
                        continue;
                    }

                    // Switch tabs (Shift+Tab)
                    (KeyCode::BackTab, _) => {
                        app.next_tab();
                    }

                    // Tab-specific key handling
                    _ => match app.active_tab {
                        ActiveTab::Chat => {
                            match (key.code, key.modifiers) {
                                // Send message or execute slash command (Enter)
                                (KeyCode::Enter, KeyModifiers::NONE) => {
                                    let text: String =
                                        app.input.lines().join("\n").trim().to_string();
                                    if !text.is_empty() {
                                        // Any text starting with / is a slash command - never send to model
                                        if text.starts_with('/') {
                                            app.execute_slash_command(&text).await;
                                        } else {
                                            app.send_message().await;
                                        }
                                    }
                                }

                                // Accept autocomplete (Tab)
                                (KeyCode::Tab, KeyModifiers::NONE) => {
                                    if app.completion.is_some() {
                                        app.accept_completion();
                                        app.update_completion();
                                    }
                                }

                                // Scroll chat
                                (KeyCode::Up, _) => {
                                    app.scroll_up(1);
                                }
                                (KeyCode::Down, _) => {
                                    app.scroll_down(1);
                                }
                                (KeyCode::PageUp, _) => {
                                    app.scroll_up(10);
                                }
                                (KeyCode::PageDown, _) => {
                                    app.scroll_down(10);
                                }
                                (KeyCode::Home, KeyModifiers::CONTROL) => {
                                    app.scroll_to_top();
                                }
                                (KeyCode::End, KeyModifiers::CONTROL) => {
                                    app.scroll_to_bottom();
                                }

                                // Shift+Enter: insert explicit newline
                                (KeyCode::Enter, KeyModifiers::SHIFT) => {
                                    app.input.insert_newline();
                                    app.update_completion();
                                }

                                // Pass other keys to text area
                                _ => {
                                    let input = Input::from(key);
                                    app.input.input(input);
                                    app.update_completion();
                                }
                            }
                        }
                        ActiveTab::State => {
                            match (key.code, key.modifiers) {
                                // Refresh state display
                                (KeyCode::Char('r'), KeyModifiers::NONE) => {
                                    app.refresh_state();
                                }

                                // Scroll state display
                                (KeyCode::Up, _) => {
                                    app.scroll_state_up(1);
                                }
                                (KeyCode::Down, _) => {
                                    app.scroll_state_down(1);
                                }
                                (KeyCode::PageUp, _) => {
                                    app.scroll_state_up(10);
                                }
                                (KeyCode::PageDown, _) => {
                                    app.scroll_state_down(10);
                                }

                                // Ignore other keys on State tab
                                _ => {}
                            }
                        }
                        ActiveTab::Search => {
                            use crate::app::SearchFocus;

                            // Handle help modal (Esc handled globally above)
                            if app.search_state.showing_help {
                                // Any key closes help modal (Esc already handled)
                                continue;
                            }

                            // Handle ? to show help (works in any context except help modal)
                            if key.code == KeyCode::Char('?') {
                                app.toggle_search_help();
                                continue;
                            }

                            // Handle F2 to toggle search mode
                            if key.code == KeyCode::F(2) {
                                app.toggle_search_mode();
                                continue;
                            }

                            // Handle document modal (Esc handled globally above)
                            if app.search_state.showing_modal {
                                match key.code {
                                    KeyCode::Char('a') => {
                                        app.add_to_context();
                                    }
                                    KeyCode::Char('r') => {
                                        app.remove_from_context();
                                    }
                                    _ => {}
                                }
                            } else {
                                match app.search_state.focus {
                                    SearchFocus::Input => {
                                        match key.code {
                                            // Execute search on Enter
                                            KeyCode::Enter => {
                                                app.execute_search().await;
                                            }
                                            // Switch focus to results on Tab
                                            KeyCode::Tab => {
                                                app.toggle_search_focus();
                                            }
                                            // Backspace
                                            KeyCode::Backspace => {
                                                app.search_input_backspace();
                                            }
                                            // Character input
                                            KeyCode::Char(c) => {
                                                app.search_input_char(c);
                                            }
                                            // Arrow keys can also navigate results
                                            KeyCode::Down => {
                                                if !app.search_state.results.is_empty() {
                                                    app.toggle_search_focus();
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    SearchFocus::Results => {
                                        match key.code {
                                            // Navigate results
                                            KeyCode::Up => {
                                                app.search_select_up();
                                            }
                                            KeyCode::Down => {
                                                app.search_select_down();
                                            }
                                            KeyCode::PageUp => {
                                                for _ in 0..10 {
                                                    app.search_select_up();
                                                }
                                            }
                                            KeyCode::PageDown => {
                                                for _ in 0..10 {
                                                    app.search_select_down();
                                                }
                                            }
                                            // View document
                                            KeyCode::Enter => {
                                                app.toggle_search_modal();
                                            }
                                            // Add to context
                                            KeyCode::Char('a') => {
                                                app.add_to_context();
                                            }
                                            // Remove from context
                                            KeyCode::Char('r') => {
                                                app.remove_from_context();
                                            }
                                            // Switch focus back to input
                                            KeyCode::Tab => {
                                                app.toggle_search_focus();
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
