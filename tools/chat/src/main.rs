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

use crate::app::App;

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
                match (key.code, key.modifiers) {
                    // Quit
                    (KeyCode::Esc, _) => {
                        app.should_quit = true;
                    }
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        app.should_quit = true;
                    }

                    // Send message or execute slash command (Enter)
                    (KeyCode::Enter, KeyModifiers::NONE) => {
                        let text: String = app.input.lines().join("\n").trim().to_string();
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
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
