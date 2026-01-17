//! Olorin Settings TUI
//!
//! A terminal-based settings editor for the Olorin project.
//!
//! ## Architecture
//!
//! The application is organized into the following modules:
//!
//! - `app`: Application state and business logic
//! - `settings`: Setting definitions for all configuration tabs
//! - `ui`: User interface rendering (form, inputs, tabs)
//! - `input_handler`: Keyboard input handling (separated for testability)
//! - `validation`: Input validation logic (shared between save functions)
//! - `text_buffer`: UTF-8 safe text buffer utilities
//! - `api`: External API calls (Ollama, TTS)

mod api;
mod app;
mod input_handler;
mod settings;
mod text_buffer;
mod ui;
mod validation;

use anyhow::Result;
use app::App;
use crossterm::event::{self, Event};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use input_handler::{handle_key_event, KeyAction};
use olorin_config::Config;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
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
                let action = handle_key_event(app, key.code, key.modifiers);
                if action == KeyAction::Quit {
                    break;
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
