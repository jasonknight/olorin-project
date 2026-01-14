//! Olorin Database Inspector
//!
//! A TUI application for inspecting SQLite and ChromaDB databases
//! used in the Olorin project.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release
//! ```
//!
//! # Key Bindings
//!
//! - `Tab`: Cycle through panels (Database List -> Records -> Query Input)
//! - `Esc`: Exit the application
//! - `Up/Down` or `j/k`: Navigate in lists
//! - `PageUp/PageDown`: Fast scroll in records
//! - `Enter`: Execute query (in input panel) or select database
//! - `r`: Refresh current view

mod app;
mod db;
mod event;
mod ui;

use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use app::App;
use event::{handle_event, EventHandler};

/// Tick rate for the event loop (milliseconds)
const TICK_RATE_MS: u64 = 250;

fn main() -> Result<()> {
    // Initialize the application
    let mut app = match App::new() {
        Ok(app) => app,
        Err(e) => {
            eprintln!("Failed to initialize application: {}", e);
            eprintln!("\nMake sure you're running from the olorin project root");
            eprintln!("and that the .env file exists.");
            std::process::exit(1);
        }
    };

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create event handler
    let event_handler = EventHandler::new(Duration::from_millis(TICK_RATE_MS));

    // Main event loop
    let result = run_app(&mut terminal, &mut app, &event_handler);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    // Handle any errors from the main loop
    if let Err(e) = result {
        eprintln!("Application error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}

/// Run the main application loop
fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    event_handler: &EventHandler,
) -> Result<()> {
    loop {
        // Draw the UI
        terminal.draw(|frame| {
            ui::render(frame, app);
        })?;

        // Check for exit condition
        if app.should_quit {
            return Ok(());
        }

        // Handle events
        match event_handler.next() {
            Ok(event) => handle_event(app, event),
            Err(e) => {
                // Log error but continue running
                app.status_message = Some(format!("Event error: {}", e));
            }
        }
    }
}
