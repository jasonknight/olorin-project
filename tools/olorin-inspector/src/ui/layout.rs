//! Layout management for the TUI

use ratatui::Frame;
use ratatui::prelude::*;

/// Create the main three-panel layout
///
/// Returns (database_list_area, records_area, input_area)
pub fn create_layout(frame: &Frame) -> (Rect, Rect, Rect) {
    let area = frame.area();

    // Reserve 1 line at bottom for status bar
    let main_area = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: area.height.saturating_sub(1),
    };

    // Split into main content and input area
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),   // Main content area
            Constraint::Length(6), // Input area (4 lines + 2 for border)
        ])
        .split(main_area);

    // Split main content into left (db list) and right (records)
    let content_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // Left panel (database list)
            Constraint::Min(50),    // Right panel (records table)
        ])
        .split(main_chunks[0]);

    (content_chunks[0], content_chunks[1], main_chunks[1])
}
