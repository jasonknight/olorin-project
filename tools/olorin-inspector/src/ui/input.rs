//! Query input panel component

use ratatui::Frame;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::{App, FocusedPanel};
use crate::db::DatabaseType;

/// Render the query input panel (bottom)
pub fn render_query_input(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == FocusedPanel::QueryInput;

    // Border style based on focus
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    // Get query hint based on database type
    let db_type = app
        .current_db_info()
        .map(|i| &i.db_type)
        .cloned()
        .unwrap_or(DatabaseType::SqliteFileTracker);

    let hint = match db_type {
        DatabaseType::ChromaDB => "Enter semantic search query",
        _ => "Enter SQL query (e.g., SELECT * FROM table WHERE ...)",
    };

    let title = format!(" Query [{}] ", hint);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title)
        .title_style(if focused {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        });

    let inner = block.inner(area);

    // Display the query text with cursor
    let display_text = if focused {
        // Show cursor
        let before: String = app.query_input.chars().take(app.cursor_position).collect();
        let after: String = app.query_input.chars().skip(app.cursor_position).collect();
        format!("{}|{}", before, after)
    } else if app.query_input.is_empty() {
        "(Press Tab to focus, then type your query)".to_string()
    } else {
        app.query_input.clone()
    };

    let text_style = if focused {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let paragraph = Paragraph::new(display_text).style(text_style).block(block);

    frame.render_widget(paragraph, area);

    // Show cursor position if focused
    if focused {
        // Calculate cursor position within the input area
        let cursor_x = inner.x + (app.cursor_position as u16 % inner.width);
        let cursor_y = inner.y + (app.cursor_position as u16 / inner.width);

        // Set cursor position (ratatui will handle this)
        frame.set_cursor_position(Position::new(
            cursor_x.min(inner.x + inner.width - 1),
            cursor_y.min(inner.y + inner.height - 1),
        ));
    }
}
