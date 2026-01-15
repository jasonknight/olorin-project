//! UI components for olorin-inspector

mod db_list;
mod input;
mod layout;
mod records;

pub use db_list::render_db_list;
pub use input::render_query_input;
pub use layout::create_layout;
pub use records::render_records;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::app::{App, RecordDetailModal};

/// Render the entire UI
pub fn render(frame: &mut Frame, app: &App) {
    let (db_list_area, records_area, input_area) = create_layout(frame);

    // Render each panel
    render_db_list(frame, db_list_area, app);
    render_records(frame, records_area, app);
    render_query_input(frame, input_area, app);

    // Render status bar at the very bottom
    render_status_bar(frame, app);

    // Render modals on top if visible (detail modal takes precedence)
    if let Some(modal) = &app.detail_modal {
        render_detail_modal(frame, modal);
    } else if let Some(modal) = &app.clear_modal {
        render_clear_modal(frame, app, modal.yes_selected);
    }
}

/// Render the status bar
fn render_status_bar(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Status bar is 1 line at the very bottom
    let status_area = Rect {
        x: area.x,
        y: area.height.saturating_sub(1),
        width: area.width,
        height: 1,
    };

    let status_text = app
        .status_message
        .as_deref()
        .unwrap_or("Ready | Tab: switch panels | Esc: quit");

    let status = Paragraph::new(status_text)
        .style(Style::default().fg(Color::White).bg(Color::DarkGray));

    frame.render_widget(status, status_area);
}

/// Render the clear database confirmation modal
fn render_clear_modal(frame: &mut Frame, app: &App, yes_selected: bool) {
    let area = frame.area();

    // Modal dimensions
    let modal_width = 50;
    let modal_height = 7;

    // Center the modal
    let modal_x = (area.width.saturating_sub(modal_width)) / 2;
    let modal_y = (area.height.saturating_sub(modal_height)) / 2;

    let modal_area = Rect {
        x: modal_x,
        y: modal_y,
        width: modal_width.min(area.width),
        height: modal_height.min(area.height),
    };

    // Clear the area behind the modal
    frame.render_widget(Clear, modal_area);

    // Get database name for the message
    let db_name = app
        .current_db_info()
        .map(|info| info.name.as_str())
        .unwrap_or("database");

    // Build the modal content
    let title = format!(" Clear {} ", db_name);

    let block = Block::default()
        .title(title)
        .title_style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .style(Style::default().bg(Color::Black));

    // Create button line with selection highlight
    let yes_style = if yes_selected {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };

    let no_style = if !yes_selected {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };

    // Build the content with Line and Span
    let content = vec![
        Line::from(""),
        Line::from("  Delete all records? This cannot be undone."),
        Line::from(""),
        Line::from(vec![
            Span::raw("          "),
            Span::styled(" Yes ", yes_style),
            Span::raw("     "),
            Span::styled(" No ", no_style),
        ]),
        Line::from(""),
    ];

    let paragraph = Paragraph::new(content)
        .block(block)
        .alignment(Alignment::Left);

    frame.render_widget(paragraph, modal_area);
}

/// Render the record detail modal
fn render_detail_modal(frame: &mut Frame, modal: &RecordDetailModal) {
    let area = frame.area();

    // Modal covers most of the screen (80% width, 80% height)
    let modal_width = (area.width as f32 * 0.8) as u16;
    let modal_height = (area.height as f32 * 0.8) as u16;

    // Center the modal
    let modal_x = (area.width.saturating_sub(modal_width)) / 2;
    let modal_y = (area.height.saturating_sub(modal_height)) / 2;

    let modal_area = Rect {
        x: modal_x,
        y: modal_y,
        width: modal_width.min(area.width),
        height: modal_height.min(area.height),
    };

    // Clear the area behind the modal
    frame.render_widget(Clear, modal_area);

    // Build the content with markdown-style headers
    let mut lines: Vec<Line> = Vec::new();

    // Get fields sorted alphabetically for consistent ordering
    let mut fields: Vec<(&String, &String)> = modal.record.fields.iter().collect();
    fields.sort_by(|a, b| a.0.cmp(b.0));

    for (key, value) in fields {
        // Add header line (## field_name)
        lines.push(Line::from(vec![Span::styled(
            format!("## {}", key),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]));

        // Add empty line after header
        lines.push(Line::from(""));

        // Add value lines (word wrap long values)
        let content_width = modal_width.saturating_sub(4) as usize; // Account for borders and padding
        for line in wrap_text(value, content_width) {
            lines.push(Line::from(line));
        }

        // Add separator between fields
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "─".repeat(content_width.min(60)),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(" Record Details (Esc to close, j/k to scroll) ")
        .title_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .scroll((modal.scroll as u16, 0));

    frame.render_widget(paragraph, modal_area);
}

/// Word wrap text to fit within a given width
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();

    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }

        let mut current_line = String::new();
        for word in paragraph.split_whitespace() {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else if current_line.len() + 1 + word.len() <= max_width {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}
