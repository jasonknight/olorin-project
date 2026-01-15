//! Records table panel component with word wrapping

use ratatui::Frame;
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Cell, Row, Scrollbar, ScrollbarOrientation, ScrollbarState, Table, TableState,
};

use crate::app::{App, FocusedPanel};
use crate::db::Record;

/// Maximum width for any single column
const MAX_COLUMN_WIDTH: u16 = 40;

/// Minimum width for any single column
const MIN_COLUMN_WIDTH: u16 = 8;

/// Render the records table panel (right side)
pub fn render_records(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == FocusedPanel::RecordsTable;

    // Get columns for current database
    let columns = app.current_columns();

    if columns.is_empty() {
        render_empty_table(frame, area, focused, "No database selected");
        return;
    }

    if app.records.is_empty() {
        render_empty_table(frame, area, focused, "No records found");
        return;
    }

    // Calculate column widths
    let available_width = area.width.saturating_sub(4); // Account for borders and scrollbar
    let col_count = columns.len() as u16;
    let base_width = (available_width / col_count).clamp(MIN_COLUMN_WIDTH, MAX_COLUMN_WIDTH);

    // Create header
    let header_cells: Vec<Cell> = columns
        .iter()
        .map(|col| {
            Cell::from(col.as_str()).style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect();

    let header = Row::new(header_cells).height(1);

    // Create rows with word wrapping
    let rows: Vec<Row> = app
        .records
        .iter()
        .map(|record| create_row(record, &columns, base_width as usize))
        .collect();

    // Create width constraints
    let widths: Vec<Constraint> = columns
        .iter()
        .map(|_| Constraint::Length(base_width))
        .collect();

    // Border style based on focus
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let db_name = app
        .current_db_info()
        .map(|i| i.name.as_str())
        .unwrap_or("Unknown");

    let title = format!(
        " {} [{}/{}] ",
        db_name,
        app.record_scroll + 1,
        app.records.len()
    );

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

    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol(">> ");

    let mut state = TableState::default();
    state.select(Some(app.record_scroll));

    // Render table
    let table_area = Rect {
        x: area.x,
        y: area.y,
        width: area.width.saturating_sub(1),
        height: area.height,
    };
    frame.render_stateful_widget(table, table_area, &mut state);

    // Render scrollbar
    if app.records.len() > 1 {
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("▲"))
            .end_symbol(Some("▼"));

        let mut scrollbar_state =
            ScrollbarState::new(app.records.len()).position(app.record_scroll);

        let scrollbar_area = Rect {
            x: area.x + area.width.saturating_sub(1),
            y: area.y + 1,
            width: 1,
            height: area.height.saturating_sub(2),
        };

        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

/// Maximum characters to show in list view before truncating
const LIST_VIEW_TRUNCATE: usize = 200;

/// Create a table row from a record with word wrapping
fn create_row(record: &Record, columns: &[String], col_width: usize) -> Row<'static> {
    let cells: Vec<Cell> = columns
        .iter()
        .map(|col| {
            let value = record.fields.get(col).map(|s| s.as_str()).unwrap_or("");

            // Truncate long values for list view (full content shown in detail modal)
            let display_value = if value.len() > LIST_VIEW_TRUNCATE {
                let truncated: String = value.chars().take(LIST_VIEW_TRUNCATE).collect();
                truncated + "..."
            } else {
                value.to_string()
            };

            // Word wrap the value
            let wrapped = wrap_text(&display_value, col_width);
            Cell::from(wrapped)
        })
        .collect();

    // Calculate row height based on maximum wrapped lines (using truncated values)
    let max_lines = columns
        .iter()
        .map(|col| {
            let value = record.fields.get(col).map(|s| s.as_str()).unwrap_or("");
            let display_value = if value.len() > LIST_VIEW_TRUNCATE {
                let truncated: String = value.chars().take(LIST_VIEW_TRUNCATE).collect();
                truncated + "..."
            } else {
                value.to_string()
            };
            count_wrapped_lines(&display_value, col_width)
        })
        .max()
        .unwrap_or(1);

    Row::new(cells).height(max_lines as u16)
}

/// Wrap text to fit within a given width
fn wrap_text(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let mut result = String::new();
    let mut current_line_len = 0;

    for word in text.split_whitespace() {
        let word_len = word.chars().count();

        if current_line_len == 0 {
            // First word on the line
            if word_len > width {
                // Word is too long, need to break it
                for (i, c) in word.chars().enumerate() {
                    if i > 0 && i % width == 0 {
                        result.push('\n');
                    }
                    result.push(c);
                }
                current_line_len = word_len % width;
                if current_line_len == 0 {
                    current_line_len = width;
                }
            } else {
                result.push_str(word);
                current_line_len = word_len;
            }
        } else if current_line_len + 1 + word_len <= width {
            // Word fits on current line
            result.push(' ');
            result.push_str(word);
            current_line_len += 1 + word_len;
        } else {
            // Need to start a new line
            result.push('\n');
            if word_len > width {
                // Word is too long, need to break it
                for (i, c) in word.chars().enumerate() {
                    if i > 0 && i % width == 0 {
                        result.push('\n');
                    }
                    result.push(c);
                }
                current_line_len = word_len % width;
                if current_line_len == 0 {
                    current_line_len = width;
                }
            } else {
                result.push_str(word);
                current_line_len = word_len;
            }
        }
    }

    result
}

/// Count how many lines text will take when wrapped
fn count_wrapped_lines(text: &str, width: usize) -> usize {
    if width == 0 || text.is_empty() {
        return 1;
    }

    let wrapped = wrap_text(text, width);
    wrapped.lines().count().max(1)
}

/// Render an empty table with a message
fn render_empty_table(frame: &mut Frame, area: Rect, focused: bool, message: &str) {
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Records ");

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let text = ratatui::widgets::Paragraph::new(message)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);

    let text_area = Rect {
        x: inner.x,
        y: inner.y + inner.height / 2,
        width: inner.width,
        height: 1,
    };

    frame.render_widget(text, text_area);
}
