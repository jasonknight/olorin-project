//! Database list panel component

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Frame;

use crate::app::{App, FocusedPanel};
use crate::db::{ConnectionState, DatabaseType};

/// Render the database list panel (left side)
pub fn render_db_list(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == FocusedPanel::DatabaseList;

    // Create list items with connection status indicators
    let items: Vec<ListItem> = app
        .databases
        .iter()
        .enumerate()
        .map(|(i, db)| {
            let info = db.info();
            let is_selected = i == app.selected_db;

            // Connection status indicator for network databases
            let (status_icon, status_style) = if info.db_type == DatabaseType::ChromaDB {
                match &info.connection_state {
                    ConnectionState::Connected => ("●", Color::Green),
                    ConnectionState::Disconnected(_) => ("○", Color::Red),
                    ConnectionState::Unknown => ("?", Color::Yellow),
                }
            } else {
                // SQLite databases are always "connected"
                ("●", Color::Green)
            };

            // Create styled spans
            let line = if is_selected {
                Line::from(vec![
                    Span::styled(
                        format!("{} ", status_icon),
                        Style::default().fg(status_style),
                    ),
                    Span::styled(
                        if info.connection_state.is_available() {
                            format!("{} ({})", info.name, info.record_count)
                        } else {
                            format!("{} (offline)", info.name)
                        },
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled(
                        format!("{} ", status_icon),
                        Style::default().fg(status_style),
                    ),
                    Span::styled(
                        if info.connection_state.is_available() {
                            format!("{} ({})", info.name, info.record_count)
                        } else {
                            format!("{} (offline)", info.name)
                        },
                        Style::default().fg(if info.connection_state.is_available() {
                            Color::White
                        } else {
                            Color::DarkGray
                        }),
                    ),
                ])
            };

            ListItem::new(line)
        })
        .collect();

    // Border style based on focus
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Databases ")
        .title_style(if focused {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        });

    let list = List::new(items)
        .block(block)
        .highlight_symbol(">> ")
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::REVERSED)
                .fg(Color::Yellow),
        );

    let mut state = ListState::default();
    state.select(Some(app.selected_db));

    frame.render_stateful_widget(list, area, &mut state);
}
