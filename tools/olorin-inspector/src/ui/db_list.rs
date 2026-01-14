//! Database list panel component

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Frame;

use crate::app::{App, FocusedPanel};

/// Render the database list panel (left side)
pub fn render_db_list(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == FocusedPanel::DatabaseList;

    // Create list items
    let items: Vec<ListItem> = app
        .databases
        .iter()
        .enumerate()
        .map(|(i, db)| {
            let info = db.info();
            let text = format!("{} ({})", info.name, info.record_count);

            let style = if i == app.selected_db {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            ListItem::new(text).style(style)
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
