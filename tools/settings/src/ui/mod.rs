//! UI rendering modules

mod form;
mod inputs;
mod tabs;

use crate::app::App;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Main render function
pub fn render(frame: &mut Frame, app: &mut App) {
    // Store visible height for scrolling calculations
    app.visible_height = frame.area().height.saturating_sub(6) as usize;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Tab bar
            Constraint::Min(5),    // Form content
            Constraint::Length(2), // Status/help bar
        ])
        .split(frame.area());

    tabs::render_tab_bar(frame, app, chunks[0]);
    form::render_form(frame, app, chunks[1]);
    render_status_bar(frame, app, chunks[2]);
}

fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Status message on first line
    let status_text = app.status.as_deref().unwrap_or("");
    let status = Paragraph::new(Line::from(vec![Span::styled(
        status_text,
        Style::default().fg(Color::Yellow),
    )]));

    // Help text on second line
    let help_spans = vec![
        Span::styled("Tab", Style::default().fg(Color::Cyan)),
        Span::raw(": Next  "),
        Span::styled("Shift+Tab", Style::default().fg(Color::Cyan)),
        Span::raw(": Prev  "),
        Span::styled("Enter", Style::default().fg(Color::Cyan)),
        Span::raw(": Save  "),
        Span::styled("F5", Style::default().fg(Color::Cyan)),
        Span::raw(": Refresh  "),
        Span::styled("PgUp/Dn", Style::default().fg(Color::Cyan)),
        Span::raw(": Scroll  "),
        Span::styled("Esc", Style::default().fg(Color::Cyan)),
        Span::raw(": Quit"),
    ];

    if inner.height >= 2 {
        let status_area = Rect {
            x: inner.x,
            y: inner.y,
            width: inner.width,
            height: 1,
        };
        let help_area = Rect {
            x: inner.x,
            y: inner.y + 1,
            width: inner.width,
            height: 1,
        };

        frame.render_widget(status, status_area);
        frame.render_widget(Paragraph::new(Line::from(help_spans)), help_area);
    } else {
        frame.render_widget(status, inner);
    }
}
