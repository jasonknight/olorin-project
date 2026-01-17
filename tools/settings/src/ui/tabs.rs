//! Tab bar rendering

use crate::app::App;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

/// Render the tab bar at the top
pub fn render_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut spans = Vec::new();

    for (i, tab) in app.tabs.iter().enumerate() {
        let is_current = i == app.current_tab;

        let style = if is_current {
            Style::default()
                .fg(Color::White)
                .bg(Color::Blue)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        spans.push(Span::styled(format!(" {} ", tab.name), style));

        if i < app.tabs.len() - 1 {
            spans.push(Span::styled(" | ", Style::default().fg(Color::DarkGray)));
        }
    }

    let paragraph = Paragraph::new(Line::from(spans));
    frame.render_widget(paragraph, inner);
}
