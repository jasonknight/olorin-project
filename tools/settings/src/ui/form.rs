//! Form rendering for settings

use crate::app::{App, Focus};
use crate::ui::inputs;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::Frame;

/// Render the form with settings
pub fn render_form(frame: &mut Frame, app: &App, area: Rect) {
    if app.is_search_tab() {
        render_search_form(frame, app, area);
    } else {
        render_settings_form(frame, app, area);
    }
}

/// Render the search tab with search input and results
fn render_search_form(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue))
        .title(" Search Settings ");

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 3 {
        return;
    }

    // Render search input (first 2 lines)
    let search_label_area = Rect {
        x: inner.x,
        y: inner.y,
        width: 8,
        height: 1,
    };
    let search_input_area = Rect {
        x: inner.x + 8,
        y: inner.y,
        width: inner.width.saturating_sub(8),
        height: 1,
    };

    let is_search_focused = matches!(app.focus, Focus::SearchInput);

    // Search label
    let label_style = if is_search_focused {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    frame.render_widget(
        Paragraph::new(Span::styled("Search: ", label_style)),
        search_label_area,
    );

    // Search input with cursor
    let bg_color = if is_search_focused {
        Color::DarkGray
    } else {
        Color::Black
    };
    let fg_color = if is_search_focused {
        Color::White
    } else {
        Color::Gray
    };

    let search_spans = if is_search_focused {
        // Use UTF-8 safe cursor splitting
        let (before, cursor_char, after) = app.split_search_at_cursor();

        vec![
            Span::styled(
                format!("[{}", before),
                Style::default().fg(fg_color).bg(bg_color),
            ),
            Span::styled(
                cursor_char.to_string(),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{}]", after),
                Style::default().fg(fg_color).bg(bg_color),
            ),
        ]
    } else {
        let display = if app.search_query.is_empty() {
            "(type to search)"
        } else {
            &app.search_query
        };
        vec![Span::styled(
            format!("[{}]", display),
            Style::default().fg(fg_color).bg(bg_color),
        )]
    };

    frame.render_widget(Paragraph::new(Line::from(search_spans)), search_input_area);

    // Separator after search input
    let sep_y = inner.y + 1;
    if sep_y < inner.y + inner.height {
        let sep_area = Rect {
            x: inner.x,
            y: sep_y,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Span::styled(
                "─".repeat(inner.width as usize),
                Style::default().fg(Color::DarkGray),
            )),
            sep_area,
        );
    }

    // Render search results
    let results_area = Rect {
        x: inner.x,
        y: inner.y + 2,
        width: inner.width,
        height: inner.height.saturating_sub(2),
    };

    if app.search_query.is_empty() {
        // Show hint when no search
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Type to search across all settings...",
                Style::default().fg(Color::DarkGray),
            )),
            results_area,
        );
        return;
    }

    if app.search_results.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "No results found",
                Style::default().fg(Color::DarkGray),
            )),
            results_area,
        );
        return;
    }

    // Render search results
    let lines_per_result = 3usize;
    let total_lines = app.search_results.len() * lines_per_result;
    let visible_height = results_area.height as usize;

    let start_line = app.form_scroll;
    let start_idx = start_line / lines_per_result;

    let mut y = results_area.y;
    let label_width = 24u16;
    let input_width = results_area.width.saturating_sub(label_width + 2);

    for (i, result) in app.search_results.iter().enumerate().skip(start_idx) {
        if y >= results_area.y + results_area.height {
            break;
        }

        let is_selected = i == app.search_selected;
        let setting = match app.get_search_result_value(i) {
            Some(s) => s,
            None => continue,
        };

        // Label area with tab name prefix
        let label_area = Rect {
            x: results_area.x,
            y,
            width: label_width,
            height: 1,
        };
        let input_area = Rect {
            x: results_area.x + label_width + 1,
            y,
            width: input_width,
            height: 1,
        };

        // Label with selection highlight
        let label_style = if is_selected {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let label_text = format!("{:width$}", setting.def.label, width = label_width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(label_text, label_style)),
            label_area,
        );

        // Render input value
        inputs::render_input(frame, setting, input_area, is_selected);

        y += 1;

        // Description with tab name
        if y < results_area.y + results_area.height {
            let desc_area = Rect {
                x: results_area.x + 2,
                y,
                width: results_area.width.saturating_sub(4),
                height: 1,
            };

            let desc_spans = vec![
                Span::styled(
                    format!("[{}] ", result.tab_name),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(
                    setting.def.description,
                    Style::default().fg(Color::DarkGray),
                ),
            ];

            frame.render_widget(Paragraph::new(Line::from(desc_spans)), desc_area);
            y += 1;
        }

        // Separator
        if y < results_area.y + results_area.height {
            let sep_area = Rect {
                x: results_area.x,
                y,
                width: results_area.width,
                height: 1,
            };
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "─".repeat(results_area.width as usize),
                    Style::default().fg(Color::DarkGray),
                )),
                sep_area,
            );
            y += 1;
        }
    }

    // Scrollbar for results
    if total_lines > visible_height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("▲"))
            .end_symbol(Some("▼"));

        let mut scrollbar_state = ScrollbarState::new(total_lines.saturating_sub(visible_height))
            .position(app.form_scroll);

        let scrollbar_area = Rect {
            x: area.x,
            y: area.y + 3, // Offset for search input area
            width: area.width,
            height: area.height.saturating_sub(3),
        };

        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

/// Render normal settings form
fn render_settings_form(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue))
        .title(format!(" {} Settings ", app.tabs[app.current_tab].name));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let settings = app.current_tab_values();
    if settings.is_empty() {
        return;
    }

    // Each setting takes 3 lines: label+input, description, separator
    let lines_per_setting = 3usize;
    let total_lines = settings.len() * lines_per_setting;
    let visible_height = inner.height as usize;

    // Calculate which settings are visible
    let start_line = app.form_scroll;
    let start_idx = start_line / lines_per_setting;

    let mut y = inner.y;
    let label_width = 24u16;
    let input_width = inner.width.saturating_sub(label_width + 2);

    for (i, setting) in settings.iter().enumerate().skip(start_idx) {
        if y >= inner.y + inner.height {
            break;
        }

        let is_focused = matches!(app.focus, Focus::FormField(idx) if idx == i);

        // Calculate areas
        let label_area = Rect {
            x: inner.x,
            y,
            width: label_width,
            height: 1,
        };
        let input_area = Rect {
            x: inner.x + label_width + 1,
            y,
            width: input_width,
            height: 1,
        };

        // Render label
        let label_style = if is_focused {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let label_text = format!("{:width$}", setting.def.label, width = label_width as usize);
        let label = Paragraph::new(Span::styled(label_text, label_style));
        frame.render_widget(label, label_area);

        // Render input
        inputs::render_input(frame, setting, input_area, is_focused);

        y += 1;

        // Render description
        if y < inner.y + inner.height {
            let desc_area = Rect {
                x: inner.x + 2,
                y,
                width: inner.width.saturating_sub(4),
                height: 1,
            };

            let mut desc_spans = vec![Span::styled(
                setting.def.description,
                Style::default().fg(Color::DarkGray),
            )];

            // Add validation error if present
            if let Some(err) = &setting.validation_error {
                desc_spans.push(Span::styled(
                    format!(" [{}]", err),
                    Style::default().fg(Color::Red),
                ));
            }

            let desc = Paragraph::new(Line::from(desc_spans));
            frame.render_widget(desc, desc_area);
            y += 1;
        }

        // Render separator
        if y < inner.y + inner.height {
            let sep_area = Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            };
            let sep = Paragraph::new(Span::styled(
                "─".repeat(inner.width as usize),
                Style::default().fg(Color::DarkGray),
            ));
            frame.render_widget(sep, sep_area);
            y += 1;
        }
    }

    // Render scrollbar if needed
    if total_lines > visible_height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("▲"))
            .end_symbol(Some("▼"));

        let mut scrollbar_state = ScrollbarState::new(total_lines.saturating_sub(visible_height))
            .position(app.form_scroll);

        frame.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
    }
}
