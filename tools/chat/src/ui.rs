//! UI rendering with ratatui

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};

use crate::app::{ActiveTab, App};
use crate::message::DisplayMessage;

/// Main UI rendering function
pub fn render(frame: &mut Frame, app: &App) {
    // Create layout: tab bar + main area + status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Tab bar
            Constraint::Min(10),   // Main content area
            Constraint::Length(1), // Status bar
        ])
        .split(frame.area());

    render_tab_bar(frame, app, chunks[0]);

    match app.active_tab {
        ActiveTab::Chat => {
            // Split main area into chat display and input
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(5),    // Chat display (expands)
                    Constraint::Length(5), // Input area (3 lines + border)
                ])
                .split(chunks[1]);

            render_chat_display(frame, app, main_chunks[0]);
            render_input_area(frame, app, main_chunks[1]);
        }
        ActiveTab::State => {
            render_state_display(frame, app, chunks[1]);
        }
        ActiveTab::Search => {
            render_search_display(frame, app, chunks[1]);
            // Render modal overlays if showing
            if app.search_state.showing_modal {
                render_document_modal(frame, app);
            }
            if app.search_state.showing_help {
                render_search_help_modal(frame);
            }
            if app.search_state.showing_manual_entry {
                render_manual_entry_modal(frame, app);
            }
        }
    }

    render_status_bar(frame, app, chunks[2]);

    // Render quit confirmation modal if showing (overlays everything)
    if app.showing_quit_modal {
        render_quit_modal(frame);
    }
}

/// Render the tab bar at the top
fn render_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let active_style = Style::default()
        .fg(Color::White)
        .bg(Color::Blue)
        .add_modifier(Modifier::BOLD);
    let inactive_style = Style::default().fg(Color::DarkGray).bg(Color::Black);
    let hint_style = Style::default().fg(Color::DarkGray);

    let chat_style = if app.active_tab == ActiveTab::Chat {
        active_style
    } else {
        inactive_style
    };
    let state_style = if app.active_tab == ActiveTab::State {
        active_style
    } else {
        inactive_style
    };
    let search_style = if app.active_tab == ActiveTab::Search {
        active_style
    } else {
        inactive_style
    };

    let tabs = Line::from(vec![
        Span::styled(" Chat ", chat_style),
        Span::raw(" "),
        Span::styled(" State ", state_style),
        Span::raw(" "),
        Span::styled(" Search ", search_style),
        Span::raw("    "),
        Span::styled("(Shift+Tab to switch)", hint_style),
    ]);

    let paragraph = Paragraph::new(tabs).style(Style::default().bg(Color::Black));
    frame.render_widget(paragraph, area);
}

/// Render the state display tab
fn render_state_display(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" System State ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();

    if app.state_display.entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "No state entries found. Press 'r' to refresh.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (key, value, type_str) in &app.state_display.entries {
            // Color-code by component prefix
            let prefix_color = get_prefix_color(key);

            // Format type indicator
            let type_indicator = match type_str.as_str() {
                "String" => "[S]",
                "Int" => "[I]",
                "Float" => "[F]",
                "Bool" => "[B]",
                "Json" => "[J]",
                "Bytes" => "[Y]",
                "Null" => "[N]",
                _ => "[?]",
            };

            let type_style = Style::default().fg(Color::DarkGray);
            let key_style = Style::default()
                .fg(prefix_color)
                .add_modifier(Modifier::BOLD);
            let value_style = Style::default().fg(Color::White);

            // Truncate long values
            let display_value = if value.len() > 60 {
                format!("{}...", &value[..57])
            } else {
                value.clone()
            };

            lines.push(Line::from(vec![
                Span::styled(format!("{} ", type_indicator), type_style),
                Span::styled(format!("{}: ", key), key_style),
                Span::styled(display_value, value_style),
            ]));
        }
    }

    // Add refresh timestamp at the bottom
    lines.push(Line::from(""));
    if let Some(ref last_refresh) = app.state_display.last_refresh {
        lines.push(Line::from(Span::styled(
            format!(
                "Last refresh: {} | Press 'r' to refresh",
                last_refresh.format("%H:%M:%S")
            ),
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "Press 'r' to refresh",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let total_lines = lines.len();
    let visible_height = inner.height as usize;

    // Clamp scroll_offset to valid range
    let max_scroll = total_lines.saturating_sub(visible_height);
    let scroll_offset = app.state_display.scroll_offset.min(max_scroll);

    // Calculate scroll position (scroll_offset is lines from bottom)
    let end_line = total_lines.saturating_sub(scroll_offset);
    let start_line = end_line.saturating_sub(visible_height);

    // Get visible lines
    let visible_lines: Vec<Line> = lines
        .into_iter()
        .skip(start_line)
        .take(visible_height)
        .collect();

    let paragraph = Paragraph::new(Text::from(visible_lines));
    frame.render_widget(paragraph, inner);

    // Show scroll indicator if there are more lines above
    if start_line > 0 {
        let indicator = Paragraph::new(format!("↑ {} more", start_line))
            .style(Style::default().fg(Color::DarkGray));
        let indicator_area = Rect {
            x: inner.x + inner.width.saturating_sub(12),
            y: inner.y,
            width: 12,
            height: 1,
        };
        frame.render_widget(indicator, indicator_area);
    }
}

/// Render the search tab display
fn render_search_display(frame: &mut Frame, app: &App, area: Rect) {
    use crate::app::SearchFocus;

    // Split into search input area and results area
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // Search input (2 lines + border)
            Constraint::Min(5),    // Results list
        ])
        .split(area);

    // Render search input
    let input_border_color = if app.search_state.focus == SearchFocus::Input {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let mode_str = app.search_state.mode.as_str();
    let input_title = format!(
        " Search Query [{}] (Tab: focus, Enter: search, F2: mode) ",
        mode_str
    );
    let input_block = Block::default()
        .title(input_title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(input_border_color));

    let input_inner = input_block.inner(chunks[0]);
    frame.render_widget(input_block, chunks[0]);

    // Render the query text with word wrapping
    let query_text = if app.search_state.query.is_empty() {
        Span::styled(
            "Type your search query...",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::styled(&app.search_state.query, Style::default().fg(Color::White))
    };

    let query_paragraph =
        Paragraph::new(Line::from(query_text)).wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(query_paragraph, input_inner);

    // Show cursor if input is focused
    if app.search_state.focus == SearchFocus::Input {
        let cursor_x = input_inner.x + app.search_state.query.chars().count() as u16;
        let cursor_x = cursor_x.min(input_inner.x + input_inner.width - 1);
        frame.set_cursor_position((cursor_x, input_inner.y));
    }

    // Render results area
    let results_border_color = if app.search_state.focus == SearchFocus::Results {
        Color::Yellow
    } else {
        Color::Blue
    };

    let context_count = app.search_state.context_ids.len();
    let results_title = if app.search_state.query.is_empty() {
        format!(" Context Documents ({}) ", context_count)
    } else {
        format!(" Search Results ({}) ", app.search_state.results.len())
    };

    let results_block = Block::default()
        .title(results_title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(results_border_color));

    let results_inner = results_block.inner(chunks[1]);
    frame.render_widget(results_block, chunks[1]);

    // Render results list
    if app.search_state.results.is_empty() {
        let empty_text = if app.search_state.is_loading {
            "Searching..."
        } else if let Some(ref error) = app.search_state.error {
            error.as_str()
        } else if app.search_state.query.is_empty() {
            "No context documents. Search to find documents to add."
        } else {
            "No results found."
        };

        let empty_paragraph = Paragraph::new(Span::styled(
            empty_text,
            Style::default().fg(Color::DarkGray),
        ));
        frame.render_widget(empty_paragraph, results_inner);
    } else {
        let visible_height = results_inner.height as usize;
        let selected = app.search_state.selected_index;

        // Calculate scroll to keep selection visible
        let start_idx = if selected >= visible_height {
            selected - visible_height + 1
        } else {
            0
        };

        let mut lines: Vec<Line> = Vec::new();

        for (i, result) in app
            .search_state
            .results
            .iter()
            .enumerate()
            .skip(start_idx)
            .take(visible_height)
        {
            let is_selected = i == selected;

            // Build the line
            let context_indicator = if result.is_in_context {
                Span::styled("[+] ", Style::default().fg(Color::Green))
            } else {
                Span::styled("[ ] ", Style::default().fg(Color::DarkGray))
            };

            // Source info - constrain to max 25 characters
            const MAX_SOURCE_LEN: usize = 25;
            let source_text = result
                .source
                .as_ref()
                .map(|s| {
                    // Extract just the filename
                    let filename = s.rsplit('/').next().unwrap_or(s);
                    // Truncate if too long
                    if filename.len() > MAX_SOURCE_LEN {
                        format!("{}...", &filename[..MAX_SOURCE_LEN - 3])
                    } else {
                        filename.to_string()
                    }
                })
                .unwrap_or_else(|| "unknown".to_string());

            // Distance info
            let distance_text = result
                .distance
                .map(|d| format!(" ({:.3})", d))
                .unwrap_or_default();

            // Truncate text preview - account for source column width
            let max_preview_len = results_inner.width as usize - MAX_SOURCE_LEN - 20;
            let preview = result.text.lines().next().unwrap_or("");
            let preview = if preview.len() > max_preview_len {
                format!("{}...", &preview[..max_preview_len.saturating_sub(3)])
            } else {
                preview.to_string()
            };

            let base_style = if is_selected {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default().fg(Color::White)
            };

            let source_style = if is_selected {
                Style::default().bg(Color::DarkGray).fg(Color::Cyan)
            } else {
                Style::default().fg(Color::Cyan)
            };

            let preview_style = if is_selected {
                Style::default().bg(Color::DarkGray).fg(Color::Green)
            } else {
                Style::default().fg(Color::Green)
            };

            // Pad source text to fixed width for alignment
            let padded_source = format!("{:<width$}", source_text, width = MAX_SOURCE_LEN);

            lines.push(Line::from(vec![
                context_indicator,
                Span::styled(padded_source, source_style),
                Span::styled(distance_text, base_style),
                Span::styled(" - ", base_style),
                Span::styled(preview, preview_style),
            ]));
        }

        let paragraph = Paragraph::new(Text::from(lines));
        frame.render_widget(paragraph, results_inner);

        // Show scroll indicator if needed
        if app.search_state.results.len() > visible_height {
            let scroll_info = format!("{}/{}", selected + 1, app.search_state.results.len());
            let scroll_paragraph =
                Paragraph::new(scroll_info).style(Style::default().fg(Color::DarkGray));
            let scroll_area = Rect {
                x: results_inner.x + results_inner.width.saturating_sub(10),
                y: chunks[1].y,
                width: 10,
                height: 1,
            };
            frame.render_widget(scroll_paragraph, scroll_area);
        }
    }

    // Render help line at the bottom of results area
    let help_text = if app.search_state.focus == SearchFocus::Input {
        "Tab: focus results | Enter: search | F3: add entry | Esc: quit"
    } else {
        "Tab: input | Enter: view | a: add | r: remove | F3: new | Esc: quit"
    };

    let help_area = Rect {
        x: chunks[1].x + 1,
        y: chunks[1].y + chunks[1].height - 1,
        width: chunks[1].width - 2,
        height: 1,
    };

    let help_paragraph = Paragraph::new(Span::styled(
        help_text,
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(help_paragraph, help_area);
}

/// Render a modal overlay showing the full document text
fn render_document_modal(frame: &mut Frame, app: &App) {
    use ratatui::widgets::Clear;

    let area = frame.area();

    // Create a centered modal that takes up most of the screen
    let modal_width = (area.width as f32 * 0.85) as u16;
    let modal_height = (area.height as f32 * 0.85) as u16;
    let modal_x = (area.width - modal_width) / 2;
    let modal_y = (area.height - modal_height) / 2;

    let modal_area = Rect {
        x: modal_x,
        y: modal_y,
        width: modal_width,
        height: modal_height,
    };

    // Clear the modal area completely first
    frame.render_widget(Clear, modal_area);

    if let Some(result) = app.get_selected_search_result() {
        // Strip common directory prefix to save space in the header
        let source_text = result
            .source
            .as_deref()
            .unwrap_or("Unknown source")
            .strip_prefix("/Users/olorin/Documents/AI_IN/")
            .or_else(|| {
                result
                    .source
                    .as_deref()
                    .unwrap_or("Unknown source")
                    .strip_prefix("~/Documents/AI_IN/")
            })
            .unwrap_or_else(|| result.source.as_deref().unwrap_or("Unknown source"));

        let context_status = if result.is_in_context {
            "[In Context]"
        } else {
            "[Not in Context]"
        };

        let title = format!(" {} {} ", source_text, context_status);

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .style(Style::default().bg(Color::Black));

        let inner = block.inner(modal_area);
        frame.render_widget(block, modal_area);

        // Render the document text with word wrapping and markdown
        let wrap_width = inner.width.saturating_sub(2) as usize;
        let content_lines = render_markdown(&result.text, wrap_width);

        // Add help text at the bottom
        let mut all_lines = content_lines;
        all_lines.push(Line::from(""));
        all_lines.push(Line::from(Span::styled(
            "─".repeat(wrap_width.min(60)),
            Style::default().fg(Color::DarkGray),
        )));
        all_lines.push(Line::from(Span::styled(
            "Esc: close | a: add to context | r: remove from context",
            Style::default().fg(Color::DarkGray),
        )));

        let paragraph = Paragraph::new(Text::from(all_lines))
            .wrap(ratatui::widgets::Wrap { trim: false })
            .style(Style::default().bg(Color::Black));

        frame.render_widget(paragraph, inner);
    }
}

/// Render a quit confirmation modal
fn render_quit_modal(frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();

    // Create a compact centered modal
    let modal_width = 22;
    let modal_height = 3;
    let modal_x = (area.width.saturating_sub(modal_width)) / 2;
    let modal_y = (area.height.saturating_sub(modal_height)) / 2;

    let modal_area = Rect {
        x: modal_x,
        y: modal_y,
        width: modal_width,
        height: modal_height,
    };

    // Clear the modal area
    frame.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(" Quit? ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    // Button-style layout: [ Yes (y) ]  [ No (n) ]
    let button_style = Style::default().fg(Color::White).bg(Color::DarkGray);
    let key_style = Style::default()
        .fg(Color::Yellow)
        .bg(Color::DarkGray)
        .add_modifier(Modifier::BOLD);

    let text = Line::from(vec![
        Span::styled(" Yes ", button_style),
        Span::styled("y", key_style),
        Span::styled(" ", button_style),
        Span::raw("  "),
        Span::styled(" No ", button_style),
        Span::styled("n", key_style),
        Span::styled(" ", button_style),
    ]);

    let paragraph = Paragraph::new(text)
        .style(Style::default().bg(Color::Black))
        .alignment(ratatui::layout::Alignment::Center);
    frame.render_widget(paragraph, inner);
}

/// Render the search help modal
fn render_search_help_modal(frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();

    // Create a centered modal taking 80% of width, 85% of height
    let modal_width = (area.width as f32 * 0.80) as u16;
    let modal_height = (area.height as f32 * 0.85) as u16;
    let modal_x = (area.width - modal_width) / 2;
    let modal_y = (area.height - modal_height) / 2;

    let modal_area = Rect {
        x: modal_x,
        y: modal_y,
        width: modal_width,
        height: modal_height,
    };

    // Clear the modal area
    frame.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(" Search Help (Press Esc to close) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    let header_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let key_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let text_style = Style::default().fg(Color::White);
    let dim_style = Style::default().fg(Color::DarkGray);

    let help_lines: Vec<Line> = vec![
        Line::from(Span::styled("SEARCH MODES", header_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("Press ", text_style),
            Span::styled("F2", key_style),
            Span::styled(" to toggle between search modes:", text_style),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Semantic  ", key_style),
            Span::styled("Finds documents by meaning using embeddings", text_style),
        ]),
        Line::from(vec![
            Span::styled("  Source    ", key_style),
            Span::styled("Finds documents by filename/path substring", text_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("SEMANTIC SEARCH", header_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("Uses ", text_style),
            Span::styled("semantic embeddings", key_style),
            Span::styled(" to find documents by meaning:", text_style),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  • ", dim_style),
            Span::styled("\"how to cook pasta\"", key_style),
            Span::styled(" finds docs about ", text_style),
            Span::styled("making noodles", key_style),
        ]),
        Line::from(vec![
            Span::styled("  • ", dim_style),
            Span::styled("\"error handling\"", key_style),
            Span::styled(" finds docs about ", text_style),
            Span::styled("exceptions, try/catch", key_style),
        ]),
        Line::from(vec![
            Span::styled("  • ", dim_style),
            Span::styled("Distance score", text_style),
            Span::styled(" (0.xxx) shows similarity - ", dim_style),
            Span::styled("lower = more similar", key_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("CONTEXT DOCUMENTS", header_style)),
        Line::from(""),
        Line::from(Span::styled(
            "Documents added to context are included in AI conversations.",
            text_style,
        )),
        Line::from(Span::styled(
            "The token count shows total size of context for model limits.",
            text_style,
        )),
        Line::from(vec![
            Span::styled("[+]", Style::default().fg(Color::Green)),
            Span::styled(" = in context, ", dim_style),
            Span::styled("[ ]", dim_style),
            Span::styled(" = not in context", dim_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("KEYBOARD SHORTCUTS", header_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  F2          ", key_style),
            Span::styled("Toggle search mode (Semantic/Source)", text_style),
        ]),
        Line::from(vec![
            Span::styled("  F3          ", key_style),
            Span::styled("Add manual entry to ChromaDB", text_style),
        ]),
        Line::from(vec![
            Span::styled("  Tab         ", key_style),
            Span::styled("Switch between search input and results", text_style),
        ]),
        Line::from(vec![
            Span::styled("  Enter       ", key_style),
            Span::styled("Execute search / View selected document", text_style),
        ]),
        Line::from(vec![
            Span::styled("  ↑/↓         ", key_style),
            Span::styled("Navigate results", text_style),
        ]),
        Line::from(vec![
            Span::styled("  PgUp/PgDn   ", key_style),
            Span::styled("Jump 10 results", text_style),
        ]),
        Line::from(vec![
            Span::styled("  a           ", key_style),
            Span::styled("Add selected document to context", text_style),
        ]),
        Line::from(vec![
            Span::styled("  r           ", key_style),
            Span::styled("Remove selected document from context", text_style),
        ]),
        Line::from(vec![
            Span::styled("  ?           ", key_style),
            Span::styled("Show this help", text_style),
        ]),
        Line::from(vec![
            Span::styled("  Esc         ", key_style),
            Span::styled("Close modal / Quit", text_style),
        ]),
        Line::from(vec![
            Span::styled("  Shift+Tab   ", key_style),
            Span::styled("Switch to other tabs", text_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("EMPTY SEARCH", header_style)),
        Line::from(""),
        Line::from(Span::styled(
            "When the search query is empty, the results area shows all",
            text_style,
        )),
        Line::from(Span::styled(
            "documents currently in your context.",
            text_style,
        )),
    ];

    let paragraph = Paragraph::new(Text::from(help_lines))
        .wrap(ratatui::widgets::Wrap { trim: false })
        .style(Style::default().bg(Color::Black));

    frame.render_widget(paragraph, inner);
}

/// Render the manual entry modal for adding documents to ChromaDB
fn render_manual_entry_modal(frame: &mut Frame, app: &App) {
    use crate::app::ManualEntryFocus;
    use ratatui::widgets::Clear;

    let area = frame.area();

    // Create a centered modal taking 80% of width, 70% of height
    let modal_width = (area.width as f32 * 0.80) as u16;
    let modal_height = (area.height as f32 * 0.70) as u16;
    let modal_x = (area.width - modal_width) / 2;
    let modal_y = (area.height - modal_height) / 2;

    let modal_area = Rect {
        x: modal_x,
        y: modal_y,
        width: modal_width,
        height: modal_height,
    };

    // Clear the modal area
    frame.render_widget(Clear, modal_area);

    let title = if app.search_state.manual_entry_loading {
        " Add to ChromaDB (submitting...) "
    } else {
        " Add to ChromaDB (F3) "
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    // Split inner area into sections
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Source field (1 line + border)
            Constraint::Min(5),    // Text field (multi-line + border)
            Constraint::Length(2), // Help text
        ])
        .split(inner);

    // Render source field
    let source_border_color = if app.search_state.manual_entry_focus == ManualEntryFocus::Source {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let source_block = Block::default()
        .title(" Source ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(source_border_color));

    let source_inner = source_block.inner(chunks[0]);
    frame.render_widget(source_block, chunks[0]);

    // Render source with visible cursor
    let source_text = &app.search_state.manual_entry_source;
    let source_cursor = app.search_state.manual_entry_source_cursor;
    let source_focused = app.search_state.manual_entry_focus == ManualEntryFocus::Source;

    let source_line = if source_text.is_empty() {
        if source_focused {
            Line::from(vec![
                Span::styled("│", Style::default().fg(Color::Yellow)),
                Span::styled("User Context", Style::default().fg(Color::DarkGray)),
            ])
        } else {
            Line::from(Span::styled(
                "User Context",
                Style::default().fg(Color::DarkGray),
            ))
        }
    } else {
        let mut spans: Vec<Span> = Vec::new();
        for (i, c) in source_text.chars().enumerate() {
            if source_focused && i == source_cursor {
                spans.push(Span::styled("│", Style::default().fg(Color::Yellow)));
            }
            spans.push(Span::styled(
                c.to_string(),
                Style::default().fg(Color::White),
            ));
        }
        // Cursor at end
        if source_focused && source_cursor >= source_text.chars().count() {
            spans.push(Span::styled("│", Style::default().fg(Color::Yellow)));
        }
        Line::from(spans)
    };

    let source_paragraph = Paragraph::new(source_line);
    frame.render_widget(source_paragraph, source_inner);

    // Render text field
    let text_border_color = if app.search_state.manual_entry_focus == ManualEntryFocus::Text {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let text_block = Block::default()
        .title(" Text Content ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(text_border_color));

    let text_inner = text_block.inner(chunks[1]);
    frame.render_widget(text_block, chunks[1]);

    // Render text with visible cursor and scrolling support
    let text = &app.search_state.manual_entry_text;
    let cursor_pos = app.search_state.manual_entry_text_cursor;
    let is_focused = app.search_state.manual_entry_focus == ManualEntryFocus::Text;
    let wrap_width = text_inner.width as usize;

    // Helper function for character-based wrapping (same as in render_input_area)
    fn wrap_line_manual(line: &str, width: usize) -> Vec<String> {
        if line.is_empty() || width == 0 {
            return vec![line.to_string()];
        }
        let mut wrapped = Vec::new();
        let mut current = String::new();
        let mut current_width = 0;

        for ch in line.chars() {
            let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1);
            if current_width + ch_width > width && !current.is_empty() {
                wrapped.push(current);
                current = String::new();
                current_width = 0;
            }
            current.push(ch);
            current_width += ch_width;
        }
        if !current.is_empty() || wrapped.is_empty() {
            wrapped.push(current);
        }
        wrapped
    }

    // Build wrapped display with cursor
    let text_content: Text = if text.is_empty() {
        if is_focused {
            Text::from(Line::from(vec![
                Span::styled("│", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "Enter text content...",
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        } else {
            Text::from(Span::styled(
                "Enter text content for the document...",
                Style::default().fg(Color::DarkGray),
            ))
        }
    } else if wrap_width == 0 {
        Text::from("")
    } else {
        // Split text by newlines, wrap each logical line, then build display
        let logical_lines: Vec<&str> = text.split('\n').collect();
        let mut all_wrapped_lines: Vec<String> = Vec::new();

        for logical_line in &logical_lines {
            let wrapped = wrap_line_manual(logical_line, wrap_width);
            all_wrapped_lines.extend(wrapped);
        }

        // Now build the styled text with cursor inserted at the visual position
        // We need to map cursor_pos (character index in original text) to visual position
        let mut visual_lines: Vec<Line> = Vec::new();
        let mut char_idx = 0;

        for (logical_line_idx, logical_line) in logical_lines.iter().enumerate() {
            let wrapped_parts = wrap_line_manual(logical_line, wrap_width);

            for wrapped_part in &wrapped_parts {
                let mut line_spans: Vec<Span> = Vec::new();

                for ch in wrapped_part.chars() {
                    if is_focused && char_idx == cursor_pos {
                        line_spans.push(Span::styled("│", Style::default().fg(Color::Yellow)));
                    }
                    line_spans.push(Span::styled(
                        ch.to_string(),
                        Style::default().fg(Color::White),
                    ));
                    char_idx += 1;
                }

                visual_lines.push(Line::from(line_spans));
            }

            // Account for the newline character (except for the last line)
            if logical_line_idx < logical_lines.len() - 1 {
                // If cursor is at this newline position
                if is_focused && char_idx == cursor_pos {
                    // Add cursor at end of previous line
                    if let Some(last_line) = visual_lines.last_mut() {
                        let mut spans: Vec<Span> = last_line.spans.to_vec();
                        spans.push(Span::styled("│", Style::default().fg(Color::Yellow)));
                        *last_line = Line::from(spans);
                    }
                }
                char_idx += 1; // For the '\n'
            }
        }

        // If cursor is at the very end, add it to the last line
        if is_focused && cursor_pos >= text.chars().count() {
            if let Some(last_line) = visual_lines.last_mut() {
                let mut spans: Vec<Span> = last_line.spans.to_vec();
                spans.push(Span::styled("│", Style::default().fg(Color::Yellow)));
                *last_line = Line::from(spans);
            } else {
                visual_lines.push(Line::from(Span::styled(
                    "│",
                    Style::default().fg(Color::Yellow),
                )));
            }
        }

        if visual_lines.is_empty() {
            visual_lines.push(Line::from(Span::styled(
                "│",
                Style::default().fg(Color::Yellow),
            )));
        }

        Text::from(visual_lines)
    };

    // Calculate visual cursor row for scrolling
    let visual_cursor_row: usize = if text.is_empty() || wrap_width == 0 {
        0
    } else {
        let logical_lines: Vec<&str> = text.split('\n').collect();
        let mut row = 0;
        let mut char_idx = 0;

        'outer: for (line_idx, logical_line) in logical_lines.iter().enumerate() {
            let wrapped_parts = wrap_line_manual(logical_line, wrap_width);

            for wrapped_part in &wrapped_parts {
                let part_char_count = wrapped_part.chars().count();

                if char_idx + part_char_count >= cursor_pos && char_idx <= cursor_pos {
                    // Cursor is on this wrapped line
                    break 'outer;
                }

                char_idx += part_char_count;
                row += 1;
            }

            // Account for newline
            if line_idx < logical_lines.len() - 1 {
                if char_idx == cursor_pos {
                    break 'outer;
                }
                char_idx += 1;
            }
        }
        row
    };

    // Update scroll offset to keep cursor visible
    let visible_height = text_inner.height as usize;
    let mut scroll_offset = app.search_state.manual_entry_scroll_offset.get();

    if visible_height > 0 {
        if visual_cursor_row < scroll_offset {
            scroll_offset = visual_cursor_row;
        } else if visual_cursor_row >= scroll_offset + visible_height {
            scroll_offset = visual_cursor_row - visible_height + 1;
        }
        app.search_state
            .manual_entry_scroll_offset
            .set(scroll_offset);
    }

    let text_paragraph = Paragraph::new(text_content).scroll((scroll_offset as u16, 0));
    frame.render_widget(text_paragraph, text_inner);

    // Render help text
    let help_line = Line::from(vec![
        Span::styled(
            "Tab",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": switch | ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "Enter",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": submit | ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "Shift+Enter",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": newline | ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "F4",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": copy | ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "F5",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": paste | ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "Esc",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": cancel", Style::default().fg(Color::DarkGray)),
    ]);

    let help_paragraph = Paragraph::new(help_line).alignment(ratatui::layout::Alignment::Center);
    frame.render_widget(help_paragraph, chunks[2]);
}

/// Get color for a state key based on its component prefix
fn get_prefix_color(key: &str) -> Color {
    if key.starts_with("broca.") {
        Color::Magenta
    } else if key.starts_with("cortex.") {
        Color::Green
    } else if key.starts_with("enrichener.") {
        Color::Blue
    } else if key.starts_with("hippocampus.") {
        Color::Yellow
    } else if key.starts_with("tools.") {
        Color::LightBlue
    } else if key.starts_with("system.") {
        Color::Cyan
    } else {
        Color::White
    }
}

/// Render the chat message display
fn render_chat_display(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Chat ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let wrap_width = inner.width.saturating_sub(2) as usize;
    let mut all_lines: Vec<Line> = Vec::new();

    for msg in &app.messages {
        match msg {
            DisplayMessage::Chat(chat_msg) => {
                // Add timestamp header
                let time = chat_msg.created_at.format("%H:%M:%S");
                let role_display = match chat_msg.role.as_str() {
                    "user" => "You",
                    "assistant" => "AI",
                    _ => &chat_msg.role,
                };

                let header_style = match chat_msg.role.as_str() {
                    "user" => Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                    "assistant" => Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                    _ => Style::default().add_modifier(Modifier::BOLD),
                };

                all_lines.push(Line::from(Span::styled(
                    format!("[{}] {}:", time, role_display),
                    header_style,
                )));

                // Render content with markdown for assistant, plain for user
                if chat_msg.role == "assistant" {
                    let md_lines = render_markdown(&chat_msg.content, wrap_width);
                    all_lines.extend(md_lines);
                } else {
                    // User messages: simple wrapped text
                    let base_style = Style::default().fg(Color::Cyan);
                    for line in textwrap::wrap(&chat_msg.content, wrap_width.max(20)) {
                        all_lines.push(Line::from(Span::styled(line.to_string(), base_style)));
                    }
                }
            }
            DisplayMessage::System(sys_msg) => {
                let style = Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::ITALIC);
                let time = sys_msg.created_at.format("%H:%M:%S");
                all_lines.push(Line::from(Span::styled(
                    format!("[{}] {}", time, sys_msg.content),
                    style,
                )));
            }
        }
        // Blank line between messages
        all_lines.push(Line::from(""));
    }

    let total_lines = all_lines.len();
    let visible_height = inner.height as usize;

    // Clamp scroll_offset to valid range
    let max_scroll = total_lines.saturating_sub(visible_height);
    let scroll_offset = app.scroll_offset.min(max_scroll);

    // Calculate scroll position (scroll_offset is lines from bottom)
    let end_line = total_lines.saturating_sub(scroll_offset);
    let start_line = end_line.saturating_sub(visible_height);

    // Get visible lines
    let visible_lines: Vec<Line> = all_lines
        .into_iter()
        .skip(start_line)
        .take(visible_height)
        .collect();

    let paragraph = Paragraph::new(Text::from(visible_lines));
    frame.render_widget(paragraph, inner);

    // Show scroll indicator if there are more lines above
    if start_line > 0 {
        let indicator = Paragraph::new(format!("↑ {} lines", start_line))
            .style(Style::default().fg(Color::DarkGray));
        let indicator_area = Rect {
            x: inner.x + inner.width.saturating_sub(14),
            y: inner.y,
            width: 14,
            height: 1,
        };
        frame.render_widget(indicator, indicator_area);
    }
}

/// Render markdown content to styled lines
fn render_markdown(content: &str, wrap_width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;
    let mut in_table = false;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut col_widths: Vec<usize> = Vec::new();

    let base_style = Style::default().fg(Color::Green);
    let code_style = Style::default()
        .fg(Color::Rgb(180, 180, 180))
        .bg(Color::Rgb(40, 40, 40));
    let header_style = Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD);
    let bold_style = Style::default()
        .fg(Color::Green)
        .add_modifier(Modifier::BOLD);
    let italic_style = Style::default()
        .fg(Color::Green)
        .add_modifier(Modifier::ITALIC);
    let table_border_style = Style::default().fg(Color::DarkGray);

    for line in content.lines() {
        // Code block handling
        if line.trim().starts_with("```") {
            if in_code_block {
                in_code_block = false;
                lines.push(Line::from(Span::styled(
                    "└─────────────────────────────────────────┘",
                    table_border_style,
                )));
            } else {
                in_code_block = true;
                let lang = line.trim().trim_start_matches("```");
                let header = if lang.is_empty() {
                    "┌─ code ─────────────────────────────────┐".to_string()
                } else {
                    format!("┌─ {} ─────────────────────────────────┐", lang)
                };
                lines.push(Line::from(Span::styled(header, table_border_style)));
            }
            continue;
        }

        if in_code_block {
            // Inside code block - render with code style
            let padded = format!(
                "│ {:<width$} │",
                line,
                width = wrap_width.saturating_sub(4).max(20)
            );
            lines.push(Line::from(Span::styled(padded, code_style)));
            continue;
        }

        // Table handling
        if line.trim().starts_with('|') && line.trim().ends_with('|') {
            let cells: Vec<String> = line
                .trim()
                .trim_matches('|')
                .split('|')
                .map(|s| s.trim().to_string())
                .collect();

            // Check if this is a separator row (|---|---|)
            if cells
                .iter()
                .all(|c| c.chars().all(|ch| ch == '-' || ch == ':'))
            {
                // Skip separator, we'll draw our own
                continue;
            }

            if !in_table {
                in_table = true;
                table_rows.clear();
                col_widths.clear();
            }

            // Update column widths
            for (i, cell) in cells.iter().enumerate() {
                if i >= col_widths.len() {
                    col_widths.push(cell.len());
                } else {
                    col_widths[i] = col_widths[i].max(cell.len());
                }
            }
            table_rows.push(cells);
            continue;
        } else if in_table {
            // End of table - render it
            lines.extend(render_table(&table_rows, &col_widths, wrap_width));
            in_table = false;
            table_rows.clear();
            col_widths.clear();
        }

        // Headers
        if let Some(stripped) = line.strip_prefix("### ") {
            lines.push(Line::from(Span::styled(
                format!("   {}", stripped),
                header_style,
            )));
            continue;
        }
        if let Some(stripped) = line.strip_prefix("## ") {
            lines.push(Line::from(Span::styled(
                format!("  {}", stripped),
                header_style,
            )));
            continue;
        }
        if let Some(stripped) = line.strip_prefix("# ") {
            lines.push(Line::from(Span::styled(
                stripped.to_string(),
                header_style.add_modifier(Modifier::UNDERLINED),
            )));
            continue;
        }

        // List items
        if line.trim().starts_with("- ") || line.trim().starts_with("* ") {
            let indent = line.len() - line.trim_start().len();
            let bullet_line = format!("{}• {}", " ".repeat(indent), &line.trim()[2..]);
            lines.extend(wrap_styled_line(
                &bullet_line,
                wrap_width,
                base_style,
                bold_style,
                italic_style,
                code_style,
            ));
            continue;
        }

        // Numbered lists
        if let Some(num_end) = line.trim().find(". ") {
            let prefix = &line.trim()[..num_end];
            if prefix.chars().all(|c| c.is_ascii_digit()) {
                let indent = line.len() - line.trim_start().len();
                let num_line = format!("{}{}", " ".repeat(indent), line.trim());
                lines.extend(wrap_styled_line(
                    &num_line,
                    wrap_width,
                    base_style,
                    bold_style,
                    italic_style,
                    code_style,
                ));
                continue;
            }
        }

        // Regular paragraph with inline formatting
        if line.trim().is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.extend(wrap_styled_line(
                line,
                wrap_width,
                base_style,
                bold_style,
                italic_style,
                code_style,
            ));
        }
    }

    // Flush any remaining table
    if in_table {
        lines.extend(render_table(&table_rows, &col_widths, wrap_width));
    }

    lines
}

/// Render a table with box drawing characters
fn render_table(
    rows: &[Vec<String>],
    col_widths: &[usize],
    max_width: usize,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let border_style = Style::default().fg(Color::DarkGray);
    let header_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let cell_style = Style::default().fg(Color::Green);

    if rows.is_empty() || col_widths.is_empty() {
        return lines;
    }

    // Adjust column widths to fit
    let total_width: usize = col_widths.iter().sum::<usize>() + col_widths.len() * 3 + 1;
    let scale = if total_width > max_width {
        max_width as f64 / total_width as f64
    } else {
        1.0
    };
    let adjusted_widths: Vec<usize> = col_widths
        .iter()
        .map(|w| ((*w as f64 * scale) as usize).max(3))
        .collect();

    // Top border
    let top_border: String = adjusted_widths
        .iter()
        .map(|w| "─".repeat(*w + 2))
        .collect::<Vec<_>>()
        .join("┬");
    lines.push(Line::from(Span::styled(
        format!("┌{}┐", top_border),
        border_style,
    )));

    for (row_idx, row) in rows.iter().enumerate() {
        // Row content
        let mut spans: Vec<Span<'static>> = vec![Span::styled("│", border_style)];
        for (col_idx, cell) in row.iter().enumerate() {
            let width = adjusted_widths.get(col_idx).copied().unwrap_or(10);
            let truncated = if cell.len() > width {
                format!("{}…", &cell[..width.saturating_sub(1)])
            } else {
                cell.clone()
            };
            let style = if row_idx == 0 {
                header_style
            } else {
                cell_style
            };
            spans.push(Span::styled(
                format!(" {:<width$} ", truncated, width = width),
                style,
            ));
            spans.push(Span::styled("│", border_style));
        }
        lines.push(Line::from(spans));

        // Separator after header
        if row_idx == 0 && rows.len() > 1 {
            let sep: String = adjusted_widths
                .iter()
                .map(|w| "─".repeat(*w + 2))
                .collect::<Vec<_>>()
                .join("┼");
            lines.push(Line::from(Span::styled(format!("├{}┤", sep), border_style)));
        }
    }

    // Bottom border
    let bottom_border: String = adjusted_widths
        .iter()
        .map(|w| "─".repeat(*w + 2))
        .collect::<Vec<_>>()
        .join("┴");
    lines.push(Line::from(Span::styled(
        format!("└{}┘", bottom_border),
        border_style,
    )));

    lines
}

/// Wrap a line with inline markdown formatting (bold, italic, code)
fn wrap_styled_line(
    text: &str,
    wrap_width: usize,
    base_style: Style,
    bold_style: Style,
    italic_style: Style,
    code_style: Style,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Parse inline formatting and create spans
    let spans = parse_inline_markdown(text, base_style, bold_style, italic_style, code_style);

    // Simple wrapping: just join all text and wrap
    let plain_text: String = spans.iter().map(|s| s.content.as_ref()).collect();

    if plain_text.len() <= wrap_width {
        lines.push(Line::from(spans));
    } else {
        // For long lines, wrap by breaking into chunks
        // This is simplified - ideally we'd preserve styling across wraps
        for wrapped in textwrap::wrap(&plain_text, wrap_width.max(20)) {
            lines.push(Line::from(Span::styled(wrapped.to_string(), base_style)));
        }
    }

    lines
}

/// Parse inline markdown formatting into styled spans
fn parse_inline_markdown(
    text: &str,
    base_style: Style,
    bold_style: Style,
    italic_style: Style,
    code_style: Style,
) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '`' => {
                // Inline code
                if !current.is_empty() {
                    spans.push(Span::styled(current.clone(), base_style));
                    current.clear();
                }
                let mut code = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '`' {
                        chars.next();
                        break;
                    }
                    code.push(chars.next().unwrap());
                }
                spans.push(Span::styled(format!(" {} ", code), code_style));
            }
            '*' => {
                // Check for bold (**) or italic (*)
                if chars.peek() == Some(&'*') {
                    chars.next(); // consume second *
                    if !current.is_empty() {
                        spans.push(Span::styled(current.clone(), base_style));
                        current.clear();
                    }
                    let mut bold_text = String::new();
                    while let Some(&next) = chars.peek() {
                        if next == '*' {
                            chars.next();
                            if chars.peek() == Some(&'*') {
                                chars.next();
                                break;
                            }
                            bold_text.push('*');
                        } else {
                            bold_text.push(chars.next().unwrap());
                        }
                    }
                    spans.push(Span::styled(bold_text, bold_style));
                } else {
                    // Single * for italic
                    if !current.is_empty() {
                        spans.push(Span::styled(current.clone(), base_style));
                        current.clear();
                    }
                    let mut italic_text = String::new();
                    while let Some(&next) = chars.peek() {
                        if next == '*' {
                            chars.next();
                            break;
                        }
                        italic_text.push(chars.next().unwrap());
                    }
                    spans.push(Span::styled(italic_text, italic_style));
                }
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        spans.push(Span::styled(current, base_style));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base_style));
    }

    spans
}

/// Render the input text area with visual soft-wrapping and ghost text completion
fn render_input_area(frame: &mut Frame, app: &App, area: Rect) {
    use ratatui::text::Line as TextLine;
    use ratatui::text::Text;

    let block = Block::default()
        .title(" Input ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Get text and cursor from tui-textarea
    let lines = app.input.lines();
    let (cursor_row, cursor_col) = app.input.cursor();

    let text_style = Style::default().fg(Color::White);
    let ghost_style = Style::default().fg(Color::DarkGray);

    // Calculate wrap width for manual character-based wrapping
    let wrap_width = inner.width as usize;
    if wrap_width == 0 {
        return;
    }

    // Manually wrap text at exact character boundaries to match cursor calculation
    // This ensures the visual cursor position matches the rendered text
    fn wrap_line(line: &str, width: usize) -> Vec<String> {
        if line.is_empty() || width == 0 {
            return vec![line.to_string()];
        }
        let mut wrapped = Vec::new();
        let mut current = String::new();
        let mut current_width = 0;

        for ch in line.chars() {
            // Use unicode width for proper character width calculation
            let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1);
            if current_width + ch_width > width && !current.is_empty() {
                wrapped.push(current);
                current = String::new();
                current_width = 0;
            }
            current.push(ch);
            current_width += ch_width;
        }
        if !current.is_empty() || wrapped.is_empty() {
            wrapped.push(current);
        }
        wrapped
    }

    // Build pre-wrapped display text
    let wrapped_display: Text = if lines.is_empty() {
        Text::from("")
    } else if lines.len() == 1 {
        // Single line - may have ghost completion
        let full_line = if let Some(ref completion) = app.completion {
            format!("{}{}", lines[0], completion)
        } else {
            lines[0].to_string()
        };
        let wrapped_parts = wrap_line(&full_line, wrap_width);
        let user_char_count = lines[0].chars().count();

        // Build styled lines, handling the transition from user text to ghost text
        let styled_lines: Vec<TextLine> = wrapped_parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                // Calculate character offset at start of this wrapped line
                let start_offset: usize =
                    wrapped_parts[..i].iter().map(|s| s.chars().count()).sum();
                let part_char_count = part.chars().count();

                if app.completion.is_some() {
                    // Determine where user text ends within this line
                    if start_offset >= user_char_count {
                        // Entire line is ghost text
                        TextLine::from(Span::styled(part.clone(), ghost_style))
                    } else if start_offset + part_char_count <= user_char_count {
                        // Entire line is user text
                        TextLine::from(Span::styled(part.clone(), text_style))
                    } else {
                        // Mixed: part user text, part ghost text
                        let user_chars_in_line = user_char_count - start_offset;
                        let (user_part, ghost_part): (String, String) = {
                            let chars: Vec<char> = part.chars().collect();
                            (
                                chars[..user_chars_in_line].iter().collect(),
                                chars[user_chars_in_line..].iter().collect(),
                            )
                        };
                        TextLine::from(vec![
                            Span::styled(user_part, text_style),
                            Span::styled(ghost_part, ghost_style),
                        ])
                    }
                } else {
                    TextLine::from(Span::styled(part.clone(), text_style))
                }
            })
            .collect();
        Text::from(styled_lines)
    } else {
        // Multi-line: wrap each line separately, no ghost text
        let mut all_lines: Vec<TextLine> = Vec::new();
        for line in lines.iter() {
            let wrapped_parts = wrap_line(line, wrap_width);
            for part in wrapped_parts {
                all_lines.push(TextLine::from(Span::styled(part, text_style)));
            }
        }
        Text::from(all_lines)
    };

    // Calculate visual cursor position using the same wrapping logic
    let mut visual_row: u16 = 0;
    let mut visual_col: u16 = 0;

    for (line_idx, line) in lines.iter().enumerate() {
        let wrapped_parts = wrap_line(line, wrap_width);

        if line_idx < cursor_row {
            // Count how many visual rows this line takes
            visual_row += wrapped_parts.len() as u16;
        } else {
            // This is the cursor line - find cursor position within wrapped parts
            let mut chars_remaining = cursor_col;
            for (wrap_idx, part) in wrapped_parts.iter().enumerate() {
                let part_char_count = part.chars().count();
                if chars_remaining <= part_char_count {
                    // Cursor is on this wrapped line
                    visual_row += wrap_idx as u16;
                    // Calculate visual column using unicode width
                    visual_col = part
                        .chars()
                        .take(chars_remaining)
                        .map(|c| unicode_width::UnicodeWidthChar::width(c).unwrap_or(1) as u16)
                        .sum();
                    break;
                }
                chars_remaining -= part_char_count;
            }
            break;
        }
    }

    // Update scroll offset to keep cursor visible (uses Cell for interior mutability)
    let visible_height = inner.height as usize;
    let mut scroll_offset = app.input_scroll_offset.get();

    if visible_height > 0 {
        let cursor_row_usize = visual_row as usize;

        // If cursor is above visible area, scroll up
        if cursor_row_usize < scroll_offset {
            scroll_offset = cursor_row_usize;
        }
        // If cursor is below visible area, scroll down
        else if cursor_row_usize >= scroll_offset + visible_height {
            scroll_offset = cursor_row_usize - visible_height + 1;
        }

        app.input_scroll_offset.set(scroll_offset);
    }

    // Render pre-wrapped text with scroll offset applied
    let paragraph = Paragraph::new(wrapped_display).scroll((scroll_offset as u16, 0));
    frame.render_widget(paragraph, inner);

    // Position cursor relative to scroll offset
    let adjusted_row = visual_row.saturating_sub(scroll_offset as u16);
    if (adjusted_row as usize) < visible_height {
        let cursor_x = inner
            .x
            .saturating_add(visual_col)
            .min(inner.x + inner.width - 1);
        let cursor_y = inner.y.saturating_add(adjusted_row);
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

/// Render the status bar
fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let status_text = match app.active_tab {
        ActiveTab::Chat => {
            let scroll_indicator = if app.auto_scroll { "" } else { " [SCROLLED] " };
            let tab_hint = if app.completion.is_some() {
                "Tab: Complete | "
            } else {
                ""
            };

            // Format token count with K suffix for thousands
            let token_display = if app.token_count >= 1000 {
                format!("{:.1}K", app.token_count as f64 / 1000.0)
            } else {
                format!("{}", app.token_count)
            };

            format!(
                " {} | Msgs: {} | Tokens: {}{} | {}Enter: Send | Shift+Enter: Newline | Esc: Quit ",
                app.status,
                app.message_count(),
                token_display,
                scroll_indicator,
                tab_hint
            )
        }
        ActiveTab::State => {
            format!(
                " {} | Entries: {} | r: Refresh | Up/Down: Scroll | Esc: Quit ",
                app.status,
                app.state_display.entries.len()
            )
        }
        ActiveTab::Search => {
            let context_count = app.search_state.context_ids.len();
            // Format token count with K suffix for thousands
            let token_display = if app.search_state.context_token_count >= 1000 {
                format!(
                    "{:.1}K",
                    app.search_state.context_token_count as f64 / 1000.0
                )
            } else {
                format!("{}", app.search_state.context_token_count)
            };
            format!(
                " {} | Results: {} | Context: {} ({} tokens) | ?: help ",
                app.status,
                app.search_state.results.len(),
                context_count,
                token_display
            )
        }
    };

    let style = if app.is_sending {
        Style::default().fg(Color::Yellow).bg(Color::DarkGray)
    } else if app.active_tab == ActiveTab::Chat && !app.auto_scroll {
        Style::default().fg(Color::Cyan).bg(Color::DarkGray)
    } else if app.status.starts_with("Error") || app.status.starts_with("Failed") {
        Style::default().fg(Color::Red).bg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White).bg(Color::DarkGray)
    };

    let paragraph = Paragraph::new(status_text).style(style);
    frame.render_widget(paragraph, area);
}
