//! UI rendering with ratatui

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};

use crate::app::App;
use crate::message::DisplayMessage;

/// Main UI rendering function
pub fn render(frame: &mut Frame, app: &App) {
    // Create layout: main area + status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),   // Chat + input area
            Constraint::Length(1), // Status bar
        ])
        .split(frame.area());

    // Split main area into chat display and input
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // Chat display (expands)
            Constraint::Length(5), // Input area (3 lines + border)
        ])
        .split(chunks[0]);

    render_chat_display(frame, app, main_chunks[0]);
    render_input_area(frame, app, main_chunks[1]);
    render_status_bar(frame, app, chunks[1]);
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

    // Render pre-wrapped text without additional wrapping
    let paragraph = Paragraph::new(wrapped_display);
    frame.render_widget(paragraph, inner);

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

    // Position cursor (clamped to visible area)
    let visible_height = inner.height;
    if visual_row < visible_height {
        let cursor_x = inner
            .x
            .saturating_add(visual_col)
            .min(inner.x + inner.width - 1);
        let cursor_y = inner.y.saturating_add(visual_row);
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

/// Render the status bar
fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let scroll_indicator = if app.auto_scroll { "" } else { " [SCROLLED] " };
    let tab_hint = if app.completion.is_some() {
        "Tab: Complete | "
    } else {
        ""
    };

    let status_text = format!(
        " {} | Msgs: {}{} | {}Enter: Send | Shift+Enter: Newline | Esc: Quit ",
        app.status,
        app.message_count(),
        scroll_indicator,
        tab_hint
    );

    let style = if app.is_sending {
        Style::default().fg(Color::Yellow).bg(Color::DarkGray)
    } else if !app.auto_scroll {
        Style::default().fg(Color::Cyan).bg(Color::DarkGray)
    } else if app.status.starts_with("Error") {
        Style::default().fg(Color::Red).bg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White).bg(Color::DarkGray)
    };

    let paragraph = Paragraph::new(status_text).style(style);
    frame.render_widget(paragraph, area);
}
