//! Input widget rendering

use crate::app::SettingValue;
use crate::settings::InputType;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

/// Render an input field based on its type
pub fn render_input(frame: &mut Frame, setting: &SettingValue, area: Rect, is_focused: bool) {
    match &setting.def.input_type {
        InputType::Text | InputType::NullableText => {
            render_text_input(frame, setting, area, is_focused);
        }
        InputType::Textarea => {
            render_textarea_input(frame, setting, area, is_focused);
        }
        InputType::Select(_) | InputType::DynamicSelect(_) => {
            render_select_input(frame, setting, area, is_focused);
        }
        InputType::IntNumber { .. }
        | InputType::FloatNumber { .. }
        | InputType::NullableInt { .. } => {
            render_number_input(frame, setting, area, is_focused);
        }
        InputType::Toggle => {
            render_toggle_input(frame, setting, area, is_focused);
        }
    }
}

fn render_text_input(frame: &mut Frame, setting: &SettingValue, area: Rect, is_focused: bool) {
    let bg_color = if is_focused {
        Color::DarkGray
    } else {
        Color::Black
    };
    let fg_color = if is_focused {
        Color::White
    } else {
        Color::Gray
    };

    let display_text = if setting.input_buffer.is_empty() {
        if matches!(setting.def.input_type, InputType::NullableText) {
            "(null)".to_string()
        } else {
            String::new()
        }
    } else {
        setting.input_buffer.clone()
    };

    // Truncate if too long (by character count, not bytes)
    let max_len = area.width.saturating_sub(2) as usize;
    let char_count = display_text.chars().count();
    let truncated = if char_count > max_len {
        let truncated_str: String = display_text
            .chars()
            .take(max_len.saturating_sub(1))
            .collect();
        format!("{}…", truncated_str)
    } else {
        display_text
    };

    let mut spans = vec![];

    if is_focused && setting.is_editing {
        // Show cursor using UTF-8 safe method
        let (before, cursor_char, after) = setting.split_at_cursor();

        spans.push(Span::styled(
            format!("[{}", before),
            Style::default().fg(fg_color).bg(bg_color),
        ));
        spans.push(Span::styled(
            cursor_char.to_string(),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("{}]", after),
            Style::default().fg(fg_color).bg(bg_color),
        ));
    } else {
        spans.push(Span::styled(
            format!("[{}]", truncated),
            Style::default().fg(fg_color).bg(bg_color),
        ));
    }

    let paragraph = Paragraph::new(Line::from(spans));
    frame.render_widget(paragraph, area);
}

fn render_textarea_input(frame: &mut Frame, setting: &SettingValue, area: Rect, is_focused: bool) {
    let bg_color = if is_focused {
        Color::DarkGray
    } else {
        Color::Black
    };
    let fg_color = if is_focused {
        Color::White
    } else {
        Color::Gray
    };

    let line_count = setting.input_buffer.lines().count();
    let preview = if line_count > 0 {
        format!("[{} items]", line_count)
    } else {
        "[empty]".to_string()
    };

    let paragraph = Paragraph::new(Span::styled(
        preview,
        Style::default().fg(fg_color).bg(bg_color),
    ));
    frame.render_widget(paragraph, area);
}

fn render_select_input(frame: &mut Frame, setting: &SettingValue, area: Rect, is_focused: bool) {
    let options = setting.get_options();
    let current = &setting.input_buffer;

    let bg_color = if is_focused {
        Color::DarkGray
    } else {
        Color::Black
    };
    let fg_color = if is_focused { Color::Cyan } else { Color::Gray };

    let arrows = if is_focused { "◀ " } else { "" };
    let arrows_end = if is_focused { " ▶" } else { "" };

    let display = format!("{}{}{}", arrows, current, arrows_end);

    // Show position indicator if focused
    let indicator = if is_focused && !options.is_empty() {
        format!(" ({}/{})", setting.select_index + 1, options.len())
    } else {
        String::new()
    };

    let paragraph = Paragraph::new(Line::from(vec![
        Span::styled(display, Style::default().fg(fg_color).bg(bg_color)),
        Span::styled(indicator, Style::default().fg(Color::DarkGray)),
    ]));
    frame.render_widget(paragraph, area);
}

fn render_number_input(frame: &mut Frame, setting: &SettingValue, area: Rect, is_focused: bool) {
    let bg_color = if is_focused {
        Color::DarkGray
    } else {
        Color::Black
    };
    let fg_color = if setting.validation_error.is_some() {
        Color::Red
    } else if is_focused {
        Color::White
    } else {
        Color::Gray
    };

    let display_text = if setting.input_buffer.is_empty() {
        if matches!(setting.def.input_type, InputType::NullableInt { .. }) {
            "(null)".to_string()
        } else {
            "0".to_string()
        }
    } else {
        setting.input_buffer.clone()
    };

    let mut spans = vec![];

    if is_focused && setting.is_editing {
        // Show cursor using UTF-8 safe method
        let (before, cursor_char, after) = setting.split_at_cursor();

        spans.push(Span::styled(
            format!("[{}", before),
            Style::default().fg(fg_color).bg(bg_color),
        ));
        spans.push(Span::styled(
            cursor_char.to_string(),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("{}]", after),
            Style::default().fg(fg_color).bg(bg_color),
        ));
    } else {
        spans.push(Span::styled(
            format!("[{}]", display_text),
            Style::default().fg(fg_color).bg(bg_color),
        ));
    }

    let paragraph = Paragraph::new(Line::from(spans));
    frame.render_widget(paragraph, area);
}

fn render_toggle_input(frame: &mut Frame, setting: &SettingValue, area: Rect, is_focused: bool) {
    let is_true = setting.input_buffer == "true";

    let bg_color = if is_focused {
        Color::DarkGray
    } else {
        Color::Black
    };

    let (checkbox, label, color) = if is_true {
        ("[x]", " true", Color::Green)
    } else {
        ("[ ]", " false", Color::Red)
    };

    let checkbox_style = Style::default()
        .fg(if is_focused { Color::Yellow } else { color })
        .bg(bg_color);

    let label_style = Style::default().fg(color).bg(bg_color);

    let paragraph = Paragraph::new(Line::from(vec![
        Span::styled(checkbox, checkbox_style),
        Span::styled(label, label_style),
    ]));
    frame.render_widget(paragraph, area);
}
