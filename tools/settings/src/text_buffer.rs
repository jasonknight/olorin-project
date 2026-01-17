//! UTF-8 safe text buffer with cursor management
//!
//! This module provides a text buffer that handles cursor positions correctly
//! for multi-byte UTF-8 characters. The cursor position is tracked as a character
//! index, not a byte index, preventing panics from invalid UTF-8 boundary access.
//!
//! This module is kept for unit testing and potential future use.

#![allow(dead_code)]

/// A text buffer with UTF-8 safe cursor handling
#[derive(Debug, Clone, Default)]
pub struct TextBuffer {
    /// The text content
    text: String,
    /// Cursor position as character index (not byte index)
    cursor_char_idx: usize,
}

impl TextBuffer {
    /// Create a new text buffer with the given content
    pub fn new(text: String) -> Self {
        let cursor_char_idx = text.chars().count();
        Self {
            text,
            cursor_char_idx,
        }
    }

    /// Create an empty text buffer
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            cursor_char_idx: 0,
        }
    }

    /// Get the text content
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Set the text content and reset cursor to end
    pub fn set_text(&mut self, text: String) {
        self.cursor_char_idx = text.chars().count();
        self.text = text;
    }

    /// Get cursor position as character index
    pub fn cursor_pos(&self) -> usize {
        self.cursor_char_idx
    }

    /// Set cursor position (clamped to valid range)
    pub fn set_cursor_pos(&mut self, pos: usize) {
        let char_count = self.text.chars().count();
        self.cursor_char_idx = pos.min(char_count);
    }

    /// Move cursor to end of text
    pub fn cursor_to_end(&mut self) {
        self.cursor_char_idx = self.text.chars().count();
    }

    /// Move cursor to start of text
    pub fn cursor_to_start(&mut self) {
        self.cursor_char_idx = 0;
    }

    /// Get the number of characters in the buffer
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Get the byte length of the buffer
    pub fn byte_len(&self) -> usize {
        self.text.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Convert character index to byte index
    fn char_idx_to_byte_idx(&self, char_idx: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_idx)
            .map(|(byte_idx, _)| byte_idx)
            .unwrap_or(self.text.len())
    }

    /// Insert a character at the cursor position
    pub fn insert_char(&mut self, c: char) {
        let byte_idx = self.char_idx_to_byte_idx(self.cursor_char_idx);
        self.text.insert(byte_idx, c);
        self.cursor_char_idx += 1;
    }

    /// Delete the character before the cursor (backspace)
    /// Returns true if a character was deleted
    pub fn delete_char_before(&mut self) -> bool {
        if self.cursor_char_idx > 0 {
            self.cursor_char_idx -= 1;
            let byte_idx = self.char_idx_to_byte_idx(self.cursor_char_idx);
            // Find the byte length of the character at this position
            if let Some(c) = self.text[byte_idx..].chars().next() {
                self.text
                    .replace_range(byte_idx..byte_idx + c.len_utf8(), "");
                return true;
            }
        }
        false
    }

    /// Delete the character at the cursor (delete forward)
    /// Returns true if a character was deleted
    pub fn delete_char_at(&mut self) -> bool {
        let char_count = self.text.chars().count();
        if self.cursor_char_idx < char_count {
            let byte_idx = self.char_idx_to_byte_idx(self.cursor_char_idx);
            if let Some(c) = self.text[byte_idx..].chars().next() {
                self.text
                    .replace_range(byte_idx..byte_idx + c.len_utf8(), "");
                return true;
            }
        }
        false
    }

    /// Move cursor left by one character
    /// Returns true if cursor moved
    pub fn cursor_left(&mut self) -> bool {
        if self.cursor_char_idx > 0 {
            self.cursor_char_idx -= 1;
            true
        } else {
            false
        }
    }

    /// Move cursor right by one character
    /// Returns true if cursor moved
    pub fn cursor_right(&mut self) -> bool {
        let char_count = self.text.chars().count();
        if self.cursor_char_idx < char_count {
            self.cursor_char_idx += 1;
            true
        } else {
            false
        }
    }

    /// Get text before cursor and text from cursor onwards (for rendering with cursor)
    /// Returns (before_cursor, cursor_char, after_cursor)
    pub fn split_at_cursor(&self) -> (&str, char, &str) {
        let byte_idx = self.char_idx_to_byte_idx(self.cursor_char_idx);
        let before = &self.text[..byte_idx];

        if self.cursor_char_idx < self.text.chars().count() {
            let cursor_char = self.text[byte_idx..].chars().next().unwrap();
            let after_byte_idx = byte_idx + cursor_char.len_utf8();
            let after = &self.text[after_byte_idx..];
            (before, cursor_char, after)
        } else {
            (before, ' ', "")
        }
    }

    /// Check if a character is valid for this buffer at the current position
    /// Used for numeric input validation
    pub fn is_valid_numeric_char(&self, c: char, allow_float: bool, allow_negative: bool) -> bool {
        match c {
            '0'..='9' => true,
            '-' if allow_negative => {
                // Only allow minus at the start and only one
                self.cursor_char_idx == 0 && !self.text.contains('-')
            }
            '.' if allow_float => {
                // Only allow one decimal point
                !self.text.contains('.')
            }
            _ => false,
        }
    }
}

impl From<String> for TextBuffer {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for TextBuffer {
    fn from(text: &str) -> Self {
        Self::new(text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buf = TextBuffer::new("hello".to_string());
        assert_eq!(buf.text(), "hello");
        assert_eq!(buf.cursor_pos(), 5);
    }

    #[test]
    fn test_insert_char() {
        let mut buf = TextBuffer::new("helo".to_string());
        buf.set_cursor_pos(3);
        buf.insert_char('l');
        assert_eq!(buf.text(), "hello");
        assert_eq!(buf.cursor_pos(), 4);
    }

    #[test]
    fn test_insert_char_unicode() {
        let mut buf = TextBuffer::new("héllo".to_string());
        assert_eq!(buf.char_count(), 5);
        buf.set_cursor_pos(2);
        buf.insert_char('x');
        assert_eq!(buf.text(), "héxllo");
        assert_eq!(buf.cursor_pos(), 3);
    }

    #[test]
    fn test_delete_char_before() {
        let mut buf = TextBuffer::new("hello".to_string());
        buf.set_cursor_pos(3);
        buf.delete_char_before();
        assert_eq!(buf.text(), "helo");
        assert_eq!(buf.cursor_pos(), 2);
    }

    #[test]
    fn test_delete_unicode() {
        let mut buf = TextBuffer::new("hé世界".to_string());
        buf.set_cursor_pos(3);
        buf.delete_char_before();
        assert_eq!(buf.text(), "hé界");
        assert_eq!(buf.cursor_pos(), 2);
    }

    #[test]
    fn test_cursor_movement() {
        let mut buf = TextBuffer::new("ab".to_string());
        buf.set_cursor_pos(1);
        assert!(buf.cursor_left());
        assert_eq!(buf.cursor_pos(), 0);
        assert!(!buf.cursor_left());
        assert!(buf.cursor_right());
        assert_eq!(buf.cursor_pos(), 1);
    }

    #[test]
    fn test_split_at_cursor() {
        let buf = TextBuffer::new("hello".to_string());
        let mut buf2 = buf.clone();
        buf2.set_cursor_pos(2);
        let (before, cursor, after) = buf2.split_at_cursor();
        assert_eq!(before, "he");
        assert_eq!(cursor, 'l');
        assert_eq!(after, "lo");
    }

    #[test]
    fn test_split_at_cursor_unicode() {
        let mut buf = TextBuffer::new("héllo".to_string());
        buf.set_cursor_pos(1);
        let (before, cursor, after) = buf.split_at_cursor();
        assert_eq!(before, "h");
        assert_eq!(cursor, 'é');
        assert_eq!(after, "llo");
    }

    #[test]
    fn test_numeric_validation() {
        let buf = TextBuffer::new("123".to_string());
        assert!(buf.is_valid_numeric_char('4', false, false));
        assert!(!buf.is_valid_numeric_char('-', false, false));
        assert!(!buf.is_valid_numeric_char('.', false, false));

        let mut buf2 = TextBuffer::empty();
        assert!(buf2.is_valid_numeric_char('-', false, true));
        buf2.insert_char('-');
        assert!(!buf2.is_valid_numeric_char('-', false, true));
    }
}
