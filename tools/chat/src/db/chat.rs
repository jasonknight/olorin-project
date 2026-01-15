//! Chat history database monitoring

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::message::ChatMessage;

/// Chat database reader
pub struct ChatDb {
    path: PathBuf,
    seen_ids: HashSet<String>,
}

impl ChatDb {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            seen_ids: HashSet::new(),
        }
    }

    /// Check if the database file exists
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Get all messages for initial load
    pub fn get_all_messages(&mut self) -> Result<Vec<ChatMessage>> {
        if !self.exists() {
            return Ok(Vec::new());
        }

        let conn = Connection::open(&self.path)?;

        let mut stmt = conn.prepare(
            r#"
            SELECT m.id, m.role, m.content, m.created_at
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.is_active = 1
            ORDER BY m.created_at ASC
            "#,
        )?;

        let messages: Vec<ChatMessage> = stmt
            .query_map([], |row| {
                Ok(ChatMessage::from_db_row(
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Track all seen message IDs
        for msg in &messages {
            self.seen_ids.insert(msg.id.clone());
        }

        Ok(messages)
    }

    /// Get new messages since last check
    pub fn get_new_messages(&mut self) -> Result<Vec<ChatMessage>> {
        if !self.exists() {
            return Ok(Vec::new());
        }

        // If we haven't loaded yet, get all messages
        if self.seen_ids.is_empty() {
            return self.get_all_messages();
        }

        let conn = Connection::open(&self.path)?;

        let mut stmt = conn.prepare(
            r#"
            SELECT m.id, m.role, m.content, m.created_at
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.is_active = 1
            ORDER BY m.created_at ASC
            "#,
        )?;

        let all_messages: Vec<ChatMessage> = stmt
            .query_map([], |row| {
                Ok(ChatMessage::from_db_row(
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Filter to only new messages (not in seen_ids)
        let new_messages: Vec<ChatMessage> = all_messages
            .into_iter()
            .filter(|msg| !self.seen_ids.contains(&msg.id))
            .collect();

        // Track new message IDs
        for msg in &new_messages {
            self.seen_ids.insert(msg.id.clone());
        }

        Ok(new_messages)
    }

    /// Get statistics about the chat database
    pub fn get_stats(&self) -> Result<ChatStats> {
        if !self.exists() {
            return Ok(ChatStats::default());
        }

        let conn = Connection::open(&self.path)?;

        let conversation_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM conversations WHERE is_active = 1", [], |row| row.get(0))
            .unwrap_or(0);

        let message_count: i64 = conn
            .query_row(
                r#"
                SELECT COUNT(*) FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.is_active = 1
                "#,
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(ChatStats {
            conversation_count: conversation_count as usize,
            message_count: message_count as usize,
        })
    }
}

#[derive(Debug, Default)]
pub struct ChatStats {
    pub conversation_count: usize,
    pub message_count: usize,
}
