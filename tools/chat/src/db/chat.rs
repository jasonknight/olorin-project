//! Chat history database monitoring

use anyhow::Result;
use chrono::{DateTime, Local};
use rusqlite::Connection;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::message::ChatMessage;

/// Chat database reader
pub struct ChatDb {
    path: PathBuf,
    seen_ids: HashSet<String>,
    /// Last time we checked for updates (for detecting streaming updates)
    last_check_time: Option<DateTime<Local>>,
}

/// Result of checking for messages - separates new vs updated
pub struct MessageCheckResult {
    /// Completely new messages (never seen before)
    pub new_messages: Vec<ChatMessage>,
    /// Messages that existed but have been updated (streaming)
    pub updated_messages: Vec<ChatMessage>,
}

impl ChatDb {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            seen_ids: HashSet::new(),
            last_check_time: None,
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
            SELECT m.id, m.role, m.content, m.created_at,
                   m.message_type, m.metadata, m.updated_at
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
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Track all seen message IDs
        for msg in &messages {
            self.seen_ids.insert(msg.id.clone());
        }

        // Set the last check time to now for subsequent update detection
        self.last_check_time = Some(Local::now());

        Ok(messages)
    }

    /// Get new and updated messages since last check
    ///
    /// Returns a MessageCheckResult with:
    /// - new_messages: Messages with IDs we haven't seen before
    /// - updated_messages: Messages we've seen but whose updated_at is newer than our last check
    pub fn get_new_and_updated_messages(&mut self) -> Result<MessageCheckResult> {
        if !self.exists() {
            return Ok(MessageCheckResult {
                new_messages: Vec::new(),
                updated_messages: Vec::new(),
            });
        }

        // If we haven't loaded yet, get all messages as "new"
        if self.seen_ids.is_empty() {
            let messages = self.get_all_messages()?;
            return Ok(MessageCheckResult {
                new_messages: messages,
                updated_messages: Vec::new(),
            });
        }

        let conn = Connection::open(&self.path)?;

        let mut stmt = conn.prepare(
            r#"
            SELECT m.id, m.role, m.content, m.created_at,
                   m.message_type, m.metadata, m.updated_at
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
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let last_check = self.last_check_time;
        let mut new_messages = Vec::new();
        let mut updated_messages = Vec::new();

        for msg in all_messages {
            if !self.seen_ids.contains(&msg.id) {
                // Completely new message
                self.seen_ids.insert(msg.id.clone());
                new_messages.push(msg);
            } else if let (Some(last_check_time), Some(updated_at)) = (last_check, msg.updated_at) {
                // Already seen, but check if it was updated since our last check
                if updated_at > last_check_time {
                    updated_messages.push(msg);
                }
            }
        }

        // Update last check time for next iteration
        self.last_check_time = Some(Local::now());

        Ok(MessageCheckResult {
            new_messages,
            updated_messages,
        })
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
