//! Message types for the chat display

#![allow(dead_code)]

use chrono::{DateTime, Local, NaiveDateTime, TimeZone};

/// A chat message from the database
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: DateTime<Local>,
    pub message_type: Option<String>,
    pub metadata: Option<String>,
    pub updated_at: Option<DateTime<Local>>,
}

impl ChatMessage {
    pub fn from_db_row(
        id: String,
        role: String,
        content: String,
        created_at_str: String,
        message_type: Option<String>,
        metadata: Option<String>,
        updated_at_str: Option<String>,
    ) -> Self {
        // Parse the timestamp - try multiple formats
        let created_at = parse_timestamp(&created_at_str);
        let updated_at = updated_at_str.map(|s| parse_timestamp(&s));

        Self {
            id,
            role,
            content,
            created_at,
            message_type,
            metadata,
            updated_at,
        }
    }

    /// Check if this is a context injection message
    pub fn is_context_message(&self) -> bool {
        matches!(
            self.message_type.as_deref(),
            Some("context_user") | Some("context_ack")
        )
    }

    /// Format the message for display
    pub fn format_display(&self) -> String {
        let time = self.created_at.format("%H:%M:%S");
        let msg_type = self.message_type.as_deref().unwrap_or("message");

        let role_display = match (self.role.as_str(), msg_type) {
            (_, "context_user") => "Context",
            (_, "context_ack") => "AI",
            ("user", _) => "You",
            ("assistant", _) => "AI",
            _ => &self.role,
        };

        // For context messages, show a condensed indicator
        if msg_type == "context_user" {
            // Show condensed context indicator instead of full content
            let chunk_info = self.get_context_chunk_info();
            format!("[{}] {}: {}", time, role_display, chunk_info)
        } else if msg_type == "context_ack" {
            // Skip the generic ack message in display
            format!("[{}] {}: (context acknowledged)", time, role_display)
        } else {
            format!("[{}] {}: {}", time, role_display, self.content)
        }
    }

    /// Extract context chunk info from metadata for display
    fn get_context_chunk_info(&self) -> String {
        if let Some(ref metadata) = self.metadata {
            // Try to parse the JSON metadata
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(metadata) {
                let chunk_count = meta
                    .get("chunk_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let sources = meta.get("sources").and_then(|v| v.as_array());
                let distances = meta.get("distances").and_then(|v| v.as_array());

                let source_info = if let Some(srcs) = sources {
                    let unique_sources: Vec<_> = srcs
                        .iter()
                        .filter_map(|s| s.as_str())
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .take(2)
                        .collect();
                    if unique_sources.is_empty() {
                        String::new()
                    } else if unique_sources.len() == 1 {
                        format!(" from \"{}\"", unique_sources[0])
                    } else {
                        format!(" from {} sources", unique_sources.len())
                    }
                } else {
                    String::new()
                };

                let relevance_info = if let Some(dists) = distances {
                    let valid_dists: Vec<f64> = dists.iter().filter_map(|d| d.as_f64()).collect();
                    if !valid_dists.is_empty() {
                        let avg_dist = valid_dists.iter().sum::<f64>() / valid_dists.len() as f64;
                        format!(" (relevance: {:.0}%)", (1.0 - avg_dist) * 100.0)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                return format!(
                    "{} chunk(s) retrieved{}{}",
                    chunk_count, source_info, relevance_info
                );
            }
        }
        // Fallback if metadata parsing fails
        "Context chunks retrieved".to_string()
    }
}

/// A system/info message (not from DB, generated locally)
#[derive(Debug, Clone)]
pub struct SystemMessage {
    pub content: String,
    pub created_at: DateTime<Local>,
}

impl SystemMessage {
    pub fn new(content: String) -> Self {
        Self {
            content,
            created_at: Local::now(),
        }
    }

    pub fn format_display(&self) -> String {
        let time = self.created_at.format("%H:%M:%S");
        format!("[{}] {}", time, self.content)
    }
}

/// Unified display message enum
#[derive(Debug, Clone)]
pub enum DisplayMessage {
    Chat(ChatMessage),
    System(SystemMessage),
}

impl DisplayMessage {
    pub fn format_display(&self) -> String {
        match self {
            DisplayMessage::Chat(msg) => msg.format_display(),
            DisplayMessage::System(msg) => msg.format_display(),
        }
    }

    pub fn created_at(&self) -> DateTime<Local> {
        match self {
            DisplayMessage::Chat(msg) => msg.created_at,
            DisplayMessage::System(msg) => msg.created_at,
        }
    }

    pub fn is_user(&self) -> bool {
        matches!(self, DisplayMessage::Chat(msg) if msg.role == "user")
    }

    pub fn is_assistant(&self) -> bool {
        matches!(self, DisplayMessage::Chat(msg) if msg.role == "assistant")
    }

    pub fn is_system(&self) -> bool {
        matches!(self, DisplayMessage::System(_))
    }
}

/// Context retrieval info from context.db
#[derive(Debug, Clone)]
pub struct ContextInfo {
    pub prompt_id: String,
    pub chunk_count: usize,
    pub sources: Vec<String>,
    pub avg_distance: f64,
    pub retrieved_at: DateTime<Local>,
}

impl ContextInfo {
    pub fn to_system_message(&self) -> SystemMessage {
        let sources_str = if self.sources.is_empty() {
            String::new()
        } else if self.sources.len() == 1 {
            format!(" from \"{}\"", self.sources[0])
        } else {
            format!(" from {} sources", self.sources.len())
        };

        let content = format!(
            "Context: {} chunk(s){} (relevance: {:.0}%)",
            self.chunk_count,
            sources_str,
            (1.0 - self.avg_distance) * 100.0
        );

        SystemMessage {
            content,
            created_at: self.retrieved_at,
        }
    }
}

/// Parse timestamp string from database
fn parse_timestamp(s: &str) -> DateTime<Local> {
    // Try RFC3339 with timezone: "2024-01-14T10:30:45.123456+00:00"
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return dt.with_timezone(&Local);
    }

    // Try ISO format with T separator and microseconds: "2024-01-14T10:30:45.123456"
    if let Ok(naive) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        if let Some(dt) = Local.from_local_datetime(&naive).single() {
            return dt;
        }
    }

    // Try ISO format with T separator: "2024-01-14T10:30:45"
    if let Ok(naive) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        if let Some(dt) = Local.from_local_datetime(&naive).single() {
            return dt;
        }
    }

    // Try with space separator: "2024-01-14 10:30:45"
    if let Ok(naive) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        if let Some(dt) = Local.from_local_datetime(&naive).single() {
            return dt;
        }
    }

    // Try with space and microseconds: "2024-01-14 10:30:45.123456"
    if let Ok(naive) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
        if let Some(dt) = Local.from_local_datetime(&naive).single() {
            return dt;
        }
    }

    // Fallback to now
    Local::now()
}
