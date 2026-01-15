//! Core traits and types for database abstraction

#![allow(dead_code)]

use std::collections::HashMap;

/// Connection state for databases that require network connectivity
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    /// Database is connected and operational
    Connected,
    /// Database is disconnected with error message
    Disconnected(String),
    /// Connection state is unknown (not yet checked)
    Unknown,
}

impl ConnectionState {
    pub fn is_available(&self) -> bool {
        matches!(self, ConnectionState::Connected)
    }

    pub fn error_message(&self) -> Option<&str> {
        match self {
            ConnectionState::Disconnected(msg) => Some(msg),
            _ => None,
        }
    }
}

/// Represents a single record from any database
#[derive(Debug, Clone)]
pub struct Record {
    /// Column name -> value mapping
    pub fields: HashMap<String, String>,
    /// Original timestamp for sorting (ISO 8601 format)
    pub timestamp: Option<String>,
}

impl Record {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            timestamp: None,
        }
    }

    pub fn with_field(mut self, key: &str, value: String) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }

    pub fn with_timestamp(mut self, ts: String) -> Self {
        self.timestamp = Some(ts);
        self
    }
}

impl Default for Record {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a database
#[derive(Debug, Clone)]
pub struct DatabaseInfo {
    pub name: String,
    pub db_type: DatabaseType,
    pub path: String,
    pub record_count: usize,
    pub columns: Vec<String>,
    pub table_name: String,
    /// Connection state (for network-based databases like ChromaDB)
    pub connection_state: ConnectionState,
}

/// Type of database
#[derive(Debug, Clone, PartialEq)]
pub enum DatabaseType {
    /// SQLite database with processed_files schema
    SqliteFileTracker,
    /// SQLite database with contexts table
    SqliteContext,
    /// SQLite database with conversations and messages tables
    SqliteChat,
    /// ChromaDB HTTP API
    ChromaDB,
}

impl DatabaseType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DatabaseType::SqliteFileTracker => "SQLite (File Tracker)",
            DatabaseType::SqliteContext => "SQLite (Context)",
            DatabaseType::SqliteChat => "SQLite (Chat)",
            DatabaseType::ChromaDB => "ChromaDB",
        }
    }
}

/// Error type for database operations
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Database not found: {0}")]
    NotFound(String),
}

/// Common interface for all database sources
pub trait DatabaseSource {
    /// Get database metadata (immutable)
    fn info(&self) -> &DatabaseInfo;

    /// Get database metadata (mutable, for updating connection state)
    #[allow(dead_code)]
    fn info_mut(&mut self) -> &mut DatabaseInfo;

    /// Check if the database is currently reachable
    /// Updates internal connection state and returns true if healthy
    fn health_check(&mut self) -> bool;

    /// Fetch most recent records (for initial load and refresh)
    /// Returns records in descending time order
    fn fetch_recent(&self, limit: usize) -> Result<Vec<Record>, DbError>;

    /// Fetch older records for infinite scroll
    /// `before` is the timestamp to fetch records older than
    fn fetch_before(&self, before: &str, limit: usize) -> Result<Vec<Record>, DbError>;

    /// Execute a custom query (SQL for SQLite, query params for ChromaDB)
    fn execute_query(&self, query: &str) -> Result<Vec<Record>, DbError>;

    /// Refresh the record count
    fn refresh_count(&mut self) -> Result<usize, DbError>;

    /// Clear all records from the database
    /// Returns the number of records deleted
    fn clear_database(&mut self) -> Result<usize, DbError>;
}
