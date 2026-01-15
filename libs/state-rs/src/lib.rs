//! Centralized State Management Library for Olorin Project
//!
//! Provides a State struct for persistent key-value storage with typed columns.
//! Uses SQLite for storage with separate columns for each data type, enabling
//! efficient access from statically-typed languages.
//!
//! # Design
//!
//! Each value is stored in a type-appropriate column (value_int, value_float, etc.)
//! A value_type column indicates which column contains the data.
//! This avoids serialization overhead and enables type-safe reads.
//!
//! # Usage
//!
//! ```rust,ignore
//! use olorin_state::{State, get_state};
//!
//! let state = State::new(None)?;
//!
//! // Set values (type auto-detected via specific methods)
//! state.set_int("broca.audio_pid", 12345)?;
//! state.set_bool("broca.is_playing", true)?;
//! state.set_string("cortex.status", "running")?;
//!
//! // Get values with type safety
//! let pid = state.get_int("broca.audio_pid")?;  // -> Option<i64>
//! let playing = state.get_bool("broca.is_playing")?;  // -> bool
//!
//! // Delete values
//! state.delete("broca.audio_pid")?;
//! state.delete_prefix("broca.")?;  // Delete all broca.* keys
//! ```

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value as JsonValue;

/// Error types for state operations
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to find project root")]
    ProjectRootNotFound,

    #[error("Invalid value type: {0}")]
    InvalidValueType(String),
}

/// Enumeration of supported value types for cross-language compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Null,
    Int,
    Float,
    String,
    Bool,
    Json,
    Bytes,
}

impl ValueType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Int => "int",
            Self::Float => "float",
            Self::String => "string",
            Self::Bool => "bool",
            Self::Json => "json",
            Self::Bytes => "bytes",
        }
    }

    /// Try to parse from string representation
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl std::str::FromStr for ValueType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "null" => Ok(Self::Null),
            "int" => Ok(Self::Int),
            "float" => Ok(Self::Float),
            "string" => Ok(Self::String),
            "bool" => Ok(Self::Bool),
            "json" => Ok(Self::Json),
            "bytes" => Ok(Self::Bytes),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// SQL schema for the state table
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY NOT NULL,
    value_type TEXT NOT NULL DEFAULT 'null',
    value_int INTEGER,
    value_float REAL,
    value_string TEXT,
    value_bool INTEGER,
    value_json TEXT,
    value_bytes BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_state_type ON state(value_type);
CREATE INDEX IF NOT EXISTS idx_state_updated ON state(updated_at);
"#;

/// Find the project root by looking for settings.json file.
fn find_project_root() -> Option<PathBuf> {
    // Try to find from current executable location
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let mut search = parent.to_path_buf();
            for _ in 0..10 {
                if search.join("settings.json").exists() {
                    return Some(search);
                }
                if let Some(p) = search.parent() {
                    search = p.to_path_buf();
                } else {
                    break;
                }
            }
        }
    }

    // Fallback: search upward from current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let mut search = cwd;
        for _ in 0..10 {
            if search.join("settings.json").exists() {
                return Some(search);
            }
            if let Some(p) = search.parent() {
                search = p.to_path_buf();
            } else {
                break;
            }
        }
    }

    // Last resort: return current working directory
    std::env::current_dir().ok()
}

/// Metadata for a state entry
#[derive(Debug, Clone)]
pub struct StateMetadata {
    pub value_type: ValueType,
    pub created_at: String,
    pub updated_at: String,
}

// Type aliases for complex query results to satisfy clippy::type_complexity
type NumericRow = (String, Option<i64>, Option<f64>, Option<String>);
type StringRow = (
    String,
    Option<i64>,
    Option<f64>,
    Option<String>,
    Option<i64>,
);

/// Centralized state management with SQLite backend and typed columns.
///
/// Provides type-safe getters and setters for sharing state across all
/// Olorin components. Uses separate columns for each data type to enable
/// efficient access from statically-typed languages.
///
/// # Thread Safety
///
/// This struct uses an internal Mutex to ensure thread-safe access to the
/// SQLite connection. Multiple threads can safely share a single State instance.
pub struct State {
    conn: Mutex<Connection>,
    db_path: PathBuf,
    project_root: PathBuf,
}

impl State {
    /// Initialize the state manager.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to SQLite database file. Defaults to ./data/state.db
    ///   relative to project root.
    ///
    /// # Errors
    ///
    /// Returns `StateError` if the database cannot be opened or initialized.
    pub fn new(db_path: Option<PathBuf>) -> Result<Self, StateError> {
        let project_root = find_project_root().ok_or(StateError::ProjectRootNotFound)?;

        let db_path = match db_path {
            Some(p) if p.is_absolute() => p,
            Some(p) => project_root.join(p),
            None => project_root.join("data").join("state.db"),
        };

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;

        // Enable WAL mode for better concurrency
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=30000;",
        )?;

        // Initialize schema
        conn.execute_batch(SCHEMA)?;

        Ok(Self {
            conn: Mutex::new(conn),
            db_path,
            project_root,
        })
    }

    /// Set an integer value.
    pub fn set_int(&self, key: &str, value: i64) -> Result<(), StateError> {
        self.set_typed(
            key,
            ValueType::Int,
            Some(value),
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Set a float value.
    pub fn set_float(&self, key: &str, value: f64) -> Result<(), StateError> {
        self.set_typed(
            key,
            ValueType::Float,
            None,
            Some(value),
            None,
            None,
            None,
            None,
        )
    }

    /// Set a string value.
    pub fn set_string(&self, key: &str, value: &str) -> Result<(), StateError> {
        self.set_typed(
            key,
            ValueType::String,
            None,
            None,
            Some(value.to_string()),
            None,
            None,
            None,
        )
    }

    /// Set a boolean value.
    pub fn set_bool(&self, key: &str, value: bool) -> Result<(), StateError> {
        self.set_typed(
            key,
            ValueType::Bool,
            None,
            None,
            None,
            Some(value),
            None,
            None,
        )
    }

    /// Set a JSON value from a serde_json::Value.
    pub fn set_json(&self, key: &str, value: &JsonValue) -> Result<(), StateError> {
        let json_str = serde_json::to_string(value)?;
        self.set_typed(
            key,
            ValueType::Json,
            None,
            None,
            None,
            None,
            Some(json_str),
            None,
        )
    }

    /// Set a JSON value from a serializable type.
    pub fn set_json_value<T: serde::Serialize>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<(), StateError> {
        let json_str = serde_json::to_string(value)?;
        self.set_typed(
            key,
            ValueType::Json,
            None,
            None,
            None,
            None,
            Some(json_str),
            None,
        )
    }

    /// Set a bytes value.
    pub fn set_bytes(&self, key: &str, value: &[u8]) -> Result<(), StateError> {
        self.set_typed(
            key,
            ValueType::Bytes,
            None,
            None,
            None,
            None,
            None,
            Some(value.to_vec()),
        )
    }

    /// Set a null value (different from delete - the key exists but has no value).
    pub fn set_null(&self, key: &str) -> Result<(), StateError> {
        self.set_typed(key, ValueType::Null, None, None, None, None, None, None)
    }

    /// Internal method to set a typed value.
    #[allow(clippy::too_many_arguments)]
    fn set_typed(
        &self,
        key: &str,
        value_type: ValueType,
        value_int: Option<i64>,
        value_float: Option<f64>,
        value_string: Option<String>,
        value_bool: Option<bool>,
        value_json: Option<String>,
        value_bytes: Option<Vec<u8>>,
    ) -> Result<(), StateError> {
        let now = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
        let conn = self.conn.lock().expect("State mutex poisoned");

        conn.execute(
            r#"
            INSERT INTO state (
                key, value_type, value_int, value_float, value_string,
                value_bool, value_json, value_bytes, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(key) DO UPDATE SET
                value_type = excluded.value_type,
                value_int = excluded.value_int,
                value_float = excluded.value_float,
                value_string = excluded.value_string,
                value_bool = excluded.value_bool,
                value_json = excluded.value_json,
                value_bytes = excluded.value_bytes,
                updated_at = excluded.updated_at
            "#,
            params![
                key,
                value_type.as_str(),
                value_int,
                value_float,
                value_string,
                value_bool.map(|b| if b { 1i64 } else { 0i64 }),
                value_json,
                value_bytes,
                now,
                now,
            ],
        )?;

        Ok(())
    }

    /// Get an integer value.
    ///
    /// Returns `None` if the key doesn't exist or is not an integer type.
    /// Will attempt to convert from float or parse from string.
    pub fn get_int(&self, key: &str) -> Result<Option<i64>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<NumericRow> = conn
            .query_row(
                "SELECT value_type, value_int, value_float, value_string FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;

        match result {
            Some((vtype, value_int, value_float, value_string)) => match vtype.as_str() {
                "int" => Ok(value_int),
                "float" => Ok(value_float.map(|f| f as i64)),
                "string" => Ok(value_string.and_then(|s| s.parse().ok())),
                _ => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// Get an integer value with a default.
    pub fn get_int_or(&self, key: &str, default: i64) -> Result<i64, StateError> {
        Ok(self.get_int(key)?.unwrap_or(default))
    }

    /// Get a float value.
    ///
    /// Returns `None` if the key doesn't exist or is not a numeric type.
    /// Will convert from integer or parse from string.
    pub fn get_float(&self, key: &str) -> Result<Option<f64>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<NumericRow> = conn
            .query_row(
                "SELECT value_type, value_int, value_float, value_string FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;

        match result {
            Some((vtype, value_int, value_float, value_string)) => match vtype.as_str() {
                "float" => Ok(value_float),
                "int" => Ok(value_int.map(|i| i as f64)),
                "string" => Ok(value_string.and_then(|s| s.parse().ok())),
                _ => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// Get a float value with a default.
    pub fn get_float_or(&self, key: &str, default: f64) -> Result<f64, StateError> {
        Ok(self.get_float(key)?.unwrap_or(default))
    }

    /// Get a string value.
    ///
    /// Returns `None` if the key doesn't exist or is not a string type.
    /// Will convert primitives (int, float, bool) to string.
    pub fn get_string(&self, key: &str) -> Result<Option<String>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<StringRow> = conn
            .query_row(
                "SELECT value_type, value_int, value_float, value_string, value_bool FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .optional()?;

        match result {
            Some((vtype, value_int, value_float, value_string, value_bool)) => match vtype.as_str()
            {
                "string" => Ok(value_string),
                "int" => Ok(value_int.map(|i| i.to_string())),
                "float" => Ok(value_float.map(|f| f.to_string())),
                "bool" => Ok(value_bool.map(|b| if b != 0 { "true" } else { "false" }.to_string())),
                _ => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// Get a string value with a default.
    pub fn get_string_or(&self, key: &str, default: &str) -> Result<String, StateError> {
        Ok(self.get_string(key)?.unwrap_or_else(|| default.to_string()))
    }

    /// Get a boolean value.
    ///
    /// Returns `false` if the key doesn't exist.
    /// Will interpret integers (0/non-0) and strings ("true", "yes", "1", "on").
    pub fn get_bool(&self, key: &str) -> Result<bool, StateError> {
        self.get_bool_or(key, false)
    }

    /// Get a boolean value with a default.
    pub fn get_bool_or(&self, key: &str, default: bool) -> Result<bool, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<(String, Option<i64>, Option<String>)> = conn
            .query_row(
                "SELECT value_type, value_bool, value_string FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        match result {
            Some((vtype, value_bool, value_string)) => match vtype.as_str() {
                "bool" => Ok(value_bool.is_some_and(|b| b != 0)),
                "int" => Ok(value_bool.is_some_and(|i| i != 0)),
                "string" => Ok(value_string.is_some_and(|s| {
                    matches!(s.to_lowercase().as_str(), "true" | "yes" | "1" | "on")
                })),
                _ => Ok(default),
            },
            None => Ok(default),
        }
    }

    /// Get a JSON value as serde_json::Value.
    ///
    /// Returns `None` if the key doesn't exist or is not a JSON type.
    pub fn get_json(&self, key: &str) -> Result<Option<JsonValue>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<(String, Option<String>)> = conn
            .query_row(
                "SELECT value_type, value_json FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        match result {
            Some((vtype, Some(s))) if vtype == "json" => Ok(Some(serde_json::from_str(&s)?)),
            _ => Ok(None),
        }
    }

    /// Get a JSON value deserialized into a specific type.
    pub fn get_json_as<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<T>, StateError> {
        match self.get_json(key)? {
            Some(v) => Ok(Some(serde_json::from_value(v)?)),
            None => Ok(None),
        }
    }

    /// Get a bytes value.
    ///
    /// Returns `None` if the key doesn't exist or is not a bytes type.
    pub fn get_bytes(&self, key: &str) -> Result<Option<Vec<u8>>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<(String, Option<Vec<u8>>)> = conn
            .query_row(
                "SELECT value_type, value_bytes FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        match result {
            Some((vtype, value_bytes)) if vtype == "bytes" => Ok(value_bytes),
            _ => Ok(None),
        }
    }

    /// Get the type of a stored value.
    ///
    /// Returns `None` if the key doesn't exist.
    pub fn get_type(&self, key: &str) -> Result<Option<ValueType>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<String> = conn
            .query_row(
                "SELECT value_type FROM state WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;

        Ok(result.and_then(|s| s.parse().ok()))
    }

    /// Check if a key exists in the state.
    pub fn exists(&self, key: &str) -> Result<bool, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM state WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )?;

        Ok(count > 0)
    }

    /// Delete a key from the state.
    ///
    /// Returns `true` if the key was deleted, `false` if it didn't exist.
    pub fn delete(&self, key: &str) -> Result<bool, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");
        let affected = conn.execute("DELETE FROM state WHERE key = ?1", params![key])?;
        Ok(affected > 0)
    }

    /// Delete all keys with a given prefix.
    ///
    /// Returns the number of keys deleted.
    pub fn delete_prefix(&self, prefix: &str) -> Result<usize, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");
        let pattern = format!("{}%", prefix);
        let affected = conn.execute("DELETE FROM state WHERE key LIKE ?1", params![pattern])?;
        Ok(affected)
    }

    /// Get all keys, optionally filtered by prefix.
    pub fn keys(&self, prefix: Option<&str>) -> Result<Vec<String>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let keys: Vec<String> = match prefix {
            Some(p) => {
                let pattern = format!("{}%", p);
                let mut stmt =
                    conn.prepare("SELECT key FROM state WHERE key LIKE ?1 ORDER BY key")?;
                let rows = stmt.query_map(params![pattern], |row| row.get(0))?;
                rows.filter_map(|r| r.ok()).collect()
            }
            None => {
                let mut stmt = conn.prepare("SELECT key FROM state ORDER BY key")?;
                let rows = stmt.query_map([], |row| row.get(0))?;
                rows.filter_map(|r| r.ok()).collect()
            }
        };

        Ok(keys)
    }

    /// Get metadata for a key including type and timestamps.
    pub fn get_metadata(&self, key: &str) -> Result<Option<StateMetadata>, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");

        let result: Option<(String, String, String)> = conn
            .query_row(
                "SELECT value_type, created_at, updated_at FROM state WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        match result {
            Some((vtype, created_at, updated_at)) => {
                let value_type: ValueType = vtype
                    .parse()
                    .map_err(|_| StateError::InvalidValueType(vtype.clone()))?;
                Ok(Some(StateMetadata {
                    value_type,
                    created_at,
                    updated_at,
                }))
            }
            None => Ok(None),
        }
    }

    /// Delete all state entries.
    ///
    /// Returns the number of entries deleted.
    pub fn clear(&self) -> Result<usize, StateError> {
        let conn = self.conn.lock().expect("State mutex poisoned");
        let affected = conn.execute("DELETE FROM state", [])?;
        Ok(affected)
    }

    /// Return the path to the state database.
    pub fn db_path(&self) -> &PathBuf {
        &self.db_path
    }

    /// Return the project root directory.
    pub fn project_root(&self) -> &PathBuf {
        &self.project_root
    }
}

// Global singleton for convenience
static DEFAULT_STATE: OnceLock<State> = OnceLock::new();

/// Get the default State singleton.
///
/// # Arguments
///
/// * `db_path` - Path to database (only applies on first call)
///
/// # Returns
///
/// A reference to the default State instance
///
/// # Panics
///
/// Panics if the state cannot be initialized on first call.
pub fn get_state(db_path: Option<PathBuf>) -> &'static State {
    DEFAULT_STATE.get_or_init(|| State::new(db_path).expect("Failed to initialize default state"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_state() -> State {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_state.db");
        // Keep tempdir alive by leaking it (for tests only)
        std::mem::forget(dir);
        State::new(Some(db_path)).unwrap()
    }

    #[test]
    fn test_int_operations() {
        let state = create_test_state();

        state.set_int("test.int", 42).unwrap();
        assert_eq!(state.get_int("test.int").unwrap(), Some(42));
        assert_eq!(state.get_int("nonexistent").unwrap(), None);
        assert_eq!(state.get_int_or("nonexistent", 99).unwrap(), 99);
    }

    #[test]
    fn test_float_operations() {
        let state = create_test_state();

        state.set_float("test.float", 3.14).unwrap();
        let val = state.get_float("test.float").unwrap().unwrap();
        assert!((val - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_string_operations() {
        let state = create_test_state();

        state.set_string("test.string", "hello world").unwrap();
        assert_eq!(
            state.get_string("test.string").unwrap(),
            Some("hello world".to_string())
        );
    }

    #[test]
    fn test_bool_operations() {
        let state = create_test_state();

        state.set_bool("test.bool", true).unwrap();
        assert!(state.get_bool("test.bool").unwrap());

        state.set_bool("test.bool", false).unwrap();
        assert!(!state.get_bool("test.bool").unwrap());
    }

    #[test]
    fn test_json_operations() {
        let state = create_test_state();

        let json_value = serde_json::json!({
            "name": "test",
            "values": [1, 2, 3]
        });

        state.set_json("test.json", &json_value).unwrap();
        let retrieved = state.get_json("test.json").unwrap().unwrap();
        assert_eq!(retrieved["name"], "test");
        assert_eq!(retrieved["values"][0], 1);
    }

    #[test]
    fn test_bytes_operations() {
        let state = create_test_state();

        let data = vec![0x01, 0x02, 0x03, 0x04];
        state.set_bytes("test.bytes", &data).unwrap();
        assert_eq!(state.get_bytes("test.bytes").unwrap(), Some(data));
    }

    #[test]
    fn test_null_operations() {
        let state = create_test_state();

        state.set_null("test.null").unwrap();
        assert!(state.exists("test.null").unwrap());
        assert_eq!(state.get_type("test.null").unwrap(), Some(ValueType::Null));
    }

    #[test]
    fn test_delete_operations() {
        let state = create_test_state();

        state.set_int("delete.me", 1).unwrap();
        assert!(state.exists("delete.me").unwrap());

        assert!(state.delete("delete.me").unwrap());
        assert!(!state.exists("delete.me").unwrap());
        assert!(!state.delete("delete.me").unwrap()); // Second delete returns false
    }

    #[test]
    fn test_delete_prefix() {
        let state = create_test_state();

        state.set_int("prefix.one", 1).unwrap();
        state.set_int("prefix.two", 2).unwrap();
        state.set_int("other.key", 3).unwrap();

        let deleted = state.delete_prefix("prefix.").unwrap();
        assert_eq!(deleted, 2);

        assert!(!state.exists("prefix.one").unwrap());
        assert!(!state.exists("prefix.two").unwrap());
        assert!(state.exists("other.key").unwrap());
    }

    #[test]
    fn test_keys() {
        let state = create_test_state();

        state.set_int("a.one", 1).unwrap();
        state.set_int("a.two", 2).unwrap();
        state.set_int("b.one", 3).unwrap();

        let all_keys = state.keys(None).unwrap();
        assert_eq!(all_keys.len(), 3);

        let a_keys = state.keys(Some("a.")).unwrap();
        assert_eq!(a_keys.len(), 2);
        assert!(a_keys.contains(&"a.one".to_string()));
        assert!(a_keys.contains(&"a.two".to_string()));
    }

    #[test]
    fn test_metadata() {
        let state = create_test_state();

        state.set_int("meta.test", 42).unwrap();
        let meta = state.get_metadata("meta.test").unwrap().unwrap();
        assert_eq!(meta.value_type, ValueType::Int);
        assert!(!meta.created_at.is_empty());
        assert!(!meta.updated_at.is_empty());
    }

    #[test]
    fn test_type_coercion() {
        let state = create_test_state();

        // Int can be retrieved as float
        state.set_int("coerce.int", 42).unwrap();
        assert_eq!(state.get_float("coerce.int").unwrap(), Some(42.0));

        // Float can be retrieved as int (truncated)
        state.set_float("coerce.float", 3.9).unwrap();
        assert_eq!(state.get_int("coerce.float").unwrap(), Some(3));

        // Primitives can be retrieved as string
        assert_eq!(
            state.get_string("coerce.int").unwrap(),
            Some("42".to_string())
        );
    }

    #[test]
    fn test_clear() {
        let state = create_test_state();

        state.set_int("clear.one", 1).unwrap();
        state.set_int("clear.two", 2).unwrap();

        let cleared = state.clear().unwrap();
        assert_eq!(cleared, 2);
        assert!(state.keys(None).unwrap().is_empty());
    }

    #[test]
    fn test_update_existing_key() {
        let state = create_test_state();

        state.set_int("update.key", 1).unwrap();
        assert_eq!(state.get_int("update.key").unwrap(), Some(1));

        state.set_int("update.key", 2).unwrap();
        assert_eq!(state.get_int("update.key").unwrap(), Some(2));

        // Can change type
        state.set_string("update.key", "now a string").unwrap();
        assert_eq!(
            state.get_string("update.key").unwrap(),
            Some("now a string".to_string())
        );
        assert_eq!(
            state.get_type("update.key").unwrap(),
            Some(ValueType::String)
        );
    }
}
