//! Unified Configuration Library for Olorin Project
//!
//! Provides a Config struct with type-safe getters and optional hot-reload support.
//! Reads from settings.json (preferred) or falls back to .env for backward compatibility.
//!
//! # Usage
//!
//! ```rust,ignore
//! use olorin_config::Config;
//!
//! let config = Config::new(None, false).unwrap();
//! let host = config.get("CHROMADB_HOST", Some("localhost"));
//! let port = config.get_int("CHROMADB_PORT", Some(8000));
//! let enabled = config.get_bool("FEATURE_ENABLED", false);
//! let path = config.get_path("INPUT_DIR", Some("~/Documents"));
//! let patterns = config.get_list("CHAT_RESET_PATTERNS", None);
//!
//! // With hot-reload support
//! let mut config = Config::new(None, true).unwrap();
//! if config.reload() {
//!     println!("Configuration was updated");
//! }
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::SystemTime;

use serde_json::Value;

/// Error types for configuration operations
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] io::Error),

    #[error("Failed to parse JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Failed to find project root")]
    ProjectRootNotFound,
}

/// Mapping from flat keys to JSON paths for backward compatibility
fn key_to_path(key: &str) -> Option<&'static str> {
    match key {
        // Global
        "LOG_LEVEL" => Some("global.log_level"),
        "LOG_DIR" => Some("global.log_dir"),
        // Kafka
        "KAFKA_BOOTSTRAP_SERVERS" => Some("kafka.bootstrap_servers"),
        "KAFKA_SEND_TIMEOUT" => Some("kafka.send_timeout"),
        "KAFKA_MAX_RETRIES" => Some("kafka.max_retries"),
        // Exo
        "EXO_BASE_URL" => Some("exo.base_url"),
        "EXO_API_KEY" => Some("exo.api_key"),
        "MODEL_NAME" => Some("exo.model_name"),
        "TEMPERATURE" => Some("exo.temperature"),
        "MAX_TOKENS" => Some("exo.max_tokens"),
        // Broca
        "BROCA_KAFKA_TOPIC" => Some("broca.kafka_topic"),
        "BROCA_CONSUMER_GROUP" => Some("broca.consumer_group"),
        "BROCA_AUTO_OFFSET_RESET" => Some("broca.auto_offset_reset"),
        "TTS_MODEL_NAME" => Some("broca.tts.model_name"),
        "TTS_SPEAKER" => Some("broca.tts.speaker"),
        "TTS_OUTPUT_DIR" => Some("broca.tts.output_dir"),
        // Cortex
        "CORTEX_INPUT_TOPIC" => Some("cortex.input_topic"),
        "CORTEX_OUTPUT_TOPIC" => Some("cortex.output_topic"),
        "CORTEX_CONSUMER_GROUP" => Some("cortex.consumer_group"),
        "CORTEX_AUTO_OFFSET_RESET" => Some("cortex.auto_offset_reset"),
        // Hippocampus
        "INPUT_DIR" => Some("hippocampus.input_dir"),
        "CHROMADB_HOST" => Some("hippocampus.chromadb.host"),
        "CHROMADB_PORT" => Some("hippocampus.chromadb.port"),
        "CHROMADB_COLLECTION" => Some("hippocampus.chromadb.collection"),
        "EMBEDDING_MODEL" => Some("hippocampus.embedding_model"),
        "CHUNK_SIZE" => Some("hippocampus.chunking.size"),
        "CHUNK_OVERLAP" => Some("hippocampus.chunking.overlap"),
        "CHUNK_MIN_SIZE" => Some("hippocampus.chunking.min_size"),
        "POLL_INTERVAL" => Some("hippocampus.poll_interval"),
        "TRACKING_DB" => Some("hippocampus.tracking_db"),
        "REPROCESS_ON_CHANGE" => Some("hippocampus.reprocess_on_change"),
        "DELETE_AFTER_PROCESSING" => Some("hippocampus.delete_after_processing"),
        "LOG_FILE" => Some("hippocampus.log_file"),
        // Enrichener
        "ENRICHENER_INPUT_TOPIC" => Some("enrichener.input_topic"),
        "ENRICHENER_OUTPUT_TOPIC" => Some("enrichener.output_topic"),
        "ENRICHENER_CONSUMER_GROUP" => Some("enrichener.consumer_group"),
        "ENRICHENER_AUTO_OFFSET_RESET" => Some("enrichener.auto_offset_reset"),
        "ENRICHENER_THREAD_POOL_SIZE" => Some("enrichener.thread_pool_size"),
        "LLM_TIMEOUT_SECONDS" => Some("enrichener.llm_timeout_seconds"),
        "DECISION_TEMPERATURE" => Some("enrichener.decision_temperature"),
        "CHROMADB_QUERY_N_RESULTS" => Some("enrichener.chromadb_query_n_results"),
        "CONTEXT_DB_PATH" => Some("enrichener.context_db_path"),
        "CLEANUP_CONTEXT_AFTER_USE" => Some("enrichener.cleanup_context_after_use"),
        // Chat
        "CHAT_HISTORY_ENABLED" => Some("chat.history_enabled"),
        "CHAT_DB_PATH" => Some("chat.db_path"),
        "CHAT_RESET_PATTERNS" => Some("chat.reset_patterns"),
        _ => None,
    }
}

/// Find the project root by looking for settings.json or .env file.
fn find_project_root() -> Option<PathBuf> {
    // Try to find from current executable location
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let mut search = parent.to_path_buf();
            for _ in 0..10 {
                if search.join("settings.json").exists() || search.join(".env").exists() {
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
            if search.join("settings.json").exists() || search.join(".env").exists() {
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

/// Configuration manager that loads from settings.json or .env file.
///
/// Provides type-safe getters and optional hot-reload support for runtime
/// configuration changes. Prefers settings.json if available, falls back to .env.
pub struct Config {
    json_path: PathBuf,
    env_path: PathBuf,
    watch: bool,
    mtime: Option<SystemTime>,
    overrides: HashMap<String, Value>,
    data: Value,
    using_json: bool,
}

impl Config {
    /// Initialize the configuration manager.
    ///
    /// # Arguments
    ///
    /// * `config_path` - Path to settings.json or .env file. Defaults to project root.
    /// * `watch` - If true, enables hot-reload support via reload() method
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the project root cannot be found when `config_path` is None.
    pub fn new(config_path: Option<PathBuf>, watch: bool) -> Result<Self, ConfigError> {
        let root = find_project_root().ok_or(ConfigError::ProjectRootNotFound)?;

        let (json_path, env_path) = match config_path {
            Some(p) => {
                if p.extension().is_some_and(|e| e == "json") {
                    (p.clone(), p.with_extension("").with_file_name(".env"))
                } else if p.file_name().is_some_and(|n| n == ".env") {
                    (p.with_file_name("settings.json"), p)
                } else {
                    // Assume it's a directory
                    (p.join("settings.json"), p.join(".env"))
                }
            }
            None => (root.join("settings.json"), root.join(".env")),
        };

        let mut config = Self {
            json_path,
            env_path,
            watch,
            mtime: None,
            overrides: HashMap::new(),
            data: Value::Null,
            using_json: false,
        };

        config.load()?;
        Ok(config)
    }

    /// Load configuration from settings.json or fall back to .env.
    fn load(&mut self) -> Result<(), ConfigError> {
        // Prefer settings.json
        if self.json_path.exists() {
            let content = fs::read_to_string(&self.json_path)?;
            self.data = serde_json::from_str(&content)?;
            self.mtime = fs::metadata(&self.json_path)?.modified().ok();
            self.using_json = true;
            return Ok(());
        }

        // Fall back to .env
        if self.env_path.exists() {
            self.data = self.parse_env_to_nested()?;
            self.mtime = fs::metadata(&self.env_path)?.modified().ok();
            self.using_json = false;
        }

        Ok(())
    }

    /// Parse .env file into nested JSON Value structure.
    fn parse_env_to_nested(&self) -> Result<Value, ConfigError> {
        let file = fs::File::open(&self.env_path)?;
        let reader = io::BufReader::new(file);

        let mut flat_data: HashMap<String, String> = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse KEY=value format
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let mut value = value.trim();

                // Remove surrounding quotes if present
                if value.len() >= 2 {
                    let first = value.chars().next().unwrap();
                    let last = value.chars().last().unwrap();
                    if first == last && (first == '"' || first == '\'') {
                        value = &value[1..value.len() - 1];
                    }
                }

                flat_data.insert(key.to_string(), value.to_string());
            }
        }

        // Convert flat keys to nested structure
        let mut result = serde_json::Map::new();

        for (flat_key, value) in flat_data {
            if let Some(path) = key_to_path(&flat_key) {
                self.set_nested(&mut result, path, Value::String(value));
            } else {
                // Store unknown keys at root level
                result.insert(flat_key, Value::String(value));
            }
        }

        Ok(Value::Object(result))
    }

    /// Set a value in a nested JSON object using dot notation path.
    fn set_nested(&self, obj: &mut serde_json::Map<String, Value>, path: &str, value: Value) {
        let keys: Vec<&str> = path.split('.').collect();
        let mut current = obj;

        for (i, key) in keys.iter().enumerate() {
            if i == keys.len() - 1 {
                current.insert((*key).to_string(), value);
                return;
            }

            if !current.contains_key(*key) {
                current.insert((*key).to_string(), Value::Object(serde_json::Map::new()));
            }

            if let Some(Value::Object(next)) = current.get_mut(*key) {
                current = next;
            } else {
                return;
            }
        }
    }

    /// Get a value from nested JSON using dot notation path.
    fn get_nested(&self, path: &str) -> Option<&Value> {
        let keys: Vec<&str> = path.split('.').collect();
        let mut current = &self.data;

        for key in keys {
            if let Value::Object(obj) = current {
                current = obj.get(key)?;
            } else {
                return None;
            }
        }

        Some(current)
    }

    /// Reload configuration if the config file has changed.
    ///
    /// Returns true if configuration was reloaded, false otherwise.
    pub fn reload(&mut self) -> bool {
        if !self.watch {
            return false;
        }

        // Check the appropriate config file
        let config_path = if self.json_path.exists() {
            &self.json_path
        } else {
            &self.env_path
        };

        if !config_path.exists() {
            return false;
        }

        let current_mtime = fs::metadata(config_path)
            .ok()
            .and_then(|m| m.modified().ok());

        if current_mtime != self.mtime {
            if self.load().is_ok() {
                return true;
            }
        }

        false
    }

    /// Get a configuration value as a string.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key (flat like 'CHROMADB_PORT' or nested like 'hippocampus.chromadb.port')
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The configuration value or default
    pub fn get(&self, key: &str, default: Option<&str>) -> Option<String> {
        // Check overrides first
        if let Some(value) = self.overrides.get(key) {
            return self.value_to_string(value);
        }

        // Map flat key to nested path if known
        let path = key_to_path(key).unwrap_or(key);
        let value = self.get_nested(path);

        match value {
            Some(v) => self.value_to_string(v),
            None => default.map(String::from),
        }
    }

    /// Convert a JSON Value to String representation.
    fn value_to_string(&self, value: &Value) -> Option<String> {
        match value {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            Value::Array(arr) => {
                let items: Vec<String> = arr
                    .iter()
                    .filter_map(|v| self.value_to_string(v))
                    .collect();
                Some(items.join(","))
            }
            Value::Null => None,
            Value::Object(_) => None,
        }
    }

    /// Get a configuration value as an integer.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    /// * `default` - Default value if not set or invalid
    ///
    /// # Returns
    ///
    /// The configuration value as i64 or default
    pub fn get_int(&self, key: &str, default: Option<i64>) -> Option<i64> {
        // Check overrides first
        if let Some(value) = self.overrides.get(key) {
            if let Some(n) = value.as_i64() {
                return Some(n);
            }
        }

        // Try nested path for native JSON int
        let path = key_to_path(key).unwrap_or(key);
        if let Some(value) = self.get_nested(path) {
            if let Some(n) = value.as_i64() {
                return Some(n);
            }
        }

        // Fall back to string parsing
        match self.get(key, None) {
            Some(value) => value.parse().ok().or(default),
            None => default,
        }
    }

    /// Get a configuration value as a float.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    /// * `default` - Default value if not set or invalid
    ///
    /// # Returns
    ///
    /// The configuration value as f64 or default
    pub fn get_float(&self, key: &str, default: Option<f64>) -> Option<f64> {
        // Check overrides first
        if let Some(value) = self.overrides.get(key) {
            if let Some(n) = value.as_f64() {
                return Some(n);
            }
        }

        // Try nested path for native JSON number
        let path = key_to_path(key).unwrap_or(key);
        if let Some(value) = self.get_nested(path) {
            if let Some(n) = value.as_f64() {
                return Some(n);
            }
        }

        // Fall back to string parsing
        match self.get(key, None) {
            Some(value) => value.parse().ok().or(default),
            None => default,
        }
    }

    /// Get a configuration value as a boolean.
    ///
    /// Recognizes: true, yes, 1, on (case-insensitive) as true.
    /// Native JSON booleans are handled directly.
    /// Everything else (including empty string) is false.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The configuration value as bool or default
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        // Check overrides first
        if let Some(value) = self.overrides.get(key) {
            if let Some(b) = value.as_bool() {
                return b;
            }
            if let Some(s) = value.as_str() {
                let lower = s.to_lowercase();
                return matches!(lower.as_str(), "true" | "yes" | "1" | "on");
            }
        }

        // Try nested path for native JSON bool
        let path = key_to_path(key).unwrap_or(key);
        if let Some(value) = self.get_nested(path) {
            if let Some(b) = value.as_bool() {
                return b;
            }
            if let Some(s) = value.as_str() {
                let lower = s.to_lowercase();
                return matches!(lower.as_str(), "true" | "yes" | "1" | "on");
            }
        }

        default
    }

    /// Get a configuration value as an expanded path.
    ///
    /// Expands ~ to the user's home directory.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The expanded path as PathBuf or None
    pub fn get_path(&self, key: &str, default: Option<&str>) -> Option<PathBuf> {
        let value = self.get(key, default)?;

        if value.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                return Some(home.join(value[1..].trim_start_matches('/')));
            }
        }

        Some(PathBuf::from(value))
    }

    /// Get a configuration value as a list of strings.
    ///
    /// Native JSON arrays are returned directly.
    /// Comma-separated strings are split into lists.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The configuration value as Vec<String> or default
    pub fn get_list(&self, key: &str, default: Option<Vec<String>>) -> Option<Vec<String>> {
        // Check overrides first
        if let Some(value) = self.overrides.get(key) {
            if let Some(arr) = value.as_array() {
                return Some(
                    arr.iter()
                        .filter_map(|v| self.value_to_string(v))
                        .collect(),
                );
            }
            if let Some(s) = value.as_str() {
                return Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
        }

        // Try nested path for native JSON array
        let path = key_to_path(key).unwrap_or(key);
        if let Some(value) = self.get_nested(path) {
            if let Some(arr) = value.as_array() {
                return Some(
                    arr.iter()
                        .filter_map(|v| self.value_to_string(v))
                        .collect(),
                );
            }
            if let Some(s) = value.as_str() {
                return Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
        }

        default
    }

    /// Set a configuration value (in-memory only).
    ///
    /// This override takes precedence over file values.
    /// Does not modify the config file.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    /// * `value` - The value to set
    pub fn set(&mut self, key: &str, value: &str) {
        self.overrides
            .insert(key.to_string(), Value::String(value.to_string()));
    }

    /// Set a configuration value as a JSON Value (in-memory only).
    pub fn set_value(&mut self, key: &str, value: Value) {
        self.overrides.insert(key.to_string(), value);
    }

    /// Clear an in-memory override for a key.
    ///
    /// After clearing, get() will return the file value.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key
    pub fn clear_override(&mut self, key: &str) {
        self.overrides.remove(key);
    }

    /// Clear all in-memory overrides.
    pub fn clear_all_overrides(&mut self) {
        self.overrides.clear();
    }

    /// Return the path to the active config file.
    pub fn config_path(&self) -> &PathBuf {
        if self.using_json {
            &self.json_path
        } else {
            &self.env_path
        }
    }

    /// Return the path to the config file (deprecated, use config_path).
    pub fn env_path(&self) -> &PathBuf {
        self.config_path()
    }
}

// Global singleton for convenience
static DEFAULT_CONFIG: OnceLock<std::sync::Mutex<Config>> = OnceLock::new();

/// Get the default Config singleton.
///
/// # Arguments
///
/// * `watch` - Enable hot-reload support (only applies on first call)
///
/// # Returns
///
/// A reference to the default Config instance
///
/// # Panics
///
/// Panics if the config cannot be initialized on first call.
pub fn get_config(watch: bool) -> std::sync::MutexGuard<'static, Config> {
    DEFAULT_CONFIG
        .get_or_init(|| {
            std::sync::Mutex::new(
                Config::new(None, watch).expect("Failed to initialize default config"),
            )
        })
        .lock()
        .expect("Config mutex poisoned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_parse_env_file() {
        let dir = tempdir().unwrap();
        let env_path = dir.path().join(".env");

        let mut file = File::create(&env_path).unwrap();
        writeln!(file, "# Comment line").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "LOG_LEVEL=DEBUG").unwrap();
        writeln!(file, "CHROMADB_PORT=9000").unwrap();
        writeln!(file, "TEMPERATURE=0.5").unwrap();
        writeln!(file, "REPROCESS_ON_CHANGE=true").unwrap();
        writeln!(file, "TEST_QUOTED=\"quoted value\"").unwrap();
        drop(file);

        let config = Config::new(Some(env_path), false).unwrap();

        assert_eq!(config.get("LOG_LEVEL", None), Some("DEBUG".to_string()));
        assert_eq!(config.get_int("CHROMADB_PORT", None), Some(9000));
        assert_eq!(config.get_float("TEMPERATURE", None), Some(0.5));
        assert!(config.get_bool("REPROCESS_ON_CHANGE", false));
        assert_eq!(
            config.get("TEST_QUOTED", None),
            Some("quoted value".to_string())
        );
    }

    #[test]
    fn test_json_config() {
        let dir = tempdir().unwrap();
        let json_path = dir.path().join("settings.json");

        let json_content = r#"{
            "global": {
                "log_level": "WARNING"
            },
            "hippocampus": {
                "chromadb": {
                    "port": 9999
                },
                "reprocess_on_change": false
            },
            "exo": {
                "temperature": 0.3
            },
            "chat": {
                "reset_patterns": ["/reset", "start over"]
            }
        }"#;

        let mut file = File::create(&json_path).unwrap();
        write!(file, "{}", json_content).unwrap();
        drop(file);

        let config = Config::new(Some(json_path), false).unwrap();

        assert_eq!(config.get("LOG_LEVEL", None), Some("WARNING".to_string()));
        assert_eq!(config.get_int("CHROMADB_PORT", None), Some(9999));
        assert!(!config.get_bool("REPROCESS_ON_CHANGE", true));
        assert_eq!(config.get_float("TEMPERATURE", None), Some(0.3));
        assert_eq!(
            config.get_list("CHAT_RESET_PATTERNS", None),
            Some(vec!["/reset".to_string(), "start over".to_string()])
        );
        // Arrays should also work via get() as comma-separated
        assert_eq!(
            config.get("CHAT_RESET_PATTERNS", None),
            Some("/reset,start over".to_string())
        );
    }

    #[test]
    fn test_defaults() {
        let dir = tempdir().unwrap();
        let env_path = dir.path().join(".env");
        File::create(&env_path).unwrap();

        let config = Config::new(Some(env_path), false).unwrap();

        assert_eq!(
            config.get("NONEXISTENT", Some("default")),
            Some("default".to_string())
        );
        assert_eq!(config.get_int("NONEXISTENT", Some(99)), Some(99));
        assert_eq!(config.get_float("NONEXISTENT", Some(1.5)), Some(1.5));
        assert!(!config.get_bool("NONEXISTENT", false));
        assert!(config.get_bool("NONEXISTENT", true));
    }

    #[test]
    fn test_overrides() {
        let dir = tempdir().unwrap();
        let env_path = dir.path().join(".env");

        let mut file = File::create(&env_path).unwrap();
        writeln!(file, "LOG_LEVEL=INFO").unwrap();
        drop(file);

        let mut config = Config::new(Some(env_path), false).unwrap();

        assert_eq!(config.get("LOG_LEVEL", None), Some("INFO".to_string()));

        config.set("LOG_LEVEL", "ERROR");
        assert_eq!(config.get("LOG_LEVEL", None), Some("ERROR".to_string()));

        config.clear_override("LOG_LEVEL");
        assert_eq!(config.get("LOG_LEVEL", None), Some("INFO".to_string()));
    }

    #[test]
    fn test_bool_values() {
        let dir = tempdir().unwrap();
        let json_path = dir.path().join("settings.json");

        let json_content = r#"{
            "hippocampus": {
                "reprocess_on_change": true,
                "delete_after_processing": false
            }
        }"#;

        let mut file = File::create(&json_path).unwrap();
        write!(file, "{}", json_content).unwrap();
        drop(file);

        let config = Config::new(Some(json_path), false).unwrap();

        assert!(config.get_bool("REPROCESS_ON_CHANGE", false));
        assert!(!config.get_bool("DELETE_AFTER_PROCESSING", true));
    }

    #[test]
    fn test_path_expansion() {
        let dir = tempdir().unwrap();
        let json_path = dir.path().join("settings.json");

        let json_content = r#"{
            "hippocampus": {
                "input_dir": "~/Documents/AI_IN"
            }
        }"#;

        let mut file = File::create(&json_path).unwrap();
        write!(file, "{}", json_content).unwrap();
        drop(file);

        let config = Config::new(Some(json_path), false).unwrap();

        let input_path = config.get_path("INPUT_DIR", None).unwrap();
        assert!(!input_path.to_string_lossy().contains('~'));
        assert!(input_path.to_string_lossy().contains("Documents/AI_IN"));
    }

    #[test]
    fn test_json_preferred_over_env() {
        let dir = tempdir().unwrap();
        let json_path = dir.path().join("settings.json");
        let env_path = dir.path().join(".env");

        // Create both files with different values
        let mut json_file = File::create(&json_path).unwrap();
        write!(json_file, r#"{{"global": {{"log_level": "DEBUG"}}}}"#).unwrap();
        drop(json_file);

        let mut env_file = File::create(&env_path).unwrap();
        writeln!(env_file, "LOG_LEVEL=WARNING").unwrap();
        drop(env_file);

        // Config should prefer JSON
        let config = Config::new(Some(dir.path().to_path_buf()), false).unwrap();
        assert_eq!(config.get("LOG_LEVEL", None), Some("DEBUG".to_string()));
        assert!(config.using_json);
    }
}
