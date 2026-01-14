//! Unified Configuration Library for Olorin Project
//!
//! Provides a Config struct with type-safe getters and optional hot-reload support.
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
//!
//! // With hot-reload support
//! let mut config = Config::new(None, true).unwrap();
//! if config.reload() {
//!     println!("Configuration was updated");
//! }
//! ```

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::SystemTime;

/// Error types for configuration operations
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read .env file: {0}")]
    IoError(#[from] io::Error),

    #[error("Failed to find project root")]
    ProjectRootNotFound,
}

/// Find the project root by looking for the .env file or known directories.
fn find_project_root() -> Option<PathBuf> {
    // Try to find from current executable location
    if let Ok(exe_path) = env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let mut search = parent.to_path_buf();
            for _ in 0..10 {
                if search.join(".env").exists() {
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
    if let Ok(cwd) = env::current_dir() {
        let mut search = cwd;
        for _ in 0..10 {
            if search.join(".env").exists() {
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
    env::current_dir().ok()
}

/// Configuration manager that loads from a .env file.
///
/// Provides type-safe getters and optional hot-reload support for runtime
/// configuration changes.
pub struct Config {
    env_path: PathBuf,
    watch: bool,
    mtime: Option<SystemTime>,
    overrides: HashMap<String, String>,
}

impl Config {
    /// Initialize the configuration manager.
    ///
    /// # Arguments
    ///
    /// * `env_path` - Path to the .env file. Defaults to project root .env
    /// * `watch` - If true, enables hot-reload support via reload() method
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the project root cannot be found when `env_path` is None.
    pub fn new(env_path: Option<PathBuf>, watch: bool) -> Result<Self, ConfigError> {
        let env_path = match env_path {
            Some(p) => p,
            None => {
                let root = find_project_root().ok_or(ConfigError::ProjectRootNotFound)?;
                root.join(".env")
            }
        };

        let mut config = Self {
            env_path,
            watch,
            mtime: None,
            overrides: HashMap::new(),
        };

        config.load()?;
        Ok(config)
    }

    /// Load the .env file into environment variables.
    fn load(&mut self) -> Result<(), ConfigError> {
        if !self.env_path.exists() {
            return Ok(());
        }

        self.mtime = fs::metadata(&self.env_path)?.modified().ok();

        let file = fs::File::open(&self.env_path)?;
        let reader = io::BufReader::new(file);

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

                env::set_var(key, value);
            }
        }

        Ok(())
    }

    /// Reload configuration if the .env file has changed.
    ///
    /// Returns true if configuration was reloaded, false otherwise.
    pub fn reload(&mut self) -> bool {
        if !self.watch || !self.env_path.exists() {
            return false;
        }

        let current_mtime = fs::metadata(&self.env_path)
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
    /// * `key` - The environment variable name
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The configuration value or default
    pub fn get(&self, key: &str, default: Option<&str>) -> Option<String> {
        // Check overrides first
        if let Some(value) = self.overrides.get(key) {
            return Some(value.clone());
        }

        env::var(key).ok().or_else(|| default.map(String::from))
    }

    /// Get a configuration value as an integer.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name
    /// * `default` - Default value if not set or invalid
    ///
    /// # Returns
    ///
    /// The configuration value as i64 or default
    pub fn get_int(&self, key: &str, default: Option<i64>) -> Option<i64> {
        match self.get(key, None) {
            Some(value) => value.parse().ok().or(default),
            None => default,
        }
    }

    /// Get a configuration value as a float.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name
    /// * `default` - Default value if not set or invalid
    ///
    /// # Returns
    ///
    /// The configuration value as f64 or default
    pub fn get_float(&self, key: &str, default: Option<f64>) -> Option<f64> {
        match self.get(key, None) {
            Some(value) => value.parse().ok().or(default),
            None => default,
        }
    }

    /// Get a configuration value as a boolean.
    ///
    /// Recognizes: true, yes, 1, on (case-insensitive) as true.
    /// Everything else (including empty string) is false.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The configuration value as bool or default
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.get(key, None) {
            Some(value) => {
                let lower = value.to_lowercase();
                matches!(lower.as_str(), "true" | "yes" | "1" | "on")
            }
            None => default,
        }
    }

    /// Get a configuration value as an expanded path.
    ///
    /// Expands ~ to the user's home directory.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name
    /// * `default` - Default value if not set
    ///
    /// # Returns
    ///
    /// The expanded path as PathBuf or None
    pub fn get_path(&self, key: &str, default: Option<&str>) -> Option<PathBuf> {
        let value = self.get(key, default)?;

        if value.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                return Some(home.join(&value[1..].trim_start_matches('/')));
            }
        }

        Some(PathBuf::from(value))
    }

    /// Set a configuration value (in-memory only).
    ///
    /// This override takes precedence over environment variables.
    /// Does not modify the .env file.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name
    /// * `value` - The value to set
    pub fn set(&mut self, key: &str, value: &str) {
        self.overrides.insert(key.to_string(), value.to_string());
    }

    /// Clear an in-memory override for a key.
    ///
    /// After clearing, get() will return the environment variable value.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name
    pub fn clear_override(&mut self, key: &str) {
        self.overrides.remove(key);
    }

    /// Clear all in-memory overrides.
    pub fn clear_all_overrides(&mut self) {
        self.overrides.clear();
    }

    /// Return the path to the .env file.
    pub fn env_path(&self) -> &PathBuf {
        &self.env_path
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
        writeln!(file, "TEST_STRING=hello").unwrap();
        writeln!(file, "TEST_INT=42").unwrap();
        writeln!(file, "TEST_FLOAT=3.14").unwrap();
        writeln!(file, "TEST_BOOL=true").unwrap();
        writeln!(file, "TEST_QUOTED=\"quoted value\"").unwrap();
        drop(file);

        let config = Config::new(Some(env_path), false).unwrap();

        assert_eq!(config.get("TEST_STRING", None), Some("hello".to_string()));
        assert_eq!(config.get_int("TEST_INT", None), Some(42));
        assert_eq!(config.get_float("TEST_FLOAT", None), Some(3.14));
        assert!(config.get_bool("TEST_BOOL", false));
        assert_eq!(
            config.get("TEST_QUOTED", None),
            Some("quoted value".to_string())
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
        writeln!(file, "MY_KEY=original").unwrap();
        drop(file);

        let mut config = Config::new(Some(env_path), false).unwrap();

        assert_eq!(config.get("MY_KEY", None), Some("original".to_string()));

        config.set("MY_KEY", "overridden");
        assert_eq!(config.get("MY_KEY", None), Some("overridden".to_string()));

        config.clear_override("MY_KEY");
        assert_eq!(config.get("MY_KEY", None), Some("original".to_string()));
    }

    #[test]
    fn test_bool_values() {
        let dir = tempdir().unwrap();
        let env_path = dir.path().join(".env");

        let mut file = File::create(&env_path).unwrap();
        writeln!(file, "BOOL_TRUE=true").unwrap();
        writeln!(file, "BOOL_YES=yes").unwrap();
        writeln!(file, "BOOL_ONE=1").unwrap();
        writeln!(file, "BOOL_ON=on").unwrap();
        writeln!(file, "BOOL_FALSE=false").unwrap();
        writeln!(file, "BOOL_NO=no").unwrap();
        writeln!(file, "BOOL_ZERO=0").unwrap();
        writeln!(file, "BOOL_EMPTY=").unwrap();
        drop(file);

        let config = Config::new(Some(env_path), false).unwrap();

        assert!(config.get_bool("BOOL_TRUE", false));
        assert!(config.get_bool("BOOL_YES", false));
        assert!(config.get_bool("BOOL_ONE", false));
        assert!(config.get_bool("BOOL_ON", false));
        assert!(!config.get_bool("BOOL_FALSE", true));
        assert!(!config.get_bool("BOOL_NO", true));
        assert!(!config.get_bool("BOOL_ZERO", true));
        assert!(!config.get_bool("BOOL_EMPTY", true));
    }

    #[test]
    fn test_path_expansion() {
        let dir = tempdir().unwrap();
        let env_path = dir.path().join(".env");

        let mut file = File::create(&env_path).unwrap();
        writeln!(file, "HOME_PATH=~/Documents").unwrap();
        writeln!(file, "ABS_PATH=/tmp/test").unwrap();
        drop(file);

        let config = Config::new(Some(env_path), false).unwrap();

        let home_path = config.get_path("HOME_PATH", None).unwrap();
        assert!(!home_path.to_string_lossy().contains('~'));
        assert!(home_path.to_string_lossy().contains("Documents"));

        let abs_path = config.get_path("ABS_PATH", None).unwrap();
        assert_eq!(abs_path, PathBuf::from("/tmp/test"));
    }
}
