//! Control API client for slash commands
//!
//! Provides HTTP client functionality to interact with the Olorin control server
//! for fetching available commands and executing them.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Metadata for a slash command from the control API
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SlashCommand {
    /// Command name without leading slash (e.g., "reset-conversation")
    pub command: String,
    /// Full slash command (e.g., "/reset-conversation")
    pub slash_command: String,
    /// Human-readable description
    pub description: String,
}

/// Response from GET /commands endpoint
#[derive(Debug, Deserialize)]
struct CommandsResponse {
    success: bool,
    commands: Vec<SlashCommand>,
}

/// Request body for POST /execute endpoint
#[derive(Debug, Serialize)]
struct ExecuteRequest {
    command: String,
    payload: serde_json::Value,
}

/// Response from POST /execute endpoint
#[derive(Debug, Deserialize)]
pub struct ExecuteResponse {
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Fetch available slash commands from the control API
///
/// Returns an empty list if the API is unavailable or returns an error.
/// This allows the chat tool to work even when the control server is down.
pub async fn fetch_commands(base_url: &str) -> Vec<SlashCommand> {
    let url = format!("{}/commands", base_url);

    // Use a short timeout to avoid blocking startup
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    match client.get(&url).send().await {
        Ok(response) => match response.json::<CommandsResponse>().await {
            Ok(data) if data.success => data.commands,
            Ok(_) => Vec::new(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Execute a slash command via the control API
///
/// Parses the command text to extract the command name and any arguments.
/// Currently supports simple key=value argument parsing.
pub async fn execute_command(base_url: &str, command_text: &str) -> Result<ExecuteResponse> {
    let url = format!("{}/execute", base_url);

    // Parse command text: "/command-name arg1=value1 arg2=value2"
    let parts: Vec<&str> = command_text.trim().splitn(2, ' ').collect();
    let command_name = parts[0].trim_start_matches('/');

    // Parse arguments if present
    let payload = if parts.len() > 1 {
        parse_arguments(parts[1])
    } else {
        serde_json::json!({})
    };

    let request = ExecuteRequest {
        command: command_name.to_string(),
        payload,
    };

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .context("Failed to send request to control API")?;

    response
        .json::<ExecuteResponse>()
        .await
        .context("Failed to parse control API response")
}

/// Parse command arguments into a JSON object
///
/// Supports:
/// - key=value (string)
/// - key=true/false (boolean)
/// - key=123 (integer)
/// - positional arguments (bare values without =)
///
/// Positional arguments are collected into a "_positional" array in order,
/// allowing the server to map them to named parameters based on ARGUMENTS metadata.
fn parse_arguments(args_str: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    let mut positional: Vec<serde_json::Value> = Vec::new();

    for part in args_str.split_whitespace() {
        if let Some((key, value)) = part.split_once('=') {
            // Named argument: key=value
            let json_value = if value == "true" {
                serde_json::Value::Bool(true)
            } else if value == "false" {
                serde_json::Value::Bool(false)
            } else if let Ok(num) = value.parse::<i64>() {
                serde_json::Value::Number(num.into())
            } else {
                serde_json::Value::String(value.to_string())
            };
            map.insert(key.to_string(), json_value);
        } else {
            // Positional argument: bare value
            let json_value = if part == "true" {
                serde_json::Value::Bool(true)
            } else if part == "false" {
                serde_json::Value::Bool(false)
            } else if let Ok(num) = part.parse::<i64>() {
                serde_json::Value::Number(num.into())
            } else {
                serde_json::Value::String(part.to_string())
            };
            positional.push(json_value);
        }
    }

    // Add positional arguments if any were found
    if !positional.is_empty() {
        map.insert(
            "_positional".to_string(),
            serde_json::Value::Array(positional),
        );
    }

    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_arguments_empty() {
        let result = parse_arguments("");
        assert_eq!(result, serde_json::json!({}));
    }

    #[test]
    fn test_parse_arguments_string() {
        let result = parse_arguments("name=test");
        assert_eq!(result, serde_json::json!({"name": "test"}));
    }

    #[test]
    fn test_parse_arguments_bool() {
        let result = parse_arguments("force=true enabled=false");
        assert_eq!(result, serde_json::json!({"force": true, "enabled": false}));
    }

    #[test]
    fn test_parse_arguments_int() {
        let result = parse_arguments("count=42");
        assert_eq!(result, serde_json::json!({"count": 42}));
    }

    #[test]
    fn test_parse_arguments_mixed() {
        let result = parse_arguments("name=test force=true count=5");
        assert_eq!(
            result,
            serde_json::json!({"name": "test", "force": true, "count": 5})
        );
    }

    #[test]
    fn test_parse_arguments_single_positional() {
        let result = parse_arguments("myfile");
        assert_eq!(result, serde_json::json!({"_positional": ["myfile"]}));
    }

    #[test]
    fn test_parse_arguments_multiple_positional() {
        let result = parse_arguments("first second third");
        assert_eq!(
            result,
            serde_json::json!({"_positional": ["first", "second", "third"]})
        );
    }

    #[test]
    fn test_parse_arguments_positional_with_types() {
        let result = parse_arguments("myfile 42 true");
        assert_eq!(
            result,
            serde_json::json!({"_positional": ["myfile", 42, true]})
        );
    }

    #[test]
    fn test_parse_arguments_mixed_positional_and_named() {
        let result = parse_arguments("myfile force=true count=5");
        assert_eq!(
            result,
            serde_json::json!({"_positional": ["myfile"], "force": true, "count": 5})
        );
    }

    #[test]
    fn test_parse_arguments_named_then_positional() {
        let result = parse_arguments("force=true myfile");
        assert_eq!(
            result,
            serde_json::json!({"force": true, "_positional": ["myfile"]})
        );
    }
}
