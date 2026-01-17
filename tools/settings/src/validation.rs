//! Validation logic for setting values
//!
//! This module provides shared validation functions to eliminate duplication
//! between save_field and save_search_result_field.

use crate::settings::InputType;
use serde_json::Value;

/// Result of validating a setting value
#[derive(Debug)]
pub enum ValidationResult {
    /// Validation passed, contains the JSON value to save
    Valid(Value),
    /// Validation failed, contains the error message
    Invalid(String),
}

/// Validate and convert an input buffer to a JSON value based on the input type
pub fn validate_and_convert(input_buffer: &str, input_type: &InputType) -> ValidationResult {
    match input_type {
        InputType::Text => ValidationResult::Valid(Value::String(input_buffer.to_string())),

        InputType::Textarea => {
            let lines: Vec<Value> = input_buffer
                .lines()
                .map(|l| Value::String(l.to_string()))
                .collect();
            ValidationResult::Valid(Value::Array(lines))
        }

        InputType::Select(_) | InputType::DynamicSelect(_) => {
            ValidationResult::Valid(Value::String(input_buffer.to_string()))
        }

        InputType::IntNumber { min, max } => validate_int(input_buffer, *min, *max),

        InputType::FloatNumber { min, max } => validate_float(input_buffer, *min, *max),

        InputType::Toggle => ValidationResult::Valid(Value::Bool(input_buffer == "true")),

        InputType::NullableText => {
            if input_buffer.is_empty() {
                ValidationResult::Valid(Value::Null)
            } else {
                ValidationResult::Valid(Value::String(input_buffer.to_string()))
            }
        }

        InputType::NullableInt { min, max } => {
            if input_buffer.is_empty() {
                ValidationResult::Valid(Value::Null)
            } else {
                validate_int(input_buffer, *min, *max)
            }
        }
    }
}

/// Validate and convert an integer string
fn validate_int(input: &str, min: Option<i64>, max: Option<i64>) -> ValidationResult {
    match input.parse::<i64>() {
        Ok(n) => {
            if let Some(min_val) = min {
                if n < min_val {
                    return ValidationResult::Invalid(format!("Min: {}", min_val));
                }
            }
            if let Some(max_val) = max {
                if n > max_val {
                    return ValidationResult::Invalid(format!("Max: {}", max_val));
                }
            }
            ValidationResult::Valid(Value::Number(n.into()))
        }
        Err(_) => ValidationResult::Invalid("Invalid number".into()),
    }
}

/// Validate and convert a float string
fn validate_float(input: &str, min: Option<f64>, max: Option<f64>) -> ValidationResult {
    match input.parse::<f64>() {
        Ok(n) => {
            if let Some(min_val) = min {
                if n < min_val {
                    return ValidationResult::Invalid(format!("Min: {:.1}", min_val));
                }
            }
            if let Some(max_val) = max {
                if n > max_val {
                    return ValidationResult::Invalid(format!("Max: {:.1}", max_val));
                }
            }
            match serde_json::Number::from_f64(n) {
                Some(num) => ValidationResult::Valid(Value::Number(num)),
                None => ValidationResult::Invalid("Invalid float value".into()),
            }
        }
        Err(_) => ValidationResult::Invalid("Invalid number".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_text() {
        let result = validate_and_convert("hello", &InputType::Text);
        assert!(matches!(result, ValidationResult::Valid(Value::String(s)) if s == "hello"));
    }

    #[test]
    fn test_validate_textarea() {
        let result = validate_and_convert("line1\nline2", &InputType::Textarea);
        if let ValidationResult::Valid(Value::Array(arr)) = result {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("line1".into()));
            assert_eq!(arr[1], Value::String("line2".into()));
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_validate_int_valid() {
        let result = validate_and_convert(
            "42",
            &InputType::IntNumber {
                min: Some(0),
                max: Some(100),
            },
        );
        assert!(
            matches!(result, ValidationResult::Valid(Value::Number(n)) if n.as_i64() == Some(42))
        );
    }

    #[test]
    fn test_validate_int_below_min() {
        let result = validate_and_convert(
            "-5",
            &InputType::IntNumber {
                min: Some(0),
                max: Some(100),
            },
        );
        assert!(matches!(result, ValidationResult::Invalid(msg) if msg.contains("Min")));
    }

    #[test]
    fn test_validate_int_above_max() {
        let result = validate_and_convert(
            "150",
            &InputType::IntNumber {
                min: Some(0),
                max: Some(100),
            },
        );
        assert!(matches!(result, ValidationResult::Invalid(msg) if msg.contains("Max")));
    }

    #[test]
    fn test_validate_int_invalid() {
        let result = validate_and_convert(
            "abc",
            &InputType::IntNumber {
                min: None,
                max: None,
            },
        );
        assert!(matches!(result, ValidationResult::Invalid(msg) if msg.contains("Invalid")));
    }

    #[test]
    fn test_validate_float_valid() {
        let result = validate_and_convert(
            "3.14",
            &InputType::FloatNumber {
                min: Some(0.0),
                max: Some(10.0),
            },
        );
        if let ValidationResult::Valid(Value::Number(n)) = result {
            assert!((n.as_f64().unwrap() - 3.14).abs() < 0.001);
        } else {
            panic!("Expected valid float");
        }
    }

    #[test]
    fn test_validate_toggle() {
        let true_result = validate_and_convert("true", &InputType::Toggle);
        assert!(matches!(
            true_result,
            ValidationResult::Valid(Value::Bool(true))
        ));

        let false_result = validate_and_convert("false", &InputType::Toggle);
        assert!(matches!(
            false_result,
            ValidationResult::Valid(Value::Bool(false))
        ));
    }

    #[test]
    fn test_validate_nullable_text_empty() {
        let result = validate_and_convert("", &InputType::NullableText);
        assert!(matches!(result, ValidationResult::Valid(Value::Null)));
    }

    #[test]
    fn test_validate_nullable_text_value() {
        let result = validate_and_convert("hello", &InputType::NullableText);
        assert!(matches!(result, ValidationResult::Valid(Value::String(s)) if s == "hello"));
    }

    #[test]
    fn test_validate_nullable_int_empty() {
        let result = validate_and_convert(
            "",
            &InputType::NullableInt {
                min: Some(0),
                max: Some(100),
            },
        );
        assert!(matches!(result, ValidationResult::Valid(Value::Null)));
    }

    #[test]
    fn test_validate_nullable_int_value() {
        let result = validate_and_convert(
            "50",
            &InputType::NullableInt {
                min: Some(0),
                max: Some(100),
            },
        );
        assert!(
            matches!(result, ValidationResult::Valid(Value::Number(n)) if n.as_i64() == Some(50))
        );
    }
}
