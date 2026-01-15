//! AI Tool Server: Write
//!
//! A simple HTTP server that implements the Olorin tool protocol.
//! Allows the AI to write content to files in ~/Documents/AI_OUT.
//!
//! Endpoints:
//! - GET  /health   - Health check
//! - GET  /describe - Tool metadata
//! - POST /call     - Execute the tool

use axum::{
    Router,
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use tower_http::cors::{Any, CorsLayer};

/// AI Tool Server for writing files
#[derive(Parser, Debug)]
#[command(
    name = "write",
    about = "AI Tool Server: Write files to ~/Documents/AI_OUT"
)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8770")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

// === Request/Response Types ===

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Serialize)]
struct ParameterSpec {
    name: String,
    #[serde(rename = "type")]
    param_type: String,
    required: bool,
    description: String,
}

#[derive(Serialize)]
struct DescribeResponse {
    name: String,
    description: String,
    parameters: Vec<ParameterSpec>,
}

#[derive(Deserialize)]
struct CallRequest {
    content: String,
    filename: String,
}

#[derive(Serialize)]
struct CallResponseSuccess {
    success: bool,
    result: String,
}

#[derive(Serialize)]
struct ErrorInfo {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[derive(Serialize)]
struct CallResponseError {
    success: bool,
    error: ErrorInfo,
}

// === Handlers ===

async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn describe() -> impl IntoResponse {
    Json(DescribeResponse {
        name: "write".to_string(),
        description: "Write content to a file in ~/Documents/AI_OUT. Use this tool when the user asks you to save, write, or export content to a file.".to_string(),
        parameters: vec![
            ParameterSpec {
                name: "content".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "The content to write to the file".to_string(),
            },
            ParameterSpec {
                name: "filename".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "The filename (without path). Will be written to ~/Documents/AI_OUT/".to_string(),
            },
        ],
    })
}

async fn call(Json(request): Json<CallRequest>) -> impl IntoResponse {
    // Validate filename (prevent path traversal)
    if request.filename.contains('/')
        || request.filename.contains('\\')
        || request.filename.contains("..")
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::to_value(CallResponseError {
                    success: false,
                    error: ErrorInfo {
                        error_type: "ValidationError".to_string(),
                        message: "Filename cannot contain path separators or '..'".to_string(),
                    },
                })
                .unwrap(),
            ),
        );
    }

    if request.filename.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::to_value(CallResponseError {
                    success: false,
                    error: ErrorInfo {
                        error_type: "ValidationError".to_string(),
                        message: "Filename cannot be empty".to_string(),
                    },
                })
                .unwrap(),
            ),
        );
    }

    // Get output directory
    let output_dir = match get_output_dir() {
        Ok(dir) => dir,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::to_value(CallResponseError {
                        success: false,
                        error: ErrorInfo {
                            error_type: "IOError".to_string(),
                            message: format!("Failed to determine output directory: {}", e),
                        },
                    })
                    .unwrap(),
                ),
            );
        }
    };

    // Create output directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(&output_dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::to_value(CallResponseError {
                    success: false,
                    error: ErrorInfo {
                        error_type: "IOError".to_string(),
                        message: format!("Failed to create output directory: {}", e),
                    },
                })
                .unwrap(),
            ),
        );
    }

    // Build full path
    let file_path = output_dir.join(&request.filename);

    // Write the file
    match write_file(&file_path, &request.content) {
        Ok(bytes_written) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(CallResponseSuccess {
                    success: true,
                    result: format!("Wrote {} bytes to {}", bytes_written, file_path.display()),
                })
                .unwrap(),
            ),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::to_value(CallResponseError {
                    success: false,
                    error: ErrorInfo {
                        error_type: "IOError".to_string(),
                        message: format!("Failed to write file: {}", e),
                    },
                })
                .unwrap(),
            ),
        ),
    }
}

// === Helper Functions ===

fn get_output_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    Ok(home.join("Documents").join("AI_OUT"))
}

fn write_file(path: &PathBuf, content: &str) -> Result<usize, std::io::Error> {
    let mut file = fs::File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(content.len())
}

// === Main ===

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Set up CORS (allow requests from any origin for local development)
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/describe", get(describe))
        .route("/call", post(call))
        .layer(cors);

    // Parse address
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("Invalid address");

    println!("Write tool server listening on http://{}", addr);
    println!("  GET  /health   - Health check");
    println!("  GET  /describe - Tool metadata");
    println!("  POST /call     - Write file");

    // Run server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
