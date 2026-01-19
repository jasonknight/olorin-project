//! Setting definitions for all configuration tabs

use serde_json::{json, Value};

/// Input type for a setting
#[derive(Debug, Clone, PartialEq)]
pub enum InputType {
    /// Single-line text input
    Text,
    /// Multi-line text (for arrays, one item per line)
    Textarea,
    /// Fixed options (up/down to cycle)
    Select(Vec<String>),
    /// Options fetched from API
    DynamicSelect(DynamicSource),
    /// Integer with optional min/max
    IntNumber { min: Option<i64>, max: Option<i64> },
    /// Float with optional min/max
    FloatNumber { min: Option<f64>, max: Option<f64> },
    /// Boolean toggle
    Toggle,
    /// Text that can be null
    NullableText,
    /// Integer that can be null
    NullableInt { min: Option<i64>, max: Option<i64> },
}

/// Source for dynamic select values
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicSource {
    /// Fetch from Ollama /api/tags
    OllamaModels,
    /// Fetch TTS models from `tts --list_models`
    TTSModels,
    /// Fetch TTS speakers (VCTK model speakers)
    TTSSpeakers,
}

/// Definition of a single setting field
#[derive(Debug, Clone)]
pub struct SettingDef {
    /// JSON path: "hippocampus.chromadb.port"
    pub key: &'static str,
    /// Display name: "ChromaDB Port"
    pub label: &'static str,
    /// Help text
    pub description: &'static str,
    /// Input type
    pub input_type: InputType,
    /// Default value (for display when null/missing)
    pub default_value: Option<Value>,
}

/// Tab definition containing settings for one component
#[derive(Debug, Clone)]
pub struct TabDef {
    /// Tab name: "Global", "Kafka", etc.
    pub name: &'static str,
    /// Settings in this tab
    pub settings: Vec<SettingDef>,
}

/// Create all tab definitions
pub fn create_tab_definitions() -> Vec<TabDef> {
    vec![
        // Tab 0: Search (special tab - settings searched dynamically)
        TabDef {
            name: "Search",
            settings: vec![],
        },
        // Tab 1: Global
        TabDef {
            name: "Global",
            settings: vec![SettingDef {
                key: "global.log_level",
                label: "Log Level",
                description: "Logging verbosity for all components",
                input_type: InputType::Select(vec![
                    "DEBUG".into(),
                    "INFO".into(),
                    "WARNING".into(),
                    "ERROR".into(),
                    "CRITICAL".into(),
                ]),
                default_value: Some(json!("INFO")),
            }],
        },
        // Tab 2: State
        TabDef {
            name: "State",
            settings: vec![SettingDef {
                key: "state.db_path",
                label: "Database Path",
                description: "SQLite database path for runtime state storage",
                input_type: InputType::Text,
                default_value: Some(json!("./data/state.db")),
            }],
        },
        // Tab 3: Inference
        TabDef {
            name: "Inference",
            settings: vec![
                SettingDef {
                    key: "inference.backend",
                    label: "Backend",
                    description: "AI inference backend to use (exo, ollama, or anthropic)",
                    input_type: InputType::Select(vec!["exo".into(), "ollama".into(), "anthropic".into()]),
                    default_value: Some(json!("ollama")),
                },
                SettingDef {
                    key: "inference.timeout",
                    label: "Timeout",
                    description: "Request timeout in seconds",
                    input_type: InputType::IntNumber {
                        min: Some(10),
                        max: Some(600),
                    },
                    default_value: Some(json!(120)),
                },
                SettingDef {
                    key: "inference.retry_count",
                    label: "Retry Count",
                    description: "Number of retries on failure",
                    input_type: InputType::IntNumber {
                        min: Some(0),
                        max: Some(10),
                    },
                    default_value: Some(json!(3)),
                },
                SettingDef {
                    key: "inference.retry_delay",
                    label: "Retry Delay",
                    description: "Delay between retries in seconds",
                    input_type: InputType::FloatNumber {
                        min: Some(0.1),
                        max: Some(30.0),
                    },
                    default_value: Some(json!(1.0)),
                },
            ],
        },
        // Tab 4: Kafka
        TabDef {
            name: "Kafka",
            settings: vec![
                SettingDef {
                    key: "kafka.bootstrap_servers",
                    label: "Bootstrap Servers",
                    description: "Kafka broker address (host:port)",
                    input_type: InputType::Text,
                    default_value: Some(json!("localhost:9092")),
                },
                SettingDef {
                    key: "kafka.send_timeout",
                    label: "Send Timeout",
                    description: "Timeout in seconds for sending messages",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(300),
                    },
                    default_value: Some(json!(10)),
                },
                SettingDef {
                    key: "kafka.max_retries",
                    label: "Max Retries",
                    description: "Maximum retry attempts for failed sends",
                    input_type: InputType::IntNumber {
                        min: Some(0),
                        max: Some(10),
                    },
                    default_value: Some(json!(3)),
                },
            ],
        },
        // Tab 5: Exo
        TabDef {
            name: "Exo",
            settings: vec![
                SettingDef {
                    key: "exo.base_url",
                    label: "Base URL",
                    description: "Exo API endpoint URL",
                    input_type: InputType::Text,
                    default_value: Some(json!("http://localhost:52415/v1")),
                },
                SettingDef {
                    key: "exo.api_key",
                    label: "API Key",
                    description: "API key for authentication (any value accepted)",
                    input_type: InputType::Text,
                    default_value: Some(json!("dummy-key")),
                },
                SettingDef {
                    key: "exo.model_name",
                    label: "Model Name",
                    description: "Model to use (null for auto-detect from running Exo)",
                    input_type: InputType::NullableText,
                    default_value: Some(Value::Null),
                },
                SettingDef {
                    key: "exo.temperature",
                    label: "Temperature",
                    description: "Response randomness (0.0=deterministic, 2.0=creative)",
                    input_type: InputType::FloatNumber {
                        min: Some(0.0),
                        max: Some(2.0),
                    },
                    default_value: Some(json!(0.7)),
                },
                SettingDef {
                    key: "exo.max_tokens",
                    label: "Max Tokens",
                    description: "Maximum response length (null for unlimited)",
                    input_type: InputType::NullableInt {
                        min: Some(1),
                        max: Some(100000),
                    },
                    default_value: Some(Value::Null),
                },
            ],
        },
        // Tab 6: Ollama
        TabDef {
            name: "Ollama",
            settings: vec![
                SettingDef {
                    key: "ollama.base_url",
                    label: "Base URL",
                    description: "Ollama API endpoint URL",
                    input_type: InputType::Text,
                    default_value: Some(json!("http://localhost:11434")),
                },
                SettingDef {
                    key: "ollama.model_name",
                    label: "Model Name",
                    description: "Installed Ollama model to use (F5 to refresh list)",
                    input_type: InputType::DynamicSelect(DynamicSource::OllamaModels),
                    default_value: Some(json!("gemma3:27b")),
                },
                SettingDef {
                    key: "ollama.temperature",
                    label: "Temperature",
                    description: "Response randomness (0.0=deterministic, 2.0=creative)",
                    input_type: InputType::FloatNumber {
                        min: Some(0.0),
                        max: Some(2.0),
                    },
                    default_value: Some(json!(0.7)),
                },
                SettingDef {
                    key: "ollama.max_tokens",
                    label: "Max Tokens",
                    description: "Maximum response length (null for unlimited)",
                    input_type: InputType::NullableInt {
                        min: Some(1),
                        max: Some(100000),
                    },
                    default_value: Some(Value::Null),
                },
            ],
        },
        // Tab 7: Anthropic
        TabDef {
            name: "Anthropic",
            settings: vec![
                SettingDef {
                    key: "anthropic.model_name",
                    label: "Model Name",
                    description: "Claude model to use (e.g., claude-sonnet-4-20250514)",
                    input_type: InputType::Text,
                    default_value: Some(json!("claude-sonnet-4-20250514")),
                },
                SettingDef {
                    key: "anthropic.temperature",
                    label: "Temperature",
                    description: "Response randomness (0.0=deterministic, 1.0=creative)",
                    input_type: InputType::FloatNumber {
                        min: Some(0.0),
                        max: Some(1.0),
                    },
                    default_value: Some(json!(0.7)),
                },
                SettingDef {
                    key: "anthropic.max_tokens",
                    label: "Max Tokens",
                    description: "Maximum response length",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(200000),
                    },
                    default_value: Some(json!(4096)),
                },
                // NOTE: API key is not shown here - it must be set in .env file
            ],
        },
        // Tab 8: Broca (was Tab 7)
        TabDef {
            name: "Broca",
            settings: vec![
                SettingDef {
                    key: "broca.kafka_topic",
                    label: "Kafka Topic",
                    description: "Topic to consume TTS requests from",
                    input_type: InputType::Text,
                    default_value: Some(json!("ai_out")),
                },
                SettingDef {
                    key: "broca.consumer_group",
                    label: "Consumer Group",
                    description: "Kafka consumer group identifier",
                    input_type: InputType::Text,
                    default_value: Some(json!("tts-consumer-group")),
                },
                SettingDef {
                    key: "broca.auto_offset_reset",
                    label: "Offset Reset",
                    description: "Kafka offset strategy for new consumer groups",
                    input_type: InputType::Select(vec!["earliest".into(), "latest".into()]),
                    default_value: Some(json!("earliest")),
                },
                SettingDef {
                    key: "broca.tts.engine",
                    label: "TTS Engine",
                    description: "Text-to-speech backend: coqui (local neural TTS) or orca (Picovoice streaming TTS)",
                    input_type: InputType::Select(vec!["coqui".into(), "orca".into()]),
                    default_value: Some(json!("coqui")),
                },
                SettingDef {
                    key: "broca.tts.output_dir",
                    label: "Output Directory",
                    description: "Directory where audio files are saved",
                    input_type: InputType::Text,
                    default_value: Some(json!("output")),
                },
                // Coqui TTS settings
                SettingDef {
                    key: "broca.tts.model_name",
                    label: "Coqui Model",
                    description: "Coqui TTS model to load (F5 to refresh). Only used when engine=coqui",
                    input_type: InputType::DynamicSelect(DynamicSource::TTSModels),
                    default_value: Some(json!("tts_models/en/vctk/vits")),
                },
                SettingDef {
                    key: "broca.tts.speaker",
                    label: "Coqui Speaker",
                    description: "Speaker/voice ID for multi-speaker Coqui models (F5 to refresh)",
                    input_type: InputType::DynamicSelect(DynamicSource::TTSSpeakers),
                    default_value: Some(json!("p272")),
                },
                // Orca TTS settings
                SettingDef {
                    key: "broca.orca.access_key",
                    label: "Orca Access Key",
                    description: "Picovoice access key (same as Porcupine). Get from console.picovoice.ai",
                    input_type: InputType::Text,
                    default_value: Some(Value::Null),
                },
                SettingDef {
                    key: "broca.orca.voice",
                    label: "Orca Voice",
                    description: "Voice model for Orca TTS. Format: language_gender",
                    input_type: InputType::Select(vec![
                        "en_female".into(),
                        "en_male".into(),
                        "de_female".into(),
                        "de_male".into(),
                        "es_female".into(),
                        "es_male".into(),
                        "fr_female".into(),
                        "fr_male".into(),
                        "it_female".into(),
                        "it_male".into(),
                        "ja_female".into(),
                        "ko_female".into(),
                        "pt_female".into(),
                        "pt_male".into(),
                    ]),
                    default_value: Some(json!("en_female")),
                },
                SettingDef {
                    key: "broca.orca.model_path",
                    label: "Orca Model Path",
                    description: "Optional custom .pv model file path (leave null for built-in voices)",
                    input_type: InputType::NullableText,
                    default_value: Some(Value::Null),
                },
            ],
        },
        // Tab 8: Cortex
        TabDef {
            name: "Cortex",
            settings: vec![
                SettingDef {
                    key: "cortex.input_topic",
                    label: "Input Topic",
                    description: "Kafka topic to consume prompts from",
                    input_type: InputType::Text,
                    default_value: Some(json!("prompts")),
                },
                SettingDef {
                    key: "cortex.output_topic",
                    label: "Output Topic",
                    description: "Kafka topic to publish AI responses to",
                    input_type: InputType::Text,
                    default_value: Some(json!("ai_out")),
                },
                SettingDef {
                    key: "cortex.consumer_group",
                    label: "Consumer Group",
                    description: "Kafka consumer group identifier",
                    input_type: InputType::Text,
                    default_value: Some(json!("exo-consumer-group")),
                },
                SettingDef {
                    key: "cortex.auto_offset_reset",
                    label: "Offset Reset",
                    description: "Kafka offset strategy for new consumer groups",
                    input_type: InputType::Select(vec!["earliest".into(), "latest".into()]),
                    default_value: Some(json!("earliest")),
                },
            ],
        },
        // Tab 9: Hippocampus
        TabDef {
            name: "Hippocampus",
            settings: vec![
                SettingDef {
                    key: "hippocampus.input_dir",
                    label: "Input Directory",
                    description: "Directory to monitor for documents",
                    input_type: InputType::Text,
                    default_value: Some(json!("~/Documents/AI_IN")),
                },
                SettingDef {
                    key: "hippocampus.poll_interval",
                    label: "Poll Interval",
                    description: "Seconds between directory scans",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(300),
                    },
                    default_value: Some(json!(5)),
                },
                SettingDef {
                    key: "hippocampus.reprocess_on_change",
                    label: "Reprocess on Change",
                    description: "Reprocess files when content changes",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "hippocampus.delete_after_processing",
                    label: "Delete After Processing",
                    description: "Delete source files after processing",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(false)),
                },
                SettingDef {
                    key: "hippocampus.chromadb.host",
                    label: "ChromaDB Host",
                    description: "ChromaDB server hostname",
                    input_type: InputType::Text,
                    default_value: Some(json!("localhost")),
                },
                SettingDef {
                    key: "hippocampus.chromadb.port",
                    label: "ChromaDB Port",
                    description: "ChromaDB server port",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(65535),
                    },
                    default_value: Some(json!(8000)),
                },
                SettingDef {
                    key: "hippocampus.chromadb.collection",
                    label: "ChromaDB Collection",
                    description: "Collection name for document embeddings",
                    input_type: InputType::Text,
                    default_value: Some(json!("documents")),
                },
                SettingDef {
                    key: "hippocampus.chunking.size",
                    label: "Chunk Size",
                    description: "Target chunk size in characters",
                    input_type: InputType::IntNumber {
                        min: Some(100),
                        max: Some(32000),
                    },
                    default_value: Some(json!(4000)),
                },
                SettingDef {
                    key: "hippocampus.chunking.overlap",
                    label: "Chunk Overlap",
                    description: "Overlap between chunks in characters",
                    input_type: InputType::IntNumber {
                        min: Some(0),
                        max: Some(8000),
                    },
                    default_value: Some(json!(400)),
                },
                SettingDef {
                    key: "hippocampus.chunking.min_size",
                    label: "Min Chunk Size",
                    description: "Minimum chunk size threshold",
                    input_type: InputType::IntNumber {
                        min: Some(10),
                        max: Some(4000),
                    },
                    default_value: Some(json!(200)),
                },
                SettingDef {
                    key: "hippocampus.trackers.ollama.enabled",
                    label: "Ollama Filter Enabled",
                    description: "Enable LLM-based content filtering",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "hippocampus.trackers.ollama.model",
                    label: "Filter Model",
                    description: "Ollama model for content relevance scoring",
                    input_type: InputType::Text,
                    default_value: Some(json!("llama3.2:1b")),
                },
                SettingDef {
                    key: "hippocampus.trackers.ollama.threshold",
                    label: "Filter Threshold",
                    description: "Minimum relevance score (0.0-1.0)",
                    input_type: InputType::FloatNumber {
                        min: Some(0.0),
                        max: Some(1.0),
                    },
                    default_value: Some(json!(0.5)),
                },
                SettingDef {
                    key: "hippocampus.trackers.min_content_chars",
                    label: "Min Content Chars",
                    description: "Minimum character count for content",
                    input_type: InputType::IntNumber {
                        min: Some(0),
                        max: Some(10000),
                    },
                    default_value: Some(json!(100)),
                },
                SettingDef {
                    key: "hippocampus.trackers.min_content_density",
                    label: "Min Content Density",
                    description: "Minimum alphanumeric ratio (0.0-1.0)",
                    input_type: InputType::FloatNumber {
                        min: Some(0.0),
                        max: Some(1.0),
                    },
                    default_value: Some(json!(0.3)),
                },
                SettingDef {
                    key: "hippocampus.trackers.min_word_count",
                    label: "Min Word Count",
                    description: "Minimum word count for content",
                    input_type: InputType::IntNumber {
                        min: Some(0),
                        max: Some(1000),
                    },
                    default_value: Some(json!(20)),
                },
                SettingDef {
                    key: "hippocampus.trackers.notify_broca",
                    label: "Notify Broca",
                    description: "Send TTS notification when processing starts",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "hippocampus.trackers.ollama.paths",
                    label: "Ollama Paths",
                    description: "Paths to search for Ollama executable (one per line)",
                    input_type: InputType::Textarea,
                    default_value: Some(json!([
                        "ollama",
                        "/usr/local/bin/ollama",
                        "/opt/homebrew/bin/ollama",
                        "/Applications/Ollama.app/Contents/Resources/ollama"
                    ])),
                },
                SettingDef {
                    key: "hippocampus.tracking_db",
                    label: "Tracking DB",
                    description: "SQLite database for markdown file tracking",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/tracking.db")),
                },
                SettingDef {
                    key: "hippocampus.pdf_tracking_db",
                    label: "PDF Tracking DB",
                    description: "SQLite database for PDF tracking",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/pdf_tracking.db")),
                },
                SettingDef {
                    key: "hippocampus.ebook_tracking_db",
                    label: "Ebook Tracking DB",
                    description: "SQLite database for ebook tracking",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/ebook_tracking.db")),
                },
                SettingDef {
                    key: "hippocampus.txt_tracking_db",
                    label: "TXT Tracking DB",
                    description: "SQLite database for text file tracking",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/txt_tracking.db")),
                },
                SettingDef {
                    key: "hippocampus.office_tracking_db",
                    label: "Office Tracking DB",
                    description: "SQLite database for Office document tracking",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/office_tracking.db")),
                },
                SettingDef {
                    key: "hippocampus.context_db",
                    label: "Context DB",
                    description: "SQLite database for context storage",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/context.db")),
                },
                SettingDef {
                    key: "hippocampus.log_file",
                    label: "Log File",
                    description: "Path to ingest log file",
                    input_type: InputType::Text,
                    default_value: Some(json!("./data/ingest.log")),
                },
            ],
        },
        // Tab 10: Enrichener
        TabDef {
            name: "Enrichener",
            settings: vec![
                SettingDef {
                    key: "enrichener.input_topic",
                    label: "Input Topic",
                    description: "Kafka topic for raw user input",
                    input_type: InputType::Text,
                    default_value: Some(json!("ai_in")),
                },
                SettingDef {
                    key: "enrichener.output_topic",
                    label: "Output Topic",
                    description: "Kafka topic for enriched prompts",
                    input_type: InputType::Text,
                    default_value: Some(json!("prompts")),
                },
                SettingDef {
                    key: "enrichener.consumer_group",
                    label: "Consumer Group",
                    description: "Kafka consumer group identifier",
                    input_type: InputType::Text,
                    default_value: Some(json!("enrichener-consumer-group")),
                },
                SettingDef {
                    key: "enrichener.auto_offset_reset",
                    label: "Offset Reset",
                    description: "Kafka offset strategy for new consumer groups",
                    input_type: InputType::Select(vec!["earliest".into(), "latest".into()]),
                    default_value: Some(json!("earliest")),
                },
                SettingDef {
                    key: "enrichener.thread_pool_size",
                    label: "Thread Pool Size",
                    description: "Threads for parallel context retrieval",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(16),
                    },
                    default_value: Some(json!(3)),
                },
                SettingDef {
                    key: "enrichener.llm_timeout_seconds",
                    label: "LLM Timeout",
                    description: "Timeout for LLM decision calls (seconds)",
                    input_type: InputType::IntNumber {
                        min: Some(10),
                        max: Some(600),
                    },
                    default_value: Some(json!(120)),
                },
                SettingDef {
                    key: "enrichener.decision_temperature",
                    label: "Decision Temperature",
                    description: "Temperature for context decision LLM (low=deterministic)",
                    input_type: InputType::FloatNumber {
                        min: Some(0.0),
                        max: Some(2.0),
                    },
                    default_value: Some(json!(0.1)),
                },
                SettingDef {
                    key: "enrichener.chromadb_query_n_results",
                    label: "Query Results",
                    description: "Number of search results to retrieve",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(100),
                    },
                    default_value: Some(json!(10)),
                },
                SettingDef {
                    key: "enrichener.context_db_path",
                    label: "Context DB Path",
                    description: "SQLite database for context storage",
                    input_type: InputType::Text,
                    default_value: Some(json!("./hippocampus/data/context.db")),
                },
                SettingDef {
                    key: "enrichener.cleanup_context_after_use",
                    label: "Cleanup Context",
                    description: "Delete context after inference",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(false)),
                },
            ],
        },
        // Tab 11: Chat
        TabDef {
            name: "Chat",
            settings: vec![
                SettingDef {
                    key: "chat.history_enabled",
                    label: "History Enabled",
                    description: "Enable chat history persistence",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "chat.db_path",
                    label: "Database Path",
                    description: "SQLite database for chat history",
                    input_type: InputType::Text,
                    default_value: Some(json!("./cortex/data/chat.db")),
                },
                SettingDef {
                    key: "chat.reset_patterns",
                    label: "Reset Patterns",
                    description: "Text patterns that reset conversation (one per line)",
                    input_type: InputType::Textarea,
                    default_value: Some(json!([
                        "/reset",
                        "reset conversation",
                        "new conversation",
                        "start over",
                        "clear history"
                    ])),
                },
            ],
        },
        // Tab 12: Control
        TabDef {
            name: "Control",
            settings: vec![
                SettingDef {
                    key: "control.api.enabled",
                    label: "API Enabled",
                    description: "Enable slash command HTTP API server",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "control.api.port",
                    label: "API Port",
                    description: "HTTP port for slash command API",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(65535),
                    },
                    default_value: Some(json!(8765)),
                },
                SettingDef {
                    key: "control.api.host",
                    label: "API Host",
                    description: "Host/IP to bind the API server to",
                    input_type: InputType::Text,
                    default_value: Some(json!("0.0.0.0")),
                },
            ],
        },
        // Tab 13: Tools
        TabDef {
            name: "Tools",
            settings: vec![
                SettingDef {
                    key: "tools.write.enabled",
                    label: "Write Tool Enabled",
                    description: "Enable file writing tool for AI",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "tools.write.port",
                    label: "Write Tool Port",
                    description: "HTTP port for write tool server",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(65535),
                    },
                    default_value: Some(json!(8770)),
                },
                SettingDef {
                    key: "tools.embeddings.enabled",
                    label: "Embeddings Enabled",
                    description: "Enable embeddings generation tool",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "tools.embeddings.port",
                    label: "Embeddings Port",
                    description: "HTTP port for embeddings server",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(65535),
                    },
                    default_value: Some(json!(8771)),
                },
                SettingDef {
                    key: "tools.embeddings.model",
                    label: "Embeddings Model",
                    description: "Sentence-transformers model for embeddings",
                    input_type: InputType::Text,
                    default_value: Some(json!("nomic-ai/nomic-embed-text-v1.5")),
                },
                SettingDef {
                    key: "tools.embeddings.document_prefix",
                    label: "Document Prefix",
                    description: "Prefix for document encoding (model-specific)",
                    input_type: InputType::Text,
                    default_value: Some(json!("search_document: ")),
                },
                SettingDef {
                    key: "tools.embeddings.query_prefix",
                    label: "Query Prefix",
                    description: "Prefix for query encoding (model-specific)",
                    input_type: InputType::Text,
                    default_value: Some(json!("search_query: ")),
                },
                SettingDef {
                    key: "tools.search.enabled",
                    label: "Search Tool Enabled",
                    description: "Enable ChromaDB search tool for AI",
                    input_type: InputType::Toggle,
                    default_value: Some(json!(true)),
                },
                SettingDef {
                    key: "tools.search.port",
                    label: "Search Tool Port",
                    description: "HTTP port for search tool server",
                    input_type: InputType::IntNumber {
                        min: Some(1),
                        max: Some(65535),
                    },
                    default_value: Some(json!(8772)),
                },
            ],
        },
    ]
}
