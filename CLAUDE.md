# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Olorin-project is a distributed AI pipeline system composed of brain-inspired components communicating via Kafka. The system creates a complete voice-to-voice AI pipeline with document retrieval capabilities.

## System Architecture

The project follows a microservices architecture with independent components:

```
[Voice Input]                              [Document Input]
     |                                           |
     v                                           v
  Temporal (Speech-to-Text)              Hippocampus (Knowledge Base)
  - Listens for: "Hey Olorin"              - Monitors: ~/Documents/AI_IN
  - Service: Faster-Whisper                - Service: ChromaDB
  - Produces: ai_in (Kafka topic)          - Purpose: Document ingestion
     |                                           |
     v                                           |
  Enrichener (RAG)  <----------------------------+
  - Consumes: ai_in (Kafka topic)          (semantic search)
  - Adds context from ChromaDB
  - Produces: prompts (Kafka topic)
     |
     v
  Cortex (AI Processing)
  - Consumes: prompts (Kafka topic)
  - Service: Ollama/Exo (AI inference)
  - Produces: ai_out (Kafka topic)
     |
     v
  Broca (Text-to-Speech)
  - Consumes: ai_out (Kafka topic)
  - Service: Coqui TTS
  - Output: Audio playback
```

### Infrastructure Components

- **Kafka**: Message broker running on port 9092 (in Podman container)
- **ChromaDB**: Vector database running on port 8000 (in Podman container)
- **Exo**: Distributed AI inference server (OpenAI-compatible API)

## Common Commands

### Linting and Formatting Code

```bash
# Run this command when you've written new Python code
pre-commit run --all-files

# Run these commands when you've written new Rust code
# Format all Rust projects (uses edition from each Cargo.toml)
cargo fmt --manifest-path libs/config-rs/Cargo.toml
cargo fmt --manifest-path libs/state-rs/Cargo.toml
cargo fmt --manifest-path tools/chat/Cargo.toml
cargo fmt --manifest-path tools/olorin-inspector/Cargo.toml

# Run clippy on all Rust projects and fix any issues found
cargo clippy --manifest-path libs/config-rs/Cargo.toml -- -D warnings
cargo clippy --manifest-path libs/state-rs/Cargo.toml -- -D warnings
cargo clippy --manifest-path tools/chat/Cargo.toml -- -D warnings
cargo clippy --manifest-path tools/olorin-inspector/Cargo.toml -- -D warnings
``` 

### Starting/Stopping the System

```bash
# Start entire infrastructure
./up

# Check system status
./status

# Stop entire infrastructure
./down
```

The `./up` script:
1. Starts Kafka container (kafkaserver) on ports 9092-9093
2. Starts ChromaDB container on port 8000
3. Launches background daemons for each component
4. Stores PIDs in `.pids/` directory
5. Logs to `logs/` directory

### Working with Individual Components

Each component (broca/, cortex/, hippocampus/) has its own virtual environment and can be run independently.

#### Broca (TTS Consumer)
```bash
cd broca
source venv/bin/activate
python3 consumer.py           # Start TTS consumer
python3 producer.py           # Interactive REPL to send text

# Kafka topics
./create-topic ai_out
./list-topics
```

#### Cortex (AI Consumer)
```bash
cd cortex
source venv/bin/activate
python3 consumer.py           # Start AI consumer
python3 producer.py           # Interactive REPL to send prompts

# Kafka topics
./create-topic prompts
./list-topics
```

#### Hippocampus (Document Ingestion)
```bash
cd hippocampus
source venv/bin/activate

# Start ingestion pipeline (monitors ~/Documents/AI_IN)
python3 ingest.py

# PDF monitoring and conversion
python3 pdf_tracker.py

# Query the knowledge base
python3 query.py

# Database management
python3 clear_db.py           # Clear ChromaDB collection
python3 status.py             # Check database status

# Testing
pytest                        # Run all tests
pytest --cov                  # Run with coverage
pytest tests/test_markdown_chunker.py  # Run specific test
```

#### Temporal (Voice Input)
```bash
cd temporal
source venv/bin/activate
python3 consumer.py           # Start voice-activated STT consumer

# The consumer listens for "hey olorin" wake phrase
# After wake word detection, it transcribes speech until:
# - 3 seconds of silence (configurable)
# - "that's all" or other stop phrase
# Transcribed text is sent to ai_in topic
```

### Container Management

```bash
# View running containers
podman ps

# View container logs
podman logs kafkaserver
podman logs chromadb

# Restart a container
podman restart kafkaserver
podman restart chromadb
```

### Log Management

All logs are stored in `logs/` directory:
```bash
# View logs
ls logs/

# Tail specific component
tail -f logs/cortex-consumer.log
tail -f logs/broca-consumer.log
tail -f logs/hippocampus-ingest.log
tail -f logs/hippocampus-pdf-tracker.log
```

## Component Details

### Broca (Text-to-Speech)

**Purpose**: Converts AI-generated text responses to speech

**Key Files**:
- `consumer.py` - Kafka consumer that receives text and generates speech
- `producer.py` - Interactive REPL for testing
- `output/` - Generated audio files

**Configuration** (in `settings.json` `broca` section):
- `kafka_topic` - Input topic (default: ai_out)
- `consumer_group` - Consumer group (default: tts-consumer-group)
- `tts.model_name` - Coqui TTS model (default: tts_models/en/vctk/vits)
- `tts.speaker` - Voice selection (default: p272)

**Features**:
- Hot-reload on `settings.json` changes
- Multi-speaker TTS support
- Timestamps audio files

### Cortex (AI Processing)

**Purpose**: Bridges Kafka prompts to Exo AI inference service

**Key Files**:
- `consumer.py` - Consumes prompts, calls Exo API, produces responses
- `producer.py` - Interactive REPL for sending prompts

**Configuration** (in `settings.json` `cortex` and `exo` sections):
- `cortex.input_topic` - Input topic (default: prompts)
- `cortex.output_topic` - Output topic (default: ai_out)
- `cortex.consumer_group` - Consumer group (default: exo-consumer-group)
- `exo.base_url` - Exo API endpoint (default: http://localhost:52415/v1)
- `exo.model_name` - AI model name (auto-detect if null)
- `exo.temperature` - Response randomness (0.0-2.0)
- `exo.max_tokens` - Maximum response length

**Features**:
- OpenAI-compatible API client
- Dynamic configuration reloading
- Structured logging with rotation
- Error messages sent to output topic

### Hippocampus (Knowledge Base)

**Purpose**: Document ingestion, processing, and semantic search

**Key Files**:
- `ingest.py` - Main ingestion pipeline (monitors markdown files)
- `pdf_tracker.py` - PDF monitoring and conversion to markdown
- `query.py` - Hybrid search REPL (exact + semantic)
- `markdown_chunker.py` - Semantic chunking engine
- `file_tracker.py` - SQLite-based file change tracking
- `clean_md.py` - Markdown cleaning utilities
- `clear_db.py` - Database management utility
- `ebook_tracker.py` - EPUB/MOBI ebook monitoring and conversion
- `txt_tracker.py` - TXT file monitoring and conversion to markdown
- `office_tracker.py` - Office document (DOC/DOCX/ODT) monitoring and conversion

**Configuration** (in `settings.json` `hippocampus` section):
- `input_dir` - Directory to monitor (default: ~/Documents/AI_IN)
- `chromadb.host` - ChromaDB server (default: localhost)
- `chromadb.port` - ChromaDB port (default: 8000)
- `chromadb.collection` - Collection name (default: documents)
- `embedding_model` - sentence-transformers model (default: nomic-ai/nomic-embed-text-v1.5)
- `chunking.size` - Target chunk size in characters (default: 4000)
- `chunking.overlap` - Overlap between chunks (default: 400)
- `poll_interval` - Seconds between directory scans (default: 5)

**Features**:
- Continuous file monitoring
- Semantic chunking by markdown structure
- Centralized embeddings via tool server (model loaded once, shared across components)
- Content-hash based change detection
- PDF to markdown conversion with Ollama-based filtering
- EPUB/MOBI ebook to markdown conversion
- TXT to markdown conversion with structure detection
- Office document (DOC/DOCX/ODT) to markdown conversion via pandoc
- Hybrid search (exact text + semantic embeddings)
- Comprehensive test suite with pytest

**Data Storage**:
- `./data/tracking.db` - SQLite database tracking processed files
- `./data/pdf_tracking.db` - SQLite database tracking processed PDFs
- `./data/ebook_tracking.db` - SQLite database tracking processed ebooks
- `./data/txt_tracking.db` - SQLite database tracking processed text files
- `./data/office_tracking.db` - SQLite database tracking processed Office documents
- `./data/` - ChromaDB persistent storage (mounted in container)

### Temporal (Voice Input)

**Purpose**: Voice-activated speech-to-text with wake word detection

**Key Files**:
- `consumer.py` - Main voice listener daemon
- `audio_capture.py` - Microphone audio capture using sounddevice
- `stt_engine.py` - Faster-Whisper speech-to-text engine
- `vad.py` - Silero VAD for voice activity detection

**Configuration** (in `settings.json` `temporal` section):
- `output_topic` - Output Kafka topic (default: ai_in)
- `feedback_topic` - Feedback topic for Broca (default: ai_out)
- `feedback_message` - Message sent on wake word (default: "Yes?")
- `audio.sample_rate` - Audio sample rate (default: 16000)
- `audio.device` - Audio input device (default: null for system default)
- `wake_word.phrase` - Wake phrase (default: "hey olorin")
- `wake_word.buffer_seconds` - Audio buffer for detection (default: 3.0)
- `stt.model` - Whisper model size (default: small)
- `stt.device` - Compute device (default: cpu)
- `stt.language` - Language code (default: en)
- `silence.timeout_seconds` - Silence timeout (default: 3.0)
- `silence.stop_phrases` - Phrases that end recording (default: ["that's all", ...])
- `behavior.pause_during_tts` - Pause listening during Broca playback (default: true)

**Features**:
- Wake word detection using Whisper streaming
- Voice activity detection with Silero VAD
- Configurable silence timeout
- Stop phrase detection ("that's all")
- Integration with Broca for feedback ("Yes?")
- Automatic pause during TTS playback to avoid echo

**Data Storage**:
- `./temporal/data/models/` - Downloaded Whisper models

## Architecture Patterns

### Message Flow

1. **Voice Pipeline**: Mic → Temporal (wake word) → ai_in topic → Enrichener (RAG) → prompts topic → Cortex → ai_out topic → Broca → Audio
2. **Text Prompt Pipeline**: User → prompts topic → Cortex → Exo API → ai_out topic → Broca → Audio
3. **Document Pipeline**: File → ~/Documents/AI_IN → Hippocampus → ChromaDB

### Configuration Management

**Unified Configuration**: All configuration is stored in `settings.json` at the project root. A shared configuration library (`libs/config.py` for Python, `libs/config-rs` for Rust) provides type-safe access with hot-reload support. Falls back to `.env` if `settings.json` doesn't exist.

**Configuration Format** (`settings.json`):
```json
{
  "global": { "log_level": "INFO" },
  "kafka": { "bootstrap_servers": "localhost:9092" },
  "exo": { "base_url": "http://localhost:52415/v1", "temperature": 0.7 },
  "broca": { "kafka_topic": "ai_out", "tts": { "model_name": "...", "speaker": "p272" } },
  "cortex": { "input_topic": "prompts", "output_topic": "ai_out" },
  "hippocampus": { "input_dir": "~/Documents/AI_IN", "chromadb": { "host": "localhost", "port": 8000 } },
  "enrichener": { "input_topic": "ai_in", "output_topic": "prompts" },
  "chat": { "history_enabled": true, "reset_patterns": ["/reset", "start over"] }
}
```

**Python Usage**:
```python
from libs.config import Config

config = Config(watch=True)  # Enable hot-reload
host = config.get('CHROMADB_HOST', 'localhost')
port = config.get_int('CHROMADB_PORT', 8000)
enabled = config.get_bool('REPROCESS_ON_CHANGE', True)
path = config.get_path('INPUT_DIR', '~/Documents/AI_IN')
patterns = config.get_list('CHAT_RESET_PATTERNS', [])
```

**Rust Usage**:
```rust
use olorin_config::Config;

let config = Config::new(None, true)?;  // Enable hot-reload
let host = config.get("CHROMADB_HOST", Some("localhost"));
let port = config.get_int("CHROMADB_PORT", Some(8000));
let enabled = config.get_bool("REPROCESS_ON_CHANGE", true);
let path = config.get_path("INPUT_DIR", Some("~/Documents/AI_IN"));
let patterns = config.get_list("CHAT_RESET_PATTERNS", None);
```

**Config Library Features**:
- `config.get(key, default)` - Get string value
- `config.get_int(key, default)` - Get integer value
- `config.get_float(key, default)` - Get float value
- `config.get_bool(key, default)` - Get boolean (native JSON bool or string: true/yes/1/on)
- `config.get_path(key, default)` - Get absolute path (~ expanded, relative paths resolved against project root)
- `config.get_list(key, default)` - Get list (native JSON array or comma-separated string)
- `config.set(key, value)` - In-memory override
- `config.reload()` - Hot-reload if file changed (returns True if reloaded)
- `config.project_root` - Project root directory (where settings.json lives)
- `config.config_path` - Path to active config file (settings.json or .env)

**Path Resolution**: All relative paths in settings.json (e.g., `./cortex/data/chat.db`) are resolved against the project root, not the current working directory. This ensures consistent path resolution regardless of which directory a component runs from.

**Backward Compatibility**: Flat keys like `CHROMADB_PORT` are automatically mapped to nested JSON paths like `hippocampus.chromadb.port`. Existing code using flat keys continues to work without changes.

**Hot-Reload**: Consumer components (broca, cortex, enrichener) detect changes to `settings.json` and automatically reload configuration without restarting.

### State Management

**Centralized State**: All runtime state is stored in a SQLite database (`./data/state.db`) with typed columns. The shared state library (`libs/state.py` for Python, `libs/state-rs` for Rust) provides type-safe access for sharing state across all components.

**Schema Design**: Uses discriminated union pattern with separate columns for each data type:
```sql
CREATE TABLE state (
    key TEXT PRIMARY KEY,
    value_type TEXT NOT NULL,  -- 'null', 'int', 'float', 'string', 'bool', 'json', 'bytes'
    value_int INTEGER,
    value_float REAL,
    value_string TEXT,
    value_bool INTEGER,
    value_json TEXT,
    value_bytes BLOB,
    created_at TEXT,
    updated_at TEXT
);
```

**Python Usage**:
```python
from libs.state import State, get_state

state = State()  # Or use get_state() singleton

# Set values (type auto-detected)
state.set("broca.audio_pid", 12345)
state.set("broca.is_playing", True)
state.set("cortex.status", "running")
state.set("system.info", {"version": "1.0", "components": ["broca", "cortex"]})

# Get values with type safety
pid = state.get_int("broca.audio_pid")  # -> Optional[int]
playing = state.get_bool("broca.is_playing")  # -> bool
info = state.get_json("system.info")  # -> Optional[dict]

# With defaults
pid = state.get_int("broca.audio_pid", default=0)

# Delete values
state.delete("broca.audio_pid")
state.delete_prefix("broca.")  # Delete all broca.* keys

# List keys
all_keys = state.keys()
broca_keys = state.keys(prefix="broca.")
```

**Rust Usage**:
```rust
use olorin_state::{State, get_state};

let state = State::new(None)?;  // Or use get_state(None) singleton

// Set values with explicit types
state.set_int("broca.audio_pid", 12345)?;
state.set_bool("broca.is_playing", true)?;
state.set_string("cortex.status", "running")?;
state.set_json("system.info", &serde_json::json!({"version": "1.0"}))?;

// Get values with type safety
let pid = state.get_int("broca.audio_pid")?;  // -> Option<i64>
let playing = state.get_bool("broca.is_playing")?;  // -> bool
let info = state.get_json("system.info")?;  // -> Option<serde_json::Value>

// With defaults
let pid = state.get_int_or("broca.audio_pid", 0)?;

// Delete values
state.delete("broca.audio_pid")?;
state.delete_prefix("broca.")?;  // Delete all broca.* keys

// List keys
let all_keys = state.keys(None)?;
let broca_keys = state.keys(Some("broca."))?;
```

**State Library Features**:
- `state.set(key, value)` - Set with auto-detected type
- `state.set_int(key, value)` / `set_float()` / `set_string()` / `set_bool()` / `set_json()` / `set_bytes()` - Explicit type setters
- `state.get(key, default)` - Get with auto-detected type
- `state.get_int(key, default)` / `get_float()` / `get_string()` / `get_bool()` / `get_json()` / `get_bytes()` - Type-safe getters
- `state.get_type(key)` - Get the ValueType enum for a key
- `state.exists(key)` - Check if key exists
- `state.delete(key)` - Delete a single key
- `state.delete_prefix(prefix)` - Delete all keys with prefix
- `state.keys(prefix)` - List keys with optional prefix filter
- `state.items(prefix)` - Get key-value pairs with optional prefix filter
- `state.get_metadata(key)` - Get type and timestamps
- `state.clear()` - Delete all state entries
- `state.db_path` - Path to the state database

**Common State Keys** (conventions for component state):
- `broca.audio_pid` - PID of currently playing afplay process (int or null)
- `broca.is_playing` - Whether Broca is currently playing audio (bool)
- `cortex.status` - Current status of Cortex consumer (string)
- `hippocampus.last_poll` - Timestamp of last directory poll (string)
- `system.pids.<component>` - Process ID for each component (int)

**Thread Safety**: The State class uses thread-local SQLite connections with WAL mode for safe concurrent access from multiple threads and processes.

### Slash Command System vs AI Tool Use

The project has two distinct systems for executing functions - these are **NOT** connected:

#### Slash Command System (User-Invoked)

For commands triggered by the user via the chat client (e.g., `/stop`, `/write`, `/clear`).

**Components**:
- `libs/control_handlers/` - Pluggable handler modules (Python)
- `libs/control_server.py` - HTTP API server exposing handlers
- `cortex/controller.py` - Runs the control server process

**How it works**:
1. User types `/command` in chat client
2. Chat client calls control server HTTP API (`POST /execute`)
3. Server routes to appropriate handler module
4. Handler executes and returns result

**Existing handlers**: `stop-audio`, `resume-audio`, `audio-status`, `write`, `clear`

**Configuration** (in `settings.json` `control` section):
```json
{
  "control": {
    "api": {
      "enabled": true,
      "port": 8765,
      "host": "0.0.0.0"
    }
  }
}
```

#### AI Tool Use (Model-Invoked)

For functions the AI model can call during inference (OpenAI function calling). Tools are standalone HTTP servers that can be written in any language.

**Architecture**:
```
[Cortex Consumer]
     |
     | 1. On startup: GET /describe from each enabled tool
     | 2. Convert to OpenAI format, pass in API requests
     | 3. When AI returns tool_calls, POST /call to tool
     | 4. Store in chat history, send result back to AI
     |
[Tool Servers] (persistent HTTP servers)
  - GET  /health   → {"status": "ok"}
  - GET  /describe → tool metadata
  - POST /call     → execute and return result
```

**Components**:
- `tools/<name>/` - Tool server implementations (Rust or Python)
- `libs/tool_client.py` - Client for tool server communication
- `cortex/consumer.py` - Integrates tools into AI inference loop

**Tool Protocol**:

`GET /health`:
```json
{"status": "ok"}
```

`GET /describe`:
```json
{
  "name": "write",
  "description": "Write content to a file in ~/Documents/AI_OUT",
  "parameters": [
    {"name": "content", "type": "string", "required": true, "description": "..."},
    {"name": "filename", "type": "string", "required": true, "description": "..."}
  ]
}
```

`POST /call`:
```json
// Request
{"content": "Hello", "filename": "test.txt"}

// Response (success)
{"success": true, "result": "Wrote 5 bytes to ~/Documents/AI_OUT/test.txt"}

// Response (error)
{"success": false, "error": {"type": "IOError", "message": "Permission denied"}}
```

**Configuration** (in `settings.json` `tools` section):
```json
{
  "tools": {
    "write": {
      "enabled": true,
      "port": 8770
    },
    "embeddings": {
      "enabled": true,
      "port": 8771
    }
  }
}
```

**Existing Tools**:
- `write` (Rust, port 8770) - Write files to ~/Documents/AI_OUT
- `embeddings` (Python, port 8771) - Generate text embeddings for documents/queries

**Creating New Tools**:
1. Create `tools/<name>/` directory
2. Implement HTTP server with `/health`, `/describe`, `/call` endpoints
3. Add tool to `settings.json` under `tools` section
4. Tool will be auto-discovered and started by `./up`

**State keys** (for tool support detection):
- `cortex.tools_supported` - Whether current model supports tools (bool)
- `cortex.tools_model` - Model name when tool support was checked (string)
- `cortex.tools_checked_at` - Timestamp of last check (ISO string)
- `tools.<name>.status` - Tool server status (string)
- `tools.<name>.port` - Tool server port (int)

### Process Management

Background processes are managed via the orchestration scripts:
- PIDs stored in `.pids/` directory
- Logs stored in `logs/` directory
- Virtual environments per component
- Graceful shutdown with SIGTERM (10-second timeout before SIGKILL)

### Git Structure

Each component (broca, cortex, hippocampus) is its own git repository. The root directory contains orchestration scripts but is not itself a git repository.

## Development Notes

### Adding New Components

When adding a new component:
1. Create directory with its own `venv/`
2. Add configuration section to root `settings.json` file
3. Add key mappings to `libs/config.py` and `libs/config-rs/src/lib.rs` for backward compatibility
4. Import `from libs.config import Config` to access configuration
5. Create `requirements.txt` for dependencies
6. Update `./up` to start the daemon
7. Update `./down` to stop the daemon
8. Update `./status` to monitor the component
9. Run `pre-commit run --all-files` to lint and format all files created or edited, and fix lints that show up.

### Kafka Topic Management

Topics are automatically created by consumers on first connection. For manual management:
```bash
# In broca/ or cortex/
./create-topic <topic-name>
./list-topics
```

### Testing Hippocampus

The project uses pytest with fixtures. Test structure:
- `tests/conftest.py` - Shared fixtures
- `tests/test_*.py` - Test modules
- Run with `pytest --cov` for coverage reports

### Debugging

1. Check logs in `logs/` directory
2. Use `./status` to see process health and recent errors
3. Test components individually before running full system
4. Set `"log_level": "DEBUG"` in `settings.json` `global` section for verbose logging (applies to all components)

### Common Issues

**Kafka connection errors**: Ensure Kafka container is running (`podman ps`)
**ChromaDB connection errors**: Check ChromaDB container is running on port 8000
**Exo connection errors**: Start Exo before running Cortex consumer
**Timestamp errors in Kafka**: The producers use broker-assigned timestamps to avoid clock skew

### RAG Context Injection

**IMPORTANT**: When injecting RAG context into LLM prompts, the message format matters significantly. Different models handle context differently.

**Correct format** (works reliably across all tested models and backends):
```
user: [context + question combined in single message]
```

Example:
```
Use the following reference context to answer my question.

<context>
[document chunks here]
</context>

Question: What year was Caesar born?
```

**Broken format** (causes context loss with Deepseek R1 and similar models):
```
user: [context]
assistant: "I understand, I'll use this context..."
user: [question]
```

**Why**: Some models (notably Deepseek R1) lose awareness of earlier messages when an assistant message appears between context and question. The assistant "acknowledgment" message causes the model to effectively "forget" the context, responding with "no context was provided" even when 200K+ characters of context exist in the conversation. Even splitting into two consecutive user messages can cause issues with some models.

**Implementation**: Context injection is handled in `cortex/consumer.py` via `_format_context_with_prompt()`. The function combines context and prompt into a single user message with context wrapped in `<context>` tags. This single-message format works reliably across Ollama, EXO, and various model architectures.

**Model capabilities caching**: The cortex consumer caches model capabilities (context length, sliding window) in memory. When switching models, ensure the cache is invalidated by checking if `model_id` matches the current model. Stale cached data from a previous model (e.g., gemma3's 1024-token sliding window) can cause incorrect warnings for the new model.

**Sliding window attention**: Some models (e.g., Gemma, some Mistral variants) use sliding window attention which limits effective context to only the last N tokens during generation. Query model capabilities via Ollama's `/api/show` endpoint to detect this. Models with sliding window are unsuitable for large RAG contexts - the context at the beginning of the conversation becomes invisible during generation.

## Project Naming

Components are named after brain regions:
- **Temporal lobe**: Auditory processing and speech comprehension (STT)
- **Cortex**: Higher-level processing (AI inference)
- **Broca's area**: Speech production (TTS)
- **Hippocampus**: Memory formation and retrieval (document storage)
