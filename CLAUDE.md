# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Olorin-project is a distributed AI pipeline system composed of three main brain-inspired components communicating via Kafka. The system creates a complete text-to-speech AI pipeline with document retrieval capabilities.

## System Architecture

The project follows a microservices architecture with three independent components:

```
[User Input]
     |
     v
  Cortex (AI Processing)
  - Consumes: prompts (Kafka topic)
  - Produces: ai_out (Kafka topic)
  - Service: Exo (distributed AI inference)
     |
     v
  Broca (Text-to-Speech)
  - Consumes: ai_out (Kafka topic)
  - Service: Coqui TTS
  - Output: Audio playback
     |
     v
Hippocampus (Knowledge Base)
  - Monitors: ~/Documents/AI_IN directory
  - Service: ChromaDB (vector database)
  - Purpose: Document ingestion, semantic search
```

### Infrastructure Components

- **Kafka**: Message broker running on port 9092 (in Podman container)
- **ChromaDB**: Vector database running on port 8000 (in Podman container)
- **Exo**: Distributed AI inference server (OpenAI-compatible API)

## Common Commands

### Linting and Formatting Code

```
# Run this command when you've written new code
pre-commit run --all-files
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
- `embedding_model` - sentence-transformers model (default: all-MiniLM-L6-v2)
- `chunking.size` - Target chunk size in characters (default: 1000)
- `chunking.overlap` - Overlap between chunks (default: 200)
- `poll_interval` - Seconds between directory scans (default: 5)

**Features**:
- Continuous file monitoring
- Semantic chunking by markdown structure
- Local embeddings (no API keys)
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

## Architecture Patterns

### Message Flow

1. **Prompt Pipeline**: User → prompts topic → Cortex → Exo API → ai_out topic → Broca → Audio
2. **Document Pipeline**: File → ~/Documents/AI_IN → Hippocampus → ChromaDB

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

## Project Naming

Components are named after brain regions:
- **Cortex**: Higher-level processing (AI inference)
- **Broca's area**: Speech production (TTS)
- **Hippocampus**: Memory formation and retrieval (document storage)
