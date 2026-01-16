# Olorin Project

Distributed AI pipeline: Kafka-connected microservices for AI inference (Cortex), text-to-speech (Broca), document retrieval (Hippocampus), and context enrichment (Enrichener).

## Quick Reference

| Component | Purpose | Input | Output | Port |
|-----------|---------|-------|--------|------|
| Cortex | AI inference via Exo | `prompts` topic | `ai_out` topic | - |
| Broca | TTS via Coqui | `ai_out` topic | Audio playback | - |
| Hippocampus | Document ingestion | `~/Documents/AI_IN` | ChromaDB | - |
| Enrichener | Context retrieval | `ai_in` topic | `prompts` topic | - |
| Control API | Slash commands | HTTP | - | 8765 |
| Kafka | Message broker | - | - | 9092 |
| ChromaDB | Vector database | - | - | 8000 |
| Exo | LLM inference | - | - | 52415 |

### Tools (AI Function Calling)

| Tool | Purpose | Port |
|------|---------|------|
| write | Write files to ~/Documents/AI_OUT | 8770 |
| embeddings | Text embeddings (nomic-embed-text-v1.5) | 8771 |
| chat | TUI chat client | - |
| olorin-inspector | TUI database inspector | - |

## Architecture

```
                              ┌─────────────────────────────────────┐
                              │           AI Tool Servers           │
                              │  write (8770) │ embeddings (8771)   │
                              └───────────────┬─────────────────────┘
                                              │ function calls
                                              ▼
User Input → [ai_in] → Enrichener → [prompts] → Cortex → Exo API → [ai_out] → Broca → Audio
                           │                       │
                           ▼                       ▼
                       ChromaDB              Control API (8765)
                           ▲                  /stop, /clear, etc.
                           │
              ~/Documents/AI_IN → Hippocampus
                                 (pdf, ebook, office, txt trackers)
```

## Operations

```bash
./up       # Start all (Kafka, ChromaDB, tools, consumers)
./down     # Stop all
./status   # Health check with log analysis
```

### Per-Component

```bash
# Cortex (AI)
cd cortex && source venv/bin/activate
python3 consumer.py    # Run consumer
python3 producer.py    # Send test prompts

# Broca (TTS)
cd broca && source venv/bin/activate
python3 consumer.py    # Run consumer
python3 producer.py    # Send test text

# Hippocampus (Documents)
cd hippocampus && source venv/bin/activate
python3 ingest.py      # Run ingestion
python3 query.py       # Search REPL
pytest --cov           # Run tests

# Enrichener (Context)
cd hippocampus && source venv/bin/activate
python3 enrichener.py  # Run enricher
```

## File Structure

```
olorin-project/
├── up, down, status       # Orchestration scripts
├── settings.json          # Unified configuration
├── libs/                  # Shared libraries
│   ├── config.py          # Configuration (hot-reload)
│   ├── state.py           # State management (SQLite)
│   ├── chat_store.py      # Conversation history
│   ├── context_store.py   # Retrieved context
│   ├── embeddings.py      # Embeddings client
│   ├── tool_client.py     # AI tool integration
│   ├── control_server.py  # Slash command API
│   └── control_handlers/  # Command handlers
│       ├── stop_audio.py
│       ├── resume_audio.py
│       ├── audio_status.py
│       ├── clear.py
│       ├── write.py
│       └── auto_context.py
├── libs/config-rs/        # Rust config library
├── libs/state-rs/         # Rust state library
├── cortex/                # AI consumer
│   ├── consumer.py        # Kafka→Exo→Kafka + tool use
│   ├── producer.py        # Test REPL
│   └── controller.py      # Control server runner
├── broca/                 # TTS consumer
│   ├── consumer.py        # Kafka→Coqui→Audio
│   └── producer.py        # Test REPL
├── hippocampus/           # Document system
│   ├── ingest.py          # Markdown→ChromaDB
│   ├── enrichener.py      # Context retrieval consumer
│   ├── pdf_tracker.py     # PDF→Markdown
│   ├── ebook_tracker.py   # EPUB/MOBI→Markdown
│   ├── txt_tracker.py     # TXT→Markdown
│   ├── office_tracker.py  # DOC/DOCX/ODT→Markdown
│   ├── query.py           # Hybrid search REPL
│   └── markdown_chunker.py
├── tools/                 # AI tool servers
│   ├── write/             # Rust: file writer
│   ├── embeddings/        # Python: text embeddings
│   ├── chat/              # Rust: TUI chat client
│   └── olorin-inspector/  # Rust: database inspector
├── data/                  # Runtime data
│   └── state.db           # Centralized state
├── .pids/                 # Runtime PID files
└── logs/                  # Runtime logs
```

## Configuration

All components use `settings.json` with hot-reload support. Key settings:

| Section | Variable | Default |
|---------|----------|---------|
| exo | model_name | auto-detect |
| exo | temperature | 0.7 |
| exo | base_url | http://localhost:52415/v1 |
| broca.tts | model_name | tts_models/en/vctk/vits |
| broca.tts | speaker | p272 |
| hippocampus | input_dir | ~/Documents/AI_IN |
| tools.embeddings | model | nomic-ai/nomic-embed-text-v1.5 |
| control.api | port | 8765 |

Set `global.log_level` to `DEBUG` for verbose output.

## State Management

Runtime state is stored in `data/state.db` (SQLite). Components share state via typed key-value storage:

```python
from libs.state import State
state = State()
state.set("broca.is_playing", True)
playing = state.get_bool("broca.is_playing")
```

Common keys: `broca.audio_pid`, `broca.is_playing`, `cortex.tools_supported`

## Slash Commands

User-invoked commands via Control API (port 8765):

| Command | Description |
|---------|-------------|
| /stop | Stop audio playback |
| /resume | Resume audio playback |
| /clear | Clear conversation history |
| /write | Write content to file |
| /auto-context | Toggle context retrieval |

## AI Tool Use

The AI model can call tools during inference. Tools are HTTP servers with `/health`, `/describe`, `/call` endpoints.

**write**: Writes files to `~/Documents/AI_OUT`
**embeddings**: Generates text embeddings (centralized, shared across components)

## Data Storage

| Path | Purpose |
|------|---------|
| `data/state.db` | Centralized runtime state |
| `cortex/data/chat.db` | Conversation history |
| `hippocampus/data/tracking.db` | File tracking |
| `hippocampus/data/context.db` | Retrieved context |
| `hippocampus/data/pdf_tracking.db` | PDF processing state |
| `hippocampus/data/ebook_tracking.db` | Ebook processing state |
| `hippocampus/data/txt_tracking.db` | Text file processing |
| `hippocampus/data/office_tracking.db` | Office document processing |
| `broca/output/` | Generated audio files |
| `.pids/` | Process ID files |
| `logs/` | Application logs |

## Dependencies

**External services (must be running):**
- Exo server on port 52415 (before Cortex)
- Kafka container via Podman (started by `./up`)
- ChromaDB container via Podman (started by `./up`)

**Key packages:**
- cortex: kafka-python, openai, python-dotenv
- broca: kafka-python, TTS, python-dotenv
- hippocampus: chromadb, sentence-transformers, PyMuPDF, ebooklib, python-docx
- tools/write: Rust + Axum
- tools/chat: Rust + Ratatui + tiktoken-rs

## Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| Kafka connection error | `podman ps` | `./up` or `podman restart kafkaserver` |
| ChromaDB connection error | `podman ps` | `./up` or `podman restart chromadb` |
| Exo connection error | Exo running? | Start Exo before Cortex |
| Tool not responding | `curl localhost:8770/health` | Check tool logs |

## Logs

```bash
tail -f logs/cortex-consumer.log
tail -f logs/broca-consumer.log
tail -f logs/hippocampus-ingest.log
tail -f logs/enrichener.log
```
