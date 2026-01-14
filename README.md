# Olorin Project

Distributed AI pipeline: Kafka-connected microservices for AI inference (Cortex), text-to-speech (Broca), and document retrieval (Hippocampus).

## Quick Reference

| Component | Purpose | Input | Output | Port |
|-----------|---------|-------|--------|------|
| Cortex | AI inference via Exo | `prompts` topic | `ai_out` topic | - |
| Broca | TTS via Coqui | `ai_out` topic | Audio playback | - |
| Hippocampus | Document ingestion | `~/Documents/AI_IN` | ChromaDB | - |
| Kafka | Message broker | - | - | 9092 |
| ChromaDB | Vector database | - | - | 8000 |
| Exo | LLM inference server | - | - | 52415 |

## Architecture

```
User Input → [prompts] → Cortex → Exo API → [ai_out] → Broca → Audio
                                     ↑
              ~/Documents/AI_IN → Hippocampus → ChromaDB (semantic search)
```

## Operations

```bash
./up       # Start all (Kafka, ChromaDB, consumers)
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
```

## File Structure

```
olorin-project/
├── up, down, status   # Orchestration scripts
├── libs/              # Shared code
│   └── olorin_logging.py
├── cortex/            # AI consumer (git repo)
│   ├── consumer.py    # Kafka→Exo→Kafka
│   ├── producer.py    # Test REPL
│   └── .env           # Config
├── broca/             # TTS consumer (git repo)
│   ├── consumer.py    # Kafka→Coqui→Audio
│   ├── producer.py    # Test REPL
│   └── .env           # Config
├── hippocampus/       # Document system (git repo)
│   ├── ingest.py      # File→ChromaDB pipeline
│   ├── pdf_tracker.py # PDF→Markdown converter
│   ├── query.py       # Hybrid search REPL
│   ├── markdown_chunker.py
│   ├── file_tracker.py
│   └── .env           # Config
├── .pids/             # Runtime PID files
└── logs/              # Runtime logs
```

## Configuration

All components hot-reload `.env` files. Key settings:

| Component | Variable | Default |
|-----------|----------|---------|
| Cortex | `MODEL_NAME` | auto-detect from Exo |
| Cortex | `TEMPERATURE` | 0.7 |
| Cortex | `EXO_BASE_URL` | http://localhost:52415/v1 |
| Broca | `TTS_MODEL_NAME` | tts_models/en/vctk/vits |
| Broca | `TTS_SPEAKER` | p272 |
| Hippocampus | `INPUT_DIR` | ~/Documents/AI_IN |
| Hippocampus | `EMBEDDING_MODEL` | all-MiniLM-L6-v2 |

## Dependencies

**External services (must be running):**
- Exo server on port 52415 (before Cortex)
- Kafka container via Podman (started by `./up`)
- ChromaDB container via Podman (started by `./up`)

**Python packages per component:**
- cortex: kafka-python, openai, python-dotenv, requests
- broca: kafka-python, TTS, python-dotenv
- hippocampus: chromadb, sentence-transformers, kafka-python, langchain

## Known Issues / Technical Debt

| Issue | Location | Severity |
|-------|----------|----------|
| `WORD_THRESHOLD` undefined | cortex/consumer.py:653 | Bug - breaks log message |
| `logging` not imported | hippocampus/ingest.py:417 | Bug - crashes on fatal error |
| PyMuPDF auto-installed at runtime | hippocampus/pdf_tracker.py:27 | Should be in requirements.txt |
| libs/ not a proper package | libs/ | Missing `__init__.py` |
| Fragmented git repos | broca/, cortex/, hippocampus/ | Each is separate repo, root is not |

## Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| Kafka connection error | `podman ps` | `./up` or `podman restart kafkaserver` |
| ChromaDB connection error | `podman ps` | `./up` or `podman restart chromadb` |
| Exo connection error | Exo running? | Start Exo before Cortex |
| Consumer rebalancing | Long inference | Increase `max_poll_interval_ms` |
| Timestamp errors | Kafka topic config | Run `broca/fix_topic_timestamps.py` |

## Logs

```bash
tail -f logs/cortex-consumer.log
tail -f logs/broca-consumer.log
tail -f logs/hippocampus-ingest.log
tail -f logs/hippocampus-pdf-tracker.log
```

Set `LOG_LEVEL=DEBUG` in component `.env` for verbose output.

## Data Storage

- `.pids/` - Process ID files (runtime)
- `logs/` - Application logs (runtime)
- `hippocampus/data/tracking.db` - SQLite file tracking
- `hippocampus/data/pdf_tracking.db` - PDF processing state
- ChromaDB data mounted at `hippocampus/data/` in container
