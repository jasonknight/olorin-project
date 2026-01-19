# Olorin

**Your AI-powered Second Brain with a Voice**

Olorin is a personal AI assistant designed to be your Chief of Staff—always listening, always ready to help, with instant access to your documents and knowledge. Talk to it naturally, and it talks back.

## What Olorin Does

- **Voice In, Voice Out** — Say "Hey Olorin" and speak naturally. Get spoken responses back.
- **Knows Your Documents** — Drop files into a folder. Olorin reads, indexes, and recalls them when relevant.
- **Multiple AI Backends** — Use Claude, Ollama (local), or distributed inference. Your choice.
- **Takes Action** — The AI can write files, search your knowledge base, and execute tasks autonomously.
- **Remembers Everything** — Full conversation history with context tracking across sessions.

## Quick Start

```bash
# Check dependencies
python3 setup.py

# Start all services
./up

# Launch the chat interface
cd tools/chat && cargo run --release

# Or just talk — say "Hey Olorin"
```

## The Experience

```
You: "Hey Olorin"
Olorin: "Yes?"

You: "What were the key points from that product roadmap I saved last week?"
Olorin: [retrieves document, synthesizes answer, speaks response]

You: "Write up a summary and save it for me"
Olorin: [creates markdown file in ~/Documents/AI_OUT]
```

### Voice Interface

The voice pipeline listens continuously for the wake phrase. After activation:
- Speak naturally until you pause (~3 seconds) or say "that's all"
- Olorin retrieves relevant context from your knowledge base automatically
- Response plays back through your speakers
- Listening pauses during playback to prevent echo

### Chat Interface

A terminal-based interface (`tools/chat`) provides:
- Full conversation view with streaming responses
- Document search with add-to-context capability
- System state monitoring (audio status, model info)
- Slash commands for control

### Document Ingestion

Drop files into `~/Documents/AI_IN`. Olorin handles:
- **Markdown** — Native support
- **PDF** — Extracts text, converts to searchable chunks
- **Ebooks** — EPUB, MOBI
- **Office** — Word docs, ODT, plain text

Documents are chunked semantically and stored in a vector database for instant retrieval.

## Slash Commands

| Command | What it does |
|---------|--------------|
| `/stop` | Mute audio playback |
| `/resume` | Resume audio |
| `/clear` | Wipe conversation history |
| `/write [filename]` | Save last response to file |
| `/auto-context on\|off` | Toggle automatic document retrieval |

## Configuration

All settings live in `settings.json`. Key options:

| Setting | Purpose |
|---------|---------|
| `inference.backend` | Choose: `ollama`, `anthropic`, or `exo` |
| `ollama.model_name` | Local model (e.g., `deepseek-r1:70b`) |
| `anthropic.model_name` | Claude model (e.g., `claude-sonnet-4-20250514`) |
| `temporal.wake_word.phrase` | Wake phrase (default: "hey olorin") |
| `broca.tts.speaker` | Voice selection |
| `hippocampus.input_dir` | Document watch folder |

For Anthropic, add your API key to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture

Olorin is built as independent components that communicate asynchronously:

```
Voice ──► Temporal (STT) ──► Enrichener (RAG) ──► Cortex (AI) ──► Broca (TTS) ──► Audio
                                   │                  │
                                   ▼                  ▼
                              ChromaDB            AI Tools
                                   ▲              (write, search)
                                   │
              ~/Documents/AI_IN ──► Hippocampus (ingestion)
```

| Component | Role |
|-----------|------|
| **Temporal** | Listens for wake word, transcribes speech |
| **Enrichener** | Decides if context is needed, retrieves relevant documents |
| **Cortex** | Sends prompts to AI, handles tool calls, manages conversation |
| **Broca** | Converts responses to speech, plays audio |
| **Hippocampus** | Monitors folders, processes documents, maintains vector store |

### AI Tools

The AI can call tools during inference:

| Tool | Capability |
|------|------------|
| **write** | Save content to `~/Documents/AI_OUT` |
| **search** | Query the knowledge base |
| **embeddings** | Generate text embeddings |

## Operations

```bash
./up          # Start everything
./down        # Stop everything
./status      # Health check with log analysis
```

### Logs

```bash
tail -f logs/cortex-consumer.log      # AI processing
tail -f logs/broca-consumer.log       # Text-to-speech
tail -f logs/enrichener.log           # Context retrieval
tail -f logs/hippocampus-ingest.log   # Document ingestion
```

## File Structure

```
olorin-project/
├── up, down, status       # Control scripts
├── setup.py               # Dependency checker
├── settings.json          # Configuration
├── .env                   # API keys (not committed)
│
├── temporal/              # Voice input (STT)
├── cortex/                # AI inference + tools
├── broca/                 # Voice output (TTS)
├── hippocampus/           # Document processing
│   └── enrichener.py      # Context retrieval
│
├── tools/
│   ├── chat/              # Terminal chat interface
│   ├── write/             # File writing tool
│   ├── search/            # Knowledge base search
│   └── olorin-inspector/  # Database viewer
│
├── libs/                  # Shared libraries
│   ├── config.py          # Configuration management
│   ├── state.py           # Runtime state
│   ├── chat_store.py      # Conversation history
│   └── context_store.py   # Retrieved documents
│
├── data/                  # Runtime databases
└── logs/                  # Application logs
```

## Data Storage

| Location | Contents |
|----------|----------|
| `data/state.db` | Runtime state (audio status, model info) |
| `cortex/data/chat.db` | Conversation history |
| `hippocampus/data/*.db` | Document tracking |
| `~/Documents/AI_IN` | Your input documents |
| `~/Documents/AI_OUT` | AI-generated files |

## Requirements

**System:**
- Python 3.11+
- Rust (for tools)
- Podman (for services)

**AI Backend (at least one):**
- Ollama (local) — `brew install ollama`
- Anthropic API key
- Exo (distributed)

**Optional:**
- Pandoc (Office document conversion)
- FFmpeg (audio processing)

Run `python3 setup.py` to check all dependencies.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No response after wake word | Check `logs/temporal-consumer.log` for STT errors |
| Context not being retrieved | Verify documents are in `~/Documents/AI_IN` and indexed |
| Audio not playing | Run `/audio-status`, check Broca logs |
| AI not responding | Verify backend is running (Ollama/Exo/API key) |

## Why "Olorin"?

Olorin is Gandalf's original name in Valinor—a Maia spirit who walked among the peoples of Middle-earth, offering wisdom and guidance. Like its namesake, this assistant is meant to be a wise companion: always present, deeply knowledgeable, and ready to help when called upon.

---

*See `CLAUDE.md` for detailed technical documentation.*
