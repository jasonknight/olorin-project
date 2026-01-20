```
 ██████╗ ██╗      ██████╗ ██████╗ ██╗███╗   ██╗
██╔═══██╗██║     ██╔═══██╗██╔══██╗██║████╗  ██║
██║   ██║██║     ██║   ██║██████╔╝██║██╔██╗ ██║
██║   ██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║
╚██████╔╝███████╗╚██████╔╝██║  ██║██║██║ ╚████║
 ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
```

# Olorin

Olorin is what you imagined AI would be. Something you talk to, that talks back. 

Olorin is a personal AI assistant designed to be your Chief of Staff. Always listening, always ready to help, with instant access to your documents and knowledge. Talk to it naturally, and it talks back.

While Olorin is always listening, it's processing audio locally, not in the cloud. It also produces audio locally. In fact, you can use only local models if that's what you want.

> **Note: This project is in active development.** Some features described below are partially implemented or experimental. The architecture is stable, but expect rough edges. Contributions and feedback welcome.

## Vision

- **Voice In, Voice Out** — Speak naturally using a wake phrase. Get spoken responses back.
- **Knows Your Documents** — Drop files into a folder. Olorin reads, indexes, and recalls them when relevant.
- **Controls your calendar** - Schedule appointments, track to-dos, create alarms and timers.
- **Multiple Input Channels** - Use Slack, Facetime, Discord, Email to send prompts and get responses.
- **Multiple AI Backends** — Use Claude, Gemini, Deepseek, Ollama (local), or distributed inference. Your choice.
- **Takes Action** — The AI can write files, search your knowledge base, and execute tasks.
- **Remembers Everything** — Full conversation history with context tracking across sessions.

## Why "Olorin"?

Olorin is Gandalf's original name in Valinor, a Maiar spirit who walked among the peoples of Middle Earth, offering wisdom and guidance. Like his namesake, this assistant is meant to be a wise companion: always present, deeply knowledgeable, and ready to help when called upon.

## Quick Start

```bash
# Check dependencies
python3 setup.py

# Start all services
./up

# Launch the chat interface
cd tools/chat && cargo run --release

# Or use voice — say your wake phrase (e.g., "Hey Computer")
```

## The Experience

```
You: "Hey Computer"
Olorin: "Yes?"

You: "What were the key points from that product roadmap I saved last week?"
Olorin: [retrieves document, synthesizes answer, speaks response]

You: "Write up a summary and save it for me"
Olorin: [creates markdown file in ~/Documents/AI_OUT]
```

## Voice Interface

### Wake Word Detection

Olorin uses **Picovoice Porcupine** for always-on wake word detection. This runs locally on your device with minimal CPU usage.

**Setup Required:** You need to create your own wake word model:
1. Go to [Picovoice Console](https://console.picovoice.ai/)
2. Create a free account and generate a custom wake word (e.g., "Hey Computer", "Hey Assistant")
3. Download the `.ppn` model file for macOS
4. Configure the path in `settings.json` under `temporal.porcupine.model_path`

The default configuration expects a wake phrase like "Hey Computer". You can customize this to any phrase Picovoice supports.

### Voice Pipeline

After wake word activation:
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

## Document Ingestion

Drop files into `~/Documents/AI_IN`. Olorin handles:
- **Markdown** — Native support with semantic chunking
- **PDF** — Extracts text, converts to searchable chunks
- **Ebooks** — EPUB, MOBI
- **Office** — Word docs, ODT, plain text

Documents are chunked semantically and stored in a vector database for instant retrieval.

## Advanced Features

### Recursive Language Model (RLM)

For models with strong code generation capabilities, Olorin implements the **Recursive Language Model** pattern (based on [arxiv:2512.24601](https://arxiv.org/abs/2512.24601)). This allows handling contexts that exceed the model's native context window by:

1. Treating long documents as environment objects rather than stuffing them into context
2. The model writes Python code to probe, filter, and partition the input
3. Sub-LLM calls handle smaller chunks, with results stored in variables
4. Iteration continues until a final answer is synthesized

This enables answering questions over documents that would be impossible to fit in a single context window. RLM is automatically enabled when:
- The model supports code generation (detected automatically)
- The context exceeds 50% of the model's context window
- RLM is enabled in configuration (`rlm.enabled: true`)

**Note:** RLM requires models with strong coding capabilities (e.g., DeepSeek-Coder, CodeLlama, Qwen-Coder). It will not activate for chat-only models.

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
| `temporal.porcupine.model_path` | Path to your Picovoice wake word model |
| `temporal.porcupine.access_key` | Your Picovoice access key |
| `broca.tts.speaker` | Voice selection |
| `hippocampus.input_dir` | Document watch folder |
| `rlm.enabled` | Enable Recursive Language Model for large contexts |

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
| **Temporal** | Listens for wake word (Picovoice), transcribes speech (Whisper) |
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
├── temporal/              # Voice input (STT + wake word)
├── cortex/                # AI inference + tools + RLM
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
│   ├── rlm_executor.py    # Recursive Language Model
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

**Voice (optional but recommended):**
- Picovoice account and access key (free tier available)
- Custom wake word model (.ppn file)

**Optional:**
- Pandoc (Office document conversion)
- FFmpeg (audio processing)

Run `python3 setup.py` to check all dependencies.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No response after wake word | Check `logs/temporal-consumer.log`, verify Picovoice model path |
| Wake word not detected | Ensure `.ppn` file exists and access key is valid |
| Context not being retrieved | Verify documents are in `~/Documents/AI_IN` and indexed |
| Audio not playing | Run `/audio-status`, check Broca logs |
| AI not responding | Verify backend is running (Ollama/Exo/API key) |
| RLM not activating | Check model supports code generation, verify `rlm.enabled: true` |



---

*See `CLAUDE.md` for detailed technical documentation.*
