# Hippocampus - Document Ingestion Pipeline

A continuous document ingestion pipeline that monitors a directory for markdown files, chunks them semantically, generates embeddings using local models, and stores them in ChromaDB for semantic search.

## Features

- **Continuous Monitoring**: Watches `~/Documents/AI_IN` for new `.md` files
- **Semantic Chunking**: Intelligently splits markdown by structure (headers, paragraphs)
- **Local Embeddings**: Uses sentence-transformers (no API keys needed)
- **Smart Tracking**: SQLite database tracks processed files and detects changes
- **Extensible Design**: Modular architecture with clear separation of concerns
- **Configurable**: All settings via `.env` file
- **Production Ready**: Comprehensive logging, error handling, and statistics

## Architecture

```
┌──────────────────┐
│  ~/Documents/    │
│    AI_IN/        │  ← Input directory (monitored)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ingest.py       │  ← Main pipeline
│  - File monitor  │
│  - Orchestration │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ file_   │ │ markdown_    │
│ tracker │ │ chunker      │
│ .py     │ │ .py          │
└────┬────┘ └──────┬───────┘
     │             │
     │             ▼
     │      ┌──────────────┐
     │      │ sentence-    │
     │      │ transformers │
     │      └──────┬───────┘
     │             │
     └─────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  ChromaDB    │  ← Vector database
    │  (port 8000) │
    └──────────────┘
```

## Prerequisites

- Python 3.8+
- Podman or Docker
- pip

## Setup

### 1. Start ChromaDB

```bash
./run
```

This starts ChromaDB in a container on port 8000 with persistent storage in `./data`.

### 2. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `chromadb` - Vector database client
- `sentence-transformers` - Local embedding models
- `langchain` & `langchain-text-splitters` - Text chunking utilities
- `python-dotenv` - Environment configuration
- `markdown` - Markdown processing
- `watchdog` - Filesystem monitoring

The first run will download the embedding model (`all-MiniLM-L6-v2`, ~80MB).

### 4. Configure (Optional)

Edit `.env` to customize settings:

```bash
# Input directory to monitor
INPUT_DIR=~/Documents/AI_IN

# ChromaDB connection
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_COLLECTION=documents

# Embedding model (sentence-transformers)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNK_MIN_SIZE=100

# Monitoring
POLL_INTERVAL=5

# File handling
REPROCESS_ON_CHANGE=true
DELETE_AFTER_PROCESSING=false
```

### 5. Create Input Directory

```bash
mkdir -p ~/Documents/AI_IN
```

## Usage

### Start the Pipeline

```bash
python3 ingest.py
```

The pipeline will:
1. Monitor `~/Documents/AI_IN` every 5 seconds
2. Detect new or changed `.md` files
3. Chunk them semantically by markdown structure
4. Generate embeddings using local model
5. Store in ChromaDB with metadata
6. Track processing state in SQLite

### Add Documents

Simply copy markdown files to the input directory:

```bash
cp my_notes.md ~/Documents/AI_IN/
```

The pipeline will automatically detect and process them.

### Stop the Pipeline

Press `Ctrl+C` to gracefully shut down. Final statistics will be displayed.

## Testing

The project uses pytest for testing with fixtures and coverage reporting.

### Installing Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov
```

Run specific test file:
```bash
pytest tests/test_markdown_chunker.py
pytest tests/test_no_frontmatter.py
```

Run tests matching a pattern:
```bash
pytest -k "frontmatter"
```

View verbose output:
```bash
pytest -v
```

### Test Structure

```
tests/
├── __init__.py                 # Test package init
├── conftest.py                 # Shared fixtures
├── test_markdown_chunker.py    # Frontmatter parsing tests
└── test_no_frontmatter.py      # Non-frontmatter tests
```

### Writing New Tests

1. Create test files with `test_` prefix in the `tests/` directory
2. Use fixtures from `conftest.py` for common setup (chunker instances, sample data)
3. Follow naming convention: `test_<what_is_tested>`
4. Use descriptive assertions with failure messages
5. Run tests before committing changes

**Example:**
```python
def test_my_feature(chunker):
    """Test my new feature."""
    result = chunker.some_method()
    assert result is not None, "Result should not be None"
```

## Module Documentation

### `ingest.py`

Main orchestration script. Runs the continuous monitoring loop, coordinates file discovery, processing, and storage.

**Key Classes:**
- `DocumentIngestionPipeline`: Main pipeline coordinator

**Key Methods:**
- `run_continuous()`: Main monitoring loop
- `process_file(file_path)`: Process a single file
- `run_single_scan()`: Single directory scan

### `file_tracker.py`

SQLite-based file tracking system. Tracks processed files using content hashes to detect changes.

**Key Classes:**
- `FileTracker`: File state management

**Key Methods:**
- `is_file_processed(file_path)`: Check if file is tracked
- `has_file_changed(file_path)`: Detect content changes via hash
- `mark_processed(file_path, chunk_count, status)`: Record processing
- `get_statistics()`: Get processing statistics

**Database Schema:**
```sql
CREATE TABLE processed_files (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    processed_at TIMESTAMP NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'success'
);
```

### `markdown_chunker.py`

Semantic markdown chunker using LangChain. Preserves document structure by respecting headers and paragraphs.

**Key Classes:**
- `MarkdownChunker`: Semantic chunking engine

**Key Methods:**
- `chunk_markdown(content, source_file)`: Create chunks from markdown
- `extract_title(content)`: Extract document title

**Chunking Strategy:**
1. Split by markdown headers (h1-h6)
2. Further split large sections using recursive text splitter
3. Preserve header hierarchy in metadata
4. Maintain overlap between chunks for context

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `~/Documents/AI_IN` | Directory to monitor |
| `CHROMADB_HOST` | `localhost` | ChromaDB server host |
| `CHROMADB_PORT` | `8000` | ChromaDB server port |
| `CHROMADB_COLLECTION` | `documents` | Collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `1000` | Target chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (chars) |
| `CHUNK_MIN_SIZE` | `100` | Minimum chunk size (chars) |
| `POLL_INTERVAL` | `5` | Seconds between scans |
| `TRACKING_DB` | `./data/tracking.db` | SQLite database path |
| `REPROCESS_ON_CHANGE` | `true` | Reprocess changed files |
| `DELETE_AFTER_PROCESSING` | `false` | Delete files after ingestion |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `./data/ingest.log` | Log file path |

### Embedding Models

You can use any sentence-transformers model:

**Fast & Lightweight:**
- `all-MiniLM-L6-v2` (384 dim, ~80MB) - Recommended default
- `paraphrase-MiniLM-L3-v2` (384 dim, ~60MB) - Fastest

**Higher Quality:**
- `all-mpnet-base-v2` (768 dim, ~420MB) - Better accuracy
- `multi-qa-mpnet-base-dot-v1` (768 dim, ~420MB) - Optimized for Q&A

**Multilingual:**
- `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, ~420MB)

Browse all models: https://www.sbert.net/docs/pretrained_models.html

## Extending the Pipeline

### Adding New File Types

1. Create a new chunker module (e.g., `pdf_chunker.py`)
2. Implement chunking logic
3. Update `ingest.py` to detect and route different file types

```python
if file_path.endswith('.pdf'):
    chunks = pdf_chunker.chunk_pdf(content, file_path)
elif file_path.endswith('.md'):
    chunks = markdown_chunker.chunk_markdown(content, file_path)
```

### Custom Metadata

Add custom metadata in chunking:

```python
chunk = {
    'text': chunk_text,
    'metadata': {
        'source': source_file,
        'author': 'John Doe',  # Custom field
        'category': 'research',  # Custom field
        **headers
    }
}
```

### Different Embedding Models

To use a different embedding provider:

1. Install the provider's library
2. Update `ingest.py` to use the new encoder:

```python
# Example: Using OpenAI embeddings
from openai import OpenAI
client = OpenAI()

def encode_with_openai(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]
```

### Using exo for Distributed Inference

If you prefer using exo for embeddings:

1. Start exo cluster: `exo run nomic-embed-text`
2. Update `ingest.py` to connect to exo's OpenAI-compatible endpoint
3. Set `EMBEDDING_MODEL` to exo model name in `.env`

## Troubleshooting

### ChromaDB Connection Errors

```bash
# Check if ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Restart ChromaDB
./run
```

### Model Download Issues

The first run downloads the embedding model. If it fails:

```bash
# Pre-download manually
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Permission Errors

Ensure input directory is readable:

```bash
chmod 755 ~/Documents/AI_IN
```

### Check Logs

```bash
# View live logs
tail -f ./data/ingest.log

# Debug mode
LOG_LEVEL=DEBUG python3 ingest.py
```

## Performance

### Benchmarks (on M1 Mac)

- **Embedding speed**: ~500 tokens/sec (all-MiniLM-L6-v2)
- **Chunking**: ~1MB markdown/sec
- **End-to-end**: ~100 pages/minute

### Resource Usage

- **Memory**: ~500MB (model loaded)
- **CPU**: <5% idle, 20-40% during processing
- **Storage**: ~1KB per chunk in ChromaDB

## Querying the Database

Use ChromaDB client to query embeddings:

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Connect to ChromaDB
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection('documents')

# Load same embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Query
query = "What is machine learning?"
query_embedding = model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

# Print results
for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
    print(f"\n{metadata['source']}:")
    print(doc[:200] + "...")
```

## License

MIT

## Contributing

Contributions welcome! The modular design makes it easy to extend:

- Add new file type processors
- Implement different chunking strategies
- Integrate alternative embedding models
- Add preprocessing pipelines
- Create monitoring dashboards
