# Claude Code Project Guidelines

This file contains project-specific guidelines and conventions for Claude Code to follow when working on this codebase.

## Development Practices

### Temporary Files
- **Always clean up temporary/one-off scripts** created for testing or debugging features
- Don't litter the codebase with comparison scripts, test harnesses, or debug utilities
- Only permanent, useful tools should remain in the project root
- Before completing a task, proactively remove any temporary files created during development

## Project Structure

This is the Hippocampus project - a document ingestion and semantic search system using ChromaDB.

### Core Files
- `ingest.py` - Document ingestion pipeline
- `query.py` - Hybrid search interface (combines exact text + semantic search)
- `markdown_chunker.py` - Text chunking utilities
- `file_tracker.py` - File tracking system
- `pdf_tracker.py` - PDF tracking system
- `ebook_tracker.py` - EPUB/MOBI ebook tracking system
- `txt_tracker.py` - TXT to markdown conversion system
- `office_tracker.py` - Office document (DOC/DOCX/ODT) tracking system
- `clean_md.py` - Markdown cleaning utilities
- `clear_db.py` - Database management
- `status.py` - System status checking

### Key Technologies
- ChromaDB for vector storage
- sentence-transformers for embeddings
- Hybrid search: exact text matching + semantic embeddings
