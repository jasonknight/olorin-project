#!/usr/bin/env python3
"""
Status utility to check the state of the ingestion pipeline.
Shows statistics about processed files and ChromaDB collection.
"""

import os
import sys
import chromadb
from chromadb.config import Settings
from file_tracker import FileTracker
from datetime import datetime

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config

# Initialize config
config = Config()


def show_status():
    """Display pipeline status and statistics."""
    # Load configuration
    chromadb_host = config.get("CHROMADB_HOST", "localhost")
    chromadb_port = config.get_int("CHROMADB_PORT", 8000)
    collection_name = config.get("CHROMADB_COLLECTION", "documents")
    tracking_db = config.get_path("TRACKING_DB", "./hippocampus/data/tracking.db")
    input_dir = config.get_path("INPUT_DIR", "~/Documents/AI_IN")

    print("=" * 80)
    print("Hippocampus - Document Ingestion Pipeline Status")
    print("=" * 80)

    # Check ChromaDB connection
    print("\n[ChromaDB]")
    try:
        client = chromadb.HttpClient(
            host=chromadb_host,
            port=chromadb_port,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
        doc_count = collection.count()
        print("  Status: ✓ Connected")
        print(f"  Host: {chromadb_host}:{chromadb_port}")
        print(f"  Collection: {collection_name}")
        print(f"  Documents: {doc_count}")
    except Exception as e:
        print(f"  Status: ✗ Error - {e}")
        print(f"  Host: {chromadb_host}:{chromadb_port}")
        return

    # Check file tracker
    print("\n[File Tracker]")
    try:
        tracker = FileTracker(tracking_db)
        stats = tracker.get_statistics()

        print(f"  Database: {tracking_db}")
        print(f"  Total files processed: {stats['total_files']}")
        print(f"  Total chunks created: {stats['total_chunks']}")
        print(f"  Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Successful: {stats['successful']}")
        print(f"  Errors: {stats['errors']}")

        # Show recent files
        if stats["total_files"] > 0:
            print("\n  Recent files:")
            recent_files = tracker.get_all_processed_files()[:5]
            for file_info in recent_files:
                processed_at = file_info["processed_at"]
                # Parse and format timestamp
                try:
                    dt = datetime.fromisoformat(processed_at)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    time_str = processed_at

                status_icon = "✓" if file_info["status"] == "success" else "✗"
                print(f"    {status_icon} {file_info['file_path']}")
                print(f"       {time_str} - {file_info['chunk_count']} chunks")

    except Exception as e:
        print(f"  Status: ✗ Error - {e}")

    # Check input directory
    print("\n[Input Directory]")
    print(f"  Path: {input_dir}")

    if os.path.exists(input_dir):
        print("  Status: ✓ Exists")

        # Count markdown files
        md_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        print(f"  Markdown files: {len(md_files)}")

        if len(md_files) > 0:
            print("\n  Files in directory:")
            for md_file in md_files[:10]:  # Show first 10
                file_size = os.path.getsize(md_file) / 1024  # KB
                rel_path = os.path.relpath(md_file, input_dir)
                print(f"    • {rel_path} ({file_size:.1f} KB)")

            if len(md_files) > 10:
                print(f"    ... and {len(md_files) - 10} more")
    else:
        print("  Status: ✗ Directory does not exist")

    # Configuration summary
    print("\n[Configuration]")
    print(f"  Embedding model: {config.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    print(f"  Chunk size: {config.get('CHUNK_SIZE', '1000')} chars")
    print(f"  Chunk overlap: {config.get('CHUNK_OVERLAP', '200')} chars")
    print(f"  Poll interval: {config.get('POLL_INTERVAL', '5')} seconds")
    print(f"  Reprocess on change: {config.get('REPROCESS_ON_CHANGE', 'true')}")
    print(
        f"  Delete after processing: {config.get('DELETE_AFTER_PROCESSING', 'false')}"
    )

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    try:
        show_status()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
