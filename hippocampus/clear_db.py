#!/usr/bin/env python3
"""
Database reset script for Hippocampus.
Clears file tracking database and ChromaDB collection, resetting everything to a pristine state.

This is the ONLY script that should clear databases - separation of concerns from tracking modules.
"""

import os
import sys
import logging
import argparse
import sqlite3
import chromadb
from chromadb.config import Settings

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def clear_tracking_db(db_path: str, db_type: str, logger: logging.Logger) -> int:
    """
    Clear all records from a file tracking database using direct SQLite operations.

    Args:
        db_path: Path to the tracking database
        db_type: Type of database ('markdown' or 'pdf') for logging
        logger: Logger instance

    Returns:
        Number of records deleted
    """
    if not os.path.exists(db_path):
        logger.warning(f"{db_type.capitalize()} database not found: {db_path}")
        return 0

    try:
        # Direct SQLite operations - no dependency on FileTracker methods
        with sqlite3.connect(db_path) as conn:
            # Get count before deletion
            cursor = conn.execute("SELECT COUNT(*) FROM processed_files")
            count = cursor.fetchone()[0]

            if count == 0:
                logger.info(
                    f"{db_type.capitalize()} tracking database is already empty"
                )
                return 0

            # Delete all records
            conn.execute("DELETE FROM processed_files")
            conn.commit()

        logger.info(f"✓ Cleared {count} {db_type} records from {db_path}")
        return count
    except Exception as e:
        logger.error(f"✗ Failed to clear {db_type} database {db_path}: {e}")
        return 0


def clear_chromadb(
    host: str, port: int, collection_name: str, logger: logging.Logger
) -> int:
    """
    Clear all documents from ChromaDB collection.

    Args:
        host: ChromaDB host
        port: ChromaDB port
        collection_name: Name of the collection to clear
        logger: Logger instance

    Returns:
        Number of documents deleted
    """
    try:
        # Connect to ChromaDB
        logger.info(f"Connecting to ChromaDB at {host}:{port}...")
        client = chromadb.HttpClient(
            host=host, port=port, settings=Settings(anonymized_telemetry=False)
        )

        # Get collection
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"Collection '{collection_name}' does not exist")
            return 0

        # Get count before deletion
        count = collection.count()

        if count == 0:
            logger.info(f"Collection '{collection_name}' is already empty")
            return 0

        # Delete the collection entirely and recreate it
        logger.info(
            f"Deleting collection '{collection_name}' with {count} documents..."
        )
        client.delete_collection(name=collection_name)

        # Recreate the collection
        client.create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings for semantic search"},
        )

        logger.info(
            f"✓ Cleared {count} documents from ChromaDB collection '{collection_name}'"
        )
        return count

    except Exception as e:
        logger.error(f"✗ Failed to clear ChromaDB: {e}")
        logger.error("Make sure ChromaDB is running (use ./run script)")
        return 0


def main():
    """Main function to clear databases with explicit, selective options."""
    parser = argparse.ArgumentParser(
        description="Clear file tracking databases and/or ChromaDB collection",
        epilog="""
Examples:
  %(prog)s --tracker --md              Clear only markdown tracking
  %(prog)s --tracker --pdf             Clear only PDF tracking
  %(prog)s --tracker --md --pdf        Clear both tracking databases
  %(prog)s --tracker                   Clear both tracking databases (default)
  %(prog)s --chroma                    Clear only ChromaDB
  %(prog)s --tracker --chroma          Clear everything
  %(prog)s --tracker --md --chroma -y  Clear markdown tracking and ChromaDB (no confirmation)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Primary action flags
    parser.add_argument(
        "--tracker", action="store_true", help="Clear file tracking database(s)"
    )
    parser.add_argument(
        "--chroma", action="store_true", help="Clear ChromaDB collection"
    )

    # Selective tracker clearing
    parser.add_argument(
        "--md",
        action="store_true",
        help="Clear only markdown tracking database (use with --tracker)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Clear only PDF tracking database (use with --tracker)",
    )

    # Options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.tracker and not args.chroma:
        parser.error("Must specify at least one of: --tracker, --chroma")

    if (args.md or args.pdf) and not args.tracker:
        parser.error("--md and --pdf can only be used with --tracker")

    # Setup logging
    logger = setup_logging(args.verbose)

    # Initialize config
    config = Config()

    # Get configuration - all paths resolved via config
    md_tracking_db = config.get_path("TRACKING_DB", "./hippocampus/data/tracking.db")
    pdf_tracking_db = config.get_path(
        "PDF_TRACKING_DB", "./hippocampus/data/pdf_tracking.db"
    )
    chromadb_host = config.get("CHROMADB_HOST", "localhost")
    chromadb_port = config.get_int("CHROMADB_PORT", 8000)
    collection_name = config.get("CHROMADB_COLLECTION", "documents")

    # Determine what to clear
    clear_md = False
    clear_pdf = False
    clear_chroma_flag = args.chroma

    if args.tracker:
        # If neither --md nor --pdf specified, clear both
        if not args.md and not args.pdf:
            clear_md = True
            clear_pdf = True
        else:
            clear_md = args.md
            clear_pdf = args.pdf

    # Display what will be cleared
    logger.info("=" * 70)
    logger.info("Database Reset Script - Hippocampus")
    logger.info("=" * 70)

    items_to_clear = []

    if clear_md:
        items_to_clear.append(f"Markdown tracking: {md_tracking_db}")
    if clear_pdf:
        items_to_clear.append(f"PDF tracking: {pdf_tracking_db}")
    if clear_chroma_flag:
        items_to_clear.append(
            f"ChromaDB: {chromadb_host}:{chromadb_port}/{collection_name}"
        )

    logger.info("Will clear:")
    for item in items_to_clear:
        logger.info(f"  • {item}")

    logger.info("=" * 70)

    # Confirmation prompt
    if not args.yes:
        response = input(
            "\nThis will permanently delete all specified data. Continue? [y/N]: "
        )
        if response.lower() not in ["y", "yes"]:
            logger.info("Aborted by user")
            return 0

    logger.info("\nStarting database reset...\n")

    # Track totals
    total_md_records = 0
    total_pdf_records = 0
    total_chroma_docs = 0

    # Clear markdown tracking database
    if clear_md:
        logger.info("Clearing markdown tracking database...")
        total_md_records = clear_tracking_db(md_tracking_db, "markdown", logger)
        logger.info("")

    # Clear PDF tracking database
    if clear_pdf:
        logger.info("Clearing PDF tracking database...")
        total_pdf_records = clear_tracking_db(pdf_tracking_db, "pdf", logger)
        logger.info("")

    # Clear ChromaDB
    if clear_chroma_flag:
        logger.info("Clearing ChromaDB collection...")
        total_chroma_docs = clear_chromadb(
            chromadb_host, chromadb_port, collection_name, logger
        )
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("Reset Complete!")
    logger.info("=" * 70)

    summary_lines = []
    if clear_md:
        summary_lines.append(f"Markdown tracking records cleared: {total_md_records}")
    if clear_pdf:
        summary_lines.append(f"PDF tracking records cleared: {total_pdf_records}")
    if clear_chroma_flag:
        summary_lines.append(f"ChromaDB documents cleared: {total_chroma_docs}")

    for line in summary_lines:
        logger.info(line)

    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
