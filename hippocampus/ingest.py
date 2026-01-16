#!/usr/bin/env python3
"""
Continuous document ingestion pipeline for ChromaDB.
Monitors a directory for markdown files, chunks them semantically,
generates embeddings, and stores them in ChromaDB.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import chromadb
from chromadb.config import Settings

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.embeddings import get_embedder
from libs.olorin_logging import OlorinLogger

# Add broca directory to path for producer import
broca_path = os.path.join(os.path.dirname(__file__), "..", "broca")
if os.path.exists(broca_path):
    sys.path.insert(0, broca_path)
    from producer import TTSProducer

from file_tracker import FileTracker  # noqa: E402
from markdown_chunker import MarkdownChunker  # noqa: E402

# Initialize config
config = Config()


class DocumentIngestionPipeline:
    """
    Pipeline for continuous document ingestion into ChromaDB.
    """

    def __init__(self):
        """Initialize the ingestion pipeline with configuration from .env"""

        # Load configuration
        self.input_dir = config.get_path("INPUT_DIR", "~/Documents/AI_IN")
        self.chromadb_host = config.get("CHROMADB_HOST", "localhost")
        self.chromadb_port = config.get_int("CHROMADB_PORT", 8000)
        self.collection_name = config.get("CHROMADB_COLLECTION", "documents")

        self.chunk_size = config.get_int("CHUNK_SIZE", 1000)
        self.chunk_overlap = config.get_int("CHUNK_OVERLAP", 200)
        self.chunk_min_size = config.get_int("CHUNK_MIN_SIZE", 100)

        self.poll_interval = config.get_int("POLL_INTERVAL", 5)
        self.tracking_db = config.get_path(
            "TRACKING_DB", "./hippocampus/data/tracking.db"
        )

        self.reprocess_on_change = config.get_bool("REPROCESS_ON_CHANGE", True)
        self.delete_after_processing = config.get_bool("DELETE_AFTER_PROCESSING", False)

        # Setup logging
        log_level = config.get("LOG_LEVEL", "INFO")
        default_log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        log_dir = config.get("LOG_DIR", default_log_dir)
        log_file = config.get(
            "LOG_FILE", os.path.join(log_dir, "hippocampus-ingest.log")
        )

        # Initialize logger
        self.logger = OlorinLogger(
            log_file=log_file, log_level=log_level, name=__name__
        )
        self.logger.info("Initializing Document Ingestion Pipeline...")

        # Create input directory if it doesn't exist
        os.makedirs(self.input_dir, exist_ok=True)
        self.logger.info(f"Monitoring directory: {self.input_dir}")

        # Initialize file tracker
        self.file_tracker = FileTracker(self.tracking_db)

        # Initialize markdown chunker
        self.chunker = MarkdownChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.chunk_min_size,
        )

        # Initialize embedder (local or API-based depending on config)
        self.embedder = get_embedder(config=config)
        self.logger.info(
            f"Embedder ready: {self.embedder.model_name} (dimension: {self.embedder.dimension})"
        )

        # Initialize ChromaDB client
        self.logger.info(
            f"Connecting to ChromaDB at {self.chromadb_host}:{self.chromadb_port}"
        )
        self.chroma_client = chromadb.HttpClient(
            host=self.chromadb_host,
            port=self.chromadb_port,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for semantic search"},
        )
        self.logger.info(
            f"Using collection: {self.collection_name} "
            f"({self.collection.count()} existing documents)"
        )

        # Initialize Kafka producer for notifications
        try:
            if "TTSProducer" in globals():
                self.kafka_producer = TTSProducer()
                self.logger.info("Kafka producer initialized for ai_out notifications")
            else:
                self.kafka_producer = None
                self.logger.warning("TTSProducer not available, notifications disabled")
        except Exception as e:
            self.logger.warning(f"Could not initialize Kafka producer: {e}")
            self.kafka_producer = None

        self.logger.info("Pipeline initialized successfully!")

    def find_markdown_files(self) -> List[str]:
        """
        Find all markdown files in the input directory.

        Returns:
            List of absolute file paths
        """
        md_files = []
        input_path = Path(self.input_dir)

        for md_file in input_path.rglob("*.md"):
            if md_file.is_file():
                md_files.append(str(md_file.absolute()))

        return md_files

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed.

        Args:
            file_path: Path to file

        Returns:
            True if file should be processed
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return False

        # If file hasn't been processed, process it
        if not self.file_tracker.is_file_processed(file_path):
            return True

        # If reprocess on change is enabled, check if file changed
        if self.reprocess_on_change:
            return self.file_tracker.has_file_changed(file_path)

        return False

    def process_file(self, file_path: str) -> bool:
        """
        Process a single markdown file.

        Args:
            file_path: Path to file

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {file_path}")

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                self.logger.warning(f"Empty file: {file_path}")
                return False

            # Extract metadata from frontmatter for notification
            frontmatter_metadata, _ = self.chunker._parse_yaml_frontmatter(content)
            if not frontmatter_metadata:
                frontmatter_metadata = {}

            # Chunk the content
            chunks = self.chunker.chunk_markdown(content, file_path)

            if not chunks:
                self.logger.warning(f"No chunks created from: {file_path}")
                return False

            # Generate embeddings for all chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            self.logger.debug(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = self.embedder.embed_documents(chunk_texts)

            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            documents = []
            embeddings_list = []

            # Generate unique IDs for each chunk
            file_hash = self.file_tracker.compute_file_hash(file_path)[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{file_hash}_{timestamp}_{idx}"
                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                embeddings_list.append(embedding.tolist())

            # If file was processed before and changed, remove old chunks
            if self.file_tracker.is_file_processed(file_path):
                self.logger.debug(f"Removing old chunks for: {file_path}")
                # Query for old chunks from this file
                old_results = self.collection.get(where={"source": file_path})
                if old_results["ids"]:
                    self.collection.delete(ids=old_results["ids"])
                    self.logger.debug(f"Removed {len(old_results['ids'])} old chunks")

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )

            # Mark file as processed
            self.file_tracker.mark_processed(
                file_path, chunk_count=len(chunks), status="success"
            )

            self.logger.info(
                f"Successfully processed {file_path}: "
                f"{len(chunks)} chunks, "
                f"{len(content)} chars"
            )

            # Send notification to ai_out
            if self.kafka_producer:
                try:
                    title = frontmatter_metadata.get("title", "Unknown")
                    author = frontmatter_metadata.get("author", "Unknown")
                    message = f"Ingestion for {title} by {author} is now complete."
                    self.kafka_producer.send_message(message)
                    self.logger.info(
                        f"Sent completion notification to ai_out: {message}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to send notification to ai_out: {e}")

            # Delete file if configured
            if self.delete_after_processing:
                os.remove(file_path)
                self.logger.info(f"Deleted processed file: {file_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            # Mark as error so we can track failures
            try:
                self.file_tracker.mark_processed(
                    file_path, chunk_count=0, status="error"
                )
            except Exception:
                pass

            return False

    def run_single_scan(self) -> Dict[str, int]:
        """
        Run a single scan of the input directory.

        Returns:
            Dictionary with scan statistics
        """
        stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
        }

        # Find all markdown files
        md_files = self.find_markdown_files()
        stats["files_found"] = len(md_files)

        if not md_files:
            self.logger.debug(f"No markdown files found in {self.input_dir}")
            return stats

        # Process each file
        for file_path in md_files:
            if self.should_process_file(file_path):
                if self.process_file(file_path):
                    stats["files_processed"] += 1
                else:
                    stats["files_failed"] += 1
            else:
                stats["files_skipped"] += 1
                self.logger.debug(f"Skipping (already processed): {file_path}")

        return stats

    def run_continuous(self):
        """
        Run continuous monitoring and ingestion loop.
        """
        self.logger.info(
            f"Starting continuous monitoring (poll interval: {self.poll_interval}s)"
        )
        self.logger.info("Press Ctrl+C to stop")

        scan_count = 0

        try:
            while True:
                scan_count += 1
                self.logger.debug(f"Scan #{scan_count}")

                stats = self.run_single_scan()

                if stats["files_processed"] > 0 or stats["files_failed"] > 0:
                    self.logger.info(
                        f"Scan #{scan_count} complete: "
                        f"{stats['files_processed']} processed, "
                        f"{stats['files_failed']} failed, "
                        f"{stats['files_skipped']} skipped"
                    )

                    # Show tracker statistics
                    tracker_stats = self.file_tracker.get_statistics()
                    self.logger.info(
                        f"Total: {tracker_stats['total_files']} files, "
                        f"{tracker_stats['total_chunks']} chunks"
                    )

                # Wait before next scan
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("\nShutting down gracefully...")
            self._shutdown()

    def _shutdown(self):
        """Cleanup and shutdown."""
        # Close Kafka producer if initialized
        if self.kafka_producer:
            try:
                self.kafka_producer.producer.flush()
                self.kafka_producer.producer.close()
                self.logger.info("Kafka producer closed")
            except Exception as e:
                self.logger.warning(f"Error closing Kafka producer: {e}")

        # Show final statistics
        stats = self.file_tracker.get_statistics()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total files processed: {stats['total_files']}")
        self.logger.info(f"  Total chunks created: {stats['total_chunks']}")
        self.logger.info(
            f"  Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB"
        )
        self.logger.info(f"  Successful: {stats['successful']}")
        self.logger.info(f"  Errors: {stats['errors']}")
        self.logger.info(f"  ChromaDB documents: {self.collection.count()}")
        self.logger.info("=" * 60)
        self.logger.info("Pipeline stopped")


def main():
    """Main entry point."""
    try:
        pipeline = DocumentIngestionPipeline()
        pipeline.run_continuous()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
