#!/usr/bin/env python3
"""
Query utility for hybrid search in ChromaDB.
Combines direct text matching with semantic search.
Simple REPL for searching the document collection.
"""

import os
import sys
import re
import readline
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from libs.config import Config

# Initialize config
config = Config()


class QueryREPL:
    def __init__(self):
        # Load configuration
        chromadb_host = config.get('CHROMADB_HOST', 'localhost')
        chromadb_port = config.get_int('CHROMADB_PORT', 8000)
        collection_name = config.get('CHROMADB_COLLECTION', 'documents')
        embedding_model_name = config.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.default_n_results = config.get_int('DEFAULT_RESULTS', 5)

        print(f"Connecting to ChromaDB at {chromadb_host}:{chromadb_port}...")

        # Connect to ChromaDB
        self.client = chromadb.HttpClient(
            host=chromadb_host,
            port=chromadb_port,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get collection
        self.collection = self.client.get_collection(name=collection_name)
        self.total_docs = self.collection.count()
        print(f"Collection '{collection_name}' has {self.total_docs} documents")

        if self.total_docs == 0:
            print("Warning: No documents in collection. Run ingest.py first to add documents.")

        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.model = SentenceTransformer(embedding_model_name)

        # Cache all documents for direct search
        print("Caching documents for hybrid search...")
        self.all_docs = self.collection.get()
        print("Ready!\n")

    def direct_text_search(self, query_text: str, n_results: int = None):
        """
        Perform direct text search for exact matches.

        Args:
            query_text: Text to search for
            n_results: Maximum number of results to return

        Returns:
            List of tuples: (doc, metadata, match_count, match_type)
        """
        if n_results is None:
            n_results = self.default_n_results

        matches = []
        pattern = re.escape(query_text)

        for doc, metadata, doc_id in zip(
            self.all_docs['documents'],
            self.all_docs['metadatas'],
            self.all_docs['ids']
        ):
            # Case-insensitive search
            if re.search(pattern, doc, re.IGNORECASE):
                match_count = len(re.findall(pattern, doc, re.IGNORECASE))
                matches.append({
                    'doc': doc,
                    'metadata': metadata,
                    'id': doc_id,
                    'match_count': match_count,
                    'match_type': 'exact'
                })

        # Sort by match count
        matches.sort(key=lambda x: x['match_count'], reverse=True)

        return matches[:n_results * 2]  # Get extra for deduplication

    def embedding_search(self, query_text: str, n_results: int = None):
        """
        Perform embedding-based semantic search.

        Args:
            query_text: Text to search for
            n_results: Number of results to return

        Returns:
            List of dicts with results
        """
        if n_results is None:
            n_results = self.default_n_results

        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results * 3, self.total_docs)  # Get extra for deduplication
        )

        matches = []
        for doc, metadata, distance, doc_id in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
            results['ids'][0]
        ):
            matches.append({
                'doc': doc,
                'metadata': metadata,
                'id': doc_id,
                'distance': distance,
                'match_type': 'semantic'
            })

        return matches

    def query_documents(self, query_text: str, n_results: int = None):
        """
        Query the document collection using hybrid search.
        Combines direct text matching with semantic search.

        Args:
            query_text: Text to search for
            n_results: Number of results to return
        """
        if self.total_docs == 0:
            print("No documents in collection.")
            return

        if n_results is None:
            n_results = self.default_n_results

        # Perform both searches
        direct_matches = self.direct_text_search(query_text, n_results)
        embedding_matches = self.embedding_search(query_text, n_results)

        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []

        # Priority 1: Direct matches (exact text found)
        for match in direct_matches:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                combined_results.append(match)

        # Priority 2: Embedding matches (semantic similarity)
        for match in embedding_matches:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                combined_results.append(match)

        # Limit to requested number of results
        combined_results = combined_results[:n_results]

        # Display results
        print("\n" + "="*80)
        print(f"HYBRID SEARCH RESULTS")
        print("="*80)
        print(f"Query: '{query_text}'")

        if direct_matches:
            print(f"✓ Found {len(direct_matches)} exact matches")
        else:
            print("✗ No exact matches found")

        print(f"Found {len(combined_results)} total results (showing top {min(n_results, len(combined_results))})")
        print("="*80)

        for idx, result in enumerate(combined_results, start=1):
            doc = result['doc']
            metadata = result['metadata']
            match_type = result['match_type']

            # Result header with type indicator
            if match_type == 'exact':
                type_badge = f"[EXACT MATCH - {result['match_count']} occurrences]"
            else:
                type_badge = f"[SEMANTIC - distance: {result['distance']:.4f}]"

            print(f"\n[Result {idx}] {type_badge}")
            print(f"Source: {metadata.get('source', 'unknown')}")

            # Show header hierarchy if available
            headers = []
            for i in range(1, 7):
                header_key = f'h{i}'
                if header_key in metadata:
                    headers.append(f"{'#'*i} {metadata[header_key]}")

            if headers:
                print(f"Location: {' > '.join(headers)}")

            print(f"Chunk {metadata.get('chunk_index', '?')} ({metadata.get('char_count', '?')} chars)")
            print("-"*80)

            # For exact matches, show context around the match
            if match_type == 'exact':
                pattern = re.escape(query_text)
                match_obj = re.search(pattern, doc, re.IGNORECASE)
                if match_obj:
                    pos = match_obj.start()
                    start = max(0, pos - 200)
                    end = min(len(doc), pos + len(query_text) + 300)
                    context = doc[start:end]

                    # Highlight the match (simple approach)
                    highlighted = re.sub(
                        f'({pattern})',
                        r'>>> \1 <<<',
                        context,
                        flags=re.IGNORECASE
                    )
                    if start > 0:
                        highlighted = "..." + highlighted
                    if end < len(doc):
                        highlighted = highlighted + "..."

                    print(highlighted)
            else:
                # Show preview (first 500 chars) for semantic matches
                preview = doc[:500]
                if len(doc) > 500:
                    preview += "..."
                print(preview)
            print()

        print("="*80 + "\n")

    def start_repl(self):
        """Start the REPL interface"""
        # Set up readline history
        history_file = os.path.expanduser('~/.query_history')
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

        print("="*60)
        print("ChromaDB Hybrid Query REPL")
        print("="*60)
        print("Type your search query and press Enter")
        print("Hybrid search: Combines exact text matching + semantic search")
        print("Commands: 'quit' or 'exit' to exit, Ctrl+C to interrupt")
        print("Append ':N' to limit results (e.g., 'machine learning:3')")
        print("="*60 + "\n")

        try:
            while True:
                try:
                    text = input(">>> ").strip()

                    if not text:
                        continue

                    if text.lower() in ['quit', 'exit']:
                        print("Exiting...")
                        break

                    # Check if user specified a result count
                    n_results = self.default_n_results
                    if ':' in text:
                        parts = text.rsplit(':', 1)
                        try:
                            n_results = int(parts[1])
                            text = parts[0].strip()
                        except ValueError:
                            pass  # Not a number, treat the whole thing as query

                    self.query_documents(text, n_results)

                except EOFError:
                    print("\nExiting...")
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
        finally:
            # Save command history
            try:
                readline.write_history_file(history_file)
            except Exception as e:
                print(f"Warning: Could not save history: {e}", file=sys.stderr)


def main():
    """Main CLI entry point."""
    try:
        repl = QueryREPL()
        repl.start_repl()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
