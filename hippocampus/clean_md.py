#!/usr/bin/env python3
"""
Clean markdown files by removing low-quality content and fixing formatting.
Uses Ollama to identify and remove advertisements, TOCs, and low-information text.
Also detects and removes repetitive content.
"""

import argparse
import sys
import re
import hashlib
import tempfile
import shutil
import subprocess
from pathlib import Path
from collections import Counter


def calculate_hash(content):
    """Calculate SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_markdown_file(file_path):
    """Read the markdown file and return its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def detect_repetitive_lines(content, threshold=3):
    """
    Detect lines that repeat more than threshold times.
    Returns a set of lines that are considered repetitive.
    """
    lines = content.split("\n")
    # Filter out empty lines and very short lines (< 10 chars)
    meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 10]

    # Count occurrences
    line_counts = Counter(meaningful_lines)

    # Find repetitive lines
    repetitive = {line for line, count in line_counts.items() if count >= threshold}

    return repetitive


def remove_repetitive_content(content, repetitive_lines):
    """
    Remove repetitive lines from content, keeping only the first occurrence.
    """
    if not repetitive_lines:
        return content

    lines = content.split("\n")
    seen_repetitive = set()
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped in repetitive_lines:
            if stripped not in seen_repetitive:
                filtered_lines.append(line)
                seen_repetitive.add(stripped)
            # Skip subsequent occurrences
        else:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def chunk_content(content, chunk_size=4000):
    """
    Split content into chunks for processing.
    Tries to split on paragraph boundaries to maintain context.
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\n+", content)

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if current_size + para_size > chunk_size and current_chunk:
            # Save current chunk and start a new one
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for the \n\n separator

    # Add the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def get_ollama_path():
    """Find the ollama binary path."""
    paths = [
        "ollama",
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]

    for path in paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, timeout=2)
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def clean_with_ollama(chunk, model="llama3.2"):
    """
    Send a chunk to Ollama for cleaning using CLI.
    """
    prompt = (
        """You are a markdown content cleaner. Your task is to clean and improve markdown content by:

1. Identifying headers/headings that aren't properly marked with # symbols and marking them correctly
2. Removing or minimizing table of contents sections (they add clutter)
3. Removing advertisements, promotional content, and calls-to-action
4. Removing low information density text (fluff, unnecessary preambles, etc.)
5. Preserving all meaningful technical content, code blocks, and substantive information

Return ONLY the cleaned markdown content. Do not add explanations or comments about what you changed.

Content to clean:

"""
        + chunk
    )

    ollama_path = get_ollama_path()
    if not ollama_path:
        print(
            "Error: Could not find Ollama. Make sure it is installed.", file=sys.stderr
        )
        sys.exit(1)

    try:
        result = subprocess.run(
            [ollama_path, "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(
                f"Error: Ollama returned error code {result.returncode}",
                file=sys.stderr,
            )
            if result.stderr:
                print(f"Error message: {result.stderr}", file=sys.stderr)
            return chunk

    except subprocess.TimeoutExpired:
        print("Error: Ollama request timed out", file=sys.stderr)
        return chunk
    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        return chunk


def clean_markdown(
    file_path, model="llama3.2", chunk_size=4000, repetition_threshold=3
):
    """
    Main function to clean a markdown file.
    """
    print(f"Reading file: {file_path}")
    content = read_markdown_file(file_path)

    print(f"Original file size: {len(content)} characters")

    # Step 1: Detect and remove repetitive content
    print("Detecting repetitive content...")
    repetitive_lines = detect_repetitive_lines(content, threshold=repetition_threshold)
    if repetitive_lines:
        print(f"Found {len(repetitive_lines)} repetitive lines")
        content = remove_repetitive_content(content, repetitive_lines)
        print(f"After removing repetitions: {len(content)} characters")
    else:
        print("No repetitive content detected")

    # Step 2: Clean with Ollama in chunks
    print(f"\nSplitting content into chunks (max {chunk_size} chars per chunk)...")
    chunks = chunk_content(content, chunk_size=chunk_size)
    print(f"Processing {len(chunks)} chunks with Ollama model: {model}")

    cleaned_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}... ", end="", flush=True)
        cleaned = clean_with_ollama(chunk, model=model)
        cleaned_chunks.append(cleaned)
        print("done")

    # Combine chunks
    cleaned_content = "\n\n".join(cleaned_chunks)

    print(f"\nFinal cleaned size: {len(cleaned_content)} characters")
    print(
        f"Reduction: {len(content) - len(cleaned_content)} characters ({100 * (1 - len(cleaned_content) / len(content)):.1f}%)"
    )

    return cleaned_content


def main():
    parser = argparse.ArgumentParser(
        description="Clean markdown files using Ollama to remove low-quality content"
    )
    parser.add_argument(
        "file_path", type=str, help="Absolute path to the markdown file to clean"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=4000,
        help="Maximum chunk size in characters (default: 4000)",
    )
    parser.add_argument(
        "-r",
        "--repetition-threshold",
        type=int,
        default=3,
        help="Number of repetitions before a line is considered repetitive (default: 3)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.file_path)
    if not input_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: Not a file: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    # Read original content and calculate hash
    original_content = read_markdown_file(args.file_path)
    original_hash = calculate_hash(original_content)
    print(f"Original file hash: {original_hash[:16]}...")

    # Clean the markdown
    cleaned_content = clean_markdown(
        args.file_path,
        model=args.model,
        chunk_size=args.chunk_size,
        repetition_threshold=args.repetition_threshold,
    )

    # Calculate hash of cleaned content
    cleaned_hash = calculate_hash(cleaned_content)
    print(f"Cleaned file hash:  {cleaned_hash[:16]}...")

    # Compare hashes
    if original_hash == cleaned_hash:
        print("\nNo changes detected. File remains unchanged.")
        return

    print("\nContent has changed. Replacing original file...")

    # Write to temporary file first
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".md", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(cleaned_content)

        # Replace original file with temp file
        shutil.move(temp_path, args.file_path)
        print(f"Successfully updated: {args.file_path}")
        print("Done!")

    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        # Clean up temp file if it exists
        if "temp_path" in locals() and Path(temp_path).exists():
            Path(temp_path).unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
