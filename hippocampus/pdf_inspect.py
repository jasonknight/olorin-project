#!/usr/bin/env python3
"""
Simple PDF inspection tool.
Prints the content of each page to stdout for testing PDF parsing.
Supports multiple extraction methods including OCR via macOS Vision framework.
"""

import sys
import argparse

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF", file=sys.stderr)
    sys.exit(1)


def extract_with_vision(page) -> str:
    """
    Use macOS Vision framework for OCR.
    This is what Preview and other macOS apps use.
    """
    try:
        import Vision
        from Foundation import NSData
    except ImportError:
        return None

    # Render page to image
    pix = page.get_pixmap(dpi=150)
    img_data = pix.tobytes("png")

    # Create NSData from image bytes
    ns_data = NSData.dataWithBytes_length_(img_data, len(img_data))

    # Create Vision request handler
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)

    # Create text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)

    # Perform request
    success, error = handler.performRequests_error_([request], None)

    if not success:
        return None

    # Extract text from results
    results = request.results()
    if not results:
        return ""

    lines = []
    for observation in results:
        candidates = observation.topCandidates_(1)
        if candidates:
            lines.append(candidates[0].string())

    return "\n".join(lines)


def extract_text_methods(page) -> dict:
    """Try multiple extraction methods and return results."""
    results = {}

    # Standard text extraction
    results["text"] = page.get_text("text")

    # Try with different flags
    results["text_dehyphenate"] = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE)

    # Block-based extraction
    blocks = page.get_text("blocks")
    block_text = "\n".join(b[4] for b in blocks if b[6] == 0)  # type 0 = text
    results["blocks"] = block_text

    # Dictionary extraction (more detailed)
    text_dict = page.get_text("dict")
    dict_text = []
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    dict_text.append(span.get("text", ""))
    results["dict"] = " ".join(dict_text)

    # Raw text (includes more characters)
    results["rawdict"] = page.get_text("rawdict")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PDF content using various extraction methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection (tries standard extraction first)
  python pdf_inspect.py document.pdf

  # Force OCR using macOS Vision framework
  python pdf_inspect.py document.pdf --ocr

  # Show what each extraction method finds (for debugging)
  python pdf_inspect.py document.pdf --debug
        """,
    )
    parser.add_argument("pdf_file", help="Path to PDF file")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use macOS Vision framework for OCR (best for image-based PDFs)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show results from all extraction methods",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for OCR rendering (default: 150)",
    )

    args = parser.parse_args()

    try:
        doc = fitz.open(args.pdf_file)
    except Exception as e:
        print(f"Error opening PDF: {e}", file=sys.stderr)
        sys.exit(1)

    total_pages = len(doc)
    print(f"PDF: {args.pdf_file}")
    print(f"Total pages: {total_pages}")

    # Show metadata
    metadata = doc.metadata
    if metadata:
        if metadata.get("title"):
            print(f"Title: {metadata['title']}")
        if metadata.get("author"):
            print(f"Author: {metadata['author']}")

    print("=" * 60)

    for page_num in range(total_pages):
        page = doc[page_num]

        print(f"\n--- Page {page_num + 1} of {total_pages} ---\n")

        if args.debug:
            # Show all extraction methods
            results = extract_text_methods(page)
            for method, text in results.items():
                if method == "rawdict":
                    continue  # Skip rawdict in output (too verbose)
                char_count = len(text.strip()) if isinstance(text, str) else 0
                print(f"[{method}] ({char_count} chars):")
                if char_count > 0:
                    preview = text.strip()[:200]
                    print(f"  {preview}...")
                else:
                    print("  (empty)")
                print()

            # Also try Vision OCR
            print("[vision_ocr]:")
            ocr_text = extract_with_vision(page)
            if ocr_text is None:
                print("  (Vision framework not available - need pyobjc)")
            elif ocr_text:
                print(f"  ({len(ocr_text)} chars)")
                print(f"  {ocr_text[:200]}...")
            else:
                print("  (empty)")

        elif args.ocr:
            # Use Vision OCR
            text = extract_with_vision(page)
            if text is None:
                print(
                    "Vision framework not available. Install with:",
                    file=sys.stderr,
                )
                print("  pip install pyobjc-framework-Vision", file=sys.stderr)
                sys.exit(1)
            print(text)

        else:
            # Try standard extraction first
            text = page.get_text("text")

            # If empty, try blocks
            if not text.strip():
                blocks = page.get_text("blocks")
                text = "\n".join(b[4] for b in blocks if b[6] == 0)

            # If still empty, suggest OCR
            if not text.strip():
                print("(No text extracted - try --ocr for image-based PDFs)")
            else:
                print(text)

    doc.close()


if __name__ == "__main__":
    main()
