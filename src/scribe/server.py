"""Scribe MCP Server - Document conversion server using FastMCP."""

import warnings

# Suppress pydub ffmpeg warnings since they're only needed for audio processing
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", module="pydub")

from typing import Any

from markitdown import MarkItDown
from mcp.server.fastmcp import FastMCP

from .converter import DocumentConverter
from .transcript import TranscriptProcessor
from .utils import get_file_info, validate_file_path

# Initialize FastMCP server
mcp = FastMCP("Scribe")
markitdown = MarkItDown()
converter = DocumentConverter(markitdown)
transcript_processor = TranscriptProcessor()


@mcp.tool()
def convert_document(
    file_path: str, 
    output_format: str = "markdown", 
    optimize_transcript: bool = False, 
    extract_metadata: bool = False,
    strip_images: bool = False
) -> dict[str, Any]:
    """
    Convert a document with standardized output format.

    Args:
        file_path: Path to the input document
        output_format: Output format (markdown, text, html)
        optimize_transcript: Apply transcript-specific optimizations
        extract_metadata: Extract additional metadata from documents
        strip_images: Remove image references from output

    Returns:
        Single document conversion result with content array and meta information
    """
    validate_file_path(file_path)

    # Use the same logic as convert_file for consistency
    results = converter.convert_file(
        file_path=file_path,
        optimize_transcript=optimize_transcript,
        extract_metadata=extract_metadata,
        transcript_processor=transcript_processor if optimize_transcript else None
    )

    # Apply strip_images if requested
    if strip_images and results["content"]:
        for doc in results["content"]:
            if doc["status"] == "success":
                doc["text"] = converter.strip_images(doc["text"])

    return results


@mcp.tool()
def batch_convert(
    directory: str,
    pattern: str = "*",
    output_format: str = "markdown",
    optimize_transcript: bool = False,
    extract_metadata: bool = False,
    recursive: bool = False,
) -> dict[str, Any]:
    """
    Convert multiple documents in a directory.

    Args:
        directory: Directory containing documents
        pattern: File pattern to match (e.g., "*.docx", "*.pdf")
        output_format: Output format for all files
        optimize_transcript: Apply transcript optimizations
        extract_metadata: Extract additional metadata from documents
        recursive: Search subdirectories

    Returns:
        Summary of batch conversion results with content included
    """
    validate_file_path(directory)  # Validate directory exists

    results = converter.batch_convert(
        directory=directory,
        pattern=pattern,
        output_format=output_format,
        optimize_transcript=optimize_transcript,
        extract_metadata=extract_metadata,
        recursive=recursive,
        transcript_processor=transcript_processor if optimize_transcript else None,
    )

    return results



@mcp.tool()
def get_document_info(file_path: str) -> dict[str, Any]:
    """
    Get information about a document without converting it.

    Args:
        file_path: Path to the document

    Returns:
        Document metadata and file information
    """
    validate_file_path(file_path)
    return get_file_info(file_path)


if __name__ == "__main__":
    mcp.run()
