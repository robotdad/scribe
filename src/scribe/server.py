"""Scribe MCP Server - Document conversion server using FastMCP."""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from markitdown import MarkItDown

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
    strip_images: bool = False
) -> Dict[str, Any]:
    """
    Convert a document to markdown or other text formats.
    
    Args:
        file_path: Path to the input document
        output_format: Output format (markdown, text, html)
        optimize_transcript: Apply transcript-specific optimizations
        strip_images: Remove image references from output
    
    Returns:
        Conversion results with content and metadata
    """
    validate_file_path(file_path)
    
    # Convert using markitdown
    result = converter.convert(file_path, output_format)
    
    # Apply transcript optimizations if requested
    if optimize_transcript:
        result["content"] = transcript_processor.optimize(result["content"])
        result["transcript_metadata"] = transcript_processor.extract_metadata(result["content"])
    
    # Strip images if requested
    if strip_images:
        result["content"] = converter.strip_images(result["content"])
    
    return result


@mcp.tool()
def batch_convert(
    directory: str,
    pattern: str = "*",
    output_format: str = "markdown",
    optimize_transcript: bool = False,
    recursive: bool = False
) -> Dict[str, Any]:
    """
    Convert multiple documents in a directory.
    
    Args:
        directory: Directory containing documents
        pattern: File pattern to match (e.g., "*.docx", "*.pdf")
        output_format: Output format for all files
        optimize_transcript: Apply transcript optimizations
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
        recursive=recursive,
        transcript_processor=transcript_processor if optimize_transcript else None
    )
    
    return results


@mcp.tool()
def get_document_info(file_path: str) -> Dict[str, Any]:
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