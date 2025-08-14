"""Document converter using markitdown library."""

import glob
import os
import re
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from .utils import get_file_info


class DocumentConverter:
    """Simple wrapper around markitdown for document conversion."""

    def __init__(self, markitdown: MarkItDown):
        self.markitdown = markitdown
        self.supported_extensions = {
            ".docx",
            ".pdf",
            ".pptx",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".txt",
            ".md",
            ".csv",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".mp3",
            ".wav",
            ".m4a",
        }

    def convert(self, file_path: str, output_format: str = "markdown") -> dict[str, Any]:
        """
        Convert a document using markitdown.

        Args:
            file_path: Path to input document
            output_format: Output format (currently only markdown supported)

        Returns:
            Conversion results
        """
        # Convert using markitdown
        result = self.markitdown.convert(file_path)

        # Get file info
        file_info = get_file_info(file_path)

        return {
            "content": result.text_content,
            "source_file": file_path,
            "format": output_format,
            "word_count": len(result.text_content.split()),
            "file_info": file_info,
            "conversion_successful": True,
        }

    def convert_file(
        self,
        file_path: str,
        optimize_transcript: bool = False,
        extract_metadata: bool = False,
        transcript_processor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Convert a single file with the same output format as batch_convert.

        Args:
            file_path: Path to the input document
            optimize_transcript: Apply transcript optimizations
            extract_metadata: Extract additional metadata from documents
            transcript_processor: Optional transcript processor for optimizations

        Returns:
            Single file conversion result in the same format as batch_convert
        """
        try:
            # Convert the file
            result = self.convert(file_path)

            # Apply transcript optimizations if requested
            if optimize_transcript and transcript_processor:
                result["content"] = transcript_processor.optimize(result["content"])
                result["transcript_metadata"] = transcript_processor.extract_metadata(result["content"])

            # Return in same format as batch_convert
            return {
                "content": [
                    {
                        "type": "document",
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "text": result["content"],
                        "status": "success",
                    }
                ],
                "meta": {"total_files": 1, "successful": 1, "failed": 0},
            }
        except Exception:
            return {
                "content": [
                    {
                        "type": "document",
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "text": "",
                        "status": "failed",
                    }
                ],
                "meta": {"total_files": 1, "successful": 0, "failed": 1},
            }

    def batch_convert(
        self,
        directory: str,
        pattern: str = "*",
        output_format: str = "markdown",
        optimize_transcript: bool = False,
        extract_metadata: bool = False,
        recursive: bool = False,
        transcript_processor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Convert multiple documents in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern to match
            output_format: Output format
            optimize_transcript: Apply transcript optimizations
            extract_metadata: Extract additional metadata from documents
            recursive: Search subdirectories
            transcript_processor: Optional transcript processor for optimizations

        Returns:
            Batch conversion results with content included
        """
        # Find matching files
        search_pattern = os.path.join(directory, "**" if recursive else "", pattern)
        files = glob.glob(search_pattern, recursive=recursive)

        # Filter to supported formats
        files = [f for f in files if Path(f).suffix.lower() in self.supported_extensions]

        results = []
        successful = 0
        failed = 0

        for file_path in files:
            try:
                result = self.convert(file_path, output_format)

                # Apply transcript optimizations if requested
                if optimize_transcript and transcript_processor:
                    result["content"] = transcript_processor.optimize(result["content"])
                    result["transcript_metadata"] = transcript_processor.extract_metadata(result["content"])

                results.append(
                    {
                        "type": "document",
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "text": result["content"],
                        "status": "success",
                    }
                )
                successful += 1
            except Exception:
                results.append(
                    {
                        "type": "document",
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "text": "",
                        "status": "failed",
                    }
                )
                failed += 1

        return {"content": results, "meta": {"total_files": len(files), "successful": successful, "failed": failed}}

    def strip_images(self, content: str) -> str:
        """Remove image references from markdown content."""
        # Remove image markdown syntax
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
        # Remove HTML img tags
        content = re.sub(r"<img[^>]*>", "", content)
        return content
