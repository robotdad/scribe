"""Document converter using markitdown library."""

import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import glob

from markitdown import MarkItDown
from .utils import get_file_info


class DocumentConverter:
    """Simple wrapper around markitdown for document conversion."""
    
    def __init__(self, markitdown: MarkItDown):
        self.markitdown = markitdown
        self.supported_extensions = {
            '.docx', '.pdf', '.pptx', '.xlsx', '.xls', 
            '.html', '.htm', '.txt', '.md', '.csv',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp',
            '.mp3', '.wav', '.m4a'
        }
    
    def convert(self, file_path: str, output_format: str = "markdown") -> Dict[str, Any]:
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
            "conversion_successful": True
        }
    
    def batch_convert(
        self, 
        directory: str, 
        pattern: str = "*", 
        output_format: str = "markdown",
        optimize_transcript: bool = False,
        recursive: bool = False,
        transcript_processor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Convert multiple documents in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            output_format: Output format
            optimize_transcript: Apply transcript optimizations
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
                
                results.append({
                    "source": file_path,
                    "status": "success",
                    "content": result["content"],
                    "word_count": result["word_count"],
                    "file_info": result["file_info"],
                    "transcript_metadata": result.get("transcript_metadata", {})
                })
                successful += 1
            except Exception as e:
                results.append({
                    "source": file_path,
                    "status": "failed",
                    "error": str(e)
                })
                failed += 1
        
        return {
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def strip_images(self, content: str) -> str:
        """Remove image references from markdown content."""
        # Remove image markdown syntax
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # Remove HTML img tags
        content = re.sub(r'<img[^>]*>', '', content)
        return content