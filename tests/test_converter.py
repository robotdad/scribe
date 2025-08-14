"""Tests for the document converter functionality."""

import tempfile
import os
from pathlib import Path
import pytest
from markitdown import MarkItDown

from scribe.converter import DocumentConverter


@pytest.fixture
def converter():
    """Create a DocumentConverter instance for testing."""
    markitdown = MarkItDown()
    return DocumentConverter(markitdown)


@pytest.fixture
def temp_files():
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Single test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("# Test Document\n\n[10:30] Alice: Hello everyone!\n[10:31] Bob: Good morning!")
        
        # Multiple test files
        test_file2 = os.path.join(temp_dir, "test2.txt")
        with open(test_file2, "w") as f:
            f.write("# Another Document\n\nThis is another test document.")
            
        test_file3 = os.path.join(temp_dir, "test3.txt")
        with open(test_file3, "w") as f:
            f.write("# Third Document\n\n[14:00] Charlie: Meeting started.")
        
        yield {
            "dir": temp_dir,
            "files": [test_file, test_file2, test_file3]
        }


class TestDocumentConverter:
    """Test cases for DocumentConverter class."""

    def test_convert_file_success(self, converter, temp_files):
        """Test convert_file with successful conversion."""
        result = converter.convert_file(
            file_path=temp_files["files"][0],
            optimize_transcript=False
        )
        
        # Check structure
        assert "content" in result
        assert "meta" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        
        # Check meta
        meta = result["meta"]
        assert meta["total_files"] == 1
        assert meta["successful"] == 1
        assert meta["failed"] == 0
        
        # Check content
        doc = result["content"][0]
        assert doc["type"] == "document"
        assert doc["filename"] == "test.txt"
        assert doc["status"] == "success"
        assert "Test Document" in doc["text"]
        assert "Alice: Hello everyone!" in doc["text"]

    def test_convert_file_with_transcript_optimization(self, converter, temp_files):
        """Test convert_file with transcript optimization."""
        from scribe.transcript import TranscriptProcessor
        transcript_processor = TranscriptProcessor()
        
        result = converter.convert_file(
            file_path=temp_files["files"][0],
            optimize_transcript=True,
            transcript_processor=transcript_processor
        )
        
        assert result["meta"]["successful"] == 1
        assert result["content"][0]["status"] == "success"

    def test_batch_convert_success(self, converter, temp_files):
        """Test batch_convert with successful conversions."""
        result = converter.batch_convert(
            directory=temp_files["dir"],
            pattern="*.txt",
            optimize_transcript=False
        )
        
        # Check structure
        assert "content" in result
        assert "meta" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 3
        
        # Check meta
        meta = result["meta"]
        assert meta["total_files"] == 3
        assert meta["successful"] == 3
        assert meta["failed"] == 0
        
        # Check all docs have correct structure
        for doc in result["content"]:
            assert doc["type"] == "document"
            assert doc["status"] == "success"
            assert "filename" in doc
            assert "text" in doc
            assert "source" in doc

    def test_batch_convert_with_pattern(self, converter, temp_files):
        """Test batch_convert with specific file pattern."""
        # Create a non-matching file
        other_file = os.path.join(temp_files["dir"], "other.md")
        with open(other_file, "w") as f:
            f.write("# Markdown file")
        
        result = converter.batch_convert(
            directory=temp_files["dir"],
            pattern="*.txt",  # Should only match .txt files
            optimize_transcript=False
        )
        
        # Should still find 3 .txt files, not the .md file
        assert result["meta"]["total_files"] == 3
        assert all(doc["filename"].endswith(".txt") for doc in result["content"])

    def test_convert_file_nonexistent_file(self, converter):
        """Test convert_file with non-existent file."""
        result = converter.convert_file(
            file_path="/nonexistent/file.txt",
            optimize_transcript=False
        )
        
        # Should return failure result
        assert result["meta"]["total_files"] == 1
        assert result["meta"]["successful"] == 0
        assert result["meta"]["failed"] == 1
        assert result["content"][0]["status"] == "failed"
        assert result["content"][0]["text"] == ""

    def test_strip_images(self, converter):
        """Test image stripping functionality."""
        content_with_images = """
        # Document
        
        Here is an image: ![alt text](image.png)
        
        And another: <img src="test.jpg" alt="test">
        
        Regular text.
        """
        
        stripped = converter.strip_images(content_with_images)
        
        assert "![alt text](image.png)" not in stripped
        assert '<img src="test.jpg" alt="test">' not in stripped
        assert "Regular text." in stripped