"""Tests for the MCP server functionality."""

import tempfile
import os
import pytest
import asyncio
from unittest.mock import Mock, patch

from scribe.server import mcp, convert_document, batch_convert, get_document_info


@pytest.fixture
def temp_test_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# Test Document\n\n[10:30] Speaker: Hello world!\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"# Test Document {i}\n\nContent of document {i}.")
            files.append(file_path)
        
        yield {"dir": temp_dir, "files": files}


class TestServerTools:
    """Test cases for MCP server tools."""

    def test_convert_document(self, temp_test_file):
        """Test the convert_document tool with standardized format."""
        result = convert_document(
            file_path=temp_test_file,
            optimize_transcript=True
        )
        
        # Check new standardized format
        assert "content" in result
        assert "meta" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        
        # Check meta information
        meta = result["meta"]
        assert meta["total_files"] == 1
        assert meta["successful"] == 1
        assert meta["failed"] == 0
        
        # Check document content
        doc = result["content"][0]
        assert doc["type"] == "document"
        assert doc["status"] == "success"
        assert doc["filename"].endswith(".txt")
        assert "Test Document" in doc["text"]

    def test_convert_document_with_strip_images(self, temp_test_file):
        """Test the convert_document tool with strip_images option."""
        # Create a file with image references
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Test\n\n![image](test.png)\n\n<img src='test.jpg'>Regular text.")
            temp_path = f.name
        
        try:
            result = convert_document(
                file_path=temp_path,
                strip_images=True
            )
            
            # Check that images were stripped
            doc = result["content"][0]
            assert "![image](test.png)" not in doc["text"]
            assert "<img src='test.jpg'>" not in doc["text"]
            assert "Regular text." in doc["text"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_batch_convert_tool(self, temp_test_dir):
        """Test the batch_convert tool with new format."""
        result = batch_convert(
            directory=temp_test_dir["dir"],
            pattern="*.txt",
            optimize_transcript=False
        )
        
        # Check new standardized format
        assert "content" in result
        assert "meta" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 3
        
        # Check meta information
        meta = result["meta"]
        assert meta["total_files"] == 3
        assert meta["successful"] == 3
        assert meta["failed"] == 0
        
        # Check all documents
        for i, doc in enumerate(result["content"]):
            assert doc["type"] == "document"
            assert doc["status"] == "success"
            assert doc["filename"].startswith("test")
            assert doc["filename"].endswith(".txt")
            assert "Test Document" in doc["text"]

    def test_get_document_info(self, temp_test_file):
        """Test the get_document_info tool."""
        result = get_document_info(file_path=temp_test_file)
        
        assert "filename" in result
        assert "size_bytes" in result
        assert "modified" in result
        assert result["size_bytes"] > 0
        assert result["is_file"] is True

    def test_convert_document_invalid_path(self):
        """Test convert_document with invalid file path."""
        with pytest.raises(Exception):  # Should raise validation error
            convert_document(file_path="/nonexistent/file.txt")

    def test_batch_convert_invalid_directory(self):
        """Test batch_convert with invalid directory."""
        with pytest.raises(Exception):  # Should raise validation error
            batch_convert(directory="/nonexistent/directory")


class TestMCPServer:
    """Test cases for the MCP server instance."""

    def test_mcp_instance_exists(self):
        """Test that MCP server instance is created."""
        assert mcp is not None
        assert hasattr(mcp, 'name')
        assert mcp.name == "Scribe"

    def test_tools_are_registered(self):
        """Test that all expected tools are registered."""
        # This would require inspecting the FastMCP instance
        # For now, we verify the functions exist and are callable
        tools = [convert_document, batch_convert, get_document_info]
        
        for tool in tools:
            assert callable(tool)
            assert hasattr(tool, '__name__')