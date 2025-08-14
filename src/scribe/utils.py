"""Minimal utilities for Scribe MCP server."""

from pathlib import Path
from typing import Any


def validate_file_path(file_path: str) -> None:
    """
    Validate file path for security and existence.

    Args:
        file_path: Path to validate

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is invalid or unsafe
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("Invalid file path")

    # Convert to Path object for validation
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Basic security: prevent directory traversal
    if ".." in str(path):
        raise ValueError("Directory traversal not allowed")

    # Check if it's a file or directory (both are valid)
    if not (path.is_file() or path.is_dir()):
        raise ValueError("Path must point to a file or directory")


def get_file_info(file_path: str) -> dict[str, Any]:
    """
    Get basic file information.

    Args:
        file_path: Path to the file

    Returns:
        File metadata
    """
    path = Path(file_path)
    stat = path.stat()

    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": stat.st_mtime,
        "absolute_path": str(path.absolute()),
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
    }
