"""Main entry point for Scribe MCP server."""

from .server import mcp


def main() -> None:
    """Entry point for the Scribe MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
