#!/usr/bin/env python3
"""Basic usage example for Scribe MCP server."""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_scribe():
    """Test the Scribe MCP server with basic functionality."""
    
    # Create a simple test file
    test_file = "test_document.txt"
    with open(test_file, "w") as f:
        f.write("# Test Document\n\nThis is a test document for Scribe MCP server.\n\n[10:30] Alice: Hello everyone!\n[10:31] Bob: Good morning!")
    
    try:
        # Create server parameters
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "scribe"]
        )
        
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print("Available tools:")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # Test convert_document tool
                print("\n--- Converting document ---")
                result = await session.call_tool("convert_document", {
                    "file_path": test_file,
                    "optimize_transcript": True
                })
                
                print(f"Conversion successful: {result.content}")
                
                # Test get_document_info tool
                print("\n--- Getting document info ---")
                info_result = await session.call_tool("get_document_info", {
                    "file_path": test_file
                })
                
                print(f"Document info: {info_result.content}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    asyncio.run(test_scribe())