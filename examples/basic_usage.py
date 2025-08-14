#!/usr/bin/env python3
"""Basic usage example for Scribe MCP server."""

import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_scribe():
    """Test the Scribe MCP server with basic functionality."""

    # Create test files for individual and batch testing
    test_file = "test_document.txt"
    test_dir = "test_batch"

    # Create single test file
    with open(test_file, "w") as f:
        f.write(
            "# Test Document\n\nThis is a test document for Scribe MCP server.\n\n[10:30] Alice: Hello everyone!\n[10:31] Bob: Good morning!"
        )

    # Create directory and multiple test files for batch testing
    os.makedirs(test_dir, exist_ok=True)
    batch_files = []

    for i in range(3):
        batch_file = os.path.join(test_dir, f"batch_doc_{i+1}.txt")
        batch_files.append(batch_file)
        with open(batch_file, "w") as f:
            f.write(
                f"# Batch Document {i+1}\n\nContent of document {i+1}.\n\n[11:0{i}] Speaker{i+1}: Message from document {i+1}!"
            )

    try:
        # Create server parameters
        server_params = StdioServerParameters(command="python", args=["-m", "scribe"])

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

                # Test convert_document tool (standardized format)
                print("\n--- Converting single document ---")
                try:
                    doc_result = await session.call_tool(
                        "convert_document", {"file_path": test_file, "optimize_transcript": True}
                    )

                    # Use structured content directly (no JSON parsing needed!)
                    if hasattr(doc_result, "structuredContent") and doc_result.structuredContent:
                        doc_data = doc_result.structuredContent
                        print("  ✅ Using structured content directly!")
                    else:
                        # Fallback to parsing JSON from text content
                        if doc_result.content and hasattr(doc_result.content[0], "text"):
                            doc_data = json.loads(doc_result.content[0].text)
                            print("  ⚠️  Using JSON parsing fallback")
                        else:
                            doc_data = doc_result.content

                    if isinstance(doc_data, dict) and "meta" in doc_data:
                        print(f"  Total files: {doc_data['meta']['total_files']}")
                        print(f"  Successful: {doc_data['meta']['successful']}")
                        print(f"  Content items: {len(doc_data['content'])}")
                        if doc_data["content"]:
                            first_doc = doc_data["content"][0]
                            print(f"  First doc: {first_doc['filename']} ({first_doc['status']})")
                            print(f"  Text preview: {first_doc['text'][:100]}...")
                    else:
                        print(f"  Unexpected format: {doc_data}")

                except Exception as e:
                    print(f"  Error with convert_document: {e}")

                # Test get_document_info tool
                print("\n--- Getting document info ---")
                try:
                    info_result = await session.call_tool("get_document_info", {"file_path": test_file})

                    # Parse MCP response
                    if info_result.content and hasattr(info_result.content[0], "text"):
                        info_data = json.loads(info_result.content[0].text)
                    else:
                        info_data = info_result.content

                    if isinstance(info_data, dict):
                        print(
                            f"Document info: {info_data.get('filename', 'N/A')}, {info_data.get('size_bytes', 0)} bytes"
                        )
                        print(f"File type: {info_data.get('extension', 'N/A')}")
                    else:
                        print(f"  Unexpected format: {info_data}")

                except Exception as e:
                    print(f"  Error with get_document_info: {e}")

                # Test batch_convert tool (new standardized format)
                print("\n--- Batch converting files ---")
                try:
                    batch_result = await session.call_tool(
                        "batch_convert", {"directory": test_dir, "pattern": "*.txt", "optimize_transcript": True}
                    )

                    # Use structured content directly (no JSON parsing needed!)
                    if hasattr(batch_result, "structuredContent") and batch_result.structuredContent:
                        batch_data = batch_result.structuredContent
                        print("  ✅ Using structured content directly!")
                    else:
                        # Fallback to parsing JSON from text content
                        if batch_result.content and hasattr(batch_result.content[0], "text"):
                            batch_data = json.loads(batch_result.content[0].text)
                            print("  ⚠️  Using JSON parsing fallback")
                        else:
                            batch_data = batch_result.content

                    if isinstance(batch_data, dict) and "meta" in batch_data:
                        print(f"  Total files: {batch_data['meta']['total_files']}")
                        print(f"  Successful: {batch_data['meta']['successful']}")
                        print(f"  Failed: {batch_data['meta']['failed']}")
                        print(f"  Content items: {len(batch_data['content'])}")

                        # Show details for each document
                        for i, doc in enumerate(batch_data["content"]):
                            print(f"  Doc {i+1}: {doc['filename']} ({doc['status']})")
                            if doc["status"] == "success":
                                print(f"    Text preview: {doc['text'][:50]}...")
                    else:
                        print(f"  Unexpected format: {batch_data}")

                except Exception as e:
                    print(f"  Error with batch_convert: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up test files
        if os.path.exists(test_file):
            os.remove(test_file)

        # Clean up batch test directory
        if os.path.exists(test_dir):
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    os.remove(batch_file)
            os.rmdir(test_dir)


if __name__ == "__main__":
    asyncio.run(test_scribe())
