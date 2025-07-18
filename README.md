# Scribe MCP Server

A general-purpose MCP (Model Context Protocol) server for document format conversion, with special optimizations for meeting transcripts.

## Features

- **Document Conversion**: Convert DOCX, PDF, PPTX, Excel, images, and audio files to Markdown
- **Transcript Optimization**: Special processing for meeting transcripts with speaker detection and formatting
- **Batch Processing**: Convert multiple documents at once
- **Recipe Integration**: Seamless integration with recipe-tool workflows
- **Security**: Path validation and safe file handling

## Installation

### Using uvx (Recommended)

```bash
# Run directly from GitHub
uvx --from git+https://github.com/robotdad/scribe.git scribe

# Or install locally for development
git clone https://github.com/robotdad/scribe.git
cd scribe
make install
```

### Using uv

```bash
# Install from the current directory
uv pip install -e .

# Or install dependencies manually
uv pip install mcp "markitdown[all]" pydantic
```

## Usage

### As MCP Server

```bash
# Start the server (if installed locally)
scribe

# Or run directly from GitHub
uvx --from git+https://github.com/robotdad/scribe.git scribe

# Or run from local installation
python -m scribe
```

### With Recipe Tool

```json
{
  "type": "mcp",
  "config": {
    "server": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/robotdad/scribe.git", "scribe"]
    },
    "tool_name": "convert_document",
    "arguments": {
      "file_path": "meeting_notes.docx",
      "optimize_transcript": true
    },
    "result_key": "converted_doc"
  }
}
```

## Tools

### `convert_document`

Convert a single document to markdown.

**Arguments:**
- `file_path` (str): Path to the input document
- `output_format` (str): Output format (default: "markdown")
- `optimize_transcript` (bool): Apply transcript optimizations (default: false)
- `strip_images` (bool): Remove image references (default: false)

**Returns:**
- `content`: Converted document content
- `transcript_metadata`: Speaker and timestamp data (if optimize_transcript=true)
- `file_info`: File metadata
- `word_count`: Number of words
- `conversion_successful`: Success status

### `batch_convert`

Convert multiple documents in a directory.

**Arguments:**
- `directory` (str): Directory containing documents
- `pattern` (str): File pattern to match (default: "*")
- `output_format` (str): Output format (default: "markdown")
- `optimize_transcript` (bool): Apply transcript optimizations (default: false)
- `recursive` (bool): Search subdirectories (default: false)

**Returns:**
- `total_files`: Number of files found
- `successful`: Number of successful conversions
- `failed`: Number of failed conversions
- `results`: Array of conversion results with content

### `get_document_info`

Get metadata about a document without converting it.

**Arguments:**
- `file_path` (str): Path to the document

**Returns:**
- File metadata including size, type, and modification date

## Supported Formats

- **Documents**: DOCX, PDF, PPTX, HTML, TXT, MD
- **Spreadsheets**: XLSX, XLS, CSV
- **Images**: JPG, PNG, GIF, BMP
- **Audio**: MP3, WAV, M4A
- **Other**: ZIP archives, JSON, XML

*Note: Actual format support depends on markitdown's capabilities and optional dependencies.*

## Development

```bash
# Install development dependencies
make install

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Check project health
make doctor

# Build AI context files
make ai-context-files
```

## Architecture

Built with:
- **FastMCP**: Modern MCP server implementation
- **markitdown**: Microsoft's document conversion library
- **Pydantic**: Data validation and settings management

## License

MIT