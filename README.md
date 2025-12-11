# RAGNAROK - RAG on Native Archives & Raw Offline Knowledge

A fully offline Retrieval-Augmented Generation (RAG) system with multiple interfaces (Web UI, TUI, CLI) powered by local LLMs and vector embeddings.

## Key Features

- **Offline RAG System**: Uses local LLMs (Phi-3, Llama) with vector embeddings for context-aware responses
- **Multiple Interfaces**: 
  - **Web UI** - Modern chat interface with streaming responses
  - **TUI** - Terminal-based interface with scrollable chat history (depends from `libncurses-dev` package on linux, `windows-curses` pip package on windows)
  - **CLI** - Simple command-line mode
- **Document Processing**: Automatic loading and chunking of Markdown, HTML, and TXT files
- **Vector Database**: Chroma-based persistent storage for semantic search
- **Performance Monitoring**: Real-time token generation rate and statistics
- **Streaming Responses**: Progressive token output for interactive experience

## System Architecture

```
magent.py (llmRag)           ← Main RAG agent orchestrator
    ├── LlamaCpp LLM         ← Text generation model
    └── vector.py            ← Retrieval system
        ├── LlamaCppEmbedder ← Embedding model
        └── Chroma DB        ← Vector storage

app.py                       ← Flask backend + routing
    ├── index.html           ← Web UI (dark theme chat)
    ├── tui.py               ← Terminal interface
    └── utils.py             ← Model downloading utility
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Models

Edit `config.json` to specify:
- GGUF LLM model URL (default: Phi-3-mini-4k-instruct-q4.gguf)
- GGUF Embedding model URL (default: bge-small-en-v1.5-q4_k_m.gguf)
- Document and database paths

```json
{
  "docs_path": "docs",
  "db_path": "chromadb",
  "models_path": "models"
}
```

### 3. Add Documents

Place your knowledge base files in the `docs` directory:
- Markdown files (`.md`)
- HTML files (`.html`, `.htm`)
- Text files (`.txt`)

### 4. Run the Agent

**Web UI** (default):
```bash
python app.py
# Open http://localhost:5000
```

**TUI Mode**:
```bash
python app.py --tui
```

**CLI Mode**:
```bash
python app.py --cli
```

## Project Structure

| File | Purpose |
|------|---------|
| `magent.py` | RAG agent orchestrator with streaming |
| `vector.py` | Vector database & embeddings management |
| `app.py` | Flask backend + routing for all modes |
| `tui.py` | Curses-based terminal interface |
| `index.html` | Web UI frontend with real-time stats |
| `config.json` | Model URLs and system configuration |

## Key Components

### RAG Agent (`magent.py`)
- Uses LangChain for orchestration
- Combines vector retrieval + LLM generation
- Tracks token generation statistics
- System prompt enforces document-only answers

### Vector Storage (`vector.py`)
- **LlamaCppEmbedder**: Local embedding model with CPU inference
- **Document Loading**: Supports MD, HTML, TXT with smart splitting
- **Chroma Integration**: Persistent vector storage with semantic search
- Auto-creates database on first run

### Web Interface (`index.html`)
- Dark theme chat UI with streaming support
- Real-time stats: tokens/sec, message count, scroll position
- Copy-to-clipboard for AI responses
- Auto-scrolling during generation

### Terminal UI (`tui.py`)
- Curses-based interactive chat
- Manual scrolling with Page Up/Down
- Color-coded messages (AI vs User)
- Status bar with performance metrics

## Configuration Options

**LLM Settings** (from config.json):
```json
{
  "temperature": 0.2,      // Lower = more deterministic
  "top_p": 0.9,            // Nucleus sampling
  "max_tokens": 1024,      // Max response length
  "n_ctx": 4096,           // Context window size
  "n_threads": auto        // CPU threads (divided by 3)
}
```

**Available Models**:
- **LLM**: Phi-3-mini, Llama-2, Llama-3 (edit `config.json` URLs)
- **Embeddings**: BGE-small-en, Nomic-embed-text (alternatives in config)

## Performance Metrics

The system tracks and displays:
- **TPS** (Tokens Per Second): Generation speed
- **Message Count**: Total exchanges
- **Scroll Height**: Chat content size
- **Generation Status**: Idle / In Progress

## Development

**Key Dependencies**:
- `langchain` - LLM orchestration
- `llama-cpp-python` - Local LLM inference
- `langchain-chroma` - Vector database
- `flask` - Web backend
- `curses` - Terminal UI

**Troubleshooting**:
- If models don't download: Check URLs in `config.json`
- Low performance: Increase `temperature`, reduce `max_tokens`
- Out of memory: Use smaller models or reduce `n_ctx`

## License & Usage

This project implements RAG with fully offline LLMs no external API calls. Generated responses are based solely on provided documents.

**Disclaimer**: Generated content may contain inaccuracies. Always verify critical information against source documents.