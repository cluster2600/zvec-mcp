# zvec-mcp

Claude Code MCP server that gives your AI agent a **local vector database** for RAG knowledge retrieval and long-term memory — powered by [zvec](https://github.com/alibaba/zvec).

## What it does

zvec-mcp runs as a [Model Context Protocol](https://modelcontextprotocol.io/) server over stdio. Once connected, Claude Code gets 11 new tools:

### Knowledge base (RAG)

| Tool | Description |
|------|-------------|
| `knowledge_ingest` | Chunk, embed, and store text for later retrieval |
| `knowledge_ingest_file` | Same, but reads directly from a file path |
| `knowledge_search` | Semantic search over ingested documents |
| `knowledge_delete_source` | Remove all chunks from a given source |
| `knowledge_stats` | Collection statistics |

### Memory

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a fact, preference, or observation |
| `memory_recall` | Semantic recall with optional category filter |
| `memory_forget` | Delete a specific memory by ID |
| `memory_forget_category` | Clear all memories in a category |
| `memory_stats` | Collection statistics |

### Status

| Tool | Description |
|------|-------------|
| `zvec_status` | Server config, embedding backend, and collection stats |

## Quick start

### 1. Install

```bash
# Clone
git clone https://github.com/cluster2600/zvec-mcp.git
cd zvec-mcp

# Create venv and install
uv venv --python 3.10
uv pip install -e .
```

### 2. Register with Claude Code

**Option A — user scope** (available in all projects):

```bash
claude mcp add-json zvec-mcp \
  '{"type":"stdio","command":"'"$(pwd)"'/.venv/bin/zvec-mcp","args":[],"env":{"ZVEC_MCP_DATA_DIR":"'"$HOME"'/.zvec-mcp","ZVEC_MCP_EMBEDDING":"local"}}' \
  --scope user
```

**Option B — project scope** (shared via `.mcp.json`):

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "zvec-mcp": {
      "command": "/path/to/zvec-mcp/.venv/bin/zvec-mcp",
      "args": [],
      "env": {
        "ZVEC_MCP_DATA_DIR": "${HOME}/.zvec-mcp",
        "ZVEC_MCP_EMBEDDING": "local"
      }
    }
  }
}
```

### 3. Verify

```bash
claude mcp list
# zvec-mcp: /path/to/.venv/bin/zvec-mcp  - ✓ Connected
```

## Configuration

All settings are driven by environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ZVEC_MCP_DATA_DIR` | `~/.zvec-mcp` | Root directory for zvec collections |
| `ZVEC_MCP_EMBEDDING` | `local` | Embedding backend: `local` or `openai` |
| `ZVEC_MCP_CHUNK_SIZE` | `512` | Characters per chunk for knowledge ingestion |
| `ZVEC_MCP_CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `OPENAI_API_KEY` | — | Required when `ZVEC_MCP_EMBEDDING=openai` |
| `ZVEC_MCP_OPENAI_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ZVEC_MCP_OPENAI_DIM` | `1536` | OpenAI embedding dimension |

### Embedding backends

**Local (default)** — uses `all-MiniLM-L6-v2` from sentence-transformers. 384 dimensions, runs entirely offline. Uses MPS acceleration on Apple Silicon.

**OpenAI** — uses `text-embedding-3-small` (or whichever model you configure). Requires an API key. Set:

```bash
ZVEC_MCP_EMBEDDING=openai
OPENAI_API_KEY=sk-...
```

## Architecture

```
┌─────────────┐    stdio/JSON-RPC    ┌──────────────────────────┐
│ Claude Code  │◄───────────────────►│  zvec-mcp (FastMCP)      │
│   (client)   │                     │                          │
└─────────────┘                     │  ┌──────────┐            │
                                     │  │ knowledge │ chunk →   │
                                     │  │ manager   │ embed →   │
                                     │  │           │ store     │
                                     │  └─────┬────┘            │
                                     │        │                 │
                                     │  ┌─────▼────┐            │
                                     │  │   zvec    │ vector    │
                                     │  │collection │ search    │
                                     │  └─────┬────┘            │
                                     │        │                 │
                                     │  ┌─────▼────┐            │
                                     │  │  memory   │ remember  │
                                     │  │  manager  │ recall    │
                                     │  └──────────┘            │
                                     │                          │
                                     │  ┌──────────┐            │
                                     │  │embeddings│ local /   │
                                     │  │ (lazy)   │ openai    │
                                     │  └──────────┘            │
                                     └──────────────────────────┘
```

### Source layout

```
src/zvec_mcp/
├── server.py       # FastMCP server — 11 tools registered via @mcp.tool()
├── config.py       # Env-driven dataclass config
├── embeddings.py   # Lazy-loaded embedding singleton (sentence-transformers or OpenAI)
├── knowledge.py    # RAG pipeline: chunk → embed → upsert → query
└── memory.py       # Semantic memory: remember → recall → forget
```

### How RAG ingestion works

1. Text is split into overlapping chunks at sentence boundaries (configurable size/overlap)
2. Each chunk is embedded via the configured backend
3. Chunks are stored in a zvec `Collection` with fields: `source`, `chunk_idx`, `text`, `created_at`
4. Chunk IDs are content-addressed (`sha256(source:idx)`) so re-ingesting the same source updates in place

### How memory works

1. Each fact/observation is embedded and stored with a category tag
2. Memory IDs are content-addressed (`sha256(text)`) — storing the same text twice is a no-op
3. Recall uses cosine similarity with optional category filtering
4. Categories let you organize memories (e.g. `preference`, `project`, `person`, `decision`)

## Data storage

All data lives under `ZVEC_MCP_DATA_DIR` (default `~/.zvec-mcp/`):

```
~/.zvec-mcp/
├── knowledge/    # zvec collection for RAG chunks
└── memory/       # zvec collection for memories
```

Collections are created automatically on first use. To reset, delete the directory.

## Dependencies

- [zvec](https://github.com/alibaba/zvec) — vector database engine
- [mcp](https://pypi.org/project/mcp/) — Model Context Protocol SDK (FastMCP)
- [sentence-transformers](https://www.sbert.net/) — local embeddings (default)
- [openai](https://pypi.org/project/openai/) — optional, for OpenAI embeddings

## License

Apache-2.0
