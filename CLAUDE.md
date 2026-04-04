# Novel Agent — CLAUDE.md

## Project Overview
A single LangGraph ReAct agent that handles Chinese novel Q&A, character roleplay, and novel ingestion into a Neo4j knowledge graph. Discord is the frontend; `utils/astream.py` (FastAPI) is the bridge.

## Structure
```
app/
  main.py          # NovelAgent — core ReAct loop (entry point for all agent calls)
  nodes/           # LangGraph nodes (event_extraction, character_analysis, causality_analysis, chat_response)
  chains/          # Compiled subgraphs (processing_chain: ingest novel; chat_chain: Q&A)
  tools/           # Agent-callable tools (event_tools, character_tools, reader_tools, causality_tools)
  skills/          # Higher-level capabilities (roleplay, rag, novel_processing)
helpers/           # Universal utilities (llm.py, epub_reader.py, chapter_parser.py)
utils/
  astream.py       # FastAPI server — Discord bot calls this; streams responses
db/
  base_adapter.py  # Abstract interface
  neo4j_adapter.py # Neo4j implementation (primary for graph queries)
  postgres_adapter.py  # PostgreSQL (fallback + aggregations)
  qdrant_adapter.py    # Vector search (optional)
  database_adapter.py  # Smart coordinator with query routing
  schema/          # .cypher and .sql schema files
config.py          # All config via env vars — AppConfig dataclass
models.py          # All shared data models
docker-compose.yml # neo4j + postgres + qdrant + agent
Dockerfile
requirements.txt
.env.example       # Template for .env
```

## Running
```bash
# Local dev
uvicorn utils.astream:app --reload --port 5005

# Docker
docker compose up
```

## Environment
Copy `.env.example` to `.env` and fill in:
- `DEEPSEEK_API_KEY`
- `NEO4J_PASSWORD`
- `POSTGRES_PASSWORD`

## Discord Bot
`agentic-discord-bot.js` (iCloud-evicted; needs re-downloading) calls:
- `POST /api/agent/chat` — main endpoint (set `stream=true` for streaming)
- `GET /api/agent/status` — health check
- `POST /api/agent/explore` — topic deep-dive

## Key Design Decisions
- **One agent**: `app/main.py:NovelAgent` handles chat, roleplay, and novel ingestion
- **helpers/ vs utils/**: helpers = reusable utilities; utils = frontend communication only
- **DB routing**: Causality/relationships → Neo4j; stats/text-search → PostgreSQL
- **Streaming**: `NovelAgent.stream_message()` wired to `StreamingResponse` in `astream.py`
