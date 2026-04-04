# Project Progress

## What Has Been Built

### Grand Migration
The original codebase was a collection of competing processor implementations in a `processors/` folder. Everything was migrated and consolidated into a clean, single-agent architecture.

---

## Directory Structure (current state)

```
app/
  core/          OOP foundation — base classes, factories, state schema
  nodes/         LangGraph nodes
  tools/         Agent-callable tools
  workflows/     Compiled LangGraph graph factories
  chains/        Legacy subgraphs (pre-migration, kept for reference)
  skills/        Higher-level capabilities (roleplay, RAG, novel processing)
  main.py        App entry point — lifecycle, wiring

helpers/         Universal reusable utilities (LLM client, epub reader, chapter parser)

utils/
  astream.py     FastAPI server — Discord bridge, SSE streaming endpoints
  queue/         Redis task queue + worker

infra/           Redis-powered streaming infrastructure (see below)

db/
  base_adapter.py
  neo4j_adapter.py
  postgres_adapter.py
  qdrant_adapter.py
  database_adapter.py  (smart coordinator with routing)

config.py        All env-var config in one AppConfig dataclass
models.py        Shared data models (TimelineEvent, CharacterData, etc.)
docker-compose.yml
```

---

## Component Breakdown

### `app/core/` — OOP Foundation

| File | What it does |
|------|-------------|
| `state.py` | `AgentState` TypedDict with proper LangGraph reducers. Custom `_append` (caps at 200) and `_replace` reducers. `RunConfig` frozen dataclass carrying task_id, stream_channel, per-request settings. `make_initial_state()` factory. |
| `tool_base.py` | `BaseTool` ABC. `ToolInput(BaseModel)` with `extra="forbid"` to catch LLM hallucinations. Abstract `execute()`. `safe_execute()` validates via Pydantic then calls execute, returns JSON error string on failure. `to_spec()` for system prompt generation. |
| `tool_factory.py` | `ToolFactory` class registry. `@register` decorator, `create()`, `build_all(**deps)`, `get_all_specs()`, `build_tool_docs()` (human-readable catalogue for system prompt). |
| `node_base.py` | `BaseNode` ABC with `node_id: ClassVar[str]`, abstract async `run()`, `__call__` shim for LangGraph registration. |
| `node_factory.py` | `NodeFactory` — mirrors ToolFactory pattern for nodes. |
| `graph_builder.py` | Fluent `GraphBuilder` wrapping LangGraph's `StateGraph`. Chainable: `add_node()`, `add_nodes()`, `set_entry()`, `add_edge()`, `add_conditional_edges()`, `add_terminal_edge()`, `compile(checkpointer=None)`. |

---

### `app/nodes/` — LangGraph Nodes

| File | Node ID | What it does |
|------|---------|-------------|
| `chat_node.py` | `"chat"` | Primary LLM reasoning node. Builds system prompt with tool catalogue, calls LLM, parses `TOOL_CALL: {...}` and `FINAL ANSWER:` patterns. Registered with `@NodeFactory.register`. |
| `tool_node.py` | `"tool"` | Dispatches tool calls from `state["tool_calls"]` to `ToolFactory`. Lazy-instantiates tools per run. Returns `ToolMessage` to message history. |
| `end_node.py` | `"end"` | Terminal node. Extracts final reply from last `AIMessage`, publishes to Redis stream channel via `StreamManager`, sets `is_complete=True`. |

**ReAct loop topology:**
```
START → chat ──(tool_call)──→ tool ──→ chat   (loop)
             ──(is_complete)──→ end → END
```

---

### `app/tools/` — Concrete Tools

All registered via `@ToolFactory.register`. All use `BaseTool` + `ToolInput(BaseModel)`.

| File | Tool name(s) | What it does |
|------|-------------|-------------|
| `send_message.py` | `send_message` | Publishes a message to the task's Redis stream channel. Used by agent to talk to the user mid-conversation. |
| `end_tool.py` | `end` | Signals the agent is ready to emit `FINAL ANSWER`. |
| `file_tools.py` | `read_file`, `grep_file` | Read local files and grep with ripgrep/grep. Sandboxed to project root. |
| `neo4j_tools.py` | `upload_neo4j`, `run_cypher` | MERGE a node into Neo4j; execute Cypher queries (read-only guard by default). |
| `rag_tool.py` | `semantic_search` | Vector search against Qdrant. Falls back to substring search if embedding not available. |
| `skill_tools.py` | `search_skill`, `load_skill` | Discover skills in `app/skills/` at runtime; load a skill's prompt into the conversation. |

Legacy tools in `event_tools.py`, `character_tools.py`, `reader_tools.py`, `causality_tools.py` are pre-migration bare functions — **not yet migrated to BaseTool**.

---

### `app/workflows/` — Compiled Graph

| File | What it does |
|------|-------------|
| `chat_workflow.py` | `create_chat_workflow(llm, db, qdrant, stream_manager, checkpointer)` — uses `GraphBuilder` + `NodeFactory.build_all()` to wire the full ReAct chat graph. Imports all node and tool modules to trigger `@register` decorators. |

---

### `utils/queue/` — Redis Task Queue

| File | What it does |
|------|-------------|
| `redis_queue.py` | `RedisTaskQueue` — BLPOP-based async task queue. `enqueue()` returns `task_id`. `dequeue()` blocks up to 5s. `iter_tasks()` yields forever. |
| `stream_manager.py` | `StreamManager` — thin async Redis pub/sub wrapper. `publish(channel, token)`, `subscribe(channel) → AsyncGenerator`. Publishes `__END__` sentinel when done. |
| `worker.py` | `AsyncAgentWorker` — background loop, picks tasks from queue, drives `graph.astream()`, publishes output tokens to Redis, signals `__END__` on completion. |

---

### `infra/` — Streaming Infrastructure (built last session)

The `infra/` module is the universal Redis harness powering all streaming and async execution.

| File | What it does |
|------|-------------|
| `redis_client.py` | `RedisPool` — singleton async connection pool with `publish()`, `publish_end()`, `subscribe()`, BLPOP helpers. Shared across the whole process. Key namespace documented inline. |
| `token_stream.py` | `TokenStreamer` — three publishing modes: (1) `stream_from_llm()` consumes live LLM async-generator, runs a state machine to detect `send_human_message` message content on-the-fly and streams those tokens immediately; also detects `FINAL ANSWER:` and streams everything after it. (2) `stream_string()` word-by-word streams a pre-formed string. (3) `subscribe()` async generator for SSE consumers. |
| `background_exec.py` | `BackgroundExecutor` — wraps any tool coroutine with `asyncio.wait_for(timeout=5s)`. Fast path: returns `(result, False)`. Slow path: promotes to background asyncio task, returns `(BackgroundHandle, True)`. `BackgroundHandle` stores job metadata in Redis hash `bg:job:{job_id}`, result in list `bg:result:{job_id}` (BLPOP-friendly). TTL 1 hour. |
| `harness.py` | `InfraHarness` — top-level wiring object. Owns `RedisPool`, `TokenStreamer`, `BackgroundExecutor`, `RedisTaskQueue`. High-level API: `stream_llm_to_channel()`, `stream_string_to_channel()`, `subscribe()`, `run_tool()`, `await_background_result()`, `enqueue()`. Injected into nodes/tools via `**deps`. |

**Redis key namespace:**
```
stream:{task_id}       pub/sub  user-visible tokens → frontend / Discord
thinking:{task_id}     pub/sub  raw LLM tokens (debug)
bg:job:{job_id}        hash     background job metadata (status, tool_name, task_id)
bg:result:{job_id}     list     single-element; BLPOP to retrieve completed result
agent:tasks            list     BLPOP task queue
```

---

### `utils/astream.py` — FastAPI Server (updated)

- `POST /api/agent/chat` — sync path invokes graph directly; streaming path enqueues task and returns `{task_id, stream_url}`
- `GET /api/agent/stream/{task_id}` — SSE endpoint, subscribes to `stream:{task_id}` channel, forwards tokens as `data: <token>\n\n`
- `POST /api/agent/explore` — topic deep-dive via direct graph invoke
- `POST /api/agent/analyze` — DB query passthrough
- `GET /api/agent/status`, `GET /api/characters`, `GET /api/events`

---

### Supporting Files

| File | What it does |
|------|-------------|
| `config.py` | `AppConfig` dataclass, all fields from env vars. Added `redis_url` field. |
| `helpers/llm.py` | `DeepSeekClient` with `chat()`, `generate()`, `stream()` (real SSE async generator), `roleplay()`, `analyze_character()`. |
| `helpers/epub_reader.py` | `EpubReader` — reads EPUB via stdlib `zipfile` + XML. |
| `helpers/chapter_parser.py` | `ChapterParser` — parses raw text into `VolumeNode`/`ChapterNode` tree. |
| `docker-compose.yml` | neo4j + postgres + qdrant + agent services. |

---

## What Is NOT Yet Done

### Medium priority — legacy tool migration
- `event_tools.py`, `character_tools.py`, `reader_tools.py`, `causality_tools.py` are pre-migration bare async functions — need to be wrapped as `BaseTool` subclasses with `@ToolFactory.register`

### Low priority / future
- LangGraph `MemorySaver` or Redis checkpointer for multi-turn conversation memory
- `create_processing_workflow()` — novel ingestion pipeline using the new node/tool architecture
- Test coverage for `app/core/`, `infra/`, nodes, tools
- `DEEPSEEK_API_KEY` → `config.py` `create_llm_client` call signature mismatch (takes `AppConfig`, old call passes raw key string)

---

## Design Principles Established

1. **One factory per type** — `ToolFactory` and `NodeFactory` are class-level registries; register with `@register` decorator, build with `build_all(**deps)`.
2. **Dependency injection via `**deps`** — every node and tool accepts `**deps` in `__init__`; each pulls what it needs and ignores the rest. No global singletons passed around.
3. **Pydantic at tool boundaries** — `ToolInput(BaseModel)` with `extra="forbid"` ensures the LLM cannot pass unexpected fields; validation failure returns a JSON error string, never crashes the graph.
4. **State reducers prevent unbounded growth** — `_append` capped at 200 items, `_replace` for scalars.
5. **Text-based tool protocol** — `TOOL_CALL: {...}` / `FINAL ANSWER:` keeps the agent decoupled from any provider-specific function-calling API.
6. **Infra is injected, not imported** — `InfraHarness` is the single entry point for all Redis operations; nothing outside `infra/` imports `redis.asyncio` directly.
7. **Background promotion is transparent** — `BackgroundExecutor.run()` returns a `BackgroundHandle` when a tool is slow; callers check `is_background` and decide whether to wait or fire-and-forget.
