#!/usr/bin/env bash
# start.sh — Start all services for local development
# Usage: ./start.sh [--no-docker]  (--no-docker skips docker compose if DBs already running)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NO_DOCKER=false
for arg in "$@"; do [[ "$arg" == "--no-docker" ]] && NO_DOCKER=true; done

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Prerequisites ─────────────────────────────────────────────────────────────
[[ -f .env ]] || error ".env not found — copy .env.example to .env and fill in your keys"

source .env

[[ -z "${DEEPSEEK_API_KEY:-}" ]] && error "DEEPSEEK_API_KEY is not set in .env"
[[ -z "${DISCORD_TOKEN:-}" ]]    && error "DISCORD_TOKEN is not set in .env"

command -v python3 &>/dev/null || error "python3 not found"
command -v node    &>/dev/null || error "node not found"
command -v npm     &>/dev/null || error "npm not found"

# ── Python venv ───────────────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    info "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

info "Checking Python dependencies..."
if ! python3 -c "import fastapi, langgraph, aiohttp" 2>/dev/null; then
    warn "Installing Python packages..."
    pip install -q -r requirements.txt
fi

info "Checking Node dependencies..."
if [[ ! -d node_modules ]]; then
    warn "Running npm install..."
    npm install --silent
fi

# ── Docker infrastructure ─────────────────────────────────────────────────────
if [[ "$NO_DOCKER" == false ]]; then
    command -v docker &>/dev/null || error "docker not found (use --no-docker if DBs are running externally)"

    info "Starting infrastructure (neo4j, postgres, redis, qdrant)..."
    # Start only the infra services (not the 'agent' profile container)
    docker compose up -d --remove-orphans neo4j postgres redis qdrant

    info "Waiting for databases to be healthy..."
    MAX_WAIT=120
    ELAPSED=0
    while true; do
        PG_OK=$(docker compose ps postgres --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Health',''))" 2>/dev/null || echo "")
        REDIS_OK=$(docker compose ps redis --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Health',''))" 2>/dev/null || echo "")

        # Simpler check: just wait for postgres and redis to accept connections
        PG_READY=$(docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-novel_user}" -q 2>/dev/null && echo "yes" || echo "no")
        REDIS_READY=$(docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG && echo "yes" || echo "no")

        if [[ "$PG_READY" == "yes" && "$REDIS_READY" == "yes" ]]; then
            info "Databases ready ✓"
            break
        fi

        if [[ $ELAPSED -ge $MAX_WAIT ]]; then
            warn "Timed out waiting for DBs — proceeding anyway (agent will retry internally)"
            break
        fi

        echo -n "."
        sleep 3
        ELAPSED=$((ELAPSED + 3))
    done
fi

# ── Cleanup on exit ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    info "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    info "Done."
}
trap cleanup EXIT INT TERM

# ── FastAPI backend ───────────────────────────────────────────────────────────
info "Starting FastAPI backend on port ${API_PORT:-5005}..."
python3 -m uvicorn utils.astream:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-5005}" \
    --reload \
    --log-level "$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')" \
    2>&1 | sed 's/^/[API] /' &
API_PID=$!
PIDS+=($API_PID)

# Wait for API to be up
info "Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${API_PORT:-5005}/health" >/dev/null 2>&1; then
        info "API is up ✓"
        break
    fi
    sleep 1
    if [[ $i -eq 30 ]]; then
        warn "API didn't respond in 30s — Discord bot may fail on first commands"
    fi
done

# ── Discord bot ────────────────────────────────────────────────────────────────
info "Starting Discord bot..."
node agentic-discord-bot.js 2>&1 | sed 's/^/[BOT] /' &
BOT_PID=$!
PIDS+=($BOT_PID)

info ""
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "  All services running"
info "  API:  http://localhost:${API_PORT:-5005}"
info "  Docs: http://localhost:${API_PORT:-5005}/docs"
info "  Press Ctrl+C to stop everything"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for either process to exit
wait "${API_PID}" "${BOT_PID}"
