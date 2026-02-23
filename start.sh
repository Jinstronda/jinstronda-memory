#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }

RAG_PORT="${RAG_PORT:-3847}"
MEM0_PORT="${MEM0_PORT:-3848}"

# load env
[ -f .env.local ] && set -a && source .env.local && set +a
[ -f .env ] && set -a && source .env && set +a

if [ -z "${OPENAI_API_KEY:-}" ]; then
    red "OPENAI_API_KEY not set. Run ./setup.sh first."
    exit 1
fi

# check if already running
if lsof -i ":$RAG_PORT" -t >/dev/null 2>&1; then
    echo "RAG already running on :$RAG_PORT (pid $(lsof -i :$RAG_PORT -t | head -1))"
else
    bun run src/rag-server.ts > /tmp/rag-server.log 2>&1 &
    echo "RAG starting on :$RAG_PORT (pid $!)"
fi

if lsof -i ":$MEM0_PORT" -t >/dev/null 2>&1; then
    echo "mem0 already running on :$MEM0_PORT (pid $(lsof -i :$MEM0_PORT -t | head -1))"
else
    cd "$ROOT/mem0-server"
    source .venv/bin/activate
    python server.py > /tmp/mem0-server.log 2>&1 &
    echo "mem0 starting on :$MEM0_PORT (pid $!)"
    cd "$ROOT"
fi

# wait for health
echo ""
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    RAG_OK=$(curl -sf "http://localhost:$RAG_PORT/health" 2>/dev/null && echo 1 || echo 0)
    MEM0_OK=$(curl -sf "http://localhost:$MEM0_PORT/health" 2>/dev/null && echo 1 || echo 0)
    if [ "$RAG_OK" = "1" ] && [ "$MEM0_OK" = "1" ]; then
        green "Both servers healthy"
        echo "  RAG:  http://localhost:$RAG_PORT"
        echo "  mem0: http://localhost:$MEM0_PORT"
        echo ""
        echo "  Logs: tail -f /tmp/rag-server.log /tmp/mem0-server.log"
        exit 0
    fi
done

red "Servers didn't start in time. Check logs:"
echo "  tail /tmp/rag-server.log"
echo "  tail /tmp/mem0-server.log"
exit 1
