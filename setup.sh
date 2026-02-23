#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
dim()   { printf "\033[90m%s\033[0m\n" "$1"; }

step() { printf "\n\033[1m[%s] %s\033[0m\n" "$1" "$2"; }

# 1. check prerequisites
step 1 "Checking prerequisites"

command -v bun >/dev/null 2>&1 || { red "bun not found. Install: curl -fsSL https://bun.sh/install | bash"; exit 1; }
command -v python3 >/dev/null 2>&1 || { red "python3 not found"; exit 1; }

BUN_V=$(bun --version)
PY_V=$(python3 --version 2>&1 | awk '{print $2}')
green "bun $BUN_V, python $PY_V"

# 2. env file
step 2 "Environment"

if [ ! -f .env ] && [ ! -f .env.local ]; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        red "No .env file and OPENAI_API_KEY not set."
        echo "  cp .env.example .env && edit .env"
        exit 1
    fi
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
    green "Created .env from OPENAI_API_KEY env var"
else
    green "Using existing $([ -f .env.local ] && echo '.env.local' || echo '.env')"
fi

# 3. bun deps
step 3 "Installing bun dependencies"
bun install --frozen-lockfile 2>/dev/null || bun install
green "Done"

# 4. python venv
step 4 "Setting up mem0 server"

VENV="$ROOT/mem0-server/.venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    green "Created venv at mem0-server/.venv"
else
    dim "Venv exists"
fi

"$VENV/bin/pip" install -q -r "$ROOT/mem0-server/requirements.txt"
green "Python deps installed"

# 5. data dirs
step 5 "Data directories"
mkdir -p "$ROOT/mem0-server/data"
mkdir -p "$ROOT/data/cache/rag"
green "Ready"

# 6. summary
step 6 "Setup complete"

echo ""
echo "  Start servers:"
echo "    bun run src/rag-server.ts          # RAG on :3847"
echo "    cd mem0-server && source .venv/bin/activate && python server.py  # mem0 on :3848"
echo ""
echo "  Or use the start script:"
echo "    ./start.sh"
echo ""
