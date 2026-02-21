#!/bin/bash
# Ensure RAG server is running on localhost:3847
# Called by Claude Code SessionStart hook

if curl -s --max-time 2 http://localhost:3847/health > /dev/null 2>&1; then
  exit 0
fi

cd /Users/joaopanizzutti/memorybench
nohup bun run src/rag-server.ts > /tmp/rag-server.log 2>&1 &
disown

# Wait for it to come up
for i in 1 2 3 4 5; do
  sleep 1
  if curl -s --max-time 2 http://localhost:3847/health > /dev/null 2>&1; then
    exit 0
  fi
done

exit 1
