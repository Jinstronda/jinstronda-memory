#!/usr/bin/env bash

RAG_PORT="${RAG_PORT:-3847}"
MEM0_PORT="${MEM0_PORT:-3848}"

for port in $RAG_PORT $MEM0_PORT; do
    pids=$(lsof -i ":$port" -t 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Stopping port $port (pid $pids)"
        echo "$pids" | xargs kill 2>/dev/null || true
    fi
done

echo "Servers stopped"
