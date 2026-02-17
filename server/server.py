"""Mem0 REST server for OpenClaw memory plugin."""

import os
import sys
from pathlib import Path
from typing import Optional

# load .env from openclaw workspace
env_path = Path.home() / ".openclaw" / "workspace" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        if key and val:
            os.environ.setdefault(key.strip(), val.strip())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from config import MEM0_CONFIG
from mem0 import Memory

app = FastAPI(title="Mem0 OpenClaw Server")
memory = Memory.from_config(MEM0_CONFIG)


class AddRequest(BaseModel):
    messages: str
    user_id: str
    metadata: Optional[dict] = None
    infer: bool = True


class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = 10


class DeleteRequest(BaseModel):
    memory_id: str


class ResetRequest(BaseModel):
    user_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/memories/")
def add_memory(req: AddRequest):
    result = memory.add(
        req.messages,
        user_id=req.user_id,
        metadata=req.metadata,
        infer=req.infer,
    )
    # mem0 returns {"results": [{"id": ..., "memory": ..., "event": "ADD"|"UPDATE"|...}]}
    results = result.get("results", [])
    ids = [r.get("id", "") for r in results if r.get("id")]
    return {"results": results, "ids": ids}


@app.post("/v1/memories/search/")
def search_memories(req: SearchRequest):
    results = memory.search(
        req.query,
        user_id=req.user_id,
        limit=req.top_k,
    )
    # mem0 search returns {"results": [{"id", "memory", "score", "metadata", ...}]}
    if isinstance(results, dict):
        return results
    return {"results": results}


@app.get("/v1/memories/")
def list_memories(user_id: str, limit: int = 100):
    results = memory.get_all(user_id=user_id, limit=limit)
    if isinstance(results, dict):
        return results
    return {"results": results}


@app.get("/v1/memories/{memory_id}")
def get_memory(memory_id: str):
    result = memory.get(memory_id)
    return result


@app.delete("/v1/memories/{memory_id}")
def delete_memory(memory_id: str):
    memory.delete(memory_id)
    return {"id": memory_id, "deleted": True}


@app.delete("/v1/memories/")
def reset_memories(user_id: str):
    memory.delete_all(user_id=user_id)
    return {"user_id": user_id, "reset": True}


@app.get("/v1/memories/{memory_id}/history")
def memory_history(memory_id: str):
    result = memory.history(memory_id)
    return {"history": result}


if __name__ == "__main__":
    port = int(os.environ.get("MEM0_PORT", "8080"))
    print(f"mem0 server starting on port {port}")
    print(f"llm: {MEM0_CONFIG['llm']['provider']}/{MEM0_CONFIG['llm']['config']['model']}")
    print(f"embedder: {MEM0_CONFIG['embedder']['config']['model']}")
    print(f"data: {MEM0_CONFIG['vector_store']['config']['path']}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, limit_concurrency=100)
