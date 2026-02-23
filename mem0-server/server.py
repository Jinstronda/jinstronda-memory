from __future__ import annotations

import os
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mem0 import Memory
from config import get_mem0_config
from graph_traversal import bfs_traverse

memory: Optional[Memory] = None
_memory_lock = threading.Lock()


def get_memory() -> Memory:
    global memory
    if memory is None:
        with _memory_lock:
            if memory is None:
                data_dir = os.getenv("MEM0_DATA_DIR", "./data")
                os.makedirs(data_dir, exist_ok=True)
                config = get_mem0_config(data_dir)
                memory = Memory.from_config(config_dict=config)
    return memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(get_memory)
    yield


app = FastAPI(lifespan=lifespan)


class AddRequest(BaseModel):
    messages: Optional[list] = None
    text: Optional[str] = None
    user_id: str
    metadata: Optional[dict] = None


@app.get("/health")
async def health():
    return {"ok": True, "provider": "mem0"}


@app.post("/add")
async def add_memory(req: AddRequest):
    m = get_memory()
    if req.text:
        result = await asyncio.to_thread(m.add, req.text, user_id=req.user_id, metadata=req.metadata)
    elif req.messages:
        result = await asyncio.to_thread(m.add, req.messages, user_id=req.user_id, metadata=req.metadata)
    else:
        return JSONResponse({"error": "Provide messages or text"}, status_code=400)
    return result


@app.get("/search")
async def search_memories(
    query: str,
    user_id: str,
    limit: int = Query(default=10),
):
    results = await asyncio.to_thread(get_memory().search, query, user_id=user_id, limit=limit)
    if isinstance(results, dict):
        return {"results": results.get("results", [])}
    return {"results": results}


@app.get("/memories")
async def get_all(user_id: str):
    result = await asyncio.to_thread(get_memory().get_all, user_id=user_id)
    if isinstance(result, dict):
        return {"memories": result.get("results", [])}
    return {"memories": result}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = Query(...)):
    m = get_memory()
    mem = await asyncio.to_thread(m.get, memory_id)
    if not mem or mem.get("user_id") != user_id:
        return JSONResponse({"error": "Memory not found or not owned by user"}, status_code=404)
    await asyncio.to_thread(m.delete, memory_id)
    return {"ok": True}


@app.delete("/memories")
async def delete_all(user_id: str):
    await asyncio.to_thread(get_memory().delete_all, user_id=user_id)
    return {"ok": True}


@app.get("/graph")
async def graph_search(
    query: str,
    user_id: str,
    limit: int = Query(default=10),
):
    results = await asyncio.to_thread(get_memory().search, query, user_id=user_id, limit=limit)
    relations = []
    if isinstance(results, dict) and "relations" in results:
        relations = results["relations"] or []
    return {"relations": relations}


@app.get("/graph/deep")
async def graph_deep_search(
    query: str,
    user_id: str,
    max_hops: int = Query(default=2),
):
    m = get_memory()
    search_result = await asyncio.to_thread(m.search, query, user_id=user_id, limit=5)
    relations = []
    if isinstance(search_result, dict):
        relations = search_result.get("relations", []) or []

    seed_entities = set()
    for rel in relations:
        seed_entities.add(rel.get("source", ""))
        seed_entities.add(rel.get("destination", ""))
    seed_entities.discard("")

    if not seed_entities:
        return {"entities": [], "relationships": []}

    graph_conn = None
    if hasattr(m, "graph") and m.graph:
        graph_conn = m.graph.graph

    return await asyncio.to_thread(bfs_traverse, graph_conn, list(seed_entities), user_id, max_hops)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MEM0_PORT", "3848"))
    uvicorn.run(app, host="0.0.0.0", port=port)
