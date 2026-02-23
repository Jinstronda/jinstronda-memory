from __future__ import annotations

import logging
import os
import asyncio
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Patch sqlite3 BEFORE mem0 import: allow cross-thread usage.
# Qdrant local stores metadata in SQLite, mem0 spawns internal threads.
_orig_connect = sqlite3.connect
def _patched_connect(*args, **kwargs):
    kwargs["check_same_thread"] = False
    return _orig_connect(*args, **kwargs)
sqlite3.connect = _patched_connect

from mem0 import Memory
from config import get_mem0_config
from graph_traversal import bfs_traverse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mem0-server")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "mem0")

memory: Optional[Memory] = None
_memory_lock = threading.Lock()
_add_total = 0
_add_ok = 0
_add_fail = 0
_add_sem: Optional[asyncio.Semaphore] = None
ADD_CONCURRENCY = int(os.getenv("MEM0_ADD_CONCURRENCY", "10"))


def _get_add_sem() -> asyncio.Semaphore:
    global _add_sem
    if _add_sem is None:
        _add_sem = asyncio.Semaphore(ADD_CONCURRENCY)
        log.info(f"add concurrency={ADD_CONCURRENCY}")
    return _add_sem


def get_data_dir() -> str:
    d = os.getenv("MEM0_DATA_DIR", DEFAULT_DATA_DIR)
    d = os.path.abspath(d)
    log.info(f"data_dir={d}")
    return d


def get_memory() -> Memory:
    global memory
    if memory is None:
        with _memory_lock:
            if memory is None:
                data_dir = get_data_dir()
                os.makedirs(data_dir, exist_ok=True)
                config = get_mem0_config(data_dir)
                memory = Memory.from_config(config_dict=config)
                log.info(f"mem0 initialized (qdrant={data_dir}/qdrant, graph={data_dir}/graph.kuzu)")
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
    global _add_total, _add_ok, _add_fail
    _add_total += 1
    req_id = _add_total

    async with _get_add_sem():
        t0 = time.time()
        log.info(f"[add #{req_id}] started")
        try:
            m = get_memory()
            if req.text:
                result = await asyncio.to_thread(m.add, req.text, user_id=req.user_id, metadata=req.metadata)
            elif req.messages:
                result = await asyncio.to_thread(m.add, req.messages, user_id=req.user_id, metadata=req.metadata)
            else:
                return JSONResponse({"error": "Provide messages or text"}, status_code=400)
            elapsed = time.time() - t0
            _add_ok += 1
            log.info(f"[add #{req_id}] done in {elapsed:.1f}s (ok={_add_ok}, fail={_add_fail})")
            return result
        except Exception as e:
            elapsed = time.time() - t0
            _add_fail += 1
            log.error(f"[add #{req_id}] failed in {elapsed:.1f}s: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)


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
