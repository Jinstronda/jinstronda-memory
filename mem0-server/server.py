import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mem0 import Memory
from config import get_mem0_config
from graph_traversal import bfs_traverse

memory: Memory | None = None


def get_memory() -> Memory:
    global memory
    if memory is None:
        data_dir = os.getenv("MEM0_DATA_DIR", "./data")
        os.makedirs(data_dir, exist_ok=True)
        config = get_mem0_config(data_dir)
        memory = Memory.from_config(config_dict=config)
    return memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_memory()
    yield


app = FastAPI(lifespan=lifespan)


class AddRequest(BaseModel):
    messages: list[dict] | None = None
    text: str | None = None
    user_id: str
    metadata: dict | None = None


@app.get("/health")
def health():
    return {"ok": True, "provider": "mem0"}


@app.post("/add")
def add_memory(req: AddRequest):
    m = get_memory()
    if req.text:
        result = m.add(req.text, user_id=req.user_id, metadata=req.metadata)
    elif req.messages:
        result = m.add(req.messages, user_id=req.user_id, metadata=req.metadata)
    else:
        return JSONResponse({"error": "Provide messages or text"}, status_code=400)
    return result


@app.get("/search")
def search_memories(
    query: str,
    user_id: str,
    limit: int = Query(default=10),
):
    results = get_memory().search(query, user_id=user_id, limit=limit)
    if isinstance(results, dict):
        return {"results": results.get("results", [])}
    return {"results": results}


@app.get("/memories")
def get_all(user_id: str):
    result = get_memory().get_all(user_id=user_id)
    if isinstance(result, dict):
        return {"memories": result.get("results", [])}
    return {"memories": result}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    get_memory().delete(memory_id)
    return {"ok": True}


@app.delete("/memories")
def delete_all(user_id: str):
    get_memory().delete_all(user_id=user_id)
    return {"ok": True}


@app.get("/graph")
def graph_search(
    query: str,
    user_id: str,
    limit: int = Query(default=10),
):
    results = get_memory().search(query, user_id=user_id, limit=limit)
    relations = []
    if isinstance(results, dict) and "relations" in results:
        relations = results["relations"] or []
    return {"relations": relations}


@app.get("/graph/deep")
def graph_deep_search(
    query: str,
    user_id: str,
    max_hops: int = Query(default=2),
):
    m = get_memory()
    search_result = m.search(query, user_id=user_id, limit=5)
    relations = []
    if isinstance(search_result, dict):
        relations = search_result.get("relations", []) or []

    seed_entities = set()
    for rel in relations:
        seed_entities.add(rel.get("source", ""))
        seed_entities.add(rel.get("destination", ""))
    seed_entities.discard("")

    if not seed_entities:
        seed_entities = {w.lower().replace(" ", "_") for w in query.split()}

    graph_conn = None
    if hasattr(m, "graph") and m.graph:
        graph_conn = m.graph.graph

    return bfs_traverse(graph_conn, list(seed_entities), user_id, max_hops)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MEM0_PORT", "3848"))
    uvicorn.run(app, host="0.0.0.0", port=port)
