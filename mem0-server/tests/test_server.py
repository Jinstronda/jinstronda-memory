import pytest


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


@pytest.mark.asyncio
async def test_add_and_search(client):
    r = await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "My favorite color is blue and I work at Stripe"},
            {"role": "assistant", "content": "Got it!"},
        ],
        "user_id": "test_user",
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data

    r = await client.get("/search", params={
        "query": "favorite color",
        "user_id": "test_user",
        "limit": 5,
    })
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    assert any("blue" in m["memory"].lower() for m in results)


@pytest.mark.asyncio
async def test_add_string(client):
    r = await client.post("/add", json={
        "text": "User prefers dark mode",
        "user_id": "test_user_2",
    })
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_get_all(client):
    await client.post("/add", json={
        "text": "I love hiking",
        "user_id": "test_getall",
    })
    r = await client.get("/memories", params={"user_id": "test_getall"})
    assert r.status_code == 200
    assert len(r.json()["memories"]) > 0


@pytest.mark.asyncio
async def test_delete(client):
    result = await client.post("/add", json={
        "text": "Temporary memory",
        "user_id": "test_delete",
    })
    memories = result.json().get("results", [])
    if memories:
        mid = memories[0]["id"]
        r = await client.delete(f"/memories/{mid}", params={"user_id": "test_delete"})
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_delete_all(client):
    await client.post("/add", json={
        "text": "Will be deleted",
        "user_id": "test_delete_all",
    })
    r = await client.delete("/memories", params={"user_id": "test_delete_all"})
    assert r.status_code == 200
    remaining = await client.get("/memories", params={"user_id": "test_delete_all"})
    assert len(remaining.json()["memories"]) == 0


@pytest.mark.asyncio
async def test_graph_search(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "Alice works at Google and lives in San Francisco"},
        ],
        "user_id": "test_graph",
    })
    r = await client.get("/graph", params={
        "query": "Alice",
        "user_id": "test_graph",
    })
    assert r.status_code == 200
    data = r.json()
    assert "relations" in data
