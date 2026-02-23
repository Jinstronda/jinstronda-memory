import pytest
from httpx import AsyncClient, ASGITransport
from server import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_full_conversation_flow(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "My name is Carlos and I live in Madrid"},
            {"role": "assistant", "content": "Nice to meet you, Carlos!"},
        ],
        "user_id": "integration_test",
    })
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "I work as a data scientist at Spotify"},
            {"role": "assistant", "content": "Interesting role!"},
        ],
        "user_id": "integration_test",
    })
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "I moved from Madrid to Berlin last month"},
            {"role": "assistant", "content": "Big move!"},
        ],
        "user_id": "integration_test",
    })

    r = await client.get("/search", params={"query": "where does the user live", "user_id": "integration_test"})
    results = r.json()["results"]
    assert len(results) > 0
    texts = " ".join(m["memory"].lower() for m in results)
    assert "berlin" in texts

    r = await client.get("/graph/deep", params={"query": "Carlos", "user_id": "integration_test", "max_hops": 2})
    data = r.json()
    assert "entities" in data
    assert "relationships" in data

    await client.delete("/memories", params={"user_id": "integration_test"})


@pytest.mark.asyncio
async def test_update_overwrites_old_fact(client):
    uid = "test_update_fact"
    await client.post("/add", json={
        "text": "User's favorite language is Python",
        "user_id": uid,
    })
    await client.post("/add", json={
        "text": "User's favorite language is now Rust",
        "user_id": uid,
    })

    r = await client.get("/search", params={"query": "favorite language", "user_id": uid})
    results = r.json()["results"]
    texts = " ".join(m["memory"].lower() for m in results)
    assert "rust" in texts

    await client.delete("/memories", params={"user_id": uid})


@pytest.mark.asyncio
async def test_search_with_no_results(client):
    r = await client.get("/search", params={
        "query": "something that was never stored",
        "user_id": "nonexistent_user_12345",
    })
    assert r.status_code == 200
    assert len(r.json()["results"]) == 0
