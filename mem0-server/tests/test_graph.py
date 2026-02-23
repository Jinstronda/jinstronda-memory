import pytest


@pytest.mark.asyncio
async def test_two_hop_traversal(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "Alice works at Google. Google is headquartered in Mountain View."},
        ],
        "user_id": "test_2hop",
    })
    r = await client.get("/graph/deep", params={
        "query": "Alice",
        "user_id": "test_2hop",
        "max_hops": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert "entities" in data
    assert "relationships" in data


@pytest.mark.asyncio
async def test_two_hop_finds_indirect(client):
    await client.post("/add", json={
        "messages": [
            {"role": "user", "content": "Bob manages the AI team. The AI team built the chatbot product."},
        ],
        "user_id": "test_indirect",
    })
    r = await client.get("/graph/deep", params={
        "query": "Bob",
        "user_id": "test_indirect",
        "max_hops": 2,
    })
    data = r.json()
    assert len(data.get("relationships", [])) >= 1
