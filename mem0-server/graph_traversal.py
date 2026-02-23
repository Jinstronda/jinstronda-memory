from collections import deque


def bfs_traverse(graph_conn, seed_entities: list[str], user_id: str, max_hops: int = 2) -> dict:
    if not graph_conn:
        return {"entities": [], "relationships": []}

    visited = set()
    queue = deque()
    all_entities = []
    all_relationships = []

    for entity in seed_entities:
        normalized = entity.lower().replace(" ", "_")
        queue.append((normalized, 0))

    while queue:
        current, depth = queue.popleft()
        if current in visited or depth > max_hops:
            continue
        visited.add(current)

        neighbors = _get_neighbors(graph_conn, current, user_id)
        for neighbor in neighbors:
            all_entities.append({"name": neighbor["name"], "type": "entity"})
            all_relationships.append({
                "source": neighbor["source"],
                "relationship": neighbor["relationship"],
                "target": neighbor["target"],
            })
            if neighbor["name"] not in visited and depth + 1 <= max_hops:
                queue.append((neighbor["name"], depth + 1))

    return {
        "entities": _dedup_entities(all_entities),
        "relationships": _dedup_relationships(all_relationships),
    }


def _get_neighbors(graph_conn, entity_name: str, user_id: str) -> list[dict]:
    neighbors = []
    try:
        # outgoing edges
        result = graph_conn.execute(
            "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity) "
            "WHERE a.name = $name "
            "RETURN a.name AS source, r.name AS rel, b.name AS target",
            {"name": entity_name, "uid": user_id},
        )
        for row in result.rows_as_dict():
            neighbors.append({
                "name": row["target"],
                "source": row["source"],
                "relationship": row["rel"],
                "target": row["target"],
            })

        # incoming edges
        result = graph_conn.execute(
            "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity {user_id: $uid}) "
            "WHERE b.name = $name "
            "RETURN a.name AS source, r.name AS rel, b.name AS target",
            {"name": entity_name, "uid": user_id},
        )
        for row in result.rows_as_dict():
            neighbors.append({
                "name": row["source"],
                "source": row["source"],
                "relationship": row["rel"],
                "target": row["target"],
            })
    except Exception:
        pass

    return neighbors


def _dedup_entities(entities: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for e in entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            result.append(e)
    return result


def _dedup_relationships(rels: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for r in rels:
        key = (r["source"], r["relationship"], r["target"])
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result
