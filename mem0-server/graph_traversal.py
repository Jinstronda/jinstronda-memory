from __future__ import annotations

import logging
from collections import deque
from typing import List, Dict

log = logging.getLogger(__name__)

MAX_NEIGHBORS_PER_NODE = 30
MAX_TOTAL_RELATIONSHIPS = 200


def bfs_traverse(graph_conn, seed_entities: List[str], user_id: str, max_hops: int = 2) -> dict:
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
        if len(all_relationships) >= MAX_TOTAL_RELATIONSHIPS:
            break

        current, depth = queue.popleft()
        if current in visited or depth > max_hops:
            continue
        visited.add(current)

        neighbors = _get_neighbors(graph_conn, current, user_id)
        if len(neighbors) > MAX_NEIGHBORS_PER_NODE:
            neighbors = neighbors[:MAX_NEIGHBORS_PER_NODE]

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


def _get_neighbors(graph_conn, entity_name: str, user_id: str) -> List[Dict]:
    neighbors = []
    try:
        result = graph_conn.execute(
            "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity {user_id: $uid}) "
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
    except Exception as e:
        log.warning("graph query failed for entity=%s: %s", entity_name, e)

    return neighbors


def _dedup_entities(entities: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for e in entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            result.append(e)
    return result


def _dedup_relationships(rels: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for r in rels:
        key = (r["source"], r["relationship"], r["target"])
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result
