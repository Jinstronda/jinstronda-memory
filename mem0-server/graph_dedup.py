from __future__ import annotations

import logging
import os
import re
from typing import List, Dict, Tuple

import openai

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small")
COSINE_THRESHOLD = float(os.getenv("GRAPH_DEDUP_THRESHOLD", "0.95"))

GARBAGE_PATTERNS = [
    re.compile(r"^posted_message:", re.IGNORECASE),
    re.compile(r"^sent_message:", re.IGNORECASE),
    re.compile(r"^said:", re.IGNORECASE),
]

TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
CONTAINER_TAGS = {"openclaw_joaos_macbook_pro_local"}


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


EMBED_BATCH_SIZE = 2048


def _batch_embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = openai.OpenAI()
    all_embeddings: List[List[float]] = [[] for _ in texts]
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[start : start + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        for item in resp.data:
            all_embeddings[start + item.index] = item.embedding
    return all_embeddings


def _build_embedding_cache(all_rel_names: List[str]) -> Dict[str, List[float]]:
    unique = list(set(all_rel_names))
    if not unique:
        return {}
    log.info(f"embedding {len(unique)} unique relationship names")
    embeddings = _batch_embed(unique)
    return dict(zip(unique, embeddings))


def is_garbage_edge(rel_name: str, source: str, target: str) -> str | None:
    for p in GARBAGE_PATTERNS:
        if p.search(rel_name):
            return f"message content in relationship: {rel_name[:60]}"
    if source == target:
        return f"self-referential: {source}"
    if TIMESTAMP_PATTERN.match(source) or TIMESTAMP_PATTERN.match(target):
        return f"timestamp entity: {source} or {target}"
    if source in CONTAINER_TAGS or target in CONTAINER_TAGS:
        return f"container tag entity: {source} or {target}"
    return None


def cluster_relationships(
    rel_names: List[str],
    threshold: float = COSINE_THRESHOLD,
    embedding_cache: Dict[str, List[float]] | None = None,
) -> List[List[int]]:
    if len(rel_names) <= 1:
        return [[i] for i in range(len(rel_names))]

    if embedding_cache:
        embeddings = [embedding_cache[n] for n in rel_names]
    else:
        embeddings = _batch_embed(rel_names)
    n = len(rel_names)
    assigned = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if _cosine(embeddings[i], embeddings[j]) >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def pick_canonical(rel_names: List[str], mentions: List[int]) -> Tuple[str, int]:
    best_idx = 0
    best_mentions = mentions[0]
    for i in range(1, len(rel_names)):
        if mentions[i] > best_mentions:
            best_idx = i
            best_mentions = mentions[i]
        elif mentions[i] == best_mentions and len(rel_names[i]) < len(rel_names[best_idx]):
            best_idx = i
    return rel_names[best_idx], best_idx


def run_dedup(graph_conn, user_id: str, dry_run: bool = False) -> Dict:
    stats = {"garbage_deleted": 0, "clusters_merged": 0, "edges_deleted": 0, "edges_before": 0, "edges_after": 0}

    # count total edges
    r = graph_conn.execute(
        "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity {user_id: $uid}) "
        "RETURN count(r) AS cnt",
        {"uid": user_id},
    )
    while r.has_next():
        stats["edges_before"] = r.get_next()[0]

    # step 1: delete garbage
    r = graph_conn.execute(
        "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity {user_id: $uid}) "
        "RETURN a.name AS src, r.name AS rel, b.name AS tgt",
        {"uid": user_id},
    )
    garbage = []
    while r.has_next():
        row = r.get_next()
        reason = is_garbage_edge(row[1], row[0], row[2])
        if reason:
            garbage.append({"src": row[0], "rel": row[1], "tgt": row[2], "reason": reason})

    log.info(f"garbage edges found: {len(garbage)}")
    for g in garbage:
        if dry_run:
            stats["garbage_deleted"] += 1
            continue
        try:
            graph_conn.execute(
                "MATCH (a:Entity {name: $src, user_id: $uid})"
                "-[r:CONNECTED_TO {name: $rel}]->"
                "(b:Entity {name: $tgt, user_id: $uid}) DELETE r",
                {"src": g["src"], "rel": g["rel"], "tgt": g["tgt"], "uid": user_id},
            )
            stats["garbage_deleted"] += 1
        except Exception as e:
            log.warning(f"failed to delete garbage edge: {e}")

    # step 2: group edges by (source, target) pair
    r = graph_conn.execute(
        "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity {user_id: $uid}) "
        "RETURN a.name AS src, b.name AS tgt, r.name AS rel, "
        "coalesce(r.mentions, 1) AS mentions",
        {"uid": user_id},
    )
    pairs: Dict[Tuple[str, str], List[Dict]] = {}
    while r.has_next():
        row = r.get_next()
        key = (row[0], row[1])
        if key not in pairs:
            pairs[key] = []
        pairs[key].append({"rel": row[2], "mentions": row[3]})

    multi_pairs = {k: v for k, v in pairs.items() if len(v) > 1}
    log.info(f"entity pairs with multiple edges: {len(multi_pairs)}")

    # pre-embed all relationship names in bulk
    all_rel_names = []
    for edges in multi_pairs.values():
        all_rel_names.extend(e["rel"] for e in edges)
    embedding_cache = _build_embedding_cache(all_rel_names)

    # step 3: cosine dedup
    for (src, tgt), edges in multi_pairs.items():
        rel_names = [e["rel"] for e in edges]
        mentions = [e["mentions"] for e in edges]

        clusters = cluster_relationships(rel_names, embedding_cache=embedding_cache)
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            cluster_names = [rel_names[i] for i in cluster]
            cluster_mentions = [mentions[i] for i in cluster]
            canonical, canon_idx_in_cluster = pick_canonical(cluster_names, cluster_mentions)
            total_mentions = sum(cluster_mentions)

            to_delete = [cluster_names[i] for i in range(len(cluster)) if i != canon_idx_in_cluster]
            log.info(f"  {src}->{tgt}: merge {to_delete} into '{canonical}'")

            if dry_run:
                stats["edges_deleted"] += len(to_delete)
            else:
                try:
                    graph_conn.execute(
                        "MATCH (a:Entity {name: $src, user_id: $uid})"
                        "-[r:CONNECTED_TO {name: $rel}]->"
                        "(b:Entity {name: $tgt, user_id: $uid}) "
                        "SET r.mentions = $mentions",
                        {"src": src, "rel": canonical, "tgt": tgt, "uid": user_id, "mentions": total_mentions},
                    )
                except Exception as e:
                    log.warning(f"failed to update canonical: {e}")

                for rel in to_delete:
                    try:
                        graph_conn.execute(
                            "MATCH (a:Entity {name: $src, user_id: $uid})"
                            "-[r:CONNECTED_TO {name: $rel}]->"
                            "(b:Entity {name: $tgt, user_id: $uid}) DELETE r",
                            {"src": src, "rel": rel, "tgt": tgt, "uid": user_id},
                        )
                        stats["edges_deleted"] += 1
                    except Exception as e:
                        log.warning(f"failed to delete edge: {e}")

            stats["clusters_merged"] += 1

    # count edges after
    r = graph_conn.execute(
        "MATCH (a:Entity {user_id: $uid})-[r:CONNECTED_TO]->(b:Entity {user_id: $uid}) "
        "RETURN count(r) AS cnt",
        {"uid": user_id},
    )
    while r.has_next():
        stats["edges_after"] = r.get_next()[0]

    return stats
