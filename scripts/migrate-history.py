#!/usr/bin/env python3
"""Migrate memories from old OpenClaw mem0 history.db into the new dual-layer system.

Reads non-deleted memories from history.db, deduplicates by text content,
and sends each to both RAG (chunk storage) and mem0 (fact extraction + graph).

Both phases run concurrently via ThreadPoolExecutor.

Usage:
    python3 scripts/migrate-history.py [--rag-only] [--mem0-only] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


HISTORY_DB = os.path.expanduser("~/.openclaw/mem0/data/history.db")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:3847")
MEM0_URL = os.environ.get("MEM0_URL", "http://localhost:3848")
CONTAINER_TAG = os.environ.get("CONTAINER_TAG", "openclaw_Joaos_MacBook_Pro_local")
RAG_CONCURRENCY = 20
MEM0_CONCURRENCY = 10


def post(url: str, data: dict, timeout: int = 300) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        raise RuntimeError(f"POST {url} failed ({e.code}): {text}")


def check_health(name: str, url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("ok"):
                print(f"  {name}: OK")
                return True
    except Exception as e:
        print(f"  {name}: UNREACHABLE ({e})")
    return False


def load_memories() -> List[str]:
    if not os.path.exists(HISTORY_DB):
        print(f"history.db not found at {HISTORY_DB}")
        sys.exit(1)

    conn = sqlite3.connect(HISTORY_DB)
    cursor = conn.execute("""
        SELECT DISTINCT h.new_memory
        FROM history h
        INNER JOIN (
            SELECT memory_id, MAX(rowid) as max_rowid
            FROM history
            WHERE new_memory IS NOT NULL
            GROUP BY memory_id
        ) latest ON h.memory_id = latest.memory_id AND h.rowid = latest.max_rowid
        WHERE h.memory_id NOT IN (
            SELECT memory_id FROM history WHERE event = 'DELETE'
        )
        AND h.new_memory IS NOT NULL
        AND LENGTH(TRIM(h.new_memory)) > 0
        ORDER BY h.updated_at ASC
    """)
    memories = [row[0] for row in cursor.fetchall()]
    conn.close()
    return memories


def send_to_rag(idx: int, text: str) -> tuple:
    try:
        post(f"{RAG_URL}/store", {"containerTag": CONTAINER_TAG, "text": text})
        return (idx, True, None)
    except Exception as e:
        return (idx, False, str(e))


def send_to_mem0(idx: int, text: str) -> tuple:
    try:
        post(f"{MEM0_URL}/add", {"text": text, "user_id": CONTAINER_TAG}, timeout=600)
        return (idx, True, None)
    except Exception as e:
        return (idx, False, str(e))


def run_parallel(name: str, fn, memories: List[str], concurrency: int):
    ok = fail = 0
    start = time.time()
    total = len(memories)
    failures = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(fn, i, text): i for i, text in enumerate(memories)}
        done_count = 0

        for future in as_completed(futures):
            idx, success, err = future.result()
            done_count += 1

            if success:
                ok += 1
            else:
                fail += 1
                failures.append(idx)
                preview = memories[idx][:60]
                print(f"  {name} FAIL [{idx}]: {preview}... ({err})")

            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - start
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = (total - done_count) / rate if rate > 0 else 0
                print(f"  {name}: {done_count}/{total} ({ok} ok, {fail} fail) "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    elapsed = time.time() - start
    print(f"  {name} done: {ok} ok, {fail} fail in {elapsed:.1f}s")
    return ok, fail, failures


def main():
    parser = argparse.ArgumentParser(description="Migrate history.db to dual-layer memory")
    parser.add_argument("--rag-only", action="store_true", help="Only send to RAG server")
    parser.add_argument("--mem0-only", action="store_true", help="Only send to mem0 server")
    parser.add_argument("--dry-run", action="store_true", help="Export memories without sending")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of memories to process")
    parser.add_argument("--concurrency", type=int, default=MEM0_CONCURRENCY, help="mem0 concurrency")
    args = parser.parse_args()

    do_rag = not args.mem0_only
    do_mem0 = not args.rag_only
    mem0_conc = args.concurrency

    print("Checking servers...")
    if do_rag and not check_health("RAG", RAG_URL):
        sys.exit(1)
    if do_mem0 and not check_health("mem0", MEM0_URL):
        sys.exit(1)

    print(f"\nLoading memories from {HISTORY_DB}...")
    memories = load_memories()
    total = len(memories)
    print(f"Found {total} unique non-deleted memories")

    if args.limit > 0:
        memories = memories[:args.limit]
        print(f"Processing first {args.limit}")

    if args.dry_run:
        print("\n[DRY RUN] First 20 memories:")
        for i, m in enumerate(memories[:20]):
            preview = m[:100] + "..." if len(m) > 100 else m
            print(f"  {i+1}. {preview}")
        print(f"\nTotal: {total}")
        return

    print(f"\nContainer: {CONTAINER_TAG}")
    targets = []
    if do_rag:
        targets.append(f"RAG ({RAG_CONCURRENCY} concurrent)")
    if do_mem0:
        targets.append(f"mem0 ({mem0_conc} concurrent)")
    print(f"Targets: {', '.join(targets)}")
    print()

    if do_rag:
        print(f"Phase 1: RAG migration...")
        run_parallel("RAG", send_to_rag, memories, RAG_CONCURRENCY)
        print()

    if do_mem0:
        print(f"Phase 2: mem0 migration...")
        run_parallel("mem0", send_to_mem0, memories, mem0_conc)
        print()

    print("Migration complete.")


if __name__ == "__main__":
    main()
