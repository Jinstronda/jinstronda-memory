#!/usr/bin/env python3
"""Migrate ALL OpenClaw memories into dual-layer system (RAG + mem0 + graph).

RAG gets chunked text (6K). mem0 gets whole files (LLM extraction handles it).
Separate concurrency for each: RAG is fast, mem0 is slow (LLM per call).

Sources:
  1. Identity files from local + GitHub workspace
  2. Memory .md/.txt from local + GitHub workspace/memory
  3. Chroma DB openclaw-source facts

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/migrate-all.py
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import aiohttp

RAG_URL = os.environ.get("RAG_URL", "http://localhost:3847")
MEM0_URL = os.environ.get("MEM0_URL", "http://localhost:3848")
TAG = "openclaw_Joaos_MacBook_Pro_local"

LOCAL_WS = Path.home() / ".openclaw/workspace"
LOCAL_MEM = LOCAL_WS / "memory"
GITHUB_WS = Path("/tmp/my-clawdbot/workspace")
GITHUB_MEM = GITHUB_WS / "memory"
CHROMA_DB = Path.home() / ".openclaw/mem0/data/chroma/chroma.sqlite3"

MAX_CHUNK = 6000
MEM0_MAX_TEXT = 50000
RAG_CONCURRENCY = 30
MEM0_CONCURRENCY = 5

IDENTITY_FILES = ["MEMORY.md", "IDENTITY.md", "SOUL.md", "USER.md", "AGENTS.md", "HEARTBEAT.md", "HORMOZI.md", "TOOLS.md"]

stats = {"rag_ok": 0, "rag_fail": 0, "mem0_ok": 0, "mem0_fail": 0}


def chunk_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK:
        return [text]
    out = []
    i = 0
    while i < len(text):
        end = min(i + MAX_CHUNK, len(text))
        if end < len(text):
            nl = text.rfind("\n", i + int(MAX_CHUNK * 0.4), end)
            if nl > 0:
                end = nl + 1
        out.append(text[i:end])
        i = end
    return out


def extract_date(filename: str) -> str | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def load_chroma_facts() -> list[str]:
    if not CHROMA_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(CHROMA_DB))
        rows = conn.execute("""
            SELECT em_data.string_value
            FROM embedding_metadata em_uid
            JOIN embedding_metadata em_data ON em_uid.id = em_data.id AND em_data.key = 'data'
            JOIN embedding_metadata em_src ON em_uid.id = em_src.id AND em_src.key = 'source'
            WHERE em_uid.key = 'user_id' AND em_uid.string_value = 'jinstronda'
            AND em_src.string_value = 'openclaw'
        """).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception as e:
        print(f"Chroma read failed: {e}")
        return []


async def rag_worker(session: aiohttp.ClientSession, queue: asyncio.Queue):
    timeout = aiohttp.ClientTimeout(total=120)
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        sid, text, date = item
        body = {
            "containerTag": TAG,
            "sessionId": sid,
            "messages": [{"role": "user", "content": text}],
        }
        if date:
            body["date"] = date
        for attempt in range(3):
            try:
                async with session.post(f"{RAG_URL}/ingest", json=body, timeout=timeout) as resp:
                    if resp.status == 200:
                        stats["rag_ok"] += 1
                        break
                    if attempt == 2:
                        stats["rag_fail"] += 1
            except Exception:
                if attempt == 2:
                    stats["rag_fail"] += 1
                else:
                    await asyncio.sleep(1 * (attempt + 1))
        queue.task_done()


async def mem0_worker(session: aiohttp.ClientSession, queue: asyncio.Queue):
    timeout = aiohttp.ClientTimeout(total=180)
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        sid, text = item
        text = text[:MEM0_MAX_TEXT]
        body = {"text": text, "user_id": TAG}
        for attempt in range(3):
            try:
                async with session.post(f"{MEM0_URL}/add", json=body, timeout=timeout) as resp:
                    if resp.status == 200:
                        stats["mem0_ok"] += 1
                        break
                    if attempt == 2:
                        stats["mem0_fail"] += 1
                        print(f"  FAIL mem0 [{sid}] status={resp.status}")
            except Exception as e:
                if attempt == 2:
                    stats["mem0_fail"] += 1
                    print(f"  FAIL mem0 [{sid}] {e}")
                else:
                    await asyncio.sleep(2 * (attempt + 1))
        queue.task_done()


async def main():
    async with aiohttp.ClientSession() as session:
        for url, name in [(RAG_URL, "RAG"), (MEM0_URL, "mem0")]:
            try:
                async with session.get(f"{url}/health") as r:
                    assert r.status == 200
            except Exception:
                print(f"{name} not reachable at {url}")
                sys.exit(1)

        print(f"RAG: {RAG_URL} ({RAG_CONCURRENCY}w) | mem0: {MEM0_URL} ({MEM0_CONCURRENCY}w)")
        print(f"tag: {TAG}\n")

        seen = set()
        items: list[tuple[str, str, str | None]] = []  # (sid, text, date)
        skip_count = 0

        # 1. identity files
        id_count = 0
        for ws_dir in [LOCAL_WS, GITHUB_WS]:
            for fname in IDENTITY_FILES:
                fp = ws_dir / fname
                if not fp.exists():
                    continue
                content = fp.read_text(errors="replace")
                h = content_hash(content)
                if h in seen:
                    skip_count += 1
                    continue
                seen.add(h)
                id_count += 1
                items.append((f"id_{fp.stem}", content, "2026-02-23"))
        print(f"[1/4] {id_count} identity files")

        # 2. local memory files
        local_count = 0
        if LOCAL_MEM.exists():
            for fp in sorted(LOCAL_MEM.iterdir()):
                if fp.suffix.lower() not in (".md", ".txt") or not fp.is_file():
                    continue
                content = fp.read_text(errors="replace")
                h = content_hash(content)
                if h in seen:
                    skip_count += 1
                    continue
                seen.add(h)
                local_count += 1
                items.append((f"local_{fp.stem}", content, extract_date(fp.name)))
        print(f"[2/4] {local_count} local memory files")

        # 3. github memory files
        gh_count = 0
        if GITHUB_MEM.exists():
            for fp in sorted(GITHUB_MEM.iterdir()):
                if fp.suffix.lower() not in (".md", ".txt") or not fp.is_file():
                    continue
                content = fp.read_text(errors="replace")
                h = content_hash(content)
                if h in seen:
                    skip_count += 1
                    continue
                seen.add(h)
                gh_count += 1
                items.append((f"gh_{fp.stem}", content, extract_date(fp.name)))
        print(f"[3/4] {gh_count} GitHub memory files ({skip_count} dupes skipped)")

        # 4. chroma facts
        facts = load_chroma_facts()
        fact_count = 0
        for j, fact in enumerate(facts):
            h = content_hash(fact)
            if h in seen:
                continue
            seen.add(h)
            fact_count += 1
            items.append((f"chroma_{j}", fact, None))
        print(f"[4/4] {fact_count} Chroma facts")

        # build RAG chunk queue and mem0 file queue
        rag_queue: asyncio.Queue = asyncio.Queue()
        mem0_queue: asyncio.Queue = asyncio.Queue()

        rag_chunk_count = 0
        for sid, text, date in items:
            if not text.strip():
                continue
            chunks = chunk_text(text)
            for ci, chunk in enumerate(chunks):
                csid = sid if len(chunks) == 1 else f"{sid}_c{ci}"
                rag_queue.put_nowait((csid, chunk, date))
                rag_chunk_count += 1
            mem0_queue.put_nowait((sid, text))

        total_items = id_count + local_count + gh_count + fact_count
        mem0_count = mem0_queue.qsize()
        print(f"\nTotal: {total_items} items")
        print(f"  RAG: {rag_chunk_count} chunks ({RAG_CONCURRENCY} workers)")
        print(f"  mem0: {mem0_count} whole files ({MEM0_CONCURRENCY} workers)")
        print()

        # sentinel values to stop workers
        for _ in range(RAG_CONCURRENCY):
            rag_queue.put_nowait(None)
        for _ in range(MEM0_CONCURRENCY):
            mem0_queue.put_nowait(None)

        t0 = time.time()

        # progress reporter
        async def progress():
            while True:
                await asyncio.sleep(10)
                elapsed = int(time.time() - t0)
                rag_pct = (stats["rag_ok"] / max(rag_chunk_count, 1)) * 100
                m0_pct = (stats["mem0_ok"] / max(mem0_count, 1)) * 100
                print(
                    f"  [{elapsed}s] rag: {stats['rag_ok']}/{rag_chunk_count} ({rag_pct:.1f}%) {stats['rag_fail']}f"
                    f" | mem0: {stats['mem0_ok']}/{mem0_count} ({m0_pct:.1f}%) {stats['mem0_fail']}f"
                )

        progress_task = asyncio.create_task(progress())

        # run workers
        rag_workers = [asyncio.create_task(rag_worker(session, rag_queue)) for _ in range(RAG_CONCURRENCY)]
        mem0_workers = [asyncio.create_task(mem0_worker(session, mem0_queue)) for _ in range(MEM0_CONCURRENCY)]

        await asyncio.gather(*rag_workers, *mem0_workers)
        progress_task.cancel()

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        print(f"  RAG:  {stats['rag_ok']} ok, {stats['rag_fail']} fail ({rag_chunk_count} chunks)")
        print(f"  mem0: {stats['mem0_ok']} ok, {stats['mem0_fail']} fail ({mem0_count} items)")
        print(f"  Deduped: {skip_count}")


if __name__ == "__main__":
    asyncio.run(main())
