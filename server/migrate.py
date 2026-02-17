"""Migrate existing OpenClaw memory files into Mem0 via async REST API."""

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

MEM0_URL = "http://localhost:8080"
USER_ID = "jinstronda"
WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"
MEMORY_MD = WORKSPACE / "MEMORY.md"
CHUNK_THRESHOLD = 6000
MAX_CONCURRENT = 50


def chunk_by_sections(text: str, max_chars: int = CHUNK_THRESHOLD) -> list:
    lines = text.split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if line.startswith("## ") and current_len > 200:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        elif current_len + line_len > max_chars and current_len > 200:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return [c.strip() for c in chunks if c.strip() and len(c.strip()) > 20]


async def post(session: aiohttp.ClientSession, path: str, body: dict) -> dict:
    async with session.post(MEM0_URL + path, json=body, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        return await resp.json()


async def ingest_chunk(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    content: str,
    filename: str,
    chunk_info: str,
    metadata: dict,
    stats: dict,
):
    async with sem:
        try:
            result = await post(session, "/v1/memories/", {
                "messages": content,
                "user_id": USER_ID,
                "metadata": metadata,
                "infer": True,
            })
            facts = len(result.get("results", []))
            print(f"  {chunk_info}: {facts} facts ({len(content)} chars)", flush=True)
            stats["facts"] += facts
        except Exception as e:
            print(f"  {chunk_info}: ERROR {e}", flush=True)
            stats["errors"] += 1


async def migrate():
    stats = {"files": 0, "facts": 0, "errors": 0}
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []

    all_files = []
    if MEMORY_MD.exists():
        all_files.append(("MEMORY.md", MEMORY_MD))
    if MEMORY_DIR.exists():
        for md in sorted(MEMORY_DIR.glob("*.md")):
            all_files.append((md.name, md))

    print(f"migrating {len(all_files)} files (max {MAX_CONCURRENT} concurrent)\n", flush=True)

    async with aiohttp.ClientSession() as session:
        for filename, filepath in all_files:
            content = filepath.read_text().strip()
            if not content or len(content) < 20:
                continue

            stats["files"] += 1
            if len(content) > CHUNK_THRESHOLD:
                chunks = chunk_by_sections(content)
                print(f"  {filename} ({len(content)} chars, {len(chunks)} chunks)", flush=True)
                for i, chunk in enumerate(chunks):
                    meta = {"source": "migration", "original_file": filename, "chunk": i + 1}
                    info = f"{filename} [{i+1}/{len(chunks)}]"
                    tasks.append(ingest_chunk(session, sem, chunk, filename, info, meta, stats))
            else:
                meta = {"source": "migration", "original_file": filename}
                tasks.append(ingest_chunk(session, sem, content, filename, filename, meta, stats))

        print(f"\n{len(tasks)} total requests queued, processing...\n", flush=True)
        await asyncio.gather(*tasks)

    print(f"\ndone. {stats['files']} files, {stats['facts']} facts extracted, {stats['errors']} errors.", flush=True)


if __name__ == "__main__":
    asyncio.run(migrate())
