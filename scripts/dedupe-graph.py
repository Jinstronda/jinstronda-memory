#!/usr/bin/env python3
"""One-time graph deduplication via the mem0 server's /graph/dedupe endpoint.

Usage:
    python3 scripts/dedupe-graph.py [--dry-run]
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

MEM0_URL = "http://localhost:3848"
USER_ID = "openclaw_Joaos_MacBook_Pro_local"


def post(url: str, data: dict, timeout: int = 1800) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Deduplicate graph relationships")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying")
    args = parser.parse_args()

    # health check
    try:
        with urllib.request.urlopen(f"{MEM0_URL}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            if not data.get("ok"):
                print("mem0 server not healthy")
                sys.exit(1)
    except Exception as e:
        print(f"mem0 server unreachable: {e}")
        sys.exit(1)

    print(f"Running graph dedup {'(DRY RUN)' if args.dry_run else '(LIVE)'}...")
    print(f"Server: {MEM0_URL}")
    print(f"User: {USER_ID}")
    print()

    result = post(f"{MEM0_URL}/graph/dedupe", {"user_id": USER_ID, "dry_run": args.dry_run})

    print(f"Edges before: {result.get('edges_before', '?')}")
    print(f"Garbage deleted: {result.get('garbage_deleted', 0)}")
    print(f"Clusters merged: {result.get('clusters_merged', 0)}")
    print(f"Edges deleted (dedup): {result.get('edges_deleted', 0)}")
    print(f"Edges after: {result.get('edges_after', '?')}")

    before = result.get("edges_before", 0)
    after = result.get("edges_after", 0)
    if before > 0:
        reduction = ((before - after) / before) * 100
        print(f"\nReduction: {before - after} edges ({reduction:.1f}%)")


if __name__ == "__main__":
    main()
