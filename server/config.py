import os
from pathlib import Path

DATA_DIR = Path.home() / ".openclaw" / "mem0" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# prefer anthropic for extraction if key available, else openai
if ANTHROPIC_API_KEY:
    llm_config = {
        "provider": "anthropic",
        "config": {
            "model": "claude-opus-4-6",
            "api_key": ANTHROPIC_API_KEY,
        },
    }
else:
    llm_config = {
        "provider": "openai",
        "config": {
            "model": "gpt-5-mini",
            "api_key": OPENAI_API_KEY,
        },
    }

MEM0_CONFIG = {
    "llm": llm_config,
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": OPENAI_API_KEY,
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "openclaw_memories",
            "path": str(DATA_DIR / "chroma"),
        },
    },
    "history_db_path": str(DATA_DIR / "history.db"),
    "version": "v1.1",
}
