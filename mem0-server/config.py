import os


def get_mem0_config(data_dir: str = "./data") -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_LLM_MODEL", "gpt-5-nano"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small"),
                "embedding_dims": 1536,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "jinstronda_memories",
                "path": f"{data_dir}/qdrant",
                "embedding_model_dims": 1536,
                "on_disk": True,
            },
        },
        "graph_store": {
            "provider": "kuzu",
            "config": {
                "db": f"{data_dir}/graph.kuzu",
                "custom_prompt": (
                    "4. Use simple, canonical relationship names (1-3 words max). "
                    "Prefer 'member_of' over 'is_member_of_group' or 'participates_in_group'. "
                    "Prefer 'friend' over 'best_friend_with'. "
                    "Never include message content, quotes, or chat text in relationship names. "
                    "Never create relationships to timestamps, dates, or system identifiers. "
                    "Reuse existing relationship verbs when the meaning is equivalent."
                ),
            },
        },
    }
