import struct

from uridx.config import (
    FASTEMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    URIDX_EMBEDDINGS,
)


def _backend():
    if URIDX_EMBEDDINGS == "ollama":
        from uridx.embeddings import ollama

        return ollama, (OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    from uridx.embeddings import fastembed

    return fastembed, (FASTEMBED_MODEL,)


def get_dimension() -> int:
    mod, args = _backend()
    return mod.get_dimension(*args)


def get_embeddings_sync(texts: list[str]) -> list[list[float]]:
    mod, args = _backend()
    return mod.get_embeddings_sync(texts, *args)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    mod, args = _backend()
    return await mod.get_embeddings(texts, *args)


def serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", data))


__all__ = [
    "get_dimension",
    "get_embeddings",
    "get_embeddings_sync",
    "serialize_embedding",
    "deserialize_embedding",
]
