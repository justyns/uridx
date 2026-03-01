import struct

from uridx.config import (
    FASTEMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    URIDX_EMBEDDINGS,
)


def get_dimension() -> int:
    if URIDX_EMBEDDINGS == "ollama":
        from uridx.embeddings.ollama import get_dimension as _get_dimension

        return _get_dimension(OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    else:
        from uridx.embeddings.fastembed import get_dimension as _get_dimension

        return _get_dimension(FASTEMBED_MODEL)


def get_embeddings_sync(texts: list[str]) -> list[list[float]]:
    if URIDX_EMBEDDINGS == "ollama":
        from uridx.embeddings.ollama import get_embeddings_sync as _get_embeddings

        return _get_embeddings(texts, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    else:
        from uridx.embeddings.fastembed import get_embeddings_sync as _get_embeddings

        return _get_embeddings(texts, FASTEMBED_MODEL)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    if URIDX_EMBEDDINGS == "ollama":
        from uridx.embeddings.ollama import get_embeddings as _get_embeddings

        return await _get_embeddings(texts, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    else:
        from uridx.embeddings.fastembed import get_embeddings as _get_embeddings

        return await _get_embeddings(texts, FASTEMBED_MODEL)


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
