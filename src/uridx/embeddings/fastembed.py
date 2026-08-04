from typing import TYPE_CHECKING

from uridx.embeddings import EmbeddingBackendError

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_model: "TextEmbedding | None" = None
_model_name: str | None = None

# Texts per ONNX forward pass (fastembed defaults to 256). Every batch pads to its longest
# member, so peak memory scales with this while throughput does not.
BATCH_SIZE = 16

MODEL_DIMENSIONS = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "snowflake/snowflake-arctic-embed-s": 384,
}


def _get_model(model_name: str) -> "TextEmbedding":
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from fastembed import TextEmbedding

        # The arena never returns memory to the OS, so RSS ratchets to the widest batch's
        # high-water mark and stays there. Off, it stays flat at no cost in throughput.
        _model = TextEmbedding(model_name=model_name, enable_cpu_mem_arena=False)
        _model_name = model_name
    return _model


def get_dimension(model: str) -> int:
    if model in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model]
    embeddings = get_embeddings_sync(["test"], model)
    return len(embeddings[0])


def get_embeddings_sync(texts: list[str], model: str) -> list[list[float]]:
    try:
        embedding_model = _get_model(model)
        return [embedding.tolist() for embedding in embedding_model.embed(texts, batch_size=BATCH_SIZE)]
    except Exception as e:
        raise EmbeddingBackendError(f"fastembed failed on {len(texts)} text(s) with {model}: {e}") from e
