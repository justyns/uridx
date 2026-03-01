from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_model: "TextEmbedding | None" = None
_model_name: str | None = None

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

        _model = TextEmbedding(model_name=model_name)
        _model_name = model_name
    return _model


def get_dimension(model: str) -> int:
    if model in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model]
    embeddings = get_embeddings_sync(["test"], model)
    return len(embeddings[0])


def get_embeddings_sync(texts: list[str], model: str) -> list[list[float]]:
    embedding_model = _get_model(model)
    return [embedding.tolist() for embedding in embedding_model.embed(texts)]


async def get_embeddings(texts: list[str], model: str) -> list[list[float]]:
    return get_embeddings_sync(texts, model)
