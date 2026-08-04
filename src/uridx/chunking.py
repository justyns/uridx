import hashlib

from uridx.config import URIDX_MAX_CHUNK_CHARS
from uridx.record import TITLE_CHUNK_KEY, ChunkInput, Record


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on paragraph boundaries.

    Paragraphs (split on blank lines) are packed up to max_chars; a single paragraph
    longer than max_chars is hard-split. Empty input yields an empty list.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(para[i : i + max_chars] for i in range(0, len(para), max_chars))
        elif len(buf) + len(para) + 2 > max_chars:
            if buf:
                chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def compute_content_hash(chunks: list[ChunkInput]) -> str:
    """Compute SHA256 hash of chunk texts for change detection."""
    content = "\n".join(c.text for c in chunks)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _cap_chunk_sizes(chunks: list[ChunkInput]) -> list[ChunkInput]:
    """Split any chunk over URIDX_MAX_CHUNK_CHARS; pieces after the first get a `#n` key suffix.

    Embedding cost is quadratic in chunk length, so one uncapped chunk can demand a
    multi-GB allocation by itself.
    """
    capped = []
    for chunk in chunks:
        if len(chunk.text) <= URIDX_MAX_CHUNK_CHARS:
            capped.append(chunk)  # as-is, so an imported chunk keeps its embedding
            continue
        for i, piece in enumerate(split_text(chunk.text, URIDX_MAX_CHUNK_CHARS)):
            key = chunk.key if chunk.key is None or i == 0 else f"{chunk.key}#{i}"
            capped.append(ChunkInput(text=piece, key=key, meta=chunk.meta))
    return capped


def prepare_record(record: Record) -> str:
    """Normalize a record for indexing in place and return its content hash.

    Caps oversized chunks, then prepends the title as its own chunk so it's searchable
    (FTS + vector), unless one is already there (import round-trips). Runs before
    hashing, so retitling an item counts as a content change.
    """
    record.chunks = _cap_chunk_sizes(record.chunks)
    if record.title and not any(c.key == TITLE_CHUNK_KEY for c in record.chunks):
        # Truncated rather than split: the title chunk is one row by definition.
        record.chunks.insert(0, ChunkInput(text=record.title[:URIDX_MAX_CHUNK_CHARS], key=TITLE_CHUNK_KEY))
    return compute_content_hash(record.chunks)
