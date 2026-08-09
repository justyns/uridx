from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import select

from uridx.config import URIDX_MIN_SCORE, URIDX_RESCORE_K
from uridx.db.engine import get_session
from uridx.db.models import Chunk, Item, Tag
from uridx.embeddings import get_embeddings_sync, serialize_embedding
from uridx.record import TITLE_CHUNK_KEY
from uridx.search.query import process_query


@dataclass
class SearchResult:
    source_uri: str
    title: str | None
    source_type: str | None
    chunk_text: str
    score: float
    """Rank within this result set, in (0, 1]. Relative: the best hit always scores 1.0,
    however good or bad it is. Use it to order results, never to judge them.
    0.0 for the unranked empty-query listing, which orders by date instead."""
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    relevance: float | None = None
    """Cosine similarity to the query, in [-1, 1]. Absolute and comparable across
    searches, which is what --min-score thresholds. None for keyword-only search,
    which never embeds the query."""


def _rrf(result_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Combine ranked result lists using Reciprocal Rank Fusion.

    Each input list is [(doc_id, score)] ordered by the source ranker; only the
    ordering is used, never the raw score. Returns [(doc_id, normalized_score)] in
    (0, 1] sorted descending. Also used with a single list, to put one ranker's
    output on that same scale: the rest of the pipeline (recency mixing, min_score,
    the final sort) assumes higher is better and O(1) magnitudes.
    """
    fused: dict[int, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    if not fused:
        return []
    max_score = max(fused.values())
    fused = {doc_id: score / max_score for doc_id, score in fused.items()}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _load_tags(session, item_ids: set[int]) -> dict[int, list[str]]:
    """Map each item_id to its list of tags."""
    tags_by_item: dict[int, list[str]] = {}
    for t in session.exec(select(Tag).where(Tag.item_id.in_(item_ids))).all():
        tags_by_item.setdefault(t.item_id, []).append(t.tag)
    return tags_by_item


def _apply_item_filters(
    stmt,
    source_type: str | None,
    source_prefix: str | None,
    after: datetime | None,
    tags: list[str] | None = None,
):
    """Apply the shared source_type / source_prefix / after / tags WHERE clauses to a Chunk+Item select."""
    if source_type:
        stmt = stmt.where(Item.source_type == source_type)
    if source_prefix:
        stmt = stmt.where(Item.source_uri.startswith(source_prefix))
    if after:
        stmt = stmt.where(Item.created_at >= after)
    for t in tags or []:
        stmt = stmt.where(Item.id.in_(select(Tag.item_id).where(Tag.tag == t)))
    return stmt


def _preview_chunk_id():
    """Correlated subquery for the chunk that best previews an item.

    Body before title (the title is already displayed on its own line, so previewing
    it would just repeat it), then lowest chunk_index. Always resolves as long as the
    item has any chunk at all.
    """
    return (
        select(Chunk.id)
        .where(Chunk.item_id == Item.id)
        .order_by((Chunk.chunk_key == TITLE_CHUNK_KEY).asc(), Chunk.chunk_index)
        .limit(1)
        .correlate(Item)
        .scalar_subquery()
    )


def _list_results(
    *,
    limit: int,
    source_type: str | None,
    tags: list[str] | None,
    source_prefix: str | None,
    after: datetime | None,
) -> list[SearchResult]:
    """Empty-query listing: items matching the filters, newest first, unranked.

    Lets `search` browse a slice of the index (e.g. a whole thread via
    `--tag thread:<id>`) without a ranking query.
    """
    with get_session() as session:
        stmt = select(Chunk, Item).join(Item, Chunk.item_id == Item.id).where(Chunk.id == _preview_chunk_id())
        stmt = _apply_item_filters(stmt, source_type, source_prefix, after, tags)
        stmt = stmt.order_by(Item.created_at.desc()).limit(limit)
        rows = session.exec(stmt).all()
        tags_by_item = _load_tags(session, {item.id for _, item in rows})

        return [
            SearchResult(
                source_uri=item.source_uri,
                title=item.title,
                source_type=item.source_type,
                chunk_text=chunk.text,
                score=0.0,
                tags=tags_by_item.get(item.id, []),
                created_at=item.created_at,
            )
            for chunk, item in rows
        ]


def _vector_search(conn, embedding_blob: bytes, limit: int) -> list[tuple[int, float]]:
    """Two-stage KNN: coarse hamming scan over sign-bit vectors, exact rescore of the survivors.

    The bit vectors are 32x smaller than the floats, so the full-table pass is cheap;
    only the URIDX_RESCORE_K coarse candidates pay full float distance math.

    The coarse pass is bound by per-row overhead, not distance math: most of its time goes on
    walking rows and one scalar-function call each. Packing the codes into wide blobs scored
    by a single vectorized pass is measurably faster for identical results; tried and backed
    out because a second copy of the vectors needs its own invalidation protocol. Worth
    revisiting if this gets hot.
    """
    query_bits = conn.exec_driver_sql("SELECT vec_quantize_binary(?)", (embedding_blob,)).scalar()
    candidates = [
        row[0]
        for row in conn.exec_driver_sql(
            "SELECT chunk_id FROM chunk_embeddings_bq "
            "ORDER BY vec_distance_hamming(vec_bit(embedding), vec_bit(?)) LIMIT ?",
            (query_bits, max(URIDX_RESCORE_K, limit)),
        )
    ]
    if not candidates:
        return []
    placeholders = ",".join("?" * len(candidates))
    return conn.exec_driver_sql(
        "SELECT chunk_id, vec_distance_cosine(embedding, ?) AS distance FROM chunk_embeddings "
        f"WHERE chunk_id IN ({placeholders}) ORDER BY distance LIMIT ?",
        (embedding_blob, *candidates, limit),
    ).fetchall()


def _relevance_by_chunk(conn, embedding_blob: bytes, chunk_ids: list[int]) -> dict[int, float]:
    """Cosine similarity of each chunk to the query, in [-1, 1].

    Computed for every ranked chunk, not just the ones the vector pass surfaced, so a
    result that arrived via BM25 alone still gets an honest relevance number.
    """
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.exec_driver_sql(
        "SELECT chunk_id, 1.0 - vec_distance_cosine(embedding, ?) FROM chunk_embeddings "
        f"WHERE chunk_id IN ({placeholders})",
        (embedding_blob, *chunk_ids),
    )
    return {chunk_id: relevance for chunk_id, relevance in rows}


MAX_AGE_DAYS = 365.0


def _recency_score(created_at: datetime | None, now: datetime) -> float:
    """Freshness in [0, 1]: 1 for today, falling linearly to 0 at MAX_AGE_DAYS and older.

    Clamped at both ends. Without the upper clamp a source dated in the future (a
    restored archive, a skewed clock, a touched mtime) scores above 1 without bound
    and swamps the relevance term it is supposed to be nudging.
    """
    if created_at is None:
        return 0.0
    created = created_at.replace(tzinfo=timezone.utc) if created_at.tzinfo is None else created_at
    age_days = (now - created).total_seconds() / 86400
    return min(1.0, max(0.0, 1.0 - age_days / MAX_AGE_DAYS))


def _fts_search(conn, fts_query: str, limit: int) -> list[tuple[int, float]]:
    """Best-first BM25 matches. Note bm25() is negative and *descends* with relevance,
    so the rows are ordered ascending and the raw scores must be re-ranked before use."""
    if not fts_query:
        return []
    return conn.exec_driver_sql(
        "SELECT rowid, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (fts_query, limit * 3),
    ).fetchall()


def hybrid_search(
    query: str,
    limit: int = 10,
    source_type: str | None = None,
    tags: list[str] | None = None,
    semantic: bool = True,
    recency_boost: float = 0.3,
    min_score: float | None = None,
    source_prefix: str | None = None,
    after: datetime | None = None,
) -> list[SearchResult]:
    """Rank chunks against the query, fusing BM25 and vector search.

    Ordering comes from Reciprocal Rank Fusion, optionally nudged by recency.
    `min_score` is a floor on cosine *relevance*, not on the fused rank score: the
    top-ranked hit is always 1.0 even for a query matching nothing, so thresholding
    the rank could never drop a bad result. It therefore only applies to semantic
    search; keyword-only search never embeds the query and has no relevance to test.
    """
    if min_score is None:
        min_score = URIDX_MIN_SCORE

    if not query.strip():
        return _list_results(limit=limit, source_type=source_type, tags=tags, source_prefix=source_prefix, after=after)

    fts_query = process_query(query)
    query_blob = serialize_embedding(get_embeddings_sync([query])[0]) if semantic else None

    with get_session() as session:
        conn = session.connection()
        rankings = [_fts_search(conn, fts_query, limit)]
        if semantic:
            rankings.append(_vector_search(conn, query_blob, limit * 3))
        ranked_chunks = _rrf(rankings)
        if not ranked_chunks:
            return []

        chunk_ids = [c[0] for c in ranked_chunks]
        score_map = {c[0]: c[1] for c in ranked_chunks}

        stmt = select(Chunk, Item).join(Item, Chunk.item_id == Item.id).where(Chunk.id.in_(chunk_ids))
        stmt = _apply_item_filters(stmt, source_type, source_prefix, after, tags)

        results_data = session.exec(stmt).all()
        tags_by_item = _load_tags(session, {item.id for _, item in results_data})
        relevance = _relevance_by_chunk(conn, query_blob, chunk_ids) if semantic else {}

        results = [
            SearchResult(
                source_uri=item.source_uri,
                title=item.title,
                source_type=item.source_type,
                chunk_text=chunk.text,
                score=score_map.get(chunk.id, 0.0),
                tags=tags_by_item.get(item.id, []),
                created_at=item.created_at,
                relevance=relevance.get(chunk.id),
            )
            for chunk, item in results_data
        ]

        if recency_boost > 0 and results:
            now = datetime.now(timezone.utc)
            for r in results:
                r.score = r.score * (1 - recency_boost) + _recency_score(r.created_at, now) * recency_boost

        results.sort(key=lambda x: x.score, reverse=True)
        if min_score is not None and semantic:
            results = [r for r in results if r.relevance is not None and r.relevance >= min_score]
        return results[:limit]
