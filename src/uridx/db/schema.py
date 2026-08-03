"""The parts of the schema SQLModel does not model, and the migrations that heal them.

SQLModel's `create_all` owns the plain tables (item, chunk, tag, location, setting).
This module owns everything hand-rolled — the FTS5 virtual table and its triggers,
and the vector tables — plus the in-place migrations that bring an older database up
to the current shape. Every function takes a SQLAlchemy Connection and runs in that
connection's transaction, except _migrate_vec0_table, which commits as it goes to
keep the WAL bounded while copying a large vector table.

The migrations here are one-time, but every one of them costs a full table scan on a
large index. `apply_schema` therefore runs them only when the stored schema_version is
behind, so ordinary commands pay one indexed lookup instead of a pass over every table. The
recurring work, re-deriving the bit vectors when a writer drifts them, is split out into
`heal_bq_drift`, which only the write path calls.
"""

from sqlalchemy import Connection

VEC0_MIGRATION_PAGE = 5000

SCHEMA_VERSION = 1
"""Bump when a migration below needs to re-run on databases that already ran the others."""


def _sql_for(conn: Connection, kind: str, name: str) -> str | None:
    """The stored CREATE statement for a table/trigger, or None if it doesn't exist."""
    row = conn.exec_driver_sql("SELECT sql FROM sqlite_master WHERE type = ? AND name = ?", (kind, name)).fetchone()
    return (row[0] or "") if row else None


def create_embeddings_tables(conn: Connection) -> None:
    """Create the vector tables — the one definition of the vector schema.

    Both are plain tables on purpose (sqlite-vec is used only for its scalar
    functions): vec0 virtual tables cost ~2x more to scan row-by-row and read a
    multi-MB vector chunk per point lookup, which made rescoring 100x slower.
    chunk_embeddings holds float32 blobs (exact rescoring, export); the derived
    chunk_embeddings_bq holds sign-bit blobs for the coarse hamming pass.
    Dimension is recorded in settings, not the schema.
    """
    conn.exec_driver_sql("CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")
    conn.exec_driver_sql("CREATE TABLE chunk_embeddings_bq (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")


def _migrate_vec0_table(conn: Connection) -> None:
    """One-time rebuild of a legacy vec0 chunk_embeddings into a plain table.

    Copies in batches (keyset by chunk_id) to keep the WAL bounded; a partial
    prior attempt (chunk_embeddings_new left behind) is discarded and restarted.
    """
    conn.exec_driver_sql("DROP TABLE IF EXISTS chunk_embeddings_new")
    conn.exec_driver_sql("CREATE TABLE chunk_embeddings_new (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")
    last_id = 0
    while True:
        rows = conn.exec_driver_sql(
            "SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id > ? ORDER BY chunk_id LIMIT ?",
            (last_id, VEC0_MIGRATION_PAGE),
        ).fetchall()
        if not rows:
            break
        conn.exec_driver_sql("INSERT INTO chunk_embeddings_new VALUES (?, ?)", [tuple(r) for r in rows])
        conn.commit()
        last_id = rows[-1][0]
    conn.exec_driver_sql("DROP TABLE chunk_embeddings")
    conn.exec_driver_sql("ALTER TABLE chunk_embeddings_new RENAME TO chunk_embeddings")
    conn.commit()


def _ensure_vector_tables(conn: Connection) -> None:
    """Migrate legacy vec0 vector tables to plain ones, and ensure both exist.

    One-time structural work only. Re-deriving the bit table when a writer drifts it
    is recurring, so it lives in `heal_bq_drift` instead.
    """
    float_sql = _sql_for(conn, "table", "chunk_embeddings")
    if float_sql is None:
        create_embeddings_tables(conn)
        return
    if "VIRTUAL" in float_sql.upper():
        _migrate_vec0_table(conn)

    bq_sql = _sql_for(conn, "table", "chunk_embeddings_bq")
    if bq_sql is not None and "VIRTUAL" in bq_sql.upper():
        conn.exec_driver_sql("DROP TABLE chunk_embeddings_bq")  # early vec0-based variant
        bq_sql = None
    if bq_sql is None:
        conn.exec_driver_sql("CREATE TABLE chunk_embeddings_bq (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")


def heal_bq_drift(conn: Connection) -> None:
    """Re-derive chunk_embeddings_bq from the float table when the two have drifted.

    Ordinary ingests and deletes keep the bit table in step row by row; drift only comes
    from an older writer that filled the float table alone. Bit vectors are pure derived
    data, so a wholesale rebuild is always safe and cheap (one pass of vec_quantize_binary).

    Both counts are full table scans, so only the write path calls this. Readers never need
    to repair derived data, and `search`/`stats` would otherwise scan both tables on every
    invocation for a check that almost always finds nothing to do.
    """
    n_float = conn.exec_driver_sql("SELECT count(*) FROM chunk_embeddings").scalar()
    n_bq = conn.exec_driver_sql("SELECT count(*) FROM chunk_embeddings_bq").scalar()
    if n_bq != n_float:
        conn.exec_driver_sql("DELETE FROM chunk_embeddings_bq")
        conn.exec_driver_sql(
            "INSERT INTO chunk_embeddings_bq (chunk_id, embedding) "
            "SELECT chunk_id, vec_quantize_binary(embedding) FROM chunk_embeddings"
        )


def _ensure_fts_table(conn: Connection) -> None:
    """Ensure the FTS table and its triggers exist with the current configuration.

    Recreated (and repopulated from chunk) when an older database is missing
    contentless_delete, still carries the dropped `context` column, or has triggers
    using the pre-contentless_delete `VALUES('delete', ...)` syntax.
    """
    create_sql = _sql_for(conn, "table", "chunks_fts")
    trigger_sql = _sql_for(conn, "trigger", "chunks_ad") or ""
    stale = create_sql is not None and (
        "contentless_delete=1" not in create_sql
        or "context" in create_sql
        or "VALUES('delete'" in trigger_sql
        or "VALUES ('delete'" in trigger_sql
    )
    if create_sql is not None and not stale:
        return

    if stale:
        for trigger in ("chunks_ai", "chunks_ad", "chunks_au"):
            conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.exec_driver_sql("DROP TABLE chunks_fts")

    conn.exec_driver_sql("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='', contentless_delete=1)")
    conn.exec_driver_sql(
        "CREATE TRIGGER chunks_ai AFTER INSERT ON chunk BEGIN "
        "INSERT INTO chunks_fts(rowid, text) VALUES (NEW.id, NEW.text); END"
    )
    conn.exec_driver_sql(
        "CREATE TRIGGER chunks_ad AFTER DELETE ON chunk BEGIN DELETE FROM chunks_fts WHERE rowid = OLD.id; END"
    )
    conn.exec_driver_sql(
        "CREATE TRIGGER chunks_au AFTER UPDATE ON chunk BEGIN "
        "DELETE FROM chunks_fts WHERE rowid = OLD.id; "
        "INSERT INTO chunks_fts(rowid, text) VALUES (NEW.id, NEW.text); END"
    )
    if stale:
        conn.exec_driver_sql("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunk")


def _ensure_item_content_hash(conn: Connection) -> None:
    """Add the content_hash column and its index to databases predating them.

    create_all only indexes tables it creates, so the index needs its own IF NOT EXISTS.
    """
    columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(item)").fetchall()]
    if "content_hash" not in columns:
        conn.exec_driver_sql("ALTER TABLE item ADD COLUMN content_hash TEXT")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_item_content_hash ON item (content_hash)")


def _backfill_locations(conn: Connection) -> None:
    """Give pre-existing items the location row that ingest now writes for them."""
    conn.exec_driver_sql(
        "INSERT OR IGNORE INTO location (item_id, uri, added_at) "
        "SELECT id, source_uri, created_at FROM item WHERE id NOT IN (SELECT DISTINCT item_id FROM location)"
    )


def _stored_schema_version(conn: Connection) -> int:
    """The version this database was last migrated to; 0 if absent or unreadable.

    0 re-runs the migrations, which are idempotent, so a corrupted value costs one slow
    startup rather than silently leaving the schema unmigrated.
    """
    row = conn.exec_driver_sql("SELECT value FROM setting WHERE key = 'schema_version'").fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def apply_schema(conn: Connection) -> None:
    """Bring the hand-rolled schema up to date. Idempotent; safe on every startup.

    Every migration below is a full table scan, so they run only when the database is
    behind SCHEMA_VERSION. On an up-to-date index this is one indexed lookup.
    """
    if _stored_schema_version(conn) >= SCHEMA_VERSION:
        return

    _ensure_item_content_hash(conn)
    _backfill_locations(conn)
    _ensure_vector_tables(conn)
    _ensure_fts_table(conn)

    conn.exec_driver_sql(
        "INSERT INTO setting (key, value, updated_at) VALUES ('schema_version', ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (str(SCHEMA_VERSION),),
    )
