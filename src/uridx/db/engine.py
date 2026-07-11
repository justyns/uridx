import sqlite3
import sys
from pathlib import Path

import sqlite_vec
from sqlmodel import Session, SQLModel, create_engine

from uridx.config import URIDX_DB_PATH
from uridx.db.models import Setting
from uridx.embeddings import get_dimension, get_model_id

_engine = None


def _ensure_fts_table(cursor):
    """Ensure FTS table exists with correct configuration, migrating if needed."""
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    row = cursor.fetchone()

    needs_recreate = False
    if row:
        create_sql = row[0] or ""
        # Check if triggers use old FTS delete command syntax (VALUES('delete',...))
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='chunks_ad'")
        trigger_row = cursor.fetchone()
        trigger_sql = trigger_row[0] if trigger_row else ""
        uses_old_delete = "VALUES('delete'" in trigger_sql or "VALUES ('delete'" in trigger_sql

        # Recreate if: missing contentless_delete, has context column, or triggers use old syntax
        if "contentless_delete=1" not in create_sql or "context" in create_sql or uses_old_delete:
            needs_recreate = True
            cursor.execute("DROP TRIGGER IF EXISTS chunks_ai")
            cursor.execute("DROP TRIGGER IF EXISTS chunks_ad")
            cursor.execute("DROP TRIGGER IF EXISTS chunks_au")
            cursor.execute("DROP TABLE chunks_fts")

    if not row or needs_recreate:
        cursor.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                content='',
                contentless_delete=1
            )
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunk BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (NEW.id, NEW.text);
            END
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunk BEGIN
                DELETE FROM chunks_fts WHERE rowid = OLD.id;
            END
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER chunks_au AFTER UPDATE ON chunk BEGIN
                DELETE FROM chunks_fts WHERE rowid = OLD.id;
                INSERT INTO chunks_fts(rowid, text) VALUES (NEW.id, NEW.text);
            END
            """
        )

        if needs_recreate:
            cursor.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunk")


def create_embeddings_table(cursor) -> None:
    """Create the vector tables — the one definition of the vector schema.

    Both are plain tables on purpose (sqlite-vec is used only for its scalar
    functions): vec0 virtual tables cost ~2x more to scan row-by-row and read a
    multi-MB vector chunk per point lookup, which made rescoring 100x slower.
    chunk_embeddings holds float32 blobs (exact rescoring, export); the derived
    chunk_embeddings_bq holds sign-bit blobs for the coarse hamming pass.
    Dimension is recorded in settings, not the schema.
    """
    cursor.execute("CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")
    cursor.execute("CREATE TABLE chunk_embeddings_bq (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")


def _migrate_vec0_table(cursor) -> None:
    """One-time rebuild of a legacy vec0 chunk_embeddings into a plain table.

    Copies in batches (keyset by chunk_id) to keep the WAL bounded; a partial
    prior attempt (chunk_embeddings_new left behind) is discarded and restarted.
    """
    cursor.execute("DROP TABLE IF EXISTS chunk_embeddings_new")
    cursor.execute("CREATE TABLE chunk_embeddings_new (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")
    last_id = 0
    while True:
        rows = cursor.execute(
            "SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id > ? ORDER BY chunk_id LIMIT 5000",
            (last_id,),
        ).fetchall()
        if not rows:
            break
        cursor.executemany("INSERT INTO chunk_embeddings_new VALUES (?, ?)", rows)
        cursor.connection.commit()
        last_id = rows[-1][0]
    cursor.execute("DROP TABLE chunk_embeddings")
    cursor.execute("ALTER TABLE chunk_embeddings_new RENAME TO chunk_embeddings")
    cursor.connection.commit()


def _ensure_vector_tables(cursor) -> None:
    """Migrate legacy vec0 tables to plain ones and re-derive the bit table on drift.

    Drift happens when an older writer fills only the float table (or deletes
    from it); bit vectors are pure derived data, so a wholesale rebuild is
    always safe and cheap (one pass of vec_quantize_binary).
    """
    float_sql = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
    ).fetchone()[0]
    if "VIRTUAL" in float_sql.upper():
        _migrate_vec0_table(cursor)
    bq_row = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_embeddings_bq'"
    ).fetchone()
    if bq_row and "VIRTUAL" in (bq_row[0] or "").upper():
        cursor.execute("DROP TABLE chunk_embeddings_bq")  # early vec0-based variant
        bq_row = None
    if not bq_row:
        cursor.execute("CREATE TABLE chunk_embeddings_bq (chunk_id INTEGER PRIMARY KEY, embedding BLOB)")
    n_float = cursor.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0]
    n_bq = cursor.execute("SELECT count(*) FROM chunk_embeddings_bq").fetchone()[0]
    if n_bq != n_float:
        cursor.execute("DELETE FROM chunk_embeddings_bq")
        cursor.execute(
            "INSERT INTO chunk_embeddings_bq (chunk_id, embedding) "
            "SELECT chunk_id, vec_quantize_binary(embedding) FROM chunk_embeddings"
        )


def _set_sqlite_pragmas(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    db_path = Path(URIDX_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    from sqlalchemy import event

    event.listen(_engine, "connect", _set_sqlite_pragmas)

    return _engine


def _load_extensions(conn: sqlite3.Connection):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def get_raw_connection() -> sqlite3.Connection:
    db_path = Path(URIDX_DB_PATH)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _load_extensions(conn)
    return conn


def get_session() -> Session:
    return Session(get_engine())


def get_setting(key: str) -> str | None:
    with get_session() as session:
        setting = session.get(Setting, key)
        return setting.value if setting else None


_db_initialized = False


def require_embed_model_match():
    """Exit if the configured embedding model differs from the one the index was
    embedded with. Called by commands that read or write vectors; `reindex` (the fix)
    and vector-free commands like stats/delete skip it."""
    stored = get_setting("embed_model")
    model_id = get_model_id()
    if stored and stored != model_id:
        print(
            f"Error: this index was embedded with {stored}, "
            f"but the current config is {model_id}.\n"
            "Run `uridx reindex` to re-embed the index, or restore the original "
            "URIDX_EMBEDDINGS / FASTEMBED_MODEL / OLLAMA_EMBED_MODEL settings.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def init_db():
    global _db_initialized
    if _db_initialized:
        return
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

    with get_session() as session:
        existing = session.get(Setting, "embed_dimension")
        if existing:
            embed_dim = int(existing.value)
        else:
            embed_dim = get_dimension()
            session.add(Setting(key="embed_dimension", value=str(embed_dim)))
            session.commit()

        if session.get(Setting, "embed_model") is None:
            session.add(Setting(key="embed_model", value=get_model_id()))
            session.commit()

    conn = get_raw_connection()
    cursor = conn.cursor()

    # Migration: add content_hash column if missing (older DBs also lack its index,
    # since create_all only indexes tables it creates)
    cursor.execute("PRAGMA table_info(item)")
    columns = [row[1] for row in cursor.fetchall()]
    if "content_hash" not in columns:
        cursor.execute("ALTER TABLE item ADD COLUMN content_hash TEXT")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_item_content_hash ON item (content_hash)")

    # Migration: populate location table for existing items without locations
    # SQLModel creates the table, but we need to populate it for pre-existing items
    cursor.execute("""
        INSERT OR IGNORE INTO location (item_id, uri, added_at)
        SELECT id, source_uri, created_at FROM item
        WHERE id NOT IN (SELECT DISTINCT item_id FROM location)
    """)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'")
    if not cursor.fetchone():
        create_embeddings_table(cursor)
    else:
        _ensure_vector_tables(cursor)

    _ensure_fts_table(cursor)

    conn.commit()
    conn.close()
    _db_initialized = True
