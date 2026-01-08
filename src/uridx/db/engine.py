import sqlite3
from pathlib import Path

import sqlite_vec
from sqlmodel import Session, SQLModel, create_engine

from uridx.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, URIDX_DB_PATH
from uridx.db.models import Setting
from uridx.embeddings.ollama import get_dimension

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
    return _engine


def _load_extensions(conn: sqlite3.Connection):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def get_raw_connection() -> sqlite3.Connection:
    db_path = Path(URIDX_DB_PATH)
    conn = sqlite3.connect(str(db_path))
    _load_extensions(conn)
    return conn


def get_session() -> Session:
    return Session(get_engine())


def init_db():
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

    with get_session() as session:
        existing = session.get(Setting, "embed_dimension")
        if existing:
            embed_dim = int(existing.value)
        else:
            embed_dim = get_dimension(OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
            session.add(Setting(key="embed_model", value=OLLAMA_EMBED_MODEL))
            session.add(Setting(key="embed_dimension", value=str(embed_dim)))
            session.commit()

    conn = get_raw_connection()
    cursor = conn.cursor()

    # Migration: add content_hash column if missing
    cursor.execute("PRAGMA table_info(item)")
    columns = [row[1] for row in cursor.fetchall()]
    if "content_hash" not in columns:
        cursor.execute("ALTER TABLE item ADD COLUMN content_hash TEXT")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'")
    if not cursor.fetchone():
        cursor.execute(
            f"""
            CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{embed_dim}]
            )
            """
        )

    _ensure_fts_table(cursor)

    conn.commit()
    conn.close()
