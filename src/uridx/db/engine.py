import sqlite3
from pathlib import Path

import sqlite_vec
from sqlmodel import Session, SQLModel, create_engine

from uridx.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, URIDX_DB_PATH
from uridx.db.models import Setting
from uridx.embeddings.ollama import get_dimension

_engine = None


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

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    if not cursor.fetchone():
        cursor.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                context,
                content='',
                contentless_delete=1
            )
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunk BEGIN
                INSERT INTO chunks_fts(rowid, text, context)
                SELECT NEW.id, NEW.text, COALESCE(
                    (SELECT context FROM item WHERE id = NEW.item_id), ''
                );
            END
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunk BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, context)
                SELECT 'delete', OLD.id, OLD.text, COALESCE(
                    (SELECT context FROM item WHERE id = OLD.item_id), ''
                );
            END
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER chunks_au AFTER UPDATE ON chunk BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, context)
                SELECT 'delete', OLD.id, OLD.text, COALESCE(
                    (SELECT context FROM item WHERE id = OLD.item_id), ''
                );
                INSERT INTO chunks_fts(rowid, text, context)
                SELECT NEW.id, NEW.text, COALESCE(
                    (SELECT context FROM item WHERE id = NEW.item_id), ''
                );
            END
            """
        )

    conn.commit()
    conn.close()
