"""The single connection to the index.

Every reader and writer goes through this engine. sqlite-vec is loaded onto each
pooled connection as it opens, so its scalar functions are available to plain
`session.connection().exec_driver_sql(...)`. There is no second, separately-opened
sqlite3 handle, and raw vector SQL shares a transaction with the ORM writes around it.
"""

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from uridx.config import URIDX_DB_PATH
from uridx.db.models import Setting
from uridx.db.schema import apply_schema, heal_bq_drift
from uridx.embeddings import EmbeddingBackendError, get_dimension, get_model_id

_engine = None
_db_initialized = False


@lru_cache(maxsize=1)
def _vec_extension_path() -> str:
    """Path to the bundled sqlite-vec loadable extension, without importing the package.

    `import sqlite_vec` pulls in numpy.typing through a try-block for register_numpy,
    which uridx never uses and which is not cheap to import. Only the path is needed,
    and resolving it does not require executing the package.
    """
    spec = importlib.util.find_spec("sqlite_vec")
    if spec is None or not spec.origin:
        raise ImportError("sqlite-vec is not installed")
    return os.path.normpath(os.path.join(os.path.dirname(spec.origin), "vec0"))


def _on_connect(dbapi_conn, connection_record):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA busy_timeout=5000")
    dbapi_conn.enable_load_extension(True)
    dbapi_conn.load_extension(_vec_extension_path())
    dbapi_conn.enable_load_extension(False)


def get_engine():
    global _engine
    if _engine is None:
        db_path = Path(URIDX_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        event.listen(_engine, "connect", _on_connect)
    return _engine


def get_session() -> Session:
    return Session(get_engine())


def get_setting(key: str) -> str | None:
    with get_session() as session:
        setting = session.get(Setting, key)
        return setting.value if setting else None


def require_embed_model_match():
    """Raise if the configured embedding model differs from the one the index was
    embedded with. Called by commands that read or write vectors; `reindex` (the fix)
    and vector-free commands like stats/delete skip it."""
    stored = get_setting("embed_model")
    model_id = get_model_id()
    if stored and stored != model_id:
        raise EmbeddingBackendError(
            f"this index was embedded with {stored}, but the current config is {model_id}.\n"
            "Run `uridx reindex` to re-embed the index, or restore the original "
            "URIDX_EMBEDDINGS / FASTEMBED_MODEL / OLLAMA_EMBED_MODEL settings."
        )


def init_db():
    global _db_initialized
    if _db_initialized:
        return
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

    with get_session() as session:
        if session.get(Setting, "embed_dimension") is None:
            session.add(Setting(key="embed_dimension", value=str(get_dimension())))
        if session.get(Setting, "embed_model") is None:
            session.add(Setting(key="embed_model", value=get_model_id()))
        session.commit()

    with engine.connect() as conn:
        apply_schema(conn)
        conn.commit()

    _db_initialized = True


def heal_index():
    """Re-derive the bit vectors if a writer drifted them. Write path only, after init_db()."""
    with get_engine().connect() as conn:
        heal_bq_drift(conn)
        conn.commit()
