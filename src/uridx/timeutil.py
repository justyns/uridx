"""The two UTC representations uridx uses, and the conversions between them.

The database stores *naive* UTC datetimes; extractors, CLI options and the JSONL
wire format use *aware* ISO-8601. Anything crossing that boundary goes through
here, so there is one place that decides how a naive input is interpreted (as UTC)
and how a malformed one behaves (raises ValueError rather than filtering silently).
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Now, in the naive UTC the database stores (datetime.utcnow is deprecated)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_aware_utc(value: datetime | str | None) -> datetime | None:
    """Coerce an ISO-8601 string or datetime to aware UTC. Naive input is read as UTC.

    A trailing `Z` is accepted (fromisoformat handles it since 3.11).
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def to_naive_utc(value: datetime | str | None) -> datetime | None:
    """Coerce an ISO-8601 string or datetime to the naive UTC the database stores."""
    aware = to_aware_utc(value)
    return aware.replace(tzinfo=None) if aware is not None else None
