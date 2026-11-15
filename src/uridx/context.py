"""Descriptions attached to path and URI prefixes.

A prefix is an absolute filesystem path (matched against `file://` source URIs), a source
URI prefix such as `claude-code://`, or `/` for everything.
"""

import os
from collections.abc import Mapping

GLOBAL_PREFIX = "/"


def normalize_prefix(raw: str) -> str:
    """Strip `file://` and trailing slashes, and make `~` and relative paths absolute."""
    raw = raw.removeprefix("file://")
    if "://" in raw:
        return raw if raw.endswith("://") else raw.rstrip("/")
    return os.path.abspath(os.path.expanduser(raw))


def prefix_matches(prefix: str, source_uri: str) -> bool:
    """True if source_uri is the prefix itself or lies under it, on a path boundary."""
    if prefix == GLOBAL_PREFIX:
        return True
    if prefix.startswith("/"):
        if not source_uri.startswith("file://"):
            return False
        source_uri = source_uri.removeprefix("file://")
    return source_uri == prefix or source_uri.startswith(prefix if prefix.endswith("/") else prefix + "/")


def context_chain(contexts: Mapping[str, str], source_uri: str) -> list[dict[str, str]]:
    """Every context matching source_uri, from global down to the most specific."""
    return [{"prefix": p, "text": contexts[p]} for p in sorted(contexts, key=len) if prefix_matches(p, source_uri)]


def indexed_text(contexts: Mapping[str, str], source_uri: str, text: str) -> str:
    """The text that gets embedded and keyword-indexed, which is the context chain followed by the chunk."""
    return "\n\n".join([*(c["text"] for c in context_chain(contexts, source_uri)), text])
