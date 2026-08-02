"""Grouping a conversation's messages into User/Assistant turn chunks.

Claude Code (JSONL transcripts) and Tsugite (an event-sourced SQLite history) store
conversations completely differently, but both read back the same way: a user message
plus everything the assistant said before the next one. Each extractor reduces its own
format to a `(role, text)` stream and hands it here, so the turn shape — the chunk key,
the "User:"/"Assistant:" framing, the turn_index metadata — is defined once.

Not named `turns.py`: a leading underscore marks the shared helpers in this package
that are not themselves `extract` subcommands.
"""

from collections.abc import Iterable
from typing import Literal

from uridx.record import ChunkInput

Role = Literal["user", "assistant"]


def build_turn_chunks(messages: Iterable[tuple[Role, str]]) -> list[ChunkInput]:
    """Group a (role, text) stream into one chunk per exchange. Empty texts are dropped."""
    turns: list[ChunkInput] = []
    user: str | None = None
    assistant: list[str] = []

    def flush() -> None:
        nonlocal user, assistant
        parts = []
        if user:
            parts.append(f"User: {user}")
        if assistant:
            parts.append(f"Assistant: {' '.join(assistant)}")
        if parts:
            turns.append(ChunkInput(text="\n\n".join(parts), key=f"turn-{len(turns)}", meta={"turn_index": len(turns)}))
        user, assistant = None, []

    for role, text in messages:
        if not text:
            continue
        if role == "user":
            if user or assistant:
                flush()
            user = text
        else:
            assistant.append(text)
    flush()
    return turns
