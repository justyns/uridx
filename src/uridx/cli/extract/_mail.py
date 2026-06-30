"""Shared email parsing and junk classification for the email extractors.

Both readers (`maildir`, `mbox`) hand `iter_message_records` a stream of
`(EmailMessage, folder_label)` pairs; it parses, drops/tags junk, and yields one
Record per message. Each message is its own Item (immutable — new mail never mutates
old Items), carrying a `thread:<root-id>` tag so conversations can be searched and
listed at retrieval time. Idempotency is left to `add_item` (content-hash upsert).

Not named `email.py` — that would shadow the stdlib `email` package.
"""

import email
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from enum import Enum
from html.parser import HTMLParser

from uridx.record import ChunkInput, Record

from .base import split_text

MAX_CHARS = 2000

_ID_RE = re.compile(r"<([^>]+)>")
_RE_PREFIX = re.compile(r"^(?:(?:re|fwd?|aw|sv)\s*:\s*)+", re.IGNORECASE)
_DURATION_RE = re.compile(r"^(\d+)\s*([smhdw])$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class JunkMode(str, Enum):
    """How `--junk` treats mailing-list / bulk / spam messages."""

    skip = "skip"
    tag = "tag"
    keep = "keep"


def parse_bytes(data: bytes, label: str) -> EmailMessage | None:
    """Parse raw RFC822 bytes into an EmailMessage, or warn and skip on failure."""
    try:
        return email.message_from_bytes(data, policy=policy.default)
    except Exception as e:
        print(f"Error parsing {label}: {e}", file=sys.stderr)
        return None


@dataclass
class Msg:
    message_id: str
    thread_id: str
    date: datetime | None
    from_addr: str
    to: str
    cc: str
    subject: str
    body: str
    attachments: list[str]
    junk: set[str]
    folder: str


def _ids(raw: str | None) -> list[str]:
    return _ID_RE.findall(raw or "")


def classify_junk(msg: EmailMessage) -> set[str]:
    """Header-only junk signals: subset of {mailing-list, bulk, spam}."""
    signals: set[str] = set()
    if any(msg.get(h) for h in ("List-Id", "List-Unsubscribe", "List-Post")):
        signals.add("mailing-list")
    precedence = (msg.get("Precedence") or "").strip().lower()
    auto = (msg.get("Auto-Submitted") or "").strip().lower()
    if precedence in ("bulk", "list", "junk") or auto.startswith("auto"):
        signals.add("bulk")
    spam_flag = (msg.get("X-Spam-Flag") or "").strip().lower()
    spam_status = (msg.get("X-Spam-Status") or "").strip().lower()
    if spam_flag == "yes" or spam_status.startswith("yes"):
        signals.add("spam")
    return signals


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping <script>/<style> (convert_charrefs decodes entities)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(raw: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    parser.close()
    return "\n".join(s for ln in parser.get_text().splitlines() if (s := ln.strip()))


def _body_text(msg: EmailMessage) -> tuple[str, list[str]]:
    attachments = [name for part in msg.iter_attachments() if (name := part.get_filename())]
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return "", attachments
    try:
        content = part.get_content()
    except (LookupError, ValueError):
        content = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
    if part.get_content_subtype() == "html":
        content = _strip_html(content)
    return content.strip(), attachments


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_message(msg: EmailMessage, folder: str) -> Msg:
    from_addr = str(msg.get("From") or "")
    subject = str(msg.get("Subject") or "")
    raw_date = msg.get("Date")
    body, attachments = _body_text(msg)
    ids = _ids(msg.get("Message-Id"))
    if ids:
        message_id = ids[0]
    else:
        seed = json.dumps(
            {
                "from": from_addr,
                "date": raw_date or "",
                "subject": subject,
                "to": str(msg.get("To") or ""),
                "cc": str(msg.get("Cc") or ""),
                "folder": folder,
                "body": body,
                "attachments": attachments,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        message_id = f"synth-{hashlib.sha256(seed).hexdigest()[:16]}@uridx"
    # Thread anchor: the earliest referenced ancestor (References is root-first), so every
    # message in a conversation derives the same id without needing to see its siblings.
    refs = _ids(msg.get("References")) or _ids(msg.get("In-Reply-To"))
    return Msg(
        message_id=message_id,
        thread_id=refs[0] if refs else message_id,
        date=_parse_date(raw_date),
        from_addr=from_addr,
        to=str(msg.get("To") or ""),
        cc=str(msg.get("Cc") or ""),
        subject=subject,
        body=body,
        attachments=attachments,
        junk=classify_junk(msg),
        folder=folder,
    )


def _clean_subject(subject: str) -> str:
    return _RE_PREFIX.sub("", subject.strip()).strip()


def _message_chunks(m: Msg) -> list[ChunkInput]:
    header = [f"From: {m.from_addr}"]
    if m.to:
        header.append(f"To: {m.to}")
    if m.cc:
        header.append(f"Cc: {m.cc}")
    if m.date:
        header.append(f"Date: {m.date.isoformat()}")
    if m.subject:
        header.append(f"Subject: {m.subject}")
    text = "\n".join(header) + "\n\n" + m.body
    meta = {"message_id": m.message_id}
    return [ChunkInput(text=piece, key=str(i), meta=meta) for i, piece in enumerate(split_text(text, MAX_CHARS))]


def build_message_record(m: Msg, *, account: str, junk_signals: set[str], user_tags: list[str]) -> Record:
    title = _clean_subject(m.subject)
    junk = sorted(junk_signals)
    context = {
        "subject": title,
        "from": m.from_addr,
        "to": m.to,
        "cc": m.cc,
        "date": m.date.isoformat() if m.date else None,
        "folder": m.folder,
        "account": account,
        "message_id": m.message_id,
        "thread_id": m.thread_id,
        "attachments": m.attachments,
        "junk": junk,
    }
    tags = list(dict.fromkeys(["email", account, m.folder, f"thread:{m.thread_id}", *user_tags, *junk]))
    return Record(
        source_uri=f"email://{account}/{m.message_id}",
        chunks=_message_chunks(m),
        title=title or None,
        source_type="email",
        context=json.dumps(context),
        tags=tags,
        created_at=m.date.isoformat() if m.date else None,
    )


def iter_message_records(
    messages: Iterable[tuple[EmailMessage, str]],
    *,
    account: str,
    junk_mode: JunkMode = JunkMode.skip,
    since: datetime | None = None,
    user_tags: list[str] | None = None,
) -> Iterator[Record]:
    """Parse, filter junk, and yield one Record per message (streaming)."""
    for msg, folder in messages:
        m = parse_message(msg, folder)
        if since and (m.date is None or m.date < since):
            continue
        if m.junk and junk_mode == JunkMode.skip:
            continue
        signals = m.junk if junk_mode == JunkMode.tag else set()
        rec = build_message_record(m, account=account, junk_signals=signals, user_tags=user_tags or [])
        if rec.chunks:
            yield rec


def parse_since(value: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD filter date to an aware UTC datetime (None if unset)."""
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {value} (expected YYYY-MM-DD)") from e
    return dt.replace(tzinfo=timezone.utc)


def parse_duration(value: str | None) -> int | None:
    """Parse a duration like 30m / 2h / 7d / 1w into seconds (None if unset)."""
    if not value:
        return None
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise ValueError(f"Invalid duration: {value} (expected e.g. 30m, 2h, 7d, 1w)")
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
