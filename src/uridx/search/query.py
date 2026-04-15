"""Query processing for FTS5 search — stopword removal and term sanitization."""

from __future__ import annotations

import re

# fmt: off
STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "just", "let", "let's", "like", "ll",
    "me", "might", "more", "most", "must", "mustn't", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shall", "shan't", "she", "should", "shouldn't", "so", "some",
    "such", "t", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "ve", "very",
    "was", "wasn't", "we", "were", "weren't", "what", "what's", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
})
# fmt: on

_FTS_SPECIAL = re.compile(r'[":*^(){}[\]|\\]')
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+(?:'[a-zA-Z]+)?")


def process_query(query: str) -> str:
    """Extract keywords from a query and return an FTS5 MATCH expression."""
    if not query or not query.strip():
        return ""

    tokens = _WORD_RE.findall(query)
    keywords = list(dict.fromkeys(t.lower() for t in tokens if t.lower() not in STOP_WORDS and len(t) >= 2))

    if keywords:
        return " OR ".join(keywords)

    safe = _FTS_SPECIAL.sub("", query).strip()
    return f'"{safe}"' if safe else ""
