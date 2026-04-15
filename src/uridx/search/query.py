"""Query processing for improved FTS5 search.

Extracts keywords from natural language queries, optionally using spaCy
for lemmatization and noun chunk extraction. Falls back to basic stopword
removal when spaCy is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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

# Characters that have special meaning in FTS5 MATCH syntax
_FTS_SPECIAL = re.compile(r'[":*^(){}[\]|\\]')

# Tokenize on whitespace and punctuation (but keep alphanumeric + underscores)
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+(?:'[a-zA-Z]+)?")

_nlp = None  # cached spaCy model


@dataclass
class QueryTerms:
    """Structured representation of a processed search query."""

    original: str
    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    fts_query: str = ""


def _sanitize_fts_term(term: str) -> str:
    """Remove FTS5 special characters from a term."""
    return _FTS_SPECIAL.sub("", term).strip()


def _build_fts_expression(keywords: list[str], phrases: list[str]) -> str:
    """Build an FTS5 MATCH expression from keywords and phrases.

    Produces: keyword1 OR keyword2 OR "multi word phrase"
    """
    parts: list[str] = []
    for kw in keywords:
        clean = _sanitize_fts_term(kw)
        if clean:
            parts.append(clean)
    for phrase in phrases:
        clean = _sanitize_fts_term(phrase)
        if clean and " " in clean:
            parts.append(f'"{clean}"')
        elif clean:
            parts.append(clean)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in parts:
        low = p.lower().strip('"')
        if low not in seen:
            seen.add(low)
            deduped.append(p)
    return " OR ".join(deduped)


def _extract_basic(text: str) -> tuple[list[str], list[str]]:
    """Basic keyword extraction: tokenize, remove stopwords."""
    tokens = _WORD_RE.findall(text)
    keywords = [t.lower() for t in tokens if t.lower() not in STOP_WORDS and len(t) >= 2]
    return keywords, []


def _load_spacy():
    """Load spaCy model, returns None if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except (ImportError, OSError):
        return None


def _extract_spacy(text: str) -> tuple[list[str], list[str]]:
    """Extract keywords and phrases using spaCy.

    Returns (lemmatized_keywords, noun_chunk_phrases).
    """
    nlp = _load_spacy()
    if nlp is None:
        return _extract_basic(text)

    doc = nlp(text)

    # Lemmatized keywords: skip stopwords, punctuation, short tokens
    keywords: list[str] = []
    for token in doc:
        if token.is_stop or token.is_punct or not token.is_alpha or len(token.text) < 2:
            continue
        lemma = token.lemma_.lower()
        if lemma not in STOP_WORDS and len(lemma) >= 2:
            keywords.append(lemma)

    # Noun chunks as phrases (only multi-word ones are useful)
    phrases: list[str] = []
    for chunk in doc.noun_chunks:
        # Strip leading determiners/stopwords from the chunk
        words = [t for t in chunk if not t.is_stop and not t.is_punct]
        if len(words) >= 2:
            phrases.append(" ".join(t.lemma_.lower() for t in words))

    return keywords, phrases


def process_query(query: str, use_spacy: bool = True) -> QueryTerms:
    """Process a search query into structured terms for FTS5 and vector search.

    Args:
        query: Raw user query string.
        use_spacy: Whether to attempt spaCy extraction (falls back to basic if unavailable).

    Returns:
        QueryTerms with keywords, phrases, and a ready-to-use FTS5 query.
    """
    if not query or not query.strip():
        return QueryTerms(original=query, fts_query="")

    if use_spacy:
        keywords, phrases = _extract_spacy(query)
    else:
        keywords, phrases = _extract_basic(query)

    fts_expr = _build_fts_expression(keywords, phrases)

    # Fallback: if extraction removed everything, use original as quoted phrase
    if not fts_expr:
        safe = _sanitize_fts_term(query)
        fts_expr = f'"{safe}"' if safe else '""'

    return QueryTerms(
        original=query,
        keywords=keywords,
        phrases=phrases,
        fts_query=fts_expr,
    )
