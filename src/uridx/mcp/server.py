from fastmcp import FastMCP

from uridx.db.engine import init_db
from uridx.db.operations import add_item, delete_item, get_item
from uridx.search.hybrid import hybrid_search

mcp = FastMCP("uridx")


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    source_type: str | None = None,
    tags: list[str] | None = None,
    semantic: bool = True,
) -> list[dict]:
    """Search the uridx knowledge base for relevant content.

    Use this tool to find information stored in uridx. Supports both semantic
    (meaning-based) and keyword search. Semantic search is enabled by default
    and finds conceptually related content even if exact words don't match.

    Args:
        query: Natural language search query describing what you're looking for
        limit: Maximum number of results to return (default 10)
        source_type: Filter by type (e.g., "note", "bookmark", "document")
        tags: Filter results to items containing all specified tags
        semantic: Use semantic search (True) or keyword-only search (False)

    Returns:
        List of matching items with source_uri, title, source_type, snippet, score, and tags
    """
    results = hybrid_search(
        query=query,
        limit=limit,
        source_type=source_type,
        tags=tags,
        semantic=semantic,
    )

    return [
        {
            "source_uri": r.source_uri,
            "title": r.title,
            "source_type": r.source_type,
            "snippet": r.chunk_text,
            "score": r.score,
            "tags": r.tags,
        }
        for r in results
    ]


@mcp.tool()
def add(
    source_uri: str,
    title: str,
    text: str,
    source_type: str = "note",
    tags: list[str] | None = None,
    context: str | None = None,
) -> dict:
    """Add a new item to the uridx knowledge base.

    Use this tool to store notes, bookmarks, or other content for later retrieval.
    The text will be indexed for both semantic and keyword search.

    Args:
        source_uri: Unique identifier for this item (e.g., URL or custom URI)
        title: Human-readable title for the item
        text: The main content to store and index
        source_type: Category of content (default "note")
        tags: Optional list of tags for filtering
        context: Optional additional context to improve search relevance

    Returns:
        Confirmation with status and the source_uri
    """
    add_item(
        source_uri=source_uri,
        title=title,
        source_type=source_type,
        context=context,
        chunks=[{"text": text}],
        tags=tags,
    )

    return {"status": "added", "source_uri": source_uri, "title": title}


@mcp.tool()
def delete(source_uri: str) -> dict:
    """Delete an item from the uridx knowledge base.

    Permanently removes the item and all its indexed content.

    Args:
        source_uri: The unique identifier of the item to delete

    Returns:
        Status indicating whether the item was deleted or not found
    """
    deleted = delete_item(source_uri)

    return {
        "status": "deleted" if deleted else "not_found",
        "source_uri": source_uri,
    }


@mcp.tool()
def get(source_uri: str) -> dict | None:
    """Retrieve a specific item from the uridx knowledge base by its URI.

    Use this to get the full details of a known item, including all chunks and tags.

    Args:
        source_uri: The unique identifier of the item to retrieve

    Returns:
        Full item details or None if not found
    """
    item = get_item(source_uri)

    if not item:
        return None

    return {
        "source_uri": item.source_uri,
        "title": item.title,
        "source_type": item.source_type,
        "context": item.context,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        "chunks": [{"text": c.text, "key": c.chunk_key} for c in item.chunks],
        "tags": [t.tag for t in item.tags],
    }


def run_server():
    init_db()
    mcp.run()
