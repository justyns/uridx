from urllib.parse import urlencode

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from uridx.config import URIDX_AUTH_TOKEN, URIDX_BASE_URL, URIDX_OAUTH_PASSWORD
from uridx.db.engine import init_db
from uridx.db.operations import _get_existing_local, add_item, delete_item, get_item
from uridx.search.hybrid import hybrid_search

_oauth_provider = None


def _init_auth():
    """Initialize and return auth provider, storing OAuth provider for login route access."""
    global _oauth_provider

    if URIDX_OAUTH_PASSWORD and URIDX_BASE_URL:
        from uridx.mcp.oauth import SingleUserOAuthProvider

        _oauth_provider = SingleUserOAuthProvider(
            base_url=URIDX_BASE_URL,
            password=URIDX_OAUTH_PASSWORD,
        )
        return _oauth_provider

    if URIDX_AUTH_TOKEN:
        return StaticTokenVerifier(
            tokens={
                URIDX_AUTH_TOKEN: {
                    "client_id": "user",
                    "scopes": ["read", "write"],
                }
            }
        )

    return None


mcp = FastMCP("uridx", auth=_init_auth())


@mcp.custom_route("/exists", methods=["POST"])
async def exists_endpoint(request: Request) -> JSONResponse:
    """Check which source_uris already exist in the index."""
    data = await request.json()
    uris = data.get("source_uris", [])
    existing = _get_existing_local(uris)
    return JSONResponse({"existing": list(existing)})


LOGIN_FORM_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>uridx - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
               display: flex; justify-content: center; align-items: center;
               min-height: 100vh; margin: 0; }}
        .container {{ background: #16213e; padding: 2rem; border-radius: 8px;
                     box-shadow: 0 4px 6px rgba(0,0,0,0.3); max-width: 400px; width: 90%; }}
        h1 {{ margin: 0 0 1.5rem; font-size: 1.5rem; text-align: center; }}
        input {{ width: 100%; padding: 0.75rem; margin: 0.5rem 0; border: 1px solid #0f3460;
                background: #1a1a2e; color: #eee; border-radius: 4px; box-sizing: border-box; }}
        button {{ width: 100%; padding: 0.75rem; margin-top: 1rem; border: none;
                 background: #e94560; color: white; border-radius: 4px; cursor: pointer;
                 font-size: 1rem; }}
        button:hover {{ background: #ff6b6b; }}
        .error {{ color: #ff6b6b; margin-top: 1rem; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>uridx</h1>
        <form method="POST">
            <input type="hidden" name="client_id" value="{client_id}">
            <input type="hidden" name="redirect_uri" value="{redirect_uri}">
            <input type="hidden" name="state" value="{state}">
            <input type="hidden" name="code_challenge" value="{code_challenge}">
            <input type="hidden" name="scopes" value="{scopes}">
            <input type="hidden" name="resource" value="{resource}">
            <input type="password" name="password" placeholder="Password" required autofocus>
            <button type="submit">Login</button>
            {error}
        </form>
    </div>
</body>
</html>"""


def _render_login_form(
    client_id: str = "",
    redirect_uri: str = "",
    state: str = "",
    code_challenge: str = "",
    scopes: str = "",
    resource: str = "",
    error: str = "",
) -> str:
    return LOGIN_FORM_HTML.format(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        scopes=scopes,
        resource=resource,
        error=error,
    )


@mcp.custom_route("/oauth/login", methods=["GET", "POST"])
async def oauth_login(request: Request) -> HTMLResponse | RedirectResponse:
    """OAuth login form and handler."""
    if _oauth_provider is None:
        return HTMLResponse("<h1>OAuth not configured</h1>", status_code=500)

    if request.method == "GET":
        params = request.query_params
        return HTMLResponse(
            _render_login_form(
                client_id=params.get("client_id", ""),
                redirect_uri=params.get("redirect_uri", ""),
                state=params.get("state", ""),
                code_challenge=params.get("code_challenge", ""),
                scopes=params.get("scopes", ""),
                resource=params.get("resource", ""),
            )
        )

    form = await request.form()
    password = str(form.get("password", ""))
    client_id = str(form.get("client_id", ""))
    redirect_uri = str(form.get("redirect_uri", ""))
    state = str(form.get("state", ""))
    code_challenge = str(form.get("code_challenge", ""))
    scopes = str(form.get("scopes", ""))
    resource = str(form.get("resource", "")) or None

    scope_list = [s.strip() for s in scopes.split(",") if s.strip()] if scopes else []

    code = _oauth_provider.verify_password_and_create_code(
        password=password,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        scopes=scope_list,
        resource=resource,
    )

    if code:
        redirect_params = {"code": code}
        if state:
            redirect_params["state"] = state
        return RedirectResponse(f"{redirect_uri}?{urlencode(redirect_params)}", status_code=302)

    return HTMLResponse(
        _render_login_form(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            scopes=scopes,
            resource=resource or "",
            error='<p class="error">Invalid password</p>',
        ),
        status_code=401,
    )


def _clean_dict(**kwargs) -> dict:
    """Build dict omitting None values and empty lists."""
    return {k: v for k, v in kwargs.items() if v is not None and v != []}


@mcp.tool(annotations={"readOnlyHint": True})
def search(
    query: str,
    limit: int = 10,
    source_type: str | None = None,
    tags: list[str] | None = None,
    semantic: bool = True,
    recency_boost: float = 0.3,
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
        recency_boost: Boost recent items (0.0-1.0, default 0.3)

    Returns:
        List of matching items with source_uri, title, source_type, snippet, score, and tags
    """
    results = hybrid_search(
        query=query,
        limit=limit,
        source_type=source_type,
        tags=tags,
        semantic=semantic,
        recency_boost=recency_boost,
    )

    return [
        _clean_dict(
            source_uri=r.source_uri,
            title=r.title,
            source_type=r.source_type,
            snippet=r.chunk_text,
            score=round(r.score, 3),
            created_at=r.created_at.isoformat() if r.created_at else None,
            tags=r.tags if r.tags else None,
        )
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


@mcp.tool(annotations={"readOnlyHint": True})
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

    return _clean_dict(
        source_uri=item.source_uri,
        title=item.title,
        source_type=item.source_type,
        context=item.context,
        created_at=item.created_at.isoformat() if item.created_at else None,
        updated_at=item.updated_at.isoformat() if item.updated_at else None,
        chunks=[{"text": c.text, "key": c.chunk_key} for c in item.chunks],
        tags=[t.tag for t in item.tags] if item.tags else None,
    )


def run_server(http: bool = False, host: str = "127.0.0.1", port: int = 8000):
    init_db()
    if http:
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()
