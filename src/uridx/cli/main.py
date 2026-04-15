import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from uridx.cli.extract import app as extract_app
from uridx.config import get_machine_id
from uridx.db.engine import init_db
from uridx.db.operations import (
    add_item,
    delete_item,
    delete_items_by_prefix,
    get_item,
    get_stats,
    list_items_by_prefix,
)
from uridx.search.hybrid import hybrid_search

app = typer.Typer()
app.add_typer(extract_app, name="extract")


@app.command()
def search(
    query: str,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t")] = None,
    type: Annotated[Optional[str], typer.Option("--type")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    json_output: Annotated[bool, typer.Option("--json", "-j")] = False,
    semantic: Annotated[bool, typer.Option("--semantic/--no-semantic")] = True,
    recency_boost: Annotated[float, typer.Option("--recency-boost")] = 0.3,
    min_score: Annotated[Optional[float], typer.Option("--min-score")] = None,
    source_prefix: Annotated[Optional[str], typer.Option("--source-prefix")] = None,
    after: Annotated[Optional[datetime], typer.Option("--after")] = None,
):
    init_db()
    results = hybrid_search(
        query,
        limit=limit,
        source_type=type,
        tags=tag,
        semantic=semantic,
        recency_boost=recency_boost,
        min_score=min_score,
        source_prefix=source_prefix,
        after=after,
    )

    if json_output:

        def _json_default(o):
            if isinstance(o, datetime):
                return o.isoformat()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")

        print(json.dumps([asdict(r) for r in results], indent=2, default=_json_default))
    else:
        for r in results:
            print(f"[{r.score:.3f}] {r.source_uri}")
            if r.title:
                print(f"  Title: {r.title}")
            if r.source_type:
                print(f"  Type: {r.source_type}")
            if r.tags:
                print(f"  Tags: {', '.join(r.tags)}")
            print(f"  {r.chunk_text[:200]}...")
            print()


@app.command()
def ingest(
    jsonl: Annotated[bool, typer.Option("--jsonl")] = False,
    text: Annotated[Optional[str], typer.Option("--text")] = None,
    replace: Annotated[bool, typer.Option("--replace")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Tags for ingested items")] = None,
):
    init_db()
    console = Console(stderr=True)

    if text:
        content = sys.stdin.read()
        with console.status(f"Ingesting {text}..."):
            item = add_item(
                source_uri=text,
                chunks=[{"text": content}],
                tags=tag,
                replace=replace,
                machine=get_machine_id(),
            )
        print(json.dumps({"source_uri": item.source_uri, "chunks": len(item.chunks)}))
    else:
        lines = [line.strip() for line in sys.stdin if line.strip()]
        count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=len(lines))
            for line in lines:
                data = json.loads(line)
                source_uri = data.get("source_uri", "unknown")
                progress.update(task, description=f"{source_uri[:60]}")
                record_tags = data.get("tags") or []
                if tag:
                    record_tags = list(dict.fromkeys(record_tags + tag))
                add_item(
                    source_uri=data["source_uri"],
                    title=data.get("title"),
                    context=data.get("context"),
                    source_type=data.get("source_type"),
                    tags=record_tags or None,
                    chunks=data.get("chunks", []),
                    replace=data.get("replace", replace),
                    created_at=data.get("created_at"),
                    machine=data.get("machine") or get_machine_id(),
                )
                count += 1
                progress.advance(task)
        print(json.dumps({"ingested": count}))


@app.command()
def delete(
    uri: Annotated[Optional[str], typer.Option("--uri", help="Exact source_uri to delete")] = None,
    source_prefix: Annotated[
        Optional[str], typer.Option("--source-prefix", help="Delete all items matching prefix")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be deleted")] = False,
):
    """Delete items from the index by URI or prefix."""
    if not uri and not source_prefix:
        print("Error: must provide --uri or --source-prefix", file=sys.stderr)
        raise typer.Exit(1)
    if uri and source_prefix:
        print("Error: provide only one of --uri or --source-prefix", file=sys.stderr)
        raise typer.Exit(1)

    init_db()

    if uri:
        if dry_run:
            item = get_item(uri)
            if item:
                print(json.dumps({"would_delete": 1, "items": [uri]}))
            else:
                print(json.dumps({"would_delete": 0, "items": []}))
        else:
            deleted = delete_item(uri)
            print(json.dumps({"deleted": 1 if deleted else 0}))
    else:
        if dry_run:
            items = list_items_by_prefix(source_prefix)
            print(json.dumps({"would_delete": len(items), "items": [i.source_uri for i in items]}))
        else:
            count = delete_items_by_prefix(source_prefix)
            print(json.dumps({"deleted": count}))


@app.command()
def stats():
    init_db()
    print(json.dumps(get_stats(), indent=2))


@app.command()
def serve(
    http: Annotated[bool, typer.Option("--http", help="Run as HTTP server")] = False,
    host: Annotated[str, typer.Option("--host", "-H", help="HTTP server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="HTTP server port")] = 8000,
):
    from uridx.mcp.server import run_server

    run_server(http=http, host=host, port=port)


if __name__ == "__main__":
    app()
