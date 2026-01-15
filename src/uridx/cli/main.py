import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from uridx.cli.extract import app as extract_app
from uridx.db.engine import init_db
from uridx.db.operations import add_item, get_stats
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
):
    init_db()
    results = hybrid_search(query, limit=limit, source_type=type, tags=tag)

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
):
    init_db()
    console = Console(stderr=True)

    if text:
        content = sys.stdin.read()
        with console.status(f"Ingesting {text}..."):
            item = add_item(
                source_uri=text,
                chunks=[{"text": content}],
                replace=replace,
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
                add_item(
                    source_uri=data["source_uri"],
                    title=data.get("title"),
                    context=data.get("context"),
                    source_type=data.get("source_type"),
                    tags=data.get("tags"),
                    chunks=data.get("chunks", []),
                    replace=data.get("replace", replace),
                    created_at=data.get("created_at"),
                )
                count += 1
                progress.advance(task)
        print(json.dumps({"ingested": count}))


@app.command()
def stats():
    init_db()
    print(json.dumps(get_stats(), indent=2))


@app.command()
def serve(
    http: Annotated[bool, typer.Option("--http", help="Run as HTTP server (SSE transport)")] = False,
    port: Annotated[int, typer.Option("--port", "-p", help="HTTP server port")] = 8000,
):
    from uridx.mcp.server import run_server

    run_server(http=http, port=port)


if __name__ == "__main__":
    app()
