import json
import sys
from dataclasses import asdict
from typing import Annotated, Optional

import typer

from uridx.db.engine import init_db
from uridx.db.operations import add_item, get_stats
from uridx.search.hybrid import hybrid_search

app = typer.Typer()


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
        print(json.dumps([asdict(r) for r in results], indent=2))
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

    if text:
        content = sys.stdin.read()
        print(f"Ingesting text for {text}...", file=sys.stderr)
        item = add_item(
            source_uri=text,
            chunks=[{"text": content}],
            replace=replace,
        )
        print(json.dumps({"source_uri": item.source_uri, "chunks": len(item.chunks)}))
    else:
        count = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            print(f"Ingesting {data.get('source_uri', 'unknown')}...", file=sys.stderr)
            item = add_item(
                source_uri=data["source_uri"],
                title=data.get("title"),
                context=data.get("context"),
                source_type=data.get("source_type"),
                tags=data.get("tags"),
                chunks=data.get("chunks", []),
                replace=data.get("replace", replace),
            )
            count += 1
        print(json.dumps({"ingested": count}))


@app.command()
def stats():
    init_db()
    print(json.dumps(get_stats(), indent=2))


@app.command()
def serve():
    from uridx.mcp.server import run_server

    run_server()


if __name__ == "__main__":
    app()
