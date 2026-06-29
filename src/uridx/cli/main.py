import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from uridx.cli.extract import app as extract_app
from uridx.cli.extract.base import MissingExtractorDependency, filter_existing_files
from uridx.cli.extract.registry import load_extractor, resolve_dispatch
from uridx.config import get_machine_id
from uridx.db.engine import init_db
from uridx.db.operations import (
    add_item,
    delete_item,
    delete_items_by_prefix,
    get_item,
    get_stats,
    ingest_record,
    list_items_by_prefix,
)
from uridx.record import ChunkInput, Record
from uridx.search.hybrid import hybrid_search

app = typer.Typer()
app.add_typer(extract_app, name="extract")

SNIPPET_LIMIT = 400


def _snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """One-line preview: collapse whitespace, cut at a word boundary, ellipsize if shortened."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:  # only back up to a word boundary if it doesn't chop too much
        cut = cut[:space]
    return cut.rstrip() + "…"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search text; leave empty to list items matching the filters")] = "",
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t")] = None,
    source_type: Annotated[Optional[str], typer.Option("--type")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    json_output: Annotated[bool, typer.Option("--json", "-j")] = False,
    full: Annotated[
        bool, typer.Option("--full", help="Show complete chunk text instead of a truncated preview")
    ] = False,
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
        source_type=source_type,
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
            print(_indent(r.chunk_text) if full else f"  {_snippet(r.chunk_text)}")
            print()


@app.command()
def ingest(
    jsonl: Annotated[bool, typer.Option("--jsonl")] = False,
    text: Annotated[Optional[str], typer.Option("--text")] = None,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Tags for ingested items")] = None,
):
    init_db()
    console = Console(stderr=True)

    if text:
        content = sys.stdin.read()
        with console.status(f"Ingesting {text}..."):
            item = add_item(
                source_uri=text,
                chunks=[ChunkInput(text=content)],
                tags=tag,
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
            for i, line in enumerate(lines, 1):
                try:
                    record = Record.model_validate_json(line)
                except ValidationError as e:
                    print(f"Error: invalid record on line {i}:\n{e}", file=sys.stderr)
                    raise typer.Exit(1)
                progress.update(task, description=f"{record.source_uri[:60]}")
                ingest_record(record, extra_tags=tag)
                count += 1
                progress.advance(task)
        print(json.dumps({"ingested": count}))


@app.command()
def add(
    paths: Annotated[list[str], typer.Argument(help="Files or directories to index")],
    extractor: Annotated[
        Optional[str], typer.Option("--extractor", "-e", help="Force a specific extractor (markdown/pdf/image/docling)")
    ] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-ingest even if already indexed")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Tags to add to ingested items")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be ingested without doing it")] = False,
):
    """Add files or directories, auto-detecting the extractor by file extension.

    Porcelain over `extract | ingest`. Already-indexed files are skipped unless --force.
    Structured sources (Claude Code / Tsugite) are not auto-detected; use `extract`.
    """
    urls = [p for p in paths if p.startswith(("http://", "https://"))]
    if urls:
        print(
            f"Error: 'add' does not accept URLs ({urls[0]}). Use: uridx extract docling <url> | uridx ingest",
            file=sys.stderr,
        )
        raise typer.Exit(2)

    try:
        buckets, skipped = resolve_dispatch(paths, extractor)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(2)

    if dry_run:
        for name, files in buckets.items():
            for f in files:
                print(f"{f}\t{name}")
        for f in skipped:
            print(f"{f}\t(no extractor)")
        return

    init_db()
    console = Console(stderr=True)

    # Pre-filter each bucket so already-indexed files are dropped (and reported) before any heavy work.
    plan: list[tuple[str, list]] = []
    skipped_existing = 0
    for name, files in buckets.items():
        survivors = filter_existing_files(files, force)
        skipped_existing += len(files) - len(survivors)
        if survivors:
            plan.append((name, survivors))

    ingested = 0
    by_type: dict[str, int] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Adding...", total=None)
        for name, survivors in plan:
            module = load_extractor(name)
            try:
                for rec in module.iter_records(survivors, tag=tag):
                    ingest_record(rec)
                    ingested += 1
                    stype = rec.source_type or name
                    by_type[stype] = by_type.get(stype, 0) + 1
                    progress.update(task, description=f"{name}: {rec.source_uri[:50]}")
            except MissingExtractorDependency as e:
                print(f"Skipping {name}: {e}", file=sys.stderr)

    print(
        json.dumps(
            {
                "ingested": ingested,
                "skipped_unknown": len(skipped),
                "skipped_existing": skipped_existing,
                "by_type": by_type,
            }
        )
    )


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


if __name__ == "__main__":
    app()
