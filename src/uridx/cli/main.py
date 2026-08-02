import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from uridx.cli.extract import app as extract_app
from uridx.cli.extract.base import MissingExtractorDependency, drop_indexed_files, output
from uridx.cli.extract.registry import load_extractor, resolve_dispatch
from uridx.db.engine import init_db, require_embed_model_match
from uridx.embeddings import EmbeddingBackendError
from uridx.db.operations import (
    delete_item,
    delete_items_by_prefix,
    get_item,
    get_stats,
    ingest_record,
    ingest_records,
    list_items_by_prefix,
    reindex_embeddings,
)
from uridx.record import ChunkInput, Record, RecordError, iter_jsonl_records
from uridx.timeutil import to_naive_utc
from uridx.search.hybrid import hybrid_search
from uridx.sync import import_records, iter_export_records

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(extract_app, name="extract")

SNIPPET_LIMIT = 400


def _progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )


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


def _run_record_stream(run, verb: str, out_key: str):
    """Feed stdin JSONL records through run(records, on_result) with progress; print summary counts."""
    console = Console(stderr=True)
    try:
        with _progress(console) as progress:
            task = progress.add_task(f"{verb}...", total=None)
            counts = run(
                iter_jsonl_records(sys.stdin),
                lambda rec, _: progress.update(task, description=rec.source_uri[:60], advance=1),
            )
    except RecordError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(1)
    print(json.dumps({out_key: counts["added"], "unchanged": counts["unchanged"], "merged": counts["merged"]}))


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
    min_score: Annotated[
        Optional[float],
        typer.Option("--min-score", help="Drop results below this cosine relevance (semantic search only)"),
    ] = None,
    source_prefix: Annotated[Optional[str], typer.Option("--source-prefix")] = None,
    after: Annotated[Optional[datetime], typer.Option("--after")] = None,
):
    init_db()
    if semantic and query.strip():
        require_embed_model_match()
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
            relevance = f" (rel {r.relevance:.2f})" if r.relevance is not None else ""
            print(f"[{r.score:.3f}]{relevance} {r.source_uri}")
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
    require_embed_model_match()

    if text:
        content = sys.stdin.read()
        with Console(stderr=True).status(f"Ingesting {text}..."):
            item, _ = ingest_record(Record(source_uri=text, chunks=[ChunkInput(text=content)], tags=tag or []))
        print(json.dumps({"source_uri": item.source_uri, "chunks": len(item.chunks)}))
    else:
        _run_record_stream(
            lambda records, on_result: ingest_records(records, extra_tags=tag, on_result=on_result),
            "Ingesting",
            "ingested",
        )


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
    require_embed_model_match()
    console = Console(stderr=True)

    # Pre-filter each bucket so already-indexed files are dropped (and reported) before any heavy work.
    plan: list[tuple[str, list]] = []
    skipped_existing = 0
    for name, files in buckets.items():
        survivors = drop_indexed_files(files, force)
        skipped_existing += len(files) - len(survivors)
        if survivors:
            plan.append((name, survivors))

    ingested = 0
    by_type: dict[str, int] = {}
    with _progress(console) as progress:
        task = progress.add_task("Adding...", total=None)
        for name, survivors in plan:
            module = load_extractor(name)

            def on_result(rec, _status, name=name):
                stype = rec.source_type or name
                by_type[stype] = by_type.get(stype, 0) + 1
                progress.update(task, description=f"{name}: {rec.source_uri[:50]}", advance=1)

            try:
                counts = ingest_records(module.iter_records(survivors, tag=tag), on_result=on_result)
                ingested += counts["added"]
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
def export(
    since: Annotated[
        Optional[str], typer.Option("--since", help="Only items updated at or after this ISO timestamp")
    ] = None,
    source_prefix: Annotated[Optional[str], typer.Option("--source-prefix", help="Filter by source URI prefix")] = None,
    source_type: Annotated[Optional[str], typer.Option("--type", help="Filter by source type")] = None,
    with_embeddings: Annotated[bool, typer.Option("--with-embeddings/--no-embeddings")] = True,
):
    """Export the index as JSONL for import on another machine.

    Streams the indexed text-representation (chunks, tags, locations, optional
    embeddings) — not the original source files. Pipe to `uridx import`:
    `uridx export | ssh uridx@hub uridx import`.
    """
    init_db()
    try:
        since_ts = to_naive_utc(since)
    except ValueError:
        print(f"Error: invalid --since timestamp: {since}", file=sys.stderr)
        raise typer.Exit(2)
    records = iter_export_records(
        since=since_ts, source_prefix=source_prefix, source_type=source_type, with_embeddings=with_embeddings
    )
    for record in records:
        output(record)


@app.command("import")
def import_():
    """Import an export stream (JSONL on stdin) into the local index.

    Idempotent: identical content merges by hash, origin machines and timestamps
    are preserved, and embeddings are reused when the model matches (otherwise
    recomputed locally from chunk text).
    """
    init_db()
    require_embed_model_match()
    _run_record_stream(import_records, "Importing", "imported")


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
def reindex():
    """Re-embed all chunks with the current embedding backend/model.

    Rebuilds only the vector table (at the new model's dimension) from chunk text
    already in the DB — no re-extraction needed. Use after changing URIDX_EMBEDDINGS,
    FASTEMBED_MODEL, or OLLAMA_EMBED_MODEL.
    """
    init_db()
    console = Console(stderr=True)
    with _progress(console) as progress:
        task = progress.add_task("Reindexing...", total=None)
        count = reindex_embeddings(
            on_progress=lambda done, total: progress.update(task, completed=done, total=total),
        )
    print(json.dumps({"reindexed": count}))


def main():
    """Entry point: report embedding-backend failures as one-line errors instead of tracebacks."""
    try:
        app()
    except EmbeddingBackendError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
