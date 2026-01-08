"""Extract subcommands with plugin support.

Built-in extractors and plugin discovery via entry points.
Plugins register under 'uridx.extractors' entry point group.
"""

import sys
from importlib.metadata import entry_points

import typer

from . import claude_code, docling, image, markdown, pdf

app = typer.Typer(help="Extract content to JSONL for ingestion")

app.command("claude-code")(claude_code.extract)
app.command("docling")(docling.extract)
app.command("markdown")(markdown.extract)
app.command("pdf")(pdf.extract)
app.command("image")(image.extract)


def load_plugins():
    """Load extractor plugins from entry points."""
    try:
        eps = entry_points(group="uridx.extractors")
    except TypeError:
        eps = entry_points().get("uridx.extractors", [])

    for ep in eps:
        try:
            extractor = ep.load()
            if isinstance(extractor, typer.Typer):
                app.add_typer(extractor, name=ep.name)
            elif callable(extractor):
                app.command(ep.name)(extractor)
        except Exception as e:
            print(f"Failed to load extractor plugin '{ep.name}': {e}", file=sys.stderr)


load_plugins()
