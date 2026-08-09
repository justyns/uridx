"""Extract subcommands. Plugins register under the 'uridx.extractors' entry point group."""

import sys
from importlib.metadata import entry_points

import typer

from . import claude_code, docling, image, maildir, markdown, mbox, pdf, tsugite

app = typer.Typer(help="Extract content to JSONL for ingestion")

app.command("claude-code")(claude_code.extract)
app.command("docling")(docling.extract)
app.command("markdown")(markdown.extract)
app.command("pdf")(pdf.extract)
app.command("image")(image.extract)
app.command("tsugite")(tsugite.extract)
app.command("maildir")(maildir.extract)
app.command("mbox")(mbox.extract)


def load_plugins():
    """Load extractor plugins from entry points."""
    for ep in entry_points(group="uridx.extractors"):
        try:
            extractor = ep.load()
            if isinstance(extractor, typer.Typer):
                app.add_typer(extractor, name=ep.name)
            elif callable(extractor):
                app.command(ep.name)(extractor)
        except Exception as e:
            print(f"Failed to load extractor plugin '{ep.name}': {e}", file=sys.stderr)


load_plugins()
