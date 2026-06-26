"""Extension -> extractor dispatch for the `add` porcelain command.

`add` auto-detects which extractor handles a file by its extension, at least
for extractors that work on specific extensions.
"""

import importlib
from pathlib import Path
from typing import Optional

# Extractor name -> module path
MODULES = {
    "markdown": "uridx.cli.extract.markdown",
    "pdf": "uridx.cli.extract.pdf",
    "image": "uridx.cli.extract.image",
    "docling": "uridx.cli.extract.docling",
}

# Default extension -> extractor for auto-detect
DEFAULT_BY_EXT = {
    ".md": "markdown",
    ".mdc": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".docx": "docling",
    ".xlsx": "docling",
    ".pptx": "docling",
    ".html": "docling",
    ".htm": "docling",
    ".xhtml": "docling",
    ".tiff": "docling",
    ".adoc": "docling",
    ".csv": "docling",
}


def load_extractor(name: str):
    """Import and return an extractor module by name."""
    return importlib.import_module(MODULES[name])


def supported_extensions(name: str) -> set[str]:
    """Extensions a named extractor handles."""
    return load_extractor(name).EXTENSIONS


def resolve_dispatch(paths: list[str], extractor: Optional[str] = None) -> tuple[dict[str, list[Path]], list[Path]]:
    """Bucket input paths by which extractor handles them.

    Returns (buckets, skipped): buckets maps extractor_name -> [Path, ...] (dedup by resolved
    path); skipped lists files with no matching extractor. Directories are walked recursively.
    With `extractor` set, only files matching that extractor's extensions are taken.
    """
    if extractor is not None and extractor not in MODULES:
        raise ValueError(f"Unknown extractor '{extractor}'. Choose from: {', '.join(sorted(MODULES))}")

    exts = supported_extensions(extractor) if extractor else None
    buckets: dict[str, list[Path]] = {}
    skipped: list[Path] = []
    seen: set[Path] = set()

    def classify(path: Path) -> None:
        rp = path.resolve()
        if rp in seen:
            return
        seen.add(rp)
        ext = path.suffix.lower()
        name = (extractor if ext in exts else None) if extractor else DEFAULT_BY_EXT.get(ext)
        (buckets.setdefault(name, []) if name else skipped).append(path)

    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    classify(f)
        else:
            classify(path)  # a file, or a nonexistent path (no matching ext -> skipped)

    return buckets, skipped
