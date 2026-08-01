"""Extension -> extractor dispatch for the `add` porcelain command.

`add` auto-detects which extractor handles a file by its extension, at least
for extractors that work on specific extensions.
"""

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

# File-drop extractors, most specific first. Several claim overlapping extensions
# (docling also reads .md and images); the earliest one here wins auto-detect.
EXTRACTOR_PRIORITY = ("markdown", "pdf", "image", "docling")


def load_extractor(name: str):
    """Import and return an extractor module by name."""
    if name not in EXTRACTOR_PRIORITY:
        raise ValueError(f"Unknown extractor '{name}'. Choose from: {', '.join(sorted(EXTRACTOR_PRIORITY))}")
    return importlib.import_module(f"uridx.cli.extract.{name}")


def supported_extensions(name: str) -> set[str]:
    """Extensions a named extractor handles."""
    return load_extractor(name).EXTENSIONS


@lru_cache(maxsize=1)
def default_by_ext() -> dict[str, str]:
    """Extension -> extractor for auto-detect, derived from what each extractor claims.

    Built by walking EXTRACTOR_PRIORITY in reverse so earlier (more specific)
    extractors overwrite later ones. Deriving it keeps `add foo.x` and
    `add --extractor <name> foo.x` from disagreeing about who handles `.x`.
    """
    return {ext: name for name in reversed(EXTRACTOR_PRIORITY) for ext in supported_extensions(name)}


def resolve_dispatch(paths: list[str], extractor: Optional[str] = None) -> tuple[dict[str, list[Path]], list[Path]]:
    """Bucket input paths by which extractor handles them.

    Returns (buckets, skipped): buckets maps extractor_name -> [Path, ...] (dedup by resolved
    path); skipped lists files with no matching extractor. Directories are walked recursively.
    With `extractor` set, only files matching that extractor's extensions are taken.
    Raises ValueError for an unknown extractor name.
    """
    exts = supported_extensions(extractor) if extractor else None
    by_ext = default_by_ext()
    buckets: dict[str, list[Path]] = {}
    skipped: list[Path] = []
    seen: set[Path] = set()

    def classify(path: Path) -> None:
        rp = path.resolve()
        if rp in seen:
            return
        seen.add(rp)
        ext = path.suffix.lower()
        name = (extractor if ext in exts else None) if extractor else by_ext.get(ext)
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
