"""Exclude rules and the directory walk behind `add`, `extract` and `ingest`."""

import os
import subprocess
from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path

import pathspec

from uridx.context import normalize_prefix, prefix_matches

DEFAULT_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".claude/worktrees",
    ".tsugite-jobs",
    "*.pyc",
)


def is_glob(rule: str) -> bool:
    return any(c in rule for c in "*?[")


def normalize_exclude(raw: str) -> str:
    """Stored form. Paths normalize like context prefixes. Bare names and relative globs match at any depth."""
    if "/" not in raw and not raw.startswith("~") and raw not in (".", ".."):
        return "**/" + raw
    if not is_glob(raw):
        return normalize_prefix(raw)
    raw = os.path.expanduser(raw)
    return raw if raw.startswith(("/", "**")) else "**/" + raw


@lru_cache
def _glob_spec(globs: tuple[str, ...]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitignore", globs)


def is_excluded(uri_or_path: str, rules: Sequence[str]) -> bool:
    """True if a rule covers this source URI (a bare absolute path counts as file://)."""
    uri = f"file://{uri_or_path}" if uri_or_path.startswith("/") else uri_or_path
    if any(prefix_matches(r, uri) for r in rules if not is_glob(r)):
        return True
    globs = tuple(r for r in rules if is_glob(r))
    return (
        bool(globs)
        and uri.startswith("file://")
        and _glob_spec(globs).match_file(uri.removeprefix("file://").lstrip("/"))
    )


def iter_files(root: Path, rules: Sequence[str], *, gitignore: bool = True) -> Iterator[Path]:
    """Files under `root` that no rule covers and, with `gitignore`, that git does not ignore.

    Rules match the resolved path. A glob that covers `root` itself matches paths relative to
    `root` instead, so a named root inside `build/` is walked and `build/` dirs under it are not.
    """
    at_root = tuple(r for r in rules if is_glob(r) and is_excluded(str(root.resolve()), [r]))
    rules = [r for r in rules if r not in at_root]

    def excluded(path: Path) -> bool:
        return is_excluded(str(path.resolve()), rules) or _glob_spec(at_root).match_file(path.relative_to(root))

    yield from (f for f in _candidates(root, gitignore, excluded) if not excluded(f))


def _candidates(directory: Path, gitignore: bool, excluded: Callable[[Path], bool]) -> Iterator[Path]:
    if gitignore and _git_lists(directory):
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=directory,
            capture_output=True,
            check=True,
        ).stdout
        for path in sorted({directory / os.fsdecode(name) for name in out.split(b"\0") if name}):
            if path.is_dir() and not path.is_symlink():
                # An embedded repo or submodule, which git lists as one entry.
                if not excluded(path):
                    yield from _candidates(path, gitignore, excluded)
            elif path.is_file():
                yield path
        return
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(d for d in dirnames if not excluded(Path(dirpath, d)))
        yield from (p for f in sorted(filenames) if (p := Path(dirpath, f)).is_file())


def _git_lists(directory: Path) -> bool:
    """True if `directory` is in a git work tree that does not ignore it.

    A directory git ignores is walked without gitignore, .gitignore files inside it included.
    """
    resolved = directory.resolve()
    if not any((d / ".git").exists() for d in (resolved, *resolved.parents)):
        return False
    check = subprocess.run(["git", "check-ignore", "-q", "."], cwd=directory)
    if check.returncode > 1:
        raise subprocess.CalledProcessError(check.returncode, check.args)
    return check.returncode == 1
