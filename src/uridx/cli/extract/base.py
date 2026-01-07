"""Shared utilities for extractors."""

import json


def output(record: dict) -> None:
    """Output a single JSONL record to stdout."""
    print(json.dumps(record))
