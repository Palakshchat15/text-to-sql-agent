"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ("flight", "fraud", "retail", "sales")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    return ROOT / load_config()["paths"][key]


def load_golden() -> list[dict]:
    with open(path("golden"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_splits() -> dict[str, list[str]]:
    with open(path("splits"), encoding="utf-8") as f:
        return json.load(f)


def questions_for(split: str) -> list[dict]:
    if split == "all":
        return load_golden()
    ids = set(load_splits()[split])
    return [q for q in load_golden() if q["id"] in ids]
