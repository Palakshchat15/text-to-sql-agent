"""Schema catalogue loading, BM25 table retrieval and schema rendering for the prompt."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import bm25s
import Stemmer

from .config import path

_STEM = Stemmer.Stemmer("english")


@lru_cache(maxsize=4)
def load_catalog(p: str | None = None) -> list[dict]:
    with open(p or path("catalog"), encoding="utf-8") as f:
        return json.load(f)


def table_doc(t: dict) -> str:
    """Text indexed for one table: names (split on _), descriptions, column names/notes, samples."""
    parts = [t["fq"], t["schema"], t["table"].replace("_", " "), t["description"], t.get("project", "")]
    for c in t["columns"]:
        parts += [c["name"].replace("_", " "), c["description"]]
        parts += [str(s) for s in c["samples"] if isinstance(s, str) and len(s) < 40]
    return " ".join(p for p in parts if p)


def _tokenize(texts: list[str]):
    texts = [re.sub(r"[_\-./]", " ", x.lower()) for x in texts]
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEM, show_progress=False)


class SchemaRetriever:
    def __init__(self, catalog: list[dict] | None = None):
        self.catalog = catalog if catalog is not None else load_catalog()
        self.by_fq = {t["fq"]: t for t in self.catalog}
        self.index = bm25s.BM25()
        self.index.index(_tokenize([table_doc(t) for t in self.catalog]), show_progress=False)

    def retrieve(self, question: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k tables as (fq_name, score). Ties keep catalogue order."""
        k = min(k, len(self.catalog))
        q = _tokenize([question])
        if not q.vocab or all(len(x) == 0 for x in q.ids):
            return [(t["fq"], 0.0) for t in self.catalog[:k]]
        docs, scores = self.index.retrieve(q, k=k, show_progress=False)
        return [(self.catalog[int(i)]["fq"], float(s)) for i, s in zip(docs[0], scores[0])]

    def all_tables(self) -> list[str]:
        return [t["fq"] for t in self.catalog]

    def table_names(self) -> set[str]:
        return set(self.by_fq)


def render_schema(tables: list[dict], docs: bool = True) -> str:
    """CREATE TABLE-style schema text. docs=False gives the bare names + types (zero-shot baseline)."""
    out = []
    for t in tables:
        lines = []
        if docs:
            head = f"-- {t['fq']} ({t['rows']:,} rows)"
            desc = " ".join(x for x in [t["description"], t.get("project", "")] if x)
            lines.append(head + (f": {desc}" if desc else ""))
        lines.append(f"CREATE TABLE {t['fq']} (")
        n = len(t["columns"])
        for i, c in enumerate(t["columns"]):
            s = f"  {c['name']} {c['type']}" + ("," if i < n - 1 else "")
            if docs:
                note = c["description"].rstrip(".")
                samples = ", ".join(str(v) for v in c["samples"][:3])
                extra = ". ".join(x for x in [note, f"Examples: {samples}" if samples else ""] if x)
                if extra:
                    s += f"  -- {extra}"
            lines.append(s)
        lines.append(");")
        out.append("\n".join(lines))
    return "\n\n".join(out)
