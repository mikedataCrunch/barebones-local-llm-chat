from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import faiss


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class IndexMeta:
    docs_path: str
    docs_sha256: str
    embed_model: str
    dim: int

    def to_dict(self) -> dict:
        return {
            "docs_path": self.docs_path,
            "docs_sha256": self.docs_sha256,
            "embed_model": self.embed_model,
            "dim": int(self.dim),
        }

    @staticmethod
    def from_dict(d: dict) -> "IndexMeta":
        return IndexMeta(
            docs_path=str(d["docs_path"]),
            docs_sha256=str(d["docs_sha256"]),
            embed_model=str(d["embed_model"]),
            dim=int(d["dim"]),
        )


def index_paths(index_dir: str | os.PathLike[str]) -> dict[str, Path]:
    root = Path(index_dir)
    return {
        "root": root,
        "faiss": root / "index.faiss",
        "docs": root / "docs.jsonl",
        "meta": root / "meta.json",
    }


def save_index(
    index_dir: str | os.PathLike[str],
    *,
    index: faiss.Index,
    documents: list[str],
    meta: IndexMeta,
) -> None:
    p = index_paths(index_dir)
    p["root"].mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(p["faiss"]))

    with open(p["docs"], "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)


def load_index(index_dir: str | os.PathLike[str]) -> tuple[faiss.Index, list[str], IndexMeta]:
    p = index_paths(index_dir)
    if not (p["faiss"].exists() and p["docs"].exists() and p["meta"].exists()):
        missing = [k for k in ("faiss", "docs", "meta") if not p[k].exists()]
        raise FileNotFoundError(f"Missing index files in {p['root']}: {', '.join(missing)}")

    index = faiss.read_index(str(p["faiss"]))

    documents: list[str] = []
    with open(p["docs"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))

    with open(p["meta"], "r", encoding="utf-8") as f:
        meta = IndexMeta.from_dict(json.load(f))

    return index, documents, meta

