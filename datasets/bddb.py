from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from config import cfg


def load_bddb_metadata() -> Dict:
    json_path = Path(cfg.paths.bddb_root) / "data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"BDDB metadata file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta


def inspect_bddb_folder() -> Dict:
    root = Path(cfg.paths.bddb_root)
    if not root.exists():
        raise FileNotFoundError(f"BDDB folder not found: {root}")

    files = [p for p in root.rglob("*") if p.is_file()]
    rel_files = [str(p.relative_to(root)) for p in files]

    suffix_count = {}
    for p in files:
        suffix = p.suffix.lower()
        suffix_count[suffix] = suffix_count.get(suffix, 0) + 1

    return {
        "root": str(root),
        "num_files": len(files),
        "suffix_count": suffix_count,
        "preview_files": rel_files[:200],
    }


def print_bddb_summary() -> None:
    meta = load_bddb_metadata()
    info = inspect_bddb_folder()

    print("=== BDDB Metadata ===")
    print(f"title      : {meta.get('title', 'N/A')}")
    print(f"identifier : {meta.get('identifier', 'N/A')}")

    print("\n=== BDDB Folder Summary ===")
    print(f"root        : {info['root']}")
    print(f"num_files   : {info['num_files']}")
    print(f"suffix_count: {info['suffix_count']}")

    print("\n=== Preview files ===")
    for fp in info["preview_files"]:
        print(fp)
