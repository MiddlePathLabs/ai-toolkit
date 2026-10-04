"""Explicit, preview-first removal of regenerable dataset cache files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Sequence


CACHE_DIRECTORIES = {
    "latent": "_latent_cache",
    "text": "_t_e_cache",
    "clip": "_clip_vision_cache",
    "face": "_face_id_cache",
    "body_proportion": "_body_proportion_cache",
    "body_shape": "_body_shape_cache",
    "normal": "_normal_cache",
    "vae_anchor": "_vae_anchor_cache",
}


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _dataset_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().absolute()
    if any(_is_link(parent) for parent in (path, *path.parents)):
        raise ValueError("Dataset root must not traverse a symbolic link or junction.")
    if not path.is_dir():
        raise ValueError("Dataset root must be an existing directory.")
    return path.resolve()


def _identity(path: Path, root: Path) -> dict:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or _is_link(path):
        raise ValueError(f"Not a regular cache file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def preview_cache_cleanup(root: str | os.PathLike[str], kinds: Sequence[str] = ("latent", "text")) -> dict:
    dataset = _dataset_root(root)
    selected = sorted(set(kinds))
    if not selected or any(kind not in CACHE_DIRECTORIES for kind in selected):
        raise ValueError("Select at least one known dataset cache type.")
    directories = {CACHE_DIRECTORIES[kind] for kind in selected}
    files = []
    pending = [dataset]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.path)
                if _is_link(path) or not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name in CACHE_DIRECTORIES.values():
                    if entry.name in directories:
                        with os.scandir(path) as cached:
                            for item in cached:
                                candidate = Path(item.path)
                                if item.is_file(follow_symlinks=False) and candidate.suffix == ".safetensors" and not _is_link(candidate):
                                    files.append(_identity(candidate, dataset))
                    continue
                pending.append(path)
    files.sort(key=lambda item: item["path"])
    plan = {"root": str(dataset), "cache_types": selected, "files": files}
    token = hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()
    return {**plan, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files), "token": token}


def apply_cache_cleanup(root: str | os.PathLike[str], token: str, kinds: Sequence[str] = ("latent", "text")) -> dict:
    plan = preview_cache_cleanup(root, kinds)
    if token != plan["token"]:
        raise ValueError("Cache files changed or the preview does not match. Preview again before applying.")
    dataset = Path(plan["root"])
    for item in plan["files"]:
        path = dataset / item["path"]
        if any(_is_link(parent) for parent in (path, *path.parents)) or _identity(path, dataset) != item:
            raise ValueError("Cache path changed. Preview again before applying.")
    removed = []
    for item in plan["files"]:
        path = dataset / item["path"]
        if any(_is_link(parent) for parent in (path, *path.parents)) or _identity(path, dataset) != item:
            raise ValueError(f"Cache path changed after {len(removed)} removals. Preview again before applying.")
        path.unlink()
        removed.append(item["path"])
    return {"root": str(dataset), "removed": removed, "removed_bytes": plan["total_bytes"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview regenerable dataset caches. Stop jobs using the dataset before applying; selected current and old caches will both be removed.")
    parser.add_argument("dataset_root")
    parser.add_argument("--cache", choices=sorted(CACHE_DIRECTORIES), action="append", dest="kinds")
    parser.add_argument("--apply", metavar="PREVIEW_TOKEN", help="Delete only the unchanged files in the matching preview.")
    args = parser.parse_args(argv)
    kinds = args.kinds or ("latent", "text")
    try:
        result = apply_cache_cleanup(args.dataset_root, args.apply, kinds) if args.apply else preview_cache_cleanup(args.dataset_root, kinds)
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
