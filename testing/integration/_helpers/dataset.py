"""Dataset discovery and staging for Phase 1 integration tests."""
import os
import shutil

from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IGNORE_PATTERNS = {"__pycache__", ".cache", "output", "test_outputs", "samples"}
MAX_IMAGES = 8
MIN_IMAGES = 2


def _is_ignored(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(part in IGNORE_PATTERNS for part in parts)


def discover_images(root: str) -> list[dict]:
    """Recursively discover supported images under *root*, sorted by relative path.

    Returns a list of dicts: {path, relpath, width, height, caption_path, has_caption}.
    Skips hidden files, ignored directories, and unreadable images.
    """
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in IGNORE_PATTERNS]
        for fname in filenames:
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            if _is_ignored(rel):
                continue
            try:
                with Image.open(full) as img:
                    width, height = img.size
            except Exception:
                continue
            stem = os.path.splitext(fname)[0]
            caption = os.path.join(dirpath, stem + ".txt")
            results.append(
                {
                    "path": full,
                    "relpath": rel,
                    "width": width,
                    "height": height,
                    "caption_path": caption if os.path.isfile(caption) else None,
                    "has_caption": os.path.isfile(caption),
                }
            )
    results.sort(key=lambda x: x["relpath"])
    return results


def select_images(images: list[dict], count: int = MAX_IMAGES) -> list[dict]:
    """Deterministically select up to *count* images by sorted relative path."""
    return images[:count]


def prepare_test_dataset(images: list[dict], dest_dir: str) -> str:
    """Copy selected images and captions into *dest_dir*.

    If a caption is missing, write a minimal placeholder caption in the temp dir.
    Never modifies the source dataset.
    Returns the destination directory path.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for entry in images:
        dst_img = os.path.join(dest_dir, os.path.basename(entry["path"]))
        shutil.copy2(entry["path"], dst_img)
        stem = os.path.splitext(os.path.basename(entry["path"]))[0]
        dst_caption = os.path.join(dest_dir, stem + ".txt")
        if entry["caption_path"] and os.path.isfile(entry["caption_path"]):
            shutil.copy2(entry["caption_path"], dst_caption)
        elif not os.path.exists(dst_caption):
            with open(dst_caption, "w", encoding="utf-8") as f:
                f.write("a photo of a person\n")
    return dest_dir


def build_manifest(images: list[dict], dataset_root: str) -> dict:
    """Build a JSON-serializable manifest of selected images."""
    return {
        "dataset_root": dataset_root,
        "num_images": len(images),
        "images": [
            {
                "relpath": img["relpath"],
                "width": img["width"],
                "height": img["height"],
                "has_caption": img["has_caption"],
            }
            for img in images
        ],
    }
