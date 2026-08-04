"""LoRA state management: checksums, seed creation, and reload verification."""
import glob
import hashlib
import os

import torch
from safetensors.torch import load_file


def file_checksum(path: str) -> str:
    """SHA-256 checksum of a file, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def tensor_checksum(tensor: torch.Tensor) -> str:
    """SHA-256 checksum of a tensor's raw bytes, truncated to 16 hex chars."""
    h = hashlib.sha256()
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def find_saved_lora(save_root: str, name: str) -> str | None:
    """Find the most recently saved LoRA safetensors for *name* under *save_root*.

    Looks for {name}*.safetensors patterns, preferring step-suffixed files,
    falling back to the unsuffixed final save.
    """
    save_dir = os.path.join(save_root, name)
    search_dirs = [save_dir, save_root]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        pattern = os.path.join(d, f"{name}*.safetensors")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if files:
            return files[0]
    return None


def find_all_saved_loras(save_root: str, name: str) -> list[str]:
    """Find every saved LoRA safetensors for *name*, sorted by embedded step number.

    Used by the strict depth-only proof to compare consecutive optimizer-step
    checkpoints (mirrors evidence/compare_safetensors_delta.py). Falls back to
    mtime order when filenames lack a step suffix.
    """
    import re

    save_dir = os.path.join(save_root, name)
    search_dirs = [save_dir, save_root]
    collected: list[str] = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        collected.extend(glob.glob(os.path.join(d, f"{name}*.safetensors")))
    # dedupe while preserving a deterministic order
    collected = sorted(set(collected))

    def _step_key(path: str):
        m = re.search(r"(\d{6,})", os.path.basename(path))
        return int(m.group(1)) if m else 0

    return sorted(collected, key=_step_key)


def load_lora_tensors(path: str) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors LoRA file."""
    return load_file(path)


def verify_saved_lora(saved_path: str, live_state_dict: dict, tolerance: float = 1e-3) -> dict:
    """Verify that a saved LoRA matches the live model state.

    Returns a dict with:
      - matches: bool
      - max_abs_diff: float
      - num_keys_checked: int
      - mismatches: list of key/max_abs_diff pairs
    """
    saved = load_lora_tensors(saved_path)
    max_diff = 0.0
    mismatches = []
    checked = 0
    for key, saved_tensor in saved.items():
        if key not in live_state_dict:
            continue
        live_tensor = live_state_dict[key].detach().cpu().float()
        saved_float = saved_tensor.float()
        if saved_float.shape != live_tensor.shape:
            mismatches.append({"key": key, "error": f"shape mismatch {saved_float.shape} vs {live_tensor.shape}"})
            continue
        diff = float((saved_float - live_tensor).abs().max().item())
        max_diff = max(max_diff, diff)
        checked += 1
        if diff > tolerance:
            mismatches.append({"key": key, "max_abs_diff": diff})
    return {
        "matches": len(mismatches) == 0,
        "max_abs_diff": max_diff,
        "num_keys_checked": checked,
        "mismatches": mismatches[:10],
    }
