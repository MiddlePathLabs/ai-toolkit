"""Small checkpoint merges preserve each key's source precision."""
import json
import sys

import torch
from safetensors.torch import load_file, save_file

from ui_scripts.merge_loras import main


def test_merge_preserves_per_key_dtype_and_mixed_rank(tmp_path, monkeypatch):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    output = tmp_path / "merged.safetensors"
    save_file({
        "fp16": torch.ones(2, dtype=torch.float16),
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "fp64": torch.ones(2, dtype=torch.float64),
        "mixed": torch.ones(2, dtype=torch.bfloat16),
    }, str(first))
    save_file({"mixed": torch.ones(3, dtype=torch.float16)}, str(second))
    monkeypatch.setattr(sys, "argv", [
        "merge_loras.py", "--loras", json.dumps([
            {"path": str(first), "strength": 0.5},
            {"path": str(second), "strength": 1.0},
        ]), "--output", str(output),
    ])
    assert main() == 0
    merged = load_file(str(output))
    for key, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16),
                       ("fp64", torch.float64)):
        assert merged[key].dtype == dtype
        assert torch.equal(merged[key], torch.full((2,), 0.5, dtype=dtype))
    assert merged["mixed"].dtype == torch.float16
    assert torch.equal(merged["mixed"], torch.tensor([1.5, 1.5, 1.0], dtype=torch.float16))
