"""assistant_lora_path arch gate: config validation and static admission accept
exactly the archs whose loader attaches the adapter, and every UI model that
defaults an adapter passes."""
import re
from pathlib import Path

import pytest

from toolkit.admission import (
    RULE_PREVIEW_LORA,
    _ASSISTANT_LORA_ARCHS,
    collect_admission_diagnostics,
)
from toolkit.config_modules import ASSISTANT_LORA_ARCHS, ModelConfig

ADAPTER = "owner/repo/adapter.safetensors"
UI_MODELS = Path(__file__).resolve().parents[1] / "extensions_built_in" / "diffusion_models" / "ui.tsx"

ACCEPTED = [
    "flux",
    "flex1",
    "minimax_h3",
    "minimax_h3_omni",
    "minimax_h3_ref2va",
    "minimax_h3_vsa",
    "qwen_image_2",
    "zimage:turbo",
    "zimage_l2p",
    "ming_image",
    "krea2:turbo",
    "krea2:o_edit_turbo",
]
REJECTED = ["sdxl", "sd1", "wan22_14b", "wan21", "lumina2", "qwen_image", "chroma"]


def _assistant_lora_errors(arch, adapter=ADAPTER):
    process = {
        "model": {"name_or_path": "x", "arch": arch, "assistant_lora_path": adapter},
        "train": {"optimizer": "adamw"},
        "datasets": [{"dataset_path": "not-present"}],
    }
    result = collect_admission_diagnostics({"process": [process]})
    return [
        d for d in result["diagnostics"]
        if d["rule_id"] == RULE_PREVIEW_LORA
        and any(f.endswith("assistant_lora_path") for f in d["fields"])
    ]


@pytest.mark.parametrize("arch", ACCEPTED)
def test_supported_arch_accepts_assistant_lora(arch):
    config = ModelConfig(name_or_path="x", arch=arch, assistant_lora_path=ADAPTER)
    assert config.assistant_lora_path == ADAPTER
    assert _assistant_lora_errors(arch) == []


@pytest.mark.parametrize("arch", REJECTED)
def test_unsupported_arch_rejects_assistant_lora(arch):
    with pytest.raises(ValueError, match="assistant_lora_path is only supported"):
        ModelConfig(name_or_path="x", arch=arch, assistant_lora_path=ADAPTER)
    assert len(_assistant_lora_errors(arch)) == 1


def test_unsupported_arch_without_adapter_is_unaffected():
    ModelConfig(name_or_path="x", arch="sdxl")
    assert _assistant_lora_errors("sdxl", adapter=None) == []


def test_admission_mirror_matches_config():
    assert set(ASSISTANT_LORA_ARCHS) == _ASSISTANT_LORA_ARCHS


def test_every_ui_adapter_default_is_allowed():
    text = UI_MODELS.read_text(encoding="utf-8")
    defaults = []
    for block in re.split(r'\n\s*name: "', text)[1:]:
        name = block.split('"', 1)[0]
        if re.search(r'"config\.process\[0\]\.model\.assistant_lora_path": \[\s*"', block):
            defaults.append(name)
    assert "zimage:turbo" in defaults and "ming_image" in defaults
    assert "minimax_h3_omni" in defaults
    for name in defaults:
        assert name.split(":")[0] in ASSISTANT_LORA_ARCHS, name
