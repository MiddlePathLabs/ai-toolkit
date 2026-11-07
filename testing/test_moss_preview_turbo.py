"""MOSS-Image preview Turbo LoRA: schedule resolution + step policy, the
arch gate in ModelConfig, and the per-DiT activation that keeps Base and
SFT_ preview twins under the same turbo recipe — on a toy transformer with
the real module names."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from extensions_built_in.diffusion_models.moss_image.moss_image import (
    MossImageModel,
)
from extensions_built_in.diffusion_models.moss_image.preview_turbo import (
    moss_turbo_steps,
    resolve_moss_preview_schedule,
)
from toolkit.config_modules import ModelConfig

W = 8


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(W, 3 * W, bias=False)
        self.o = nn.Linear(W, W, bias=False)
        self.ffn_up_gate = nn.Linear(W, 2 * W, bias=False)
        self.ffn_down = nn.Linear(2 * W, W, bias=False)

    def forward(self, x):
        q, _, _ = self.qkv(x).chunk(3, dim=-1)
        return x + self.o(q)


class ToyMossTransformer(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.video_embedder = nn.Linear(4, W, bias=False)
        self.blocks = nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, x):
        x = self.video_embedder(x)
        for block in self.blocks:
            x = block(x)
        return x


def _lora_file(tmp_path, name="moss_image_dmd8.safetensors", rank=2):
    torch.manual_seed(0)
    sd = {}
    paths = ["video_embedder", "blocks.0.qkv", "blocks.1.ffn_down"]
    for path in paths:
        out_f = {"video_embedder": W, "blocks.0.qkv": 3 * W, "blocks.1.ffn_down": W}[path]
        in_f = 2 * W if path == "blocks.1.ffn_down" else (4 if path == "video_embedder" else W)
        sd[f"diffusion_model.{path}.lora_down.weight"] = torch.randn(rank, in_f) * 0.1
        sd[f"diffusion_model.{path}.lora_up.weight"] = torch.randn(out_f, rank) * 0.1
        sd[f"diffusion_model.{path}.alpha"] = torch.tensor(float(rank))
    path = tmp_path / name
    save_file(sd, str(path))
    return str(path)


def _frozen(transformer):
    for p in transformer.parameters():
        p.requires_grad_(False)
    return transformer


def _bare_model(tmp_path, *, preview_dit=None, **model_config_kwargs):
    model = object.__new__(MossImageModel)
    model.model_config = ModelConfig(
        name_or_path="x",
        arch="moss_image",
        preview_lora_path=_lora_file(tmp_path),
        **model_config_kwargs,
    )
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.target_lora_modules = ["ToyMossTransformer"]
    model._status_update_hooks = []
    model.model = _frozen(ToyMossTransformer())
    model._preview_dit = preview_dit
    model._preview_turbos = []
    model._preview_turbo_schedule = ""
    model._turbo_cfg_notice = False
    model._turbo_steps_notice = False
    return model


# ---------------------------------------------------------------- schedules
@pytest.mark.parametrize(
    "schedule, name, expected",
    [
        ("auto", "MOSS-Image-Base-8-steps-dmd", "dmd"),
        ("auto", "MOSS-Image-SFT-4-steps-v3.safetensors", "sft4"),
        ("auto", "MOSS-Image-Turbo.safetensors", "c16"),
        ("auto", "some_random_distill.safetensors", "dmd"),  # safe default
        ("sft4", "anything.safetensors", "sft4"),
        ("DMD", "x.safetensors", "dmd"),
    ],
)
def test_resolve_moss_schedule(schedule, name, expected):
    assert resolve_moss_preview_schedule(schedule, f"/dir/{name}") == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("moss-turbo-8steps.safetensors", "dmd"),  # step count beats 'turbo'
        ("moss_turbo_4step.safetensors", "sft4"),
        ("moss-16-steps.safetensors", "c16"),
        ("moss-6_steps.safetensors", "dmd"),
        ("moss-v128steps.safetensors", "dmd"),  # no count token; default
    ],
)
def test_resolve_moss_schedule_explicit_step_count(name, expected):
    assert resolve_moss_preview_schedule("auto", f"/dir/{name}") == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("akhaliq/MOSS-Image-SFT-4-steps/model.safetensors", "sft4"),
        ("Coffmeteor/MOSS-Image-Turbo/model.safetensors", "c16"),
        ("lodestones/MOSS-Image-Base-8-steps-dmd/model.safetensors", "dmd"),
        # only repo + file count: a user or local directory name cannot steer it
        ("turbo-user/plain-repo/model.safetensors", "dmd"),
        (r"D:\turbo\loras\adapters\model.safetensors", "dmd"),
    ],
)
def test_resolve_moss_schedule_reads_configured_hub_path(source, expected):
    # the resolved download keeps only the generic file name
    resolved = "/models/loras/preview_adapters/model.safetensors"
    assert resolve_moss_preview_schedule("auto", resolved, source=source) == expected


def test_resolve_moss_schedule_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_moss_preview_schedule("turbo", "x.safetensors")  # qwen recipe


def test_moss_turbo_steps_dmd_clamps_into_card_window():
    assert moss_turbo_steps("dmd", 25, 1024, 1024) == 8
    assert moss_turbo_steps("dmd", 6, 512, 512) == 6
    assert moss_turbo_steps("dmd", 3, 512, 512) == 4


def test_moss_turbo_steps_sft4_bands_by_resolution():
    assert moss_turbo_steps("sft4", 25, 1024, 1024) == 4
    assert moss_turbo_steps("sft4", 25, 1536, 1024) == 6
    assert moss_turbo_steps("sft4", 25, 2048, 2048) == 8


def test_moss_turbo_steps_c16_is_fixed():
    assert moss_turbo_steps("c16", 4, 1024, 1024) == 16
    assert moss_turbo_steps("c16", 50, 2048, 2048) == 16


# ---------------------------------------------------------------- config gate
def test_model_config_accepts_moss_preview_lora():
    config = ModelConfig(
        name_or_path="x", arch="moss_image", preview_lora_path="lodestones/MOSS-Image-Base-8-steps-dmd/lora.safetensors"
    )
    assert config.preview_lora_schedule == "auto"


def test_model_config_schedule_sets_are_arch_aware():
    ModelConfig(
        name_or_path="x", arch="moss_image",
        preview_lora_path="p.safetensors", preview_lora_schedule="sft4",
    )
    with pytest.raises(ValueError):  # qwen recipe on moss
        ModelConfig(
            name_or_path="x", arch="moss_image",
            preview_lora_path="p.safetensors", preview_lora_schedule="turbo",
        )
    with pytest.raises(ValueError):  # moss recipe on qwen
        ModelConfig(
            name_or_path="x", arch="qwen_image_2",
            preview_lora_path="p.safetensors", preview_lora_schedule="dmd",
        )
    with pytest.raises(ValueError):  # unsupported arch
        ModelConfig(name_or_path="x", arch="krea2", preview_lora_path="p.safetensors")


# ---------------------------------------------------------------- per-DiT turbo
def test_preview_turbo_loads_on_both_dits(tmp_path):
    model = _bare_model(tmp_path, preview_dit=_frozen(ToyMossTransformer()))
    assert model._preview_turbos == []
    model._maybe_load_preview_turbo()
    assert model._preview_turbo_schedule == "dmd"  # 'dmd' in the file name
    assert len(model._preview_turbos) == 2  # training DiT + preview checkpoint DiT
    assert all(t.report.matched == 3 for _, t in model._preview_turbos)
    assert all(not t.network.is_active for _, t in model._preview_turbos)


def test_preview_turbo_skipped_without_flag(tmp_path):
    model = _bare_model(tmp_path)
    model.model_config.preview_lora_path = None
    model._maybe_load_preview_turbo()
    assert model._preview_turbos == []


def test_turbo_activates_only_on_the_live_dit(tmp_path):
    model = _bare_model(tmp_path, preview_dit=_frozen(ToyMossTransformer()))
    model._maybe_load_preview_turbo()
    training_dit, sft_dit = model.model, model._preview_dit
    x = torch.randn(2, 4)
    base_training = training_dit(x)
    base_sft = sft_dit(x)

    # the Base pass: swap nothing, the live DiT is the training one
    model._enter_generate_adapters()
    assert not torch.allclose(training_dit(x), base_training)
    assert torch.equal(sft_dit(x), base_sft)  # SFT DiT untouched
    model._exit_generate_adapters()
    assert torch.equal(training_dit(x), base_training)  # exact restore

    # the SFT pass: preview_checkpoint_context swapped self.model first
    model.model = sft_dit
    model._enter_generate_adapters()
    assert not torch.allclose(sft_dit(x), base_sft)
    assert torch.equal(training_dit(x), base_training)
    model._exit_generate_adapters()
    assert torch.equal(sft_dit(x), base_sft)


# ---------------------------------------------------------------- sample recipe
def _gen_config(**kw):
    defaults = dict(
        guidance_scale=3.0, num_inference_steps=25, sigmas=None,
        width=1024, height=1024,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_turbo_sample_settings_forces_cfg1_and_schedule_steps(tmp_path):
    model = _bare_model(tmp_path)
    model._preview_turbos = [("dit", object())]  # helper only checks non-empty
    model._preview_turbo_schedule = "dmd"
    steps, sigmas, guidance = model._turbo_sample_settings(_gen_config())
    assert (steps, sigmas, guidance) == (8, None, 1.0)

    # explicit per-prompt sigmas give way too (turbo owns its window)
    steps, sigmas, guidance = model._turbo_sample_settings(
        _gen_config(sigmas=[1.0, 0.5, 0.0])
    )
    assert (steps, sigmas, guidance) == (8, None, 1.0)


def test_turbo_sample_settings_sft4_follows_resolution(tmp_path):
    model = _bare_model(tmp_path)
    model._preview_turbos = [("dit", object())]
    model._preview_turbo_schedule = "sft4"
    assert model._turbo_sample_settings(_gen_config(width=2048, height=2048))[0] == 8
    assert model._turbo_sample_settings(_gen_config(width=1024, height=1024))[0] == 4


def test_qwen_preview_lora_module_still_re_exports_shared_machinery():
    # the shared module was qwen's first home; its callers (and tests) import
    # these names from there
    from extensions_built_in.diffusion_models.qwen_image_2 import preview_lora as pl

    assert pl.QwenPreviewTurbo is pl.PreviewTurbo
    assert callable(pl.load_frozen_lora)
    assert callable(pl.resolve_preview_lora_path)
