"""Qwen-Image 2.1 frozen adapters: the Comfy-Org Turbo preview LoRA (its
schedule and its norm-weight deltas) and the training adapter
(assistant_lora_path) — on a toy transformer with the real module names."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from extensions_built_in.diffusion_models.qwen_image_2 import preview_lora as pl
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import (
    QwenImage2Model,
)
from toolkit.config_modules import ModelConfig

W = 8


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(W, W, bias=False)
        self.norm_q = _RMSNorm(W)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()

    def forward(self, x):
        return x + self.attn.norm_q(self.attn.to_q(x))


class ToyQwenTransformer(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.img_in = nn.Linear(W, W, bias=False)
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, x):
        x = self.img_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _lora_file(tmp_path, name, *, with_diffs=False, rank=2):
    torch.manual_seed(0)
    sd = {}
    for path in ("img_in", "transformer_blocks.0.attn.to_q", "transformer_blocks.2.attn.to_q"):
        sd[f"diffusion_model.{path}.lora_down.weight"] = torch.randn(rank, W) * 0.1
        sd[f"diffusion_model.{path}.lora_up.weight"] = torch.randn(W, rank) * 0.1
        sd[f"diffusion_model.{path}.alpha"] = torch.tensor(float(rank))
    if with_diffs:
        sd["diffusion_model.transformer_blocks.1.attn.norm_q.diff"] = torch.full((W,), 0.25)
        sd["diffusion_model.transformer_blocks.9.attn.norm_q.diff"] = torch.full((W,), 0.25)
    path = tmp_path / name
    save_file(sd, str(path))
    return str(path)


def _frozen(transformer):
    for p in transformer.parameters():
        p.requires_grad_(False)
    return transformer


@pytest.mark.parametrize(
    "schedule, name, expected",
    [
        ("auto", "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", "turbo"),
        ("auto", "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors", "viggle"),
        ("viggle", "anything.safetensors", "viggle"),
        ("TURBO", "x-viggle.safetensors", "turbo"),
    ],
)
def test_resolve_preview_schedule(schedule, name, expected):
    assert pl.resolve_preview_schedule(schedule, f"/some/dir/{name}") == expected


def test_resolve_preview_schedule_rejects_unknown():
    with pytest.raises(ValueError):
        pl.resolve_preview_schedule("lightning", "x.safetensors")


def test_model_config_preview_schedule_validation():
    config = ModelConfig(name_or_path="x", arch="qwen_image_2", preview_lora_path="p.safetensors")
    assert config.preview_lora_schedule == "auto"
    with pytest.raises(ValueError):
        ModelConfig(
            name_or_path="x",
            arch="qwen_image_2",
            preview_lora_path="p.safetensors",
            preview_lora_schedule="nope",
        )


def test_weight_diffs_apply_while_active_and_restore_exactly(tmp_path):
    transformer = _frozen(ToyQwenTransformer())
    path = _lora_file(tmp_path, "turbo.safetensors", with_diffs=True)
    turbo = pl.load_preview_lora(transformer, path, 2.0, target_lin_modules=["ToyQwenTransformer"])

    assert turbo.schedule == "turbo"
    assert turbo.report.matched == 3
    assert turbo.report.weight_diffs == 1
    # the diff for a block the model does not have is reported, not dropped
    assert any("transformer_blocks.9.attn.norm_q.diff" in s for s in turbo.report.skipped)

    norm = transformer.transformer_blocks[1].attn.norm_q.weight
    before = norm.detach().clone()
    turbo.activate(transformer, "cpu", torch.float32)
    assert torch.allclose(norm, before + 0.5)  # strength 2 x 0.25
    turbo.activate(transformer, "cpu", torch.float32)  # idempotent
    assert torch.allclose(norm, before + 0.5)
    turbo.deactivate()
    assert torch.equal(norm, before)
    assert not turbo.network.is_active


def test_training_adapter_never_applies_weight_diffs(tmp_path):
    transformer = _frozen(ToyQwenTransformer())
    path = _lora_file(tmp_path, "adapter.safetensors", with_diffs=True)
    adapter = pl.load_frozen_lora(
        transformer, path, 1.0,
        target_lin_modules=["ToyQwenTransformer"], apply_weight_diffs=False,
    )
    assert adapter.weight_diffs == []
    assert any("transformer_blocks.1.attn.norm_q" in s for s in adapter.report.skipped)


def _bare_qwen_model(**model_config_kwargs):
    model = object.__new__(QwenImage2Model)
    model.model_config = ModelConfig(name_or_path="x", arch="qwen_image_2", **model_config_kwargs)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.target_lora_modules = ["ToyQwenTransformer"]
    model.preview_turbo = None
    model._turbo_cfg_notice = False
    model._turbo_steps_notice = False
    model._status_update_hooks = []
    model.assistant_lora = None
    model.invert_assistant_lora = False
    return model


def test_turbo_schedule_is_the_turbo_checkpoints_own_sigmas():
    model = _bare_qwen_model()
    model.preview_turbo = SimpleNamespace(schedule="turbo")
    scheduler = model.get_sampling_scheduler()
    sigmas = model.get_sampling_sigmas(25)  # configured steps are overridden
    assert len(sigmas) == pl.TURBO_STEPS
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=0.8)
    # no dynamic shift, no shift terminal: the sigmas come out as given
    assert np.allclose(scheduler.sigmas[:-1].numpy(), pl.TURBO_SIGMAS, atol=1e-6)
    assert float(scheduler.sigmas[-1]) == 0.0


def test_viggle_schedule_keeps_its_grid_and_the_resolution_shift():
    model = _bare_qwen_model()
    model.preview_turbo = SimpleNamespace(schedule="viggle")
    assert np.allclose(model.get_sampling_sigmas(6), pl.turbo_sigma_grid(6))
    scheduler = model.get_sampling_scheduler()
    assert scheduler.config.use_dynamic_shifting
    assert scheduler.config.shift_terminal is None


def test_qwen_training_adapter_is_live_for_training_and_off_for_previews(tmp_path):
    transformer = _frozen(ToyQwenTransformer())
    path = _lora_file(tmp_path, "fizgig_qwen_adapter.safetensors")
    model = _bare_qwen_model(assistant_lora_path=path)

    x = torch.randn(2, W)
    base_out = transformer(x)
    model.load_training_adapter(transformer)

    adapter = model.assistant_lora
    assert adapter is not None and adapter.is_active
    assert model.invert_assistant_lora is False
    assert all(not p.requires_grad for p in adapter.parameters())
    assert not torch.allclose(transformer(x), base_out)

    model._enter_generate_adapters()
    assert not adapter.is_active
    assert torch.allclose(transformer(x), base_out)
    model._exit_generate_adapters()
    assert adapter.is_active


def test_assistant_lora_path_is_accepted_on_qwen_image_2():
    config = ModelConfig(name_or_path="x", arch="qwen_image_2", assistant_lora_path="a/b/c.safetensors")
    assert config.assistant_lora_path == "a/b/c.safetensors"
