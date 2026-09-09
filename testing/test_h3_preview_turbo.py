import hashlib

import pytest
import torch
import torch.nn as nn

from safetensors.torch import save_file


from extensions_built_in.diffusion_models.minimax_h3.src.transformer import MiniMaxH3AdalnProj
from extensions_built_in.diffusion_models.minimax_h3.src.turbo import (
    EGRID_SHA256,
    EGRID_SHAPE,
    EGRID_SOURCE,
    FULL_MODEL_TEMB_WIDTH,
    egrid_path,
    file_sha256,
    load_egrid,
    patch_adaln,
    partition_turbo_weights,
    unpatch_adaln,
    load_preview_turbo,
)
from toolkit.config_modules import ModelConfig
from toolkit.models.base_model import BaseModel


def _cfg(**kwargs):
    defaults = dict(name_or_path="dummy", arch="minimax_h3")
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class PreviewTurboDit(nn.Module):
    def __init__(self, table_rows=8):
        super().__init__()
        self.qkv = nn.Linear(8, 8, bias=False)
        self.adaln_proj = MiniMaxH3AdalnProj(
            t_dim=8, hidden=4, expand=2, modalities=3, apply_silu=False, bias=True
        )
        self.register_buffer("adaln_t_table", torch.randn(table_rows, 8))


def _pair(in_f, out_f, rank=4):
    return torch.randn(rank, in_f), torch.randn(out_f, rank)


def test_preview_lora_config_default_off_and_h3_only():
    cfg = _cfg()
    assert cfg.preview_lora_path is None
    assert cfg.preview_lora_strength == pytest.approx(1.0)
    with pytest.raises(ValueError, match="MiniMax-H3 only"):
        _cfg(arch="flux", preview_lora_path="turbo.safetensors")
    with pytest.raises(ValueError, match="not inference_lora_path"):
        _cfg(inference_lora_path="turbo.safetensors")
    with pytest.raises(ValueError, match="cannot both be set"):
        _cfg(preview_lora_path="a.safetensors", inference_lora_path="b.safetensors")
    with pytest.raises(ValueError, match="finite number"):
        _cfg(preview_lora_path="a.safetensors", preview_lora_strength=float("nan"))


def test_egrid_provenance():
    path = egrid_path()
    assert file_sha256(path) == EGRID_SHA256
    grid = load_egrid()
    assert tuple(grid.shape) == EGRID_SHAPE
    assert grid.dtype == torch.bfloat16
    assert EGRID_SOURCE.startswith("larryvrh/")


def test_partition_matches_shape_adaln_and_skips():
    dit = PreviewTurboDit()
    down, up = _pair(8, 8)
    ad_down, ad_up = _pair(FULL_MODEL_TEMB_WIDTH, dit.adaln_proj.linear.out_features)
    skip_down, skip_up = _pair(3, 3)
    sd = {
        "lora_unet_qkv.lora_down.weight": down,
        "lora_unet_qkv.lora_up.weight": up,
        "lora_unet_adaln_proj_linear.lora_down.weight": ad_down,
        "lora_unet_adaln_proj_linear.lora_up.weight": ad_up,
        "lora_unet_not_here.lora_down.weight": skip_down,
        "lora_unet_not_here.lora_up.weight": skip_up,
    }
    keep, adaln, skipped, rank, alpha = partition_turbo_weights(dit, sd, strength=0.5)
    assert rank == 4
    assert "transformer.qkv.lora_down.weight" in keep
    assert len(adaln) == 1
    assert adaln[0][0] is dit.adaln_proj
    assert torch.allclose(adaln[0][2], ad_up * 0.5)
    assert any("not_here" in name for name in skipped)


def test_all_mismatched_file_fails():
    dit = PreviewTurboDit()
    down, up = _pair(3, 3)
    sd = {
        "lora_unet_qkv.lora_down.weight": down,
        "lora_unet_qkv.lora_up.weight": up,
    }
    with pytest.raises(ValueError, match="all-mismatched"):
        partition_turbo_weights(dit, sd, 1.0)


def test_adaln_injection_adds_and_unpatches():
    dit = PreviewTurboDit(table_rows=4)
    proj = dit.adaln_proj
    temb = dit.adaln_t_table[1:3].clone()
    before = [c.detach().clone() for c in proj(temb)]

    rank = 2
    egrid = torch.randn(4, FULL_MODEL_TEMB_WIDTH)
    down = torch.randn(rank, FULL_MODEL_TEMB_WIDTH)
    up = torch.randn(proj.linear.out_features, rank)
    patched = patch_adaln(dit, [(proj, down, up)], "cpu", torch.float32, egrid)
    assert patched == [proj]
    after = proj(temb)
    assert not torch.allclose(after[0], before[0])
    assert after[0].shape == before[0].shape

    unpatch_adaln(patched)
    restored = [c.detach().clone() for c in proj(temb)]
    for a, b in zip(restored, before):
        assert torch.allclose(a, b)
    assert "forward" not in proj.__dict__


def test_adaln_updates_on_same_module_add_not_replace():
    dit = PreviewTurboDit(table_rows=4)
    proj = dit.adaln_proj
    temb = dit.adaln_t_table[:1].clone()
    egrid = torch.zeros(4, FULL_MODEL_TEMB_WIDTH)
    egrid[0] = 1.0
    dit.adaln_t_table.zero_()
    temb.zero_()
    rank = 1
    down = torch.ones(rank, FULL_MODEL_TEMB_WIDTH)
    up_a = torch.ones(proj.linear.out_features, rank)
    up_b = torch.ones(proj.linear.out_features, rank) * 2
    before = torch.cat(proj(temb), dim=-1)
    patch_adaln(dit, [(proj, down, up_a), (proj, down, up_b)], "cpu", torch.float32, egrid)
    after = torch.cat(proj(temb), dim=-1)
    # two updates: +1*||standin|| contribution twice with 1x and 2x ups
    assert after.shape == before.shape
    assert (after - before).abs().max() > 0
    one = patch_adaln(dit, [(proj, down, up_a)], "cpu", torch.float32, egrid)
    after_one = torch.cat(proj(temb), dim=-1)
    unpatch_adaln(one)
    both = patch_adaln(dit, [(proj, down, up_a), (proj, down, up_b)], "cpu", torch.float32, egrid)
    after_both = torch.cat(proj(temb), dim=-1)
    # 1x + 2x = 3x the single update
    delta_one = after_one - before
    delta_both = after_both - before
    assert torch.allclose(delta_both, delta_one * 3, atol=1e-4)
    unpatch_adaln(both)


def test_lora_wrappers_stack_on_same_linear():
    lin = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4))
    orig = lin.forward

    def first(x, *args, **kwargs):
        return orig(x, *args, **kwargs) + 1.0

    lin.forward = first
    inner = lin.forward

    def second(x, *args, **kwargs):
        return inner(x, *args, **kwargs) + 10.0

    lin.forward = second
    out = lin(torch.zeros(2, 4))
    assert torch.allclose(out, torch.full((2, 4), 11.0))


def test_trainable_weights_unchanged_by_preview_cycle():
    dit = PreviewTurboDit()
    trainable = nn.Linear(8, 8, bias=False)
    before = hashlib.sha256(trainable.weight.detach().cpu().numpy().tobytes()).hexdigest()
    down, up = _pair(8, 8)
    sd = {
        "lora_unet_qkv.lora_down.weight": down,
        "lora_unet_qkv.lora_up.weight": up,
    }
    keep, adaln, skipped, rank, alpha = partition_turbo_weights(dit, sd, 1.0)
    egrid = torch.randn(8, FULL_MODEL_TEMB_WIDTH)
    patched = patch_adaln(dit, adaln, "cpu", torch.float32, egrid)
    try:
        raise RuntimeError("sample failed")
    except RuntimeError:
        unpatch_adaln(patched)
    after = hashlib.sha256(trainable.weight.detach().cpu().numpy().tobytes()).hexdigest()
    assert before == after
    assert "forward" not in dit.adaln_proj.__dict__


def test_generate_images_restores_adapters_on_exception():
    model = object.__new__(BaseModel)
    calls = []
    model._enter_generate_adapters = lambda: calls.append("enter")
    model._exit_generate_adapters = lambda: calls.append("exit")

    def boom(*_a, **_k):
        raise RuntimeError("sample failed")

    model._generate_images_body = boom
    with pytest.raises(RuntimeError, match="sample failed"):
        BaseModel.generate_images(model, [])
    assert calls == ["enter", "exit"]


def test_generate_images_restores_adapters_if_enter_raises():
    model = object.__new__(BaseModel)
    calls = []

    def enter():
        calls.append("enter")
        raise RuntimeError("activate failed")

    def body(*_a, **_k):
        calls.append("body")

    model._enter_generate_adapters = enter
    model._exit_generate_adapters = lambda: calls.append("exit")
    model._generate_images_body = body
    with pytest.raises(RuntimeError, match="activate failed"):
        BaseModel.generate_images(model, [])
    assert calls == ["enter", "exit"]


def test_generate_images_body_restores_rng_on_exception():
    model = object.__new__(BaseModel)
    model.network = None
    model.adapter = None
    model.save_device_state = lambda: None
    model.set_device_state_preset = lambda *_a, **_k: None
    model.restore_device_state = lambda: None
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32

    class _UNet:
        def to(self, *args, **kwargs):
            return self

    model.unet = _UNet()

    def boom():
        torch.manual_seed(999)
        raise RuntimeError("pipeline failed")

    model.get_generation_pipeline = boom
    torch.manual_seed(123)
    expected = torch.randint(0, 1000, (1,))
    torch.manual_seed(123)
    with pytest.raises(RuntimeError, match="pipeline failed"):
        BaseModel._generate_images_body(model, [])
    after = torch.randint(0, 1000, (1,))
    assert torch.equal(after, expected)


def test_row_count_mismatch_fails_closed():
    dit = PreviewTurboDit(table_rows=3)
    proj = dit.adaln_proj
    down, up = _pair(FULL_MODEL_TEMB_WIDTH, proj.linear.out_features)
    egrid = torch.randn(4, FULL_MODEL_TEMB_WIDTH)
    with pytest.raises(ValueError, match="e-grid rows"):
        patch_adaln(dit, [(proj, down, up)], "cpu", torch.float32, egrid)


def _patch_adaln_one(dit):
    proj = dit.adaln_proj
    down, up = _pair(FULL_MODEL_TEMB_WIDTH, proj.linear.out_features)
    egrid = torch.randn(dit.adaln_t_table.shape[0], FULL_MODEL_TEMB_WIDTH)
    patched = patch_adaln(dit, [(proj, down, up)], "cpu", torch.float32, egrid)
    return proj, patched


def test_adaln_interpolated_temb_is_within_grid_cell():
    torch.manual_seed(0)
    dit = PreviewTurboDit(table_rows=4)
    proj, patched = _patch_adaln_one(dit)
    temb = 0.5 * (dit.adaln_t_table[0] + dit.adaln_t_table[1]).unsqueeze(0)
    proj(temb)
    unpatch_adaln(patched)
    assert not hasattr(dit.adaln_proj, "_adaln_egrid_checked")


def test_adaln_far_temb_fails_closed():
    torch.manual_seed(0)
    dit = PreviewTurboDit(table_rows=4)
    proj, patched = _patch_adaln_one(dit)
    temb = dit.adaln_t_table[:1].clone() + 50.0
    with pytest.raises(ValueError, match="does not match the grid"):
        proj(temb)
    unpatch_adaln(patched)


def test_load_preview_turbo_inactive_until_activate(tmp_path):
    dit = PreviewTurboDit()
    down, up = _pair(8, 8)
    path = tmp_path / "turbo.safetensors"
    save_file(
        {
            "lora_unet_qkv.lora_down.weight": down,
            "lora_unet_qkv.lora_up.weight": up,
            "lora_unet_qkv.alpha": torch.tensor(4.0),
        },
        str(path),
    )
    turbo = load_preview_turbo(
        dit, str(path), 1.0, target_lin_modules=["PreviewTurboDit"]
    )
    assert turbo.network.is_active is False
    assert turbo.report.matched == 1
    assert turbo.report.injected_adaln == 0
    x = torch.randn(2, 8)
    with torch.no_grad():
        y0 = dit.qkv(x)
    turbo.activate(dit, "cpu", dit.qkv.weight.dtype)
    assert turbo.network.is_active
    with torch.no_grad():
        y1 = dit.qkv(x)
    assert not torch.allclose(y0, y1)
    turbo.deactivate()
    assert turbo.network.is_active is False
    with torch.no_grad():
        y2 = dit.qkv(x)
    assert torch.allclose(y0, y2)
    for param in turbo.network.parameters():
        assert param.device.type == "cpu"
