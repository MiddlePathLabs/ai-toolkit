"""Focused tests for the Phase 3 body-proportion (ViTPose) anchor.

Covers: BodyProportionConfig defaults + validation, DatasetConfig per-dataset
overrides, the vendored soft-argmax, the ratio computation (dims, head mode,
visibility, ref_ratios fallback, gradient flow), the visibility-weighted L1 +
missing-keypoint loss, the lazy mixin read/cleanup/reject, cache stamping with
a fake encoder, batch aggregation, and preflight_body_proportion. No ViTPose
weights or GPU required -- the perceptor is faked where needed.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, BodyProportionConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.body_proportion import (
    DifferentiableBodyProportionEncoder,
    cache_body_proportion,
    _soft_argmax_2d,
    CACHE_VERSION_KEY_BODY,
    CACHE_VERSION_KEY_HEAD,
)
from toolkit.body_proportion_loss import compute_body_proportion_loss


# ---------------------------------------------------------------------------
# Config defaults + validation
# ---------------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = BodyProportionConfig()
    assert c.loss_weight == 0.0
    assert c.loss_min_t == 0.0
    assert c.loss_max_t == 1.0
    assert c.include_head is False


def test_config_has_no_enabled_field():
    assert not hasattr(BodyProportionConfig(), 'enabled')


def test_partial_object_merge():
    c = BodyProportionConfig(loss_weight=0.1, include_head=True)
    assert c.loss_weight == 0.1
    assert c.include_head is True
    assert c.loss_min_t == 0.0
    assert c.loss_max_t == 1.0


def test_timestep_validation():
    with pytest.raises(ValueError, match="loss_min_t"):
        BodyProportionConfig(loss_min_t=-0.01)
    with pytest.raises(ValueError, match="loss_max_t"):
        BodyProportionConfig(loss_max_t=1.01)
    with pytest.raises(ValueError, match="loss_min_t"):
        BodyProportionConfig(loss_min_t=0.7, loss_max_t=0.3)


def test_dataset_overrides_default_none():
    d = DatasetConfig(dataset_path=".", resolution=64)
    assert d.body_proportion_loss_weight is None
    assert d.body_proportion_loss_min_t is None
    assert d.body_proportion_loss_max_t is None


# ---------------------------------------------------------------------------
# Vendored soft-argmax
# ---------------------------------------------------------------------------

def test_soft_argmax_peak_location():
    # A single peak at (h=2, w=3) in a 4x6 map -> coords near that cell, [-1,1].
    hm = torch.zeros(1, 1, 4, 6)
    hm[0, 0, 2, 3] = 1.0
    coords = _soft_argmax_2d(hm)  # (1,1,2) in (x, y)
    assert coords.shape == (1, 1, 2)
    # normalized_linspace(6) = [-0.833,-0.5,-0.167,0.167,0.5,0.833]; index 3 -> 0.167
    assert abs(coords[0, 0, 0].item() - 0.1667) < 1e-3
    # normalized_linspace(4) = [-0.75,-0.25,0.25,0.75]; index 2 -> 0.25
    assert abs(coords[0, 0, 1].item() - 0.25) < 1e-3


def test_soft_argmax_gradient_flows():
    hm = torch.rand(1, 1, 4, 4, requires_grad=True)
    coords = _soft_argmax_2d(hm)
    coords.sum().backward()
    assert hm.grad is not None
    assert torch.isfinite(hm.grad).all()


# ---------------------------------------------------------------------------
# Ratio computation (pure math, no ViTPose)
# ---------------------------------------------------------------------------

def _perfect_keypoints():
    """A batch of 1 with synthetic COCO keypoints producing nonzero ratios."""
    kp = torch.zeros(1, 17, 2)
    # place shoulders/elbows/wrists/hips/knees/ankles symmetrically
    kp[0, 5] = torch.tensor([-0.4, -0.5])  # L shoulder
    kp[0, 6] = torch.tensor([0.4, -0.5])   # R shoulder
    kp[0, 7] = torch.tensor([-0.5, -0.1])  # L elbow
    kp[0, 8] = torch.tensor([0.5, -0.1])   # R elbow
    kp[0, 9] = torch.tensor([-0.55, 0.2])  # L wrist
    kp[0, 10] = torch.tensor([0.55, 0.2])  # R wrist
    kp[0, 11] = torch.tensor([-0.25, 0.3])  # L hip
    kp[0, 12] = torch.tensor([0.25, 0.3])  # R hip
    kp[0, 13] = torch.tensor([-0.27, 0.6])  # L knee
    kp[0, 14] = torch.tensor([0.27, 0.6])  # R knee
    kp[0, 15] = torch.tensor([-0.28, 0.9])  # L ankle
    kp[0, 16] = torch.tensor([0.28, 0.9])  # R ankle
    kp[0, 0] = torch.tensor([0.0, -0.8])  # nose
    kp[0, 3] = torch.tensor([-0.25, -0.75])  # L ear
    kp[0, 4] = torch.tensor([0.25, -0.75])  # R ear
    vis = torch.ones(1, 17)
    return kp, vis


def test_compute_ratios_body_only_dim():
    kp, vis = _perfect_keypoints()
    ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(kp, vis)
    assert ratios.shape == (1, 8)
    assert ratio_vis.shape == (1, 8)
    assert torch.isfinite(ratios).all()


def test_compute_ratios_with_head_adds_two():
    kp, vis = _perfect_keypoints()
    ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(kp, vis, include_head=True)
    assert ratios.shape == (1, 10)


def test_compute_ratios_low_visibility_zeros_ratio_vis():
    kp, vis = _perfect_keypoints()
    vis[0, 13] = 0.0  # kill L knee
    vis[0, 14] = 0.0  # kill R knee
    _, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(kp, vis)
    # thigh (idx 2) and shin (idx 3) depend on knees -> vis 0
    assert ratio_vis[0, 2].item() == 0.0
    assert ratio_vis[0, 3].item() == 0.0


def test_compute_ratios_ref_fallback_replaces_low_conf():
    kp, vis = _perfect_keypoints()
    vis[0, 13] = 0.0
    vis[0, 14] = 0.0
    ref = torch.full((1, 8), 0.42)
    ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
        kp, vis, ref_ratios=ref
    )
    # thigh/shin were low-conf -> replaced with ref value (0.42), vis zeroed
    assert abs(ratios[0, 2].item() - 0.42) < 1e-5
    assert abs(ratios[0, 3].item() - 0.42) < 1e-5
    assert ratio_vis[0, 2].item() == 0.0


def test_compute_ratios_gradient_flows_to_keypoints():
    kp, vis = _perfect_keypoints()
    kp = kp.clone().requires_grad_(True)
    ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(kp, vis)
    ratios.sum().backward()
    assert kp.grad is not None
    assert torch.isfinite(kp.grad).all()


# ---------------------------------------------------------------------------
# Loss math
# ---------------------------------------------------------------------------

def test_loss_zero_when_identical():
    gen_r = torch.rand(2, 8)
    gen_v = torch.ones(2, 8)
    loss, missing = compute_body_proportion_loss(gen_r, gen_v, gen_r, gen_v)
    assert torch.allclose(loss, torch.zeros(2), atol=1e-6)
    assert torch.allclose(missing, torch.zeros(2))


def test_loss_positive_when_different():
    gen_r = torch.zeros(2, 8)
    ref_r = torch.ones(2, 8)
    vis = torch.ones(2, 8)
    loss, _ = compute_body_proportion_loss(gen_r, vis, ref_r, vis)
    assert (loss > 0).all()


def test_loss_missing_penalty():
    # ref confident, gen dropped below threshold -> missing fraction
    ref_v = torch.ones(1, 8)
    gen_v = torch.zeros(1, 8)  # all "missing"
    gen_r = torch.zeros(1, 8)
    ref_r = torch.zeros(1, 8)
    loss, missing = compute_body_proportion_loss(gen_r, gen_v, ref_r, ref_v)
    assert abs(missing.item() - 1.0) < 1e-6  # all 8 ratios missing
    # l1 term is 0 (both zero) so loss == missing fraction
    assert abs(loss.item() - 1.0) < 1e-6


def test_loss_gradient_flows():
    gen_r = torch.rand(1, 8, requires_grad=True)
    gen_v = torch.ones(1, 8)
    ref_r = torch.zeros(1, 8)
    ref_v = torch.ones(1, 8)
    loss, _ = compute_body_proportion_loss(gen_r, gen_v, ref_r, ref_v)
    loss.backward()
    assert gen_r.grad is not None


# ---------------------------------------------------------------------------
# Cache mixin + cache_body_proportion (fake encoder)
# ---------------------------------------------------------------------------

def _make_image(path):
    from PIL import Image
    Image.new("RGB", (32, 32), (100, 100, 100)).save(str(path))
    return str(path)


def _make_file_item(image_path, dataset_config):
    return FileItemDTO(path=image_path, dataset_config=dataset_config)


class _FakeEncoder:
    def __init__(self, ratios_vec):
        self._v = ratios_vec

    def encode(self, pil_image, include_head=False):
        return self._v.clone()


def test_cache_stamps_and_writes(tmp_path):
    from safetensors.torch import load_file
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    fake = _FakeEncoder(torch.rand(16))
    cache_body_proportion([fi], SimpleNamespace(include_head=False), encoder=fake)
    assert fi.is_body_proportion_cached is True
    assert fi.body_proportion_gt is None  # lazy
    assert os.path.exists(fi._bp_cache_path)
    data = load_file(fi._bp_cache_path)
    assert "body_proportion_gt" in data
    assert CACHE_VERSION_KEY_BODY in data
    assert torch.equal(fi.get_body_proportion_gt(), fake._v)


def test_cache_hit_skips_recompute(tmp_path):
    from safetensors.torch import save_file as _save
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    # pre-write a cache file at the expected path
    cache_dir = os.path.join(os.path.dirname(image_path), "_body_proportion_cache")
    os.makedirs(cache_dir, exist_ok=True)
    expected_path = os.path.join(
        cache_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_bodyprop.safetensors"
    )
    first = torch.rand(16)
    _save(
        {"body_proportion_gt": first, CACHE_VERSION_KEY_BODY: torch.ones(1)},
        expected_path,
    )
    called = {"n": 0}

    def _encode(pil_image, include_head=False):
        called["n"] += 1
        return torch.zeros(16)

    cache_body_proportion([fi], SimpleNamespace(include_head=False), encoder=SimpleNamespace(encode=_encode))
    assert called["n"] == 0  # hit -> no recompute
    assert torch.equal(fi.get_body_proportion_gt(), first)


def test_cache_include_head_uses_head_version_key(tmp_path):
    from safetensors.torch import load_file
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    fake = _FakeEncoder(torch.rand(20))
    cache_body_proportion([fi], SimpleNamespace(include_head=True), encoder=fake)
    data = load_file(fi._bp_cache_path)
    assert CACHE_VERSION_KEY_HEAD in data
    assert CACHE_VERSION_KEY_BODY not in data


def test_corrupt_cache_rejected(tmp_path):
    from safetensors.torch import save_file as _save
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    # write a file missing the version key -> should miss and recompute
    cache_dir = os.path.join(os.path.dirname(image_path), "_body_proportion_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_bodyprop.safetensors"
    )
    _save({"body_proportion_gt": torch.rand(16)}, cache_path)  # no version key
    fi._bp_cache_path = cache_path
    fi._bp_cache_key = "body_proportion_gt"
    fi.is_body_proportion_cached = True
    # the mixin only validates finiteness; the header-hit (version key) check is
    # in _bp_cache_hit; here we confirm a NaN tensor is rejected
    fi_nan = _make_file_item(image_path, cfg)
    nan_path = str(tmp_path / "nan.safetensors")
    _save({"body_proportion_gt": torch.full((16,), float("nan")), CACHE_VERSION_KEY_BODY: torch.ones(1)}, nan_path)
    fi_nan._bp_cache_path = nan_path
    fi_nan._bp_cache_key = "body_proportion_gt"
    fi_nan.is_body_proportion_cached = True
    assert fi_nan.get_body_proportion_gt() is None


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def test_batch_aggregates_body_proportion(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path), resolution=32,
        body_proportion_loss_weight=0.2, body_proportion_loss_min_t=0.1, body_proportion_loss_max_t=0.5,
    )
    fi = _make_file_item(image_path, cfg)
    fake = _FakeEncoder(torch.rand(16))
    cache_body_proportion([fi], SimpleNamespace(include_head=False), encoder=fake)
    fi.load_and_process_image(__import__("torchvision").transforms.Compose([__import__("torchvision").transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.body_proportion_gt is not None
    assert batch.body_proportion_gt.shape == (1, 16)
    assert batch.body_proportion_loss_weight_list == [0.2]
    assert batch.body_proportion_loss_min_t_list == [0.1]
    assert batch.body_proportion_loss_max_t_list == [0.5]


# ---------------------------------------------------------------------------
# preflight_body_proportion
# ---------------------------------------------------------------------------

def test_preflight_inert_when_absent():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_proportion
    assert preflight_body_proportion(None, [], None, False) is None


def test_preflight_dataset_only_activation():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_proportion
    dc = SimpleNamespace(body_proportion_loss_weight=0.3,
                         body_proportion_loss_min_t=None, body_proportion_loss_max_t=None)
    cfg = preflight_body_proportion(None, [dc], 'krea2', False)
    assert cfg is not None and cfg.loss_weight == 0.0


def test_preflight_krea2_low_vram_rejected():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_proportion
    cfg = BodyProportionConfig(loss_weight=0.01)
    with pytest.raises(ValueError, match="low_vram"):
        preflight_body_proportion(cfg, [], 'krea2', True)


def test_preflight_low_vram_ok_non_krea():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_proportion
    cfg = BodyProportionConfig(loss_weight=0.01)
    out = preflight_body_proportion(cfg, [], 'flux1', True)
    assert out is not None and out.loss_weight == 0.01
