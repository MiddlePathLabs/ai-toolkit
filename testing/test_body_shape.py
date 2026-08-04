"""Focused tests for the Phase 3 body-shape (HybrIK SMPL betas) anchor.

Covers: BodyShapeConfig defaults + validation, DatasetConfig overrides, the
L1 + cosine loss math, the lazy mixin read/cleanup/reject, cache stamping with
a fake encoder, batch aggregation, and preflight_body_shape. No HybrIK weights
or GPU required -- the perceptor is faked.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, BodyShapeConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.body_shape import cache_body_shape, BETA_DIM, CACHE_VERSION_KEY
from toolkit.body_shape_loss import compute_body_shape_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = BodyShapeConfig()
    assert c.loss_weight == 0.0
    assert c.loss_min_t == 0.4
    assert c.loss_max_t == 0.8
    assert c.loss_min_cos == 0.2


def test_config_has_no_enabled_field():
    assert not hasattr(BodyShapeConfig(), 'enabled')


def test_timestep_validation():
    with pytest.raises(ValueError, match="loss_min_t"):
        BodyShapeConfig(loss_min_t=-0.01)
    with pytest.raises(ValueError, match="loss_max_t"):
        BodyShapeConfig(loss_max_t=1.01)
    with pytest.raises(ValueError, match="loss_min_t"):
        BodyShapeConfig(loss_min_t=0.7, loss_max_t=0.3)


def test_dataset_overrides_default_none():
    d = DatasetConfig(dataset_path=".", resolution=64)
    assert d.body_shape_loss_weight is None
    assert d.body_shape_loss_min_t is None
    assert d.body_shape_loss_max_t is None
    assert d.body_shape_loss_min_cos is None


# ---------------------------------------------------------------------------
# Loss math
# ---------------------------------------------------------------------------

def test_loss_zero_when_identical():
    betas = torch.randn(2, BETA_DIM)
    l1, cos = compute_body_shape_loss(betas, betas)
    assert torch.allclose(l1, torch.zeros(2), atol=1e-6)
    assert torch.allclose(cos, torch.ones(2), atol=1e-5)


def test_loss_positive_when_different():
    l1, _ = compute_body_shape_loss(torch.zeros(1, BETA_DIM), torch.ones(1, BETA_DIM))
    assert l1.item() == 1.0


def test_loss_gradient_flows():
    gen = torch.randn(1, BETA_DIM, requires_grad=True)
    ref = torch.randn(1, BETA_DIM)
    l1, _ = compute_body_shape_loss(gen, ref)
    l1.backward()
    assert gen.grad is not None


# ---------------------------------------------------------------------------
# Cache mixin + cache_body_shape (fake encoder)
# ---------------------------------------------------------------------------

def _make_image(path):
    from PIL import Image
    Image.new("RGB", (32, 32), (100, 100, 100)).save(str(path))
    return str(path)


def _make_file_item(image_path, dataset_config):
    return FileItemDTO(path=image_path, dataset_config=dataset_config)


class _FakeEncoder:
    def __init__(self, betas):
        self._betas = betas

    def encode(self, pil_image, person_bbox=None):
        return self._betas.clone()


def test_cache_stamps_and_writes(tmp_path):
    from safetensors.torch import load_file
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    fake = _FakeEncoder(torch.randn(BETA_DIM))
    cache_body_shape([fi], SimpleNamespace(), encoder=fake)
    assert fi.is_body_shape_cached is True
    assert fi.body_shape_gt is None  # lazy
    assert os.path.exists(fi._body_shape_cache_path)
    data = load_file(fi._body_shape_cache_path)
    assert "body_shape_embedding" in data and CACHE_VERSION_KEY in data
    assert torch.equal(fi.get_body_shape_gt(), fake._betas)


def test_cache_hit_skips_recompute(tmp_path):
    from safetensors.torch import save_file
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    cache_dir = os.path.join(os.path.dirname(image_path), "_body_shape_cache")
    os.makedirs(cache_dir, exist_ok=True)
    expected_path = os.path.join(
        cache_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_bodyshape.safetensors"
    )
    first = torch.randn(BETA_DIM)
    save_file({"body_shape_embedding": first, CACHE_VERSION_KEY: torch.ones(1)}, expected_path)
    called = {"n": 0}

    def _encode(pil_image, person_bbox=None):
        called["n"] += 1
        return torch.zeros(BETA_DIM)

    cache_body_shape([fi], SimpleNamespace(), encoder=SimpleNamespace(encode=_encode))
    assert called["n"] == 0
    assert torch.equal(fi.get_body_shape_gt(), first)


def test_non_finite_cache_rejected(tmp_path):
    from safetensors.torch import save_file
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    nan_path = str(tmp_path / "nan.safetensors")
    save_file(
        {"body_shape_embedding": torch.full((BETA_DIM,), float("nan")), CACHE_VERSION_KEY: torch.ones(1)},
        nan_path,
    )
    fi._body_shape_cache_path = nan_path
    fi._body_shape_cache_key = "body_shape_embedding"
    fi.is_body_shape_cached = True
    assert fi.get_body_shape_gt() is None


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def test_batch_aggregates_body_shape(tmp_path):
    from torchvision import transforms
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path), resolution=32,
        body_shape_loss_weight=0.05, body_shape_loss_min_t=0.1, body_shape_loss_max_t=0.5,
    )
    fi = _make_file_item(image_path, cfg)
    fake = _FakeEncoder(torch.randn(BETA_DIM))
    cache_body_shape([fi], SimpleNamespace(), encoder=fake)
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.body_shape_gt.shape == (1, BETA_DIM)
    assert batch.body_shape_loss_weight_list == [0.05]


# ---------------------------------------------------------------------------
# preflight_body_shape
# ---------------------------------------------------------------------------

def test_preflight_inert_when_absent():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_shape
    assert preflight_body_shape(None, [], None, False) is None


def test_preflight_dataset_only_activation():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_shape
    dc = SimpleNamespace(body_shape_loss_weight=0.1,
                         body_shape_loss_min_t=None, body_shape_loss_max_t=None, body_shape_loss_min_cos=None)
    cfg = preflight_body_shape(None, [dc], 'krea2', False)
    assert cfg is not None and cfg.loss_weight == 0.0


def test_preflight_krea2_low_vram_rejected():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_body_shape
    cfg = BodyShapeConfig(loss_weight=0.05)
    with pytest.raises(ValueError, match="low_vram"):
        preflight_body_shape(cfg, [], 'krea2', True)
