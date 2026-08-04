"""Focused tests for the Phase 3 vae-anchor (cross-VAE Flux 2 perceptual) anchor.

Covers: VAEAnchorConfig defaults + validation, DatasetConfig overrides, the
compute_loss cosine math, the lazy mixin read/cleanup/reject, batch feature
stacking, and preflight_vae_anchor. No Flux 2 VAE weights or GPU required --
the encoder is faked; compute_loss is pure tensor math.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, VAEAnchorConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.vae_anchor import VAEAnchorEncoder, FEATURE_LEVELS, CACHE_VERSION_KEY


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = VAEAnchorConfig()
    assert c.loss_weight == 0.0
    assert c.loss_min_t == 0.0
    assert c.loss_max_t == 0.5
    assert c.vae_model_path == ''


def test_config_has_no_enabled_field():
    assert not hasattr(VAEAnchorConfig(), 'enabled')


def test_timestep_validation():
    with pytest.raises(ValueError, match="loss_min_t"):
        VAEAnchorConfig(loss_min_t=-0.01)
    with pytest.raises(ValueError, match="loss_max_t"):
        VAEAnchorConfig(loss_max_t=1.01)
    with pytest.raises(ValueError, match="loss_min_t"):
        VAEAnchorConfig(loss_min_t=0.7, loss_max_t=0.3)


def test_dataset_overrides_default_none():
    d = DatasetConfig(dataset_path=".", resolution=64)
    assert d.vae_anchor_loss_weight is None
    assert d.vae_anchor_loss_min_t is None
    assert d.vae_anchor_loss_max_t is None


# ---------------------------------------------------------------------------
# compute_loss (pure tensor math)
# ---------------------------------------------------------------------------

def test_compute_loss_zero_when_identical():
    feats = {lv: torch.randn(2, 16, 8, 8) for lv in FEATURE_LEVELS}
    loss, per_level = VAEAnchorEncoder.compute_loss(feats, feats)
    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.zeros(2), atol=1e-5)
    assert set(per_level.keys()) == set(FEATURE_LEVELS)


def test_compute_loss_positive_when_different():
    pred = {lv: torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    ref = {lv: -torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    loss, _ = VAEAnchorEncoder.compute_loss(pred, ref)
    assert (loss > 0).all()


def test_compute_loss_gradient_flows():
    pred = {lv: torch.randn(1, 16, 8, 8, requires_grad=True) for lv in FEATURE_LEVELS}
    ref = {lv: torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    loss, _ = VAEAnchorEncoder.compute_loss(pred, ref)
    loss.backward()
    for lv in FEATURE_LEVELS:
        assert pred[lv].grad is not None


def test_compute_loss_handles_size_mismatch():
    pred = {lv: torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    ref = {lv: torch.randn(1, 16, 12, 12) for lv in FEATURE_LEVELS}  # different size
    loss, _ = VAEAnchorEncoder.compute_loss(pred, ref)
    assert loss.shape == (1,)  # no crash; ref interpolated


# ---------------------------------------------------------------------------
# Cache mixin read/cleanup/reject
# ---------------------------------------------------------------------------

def _make_image(path):
    from PIL import Image
    Image.new("RGB", (32, 32), (100, 100, 100)).save(str(path))
    return str(path)


def _make_file_item(image_path, dataset_config):
    return FileItemDTO(path=image_path, dataset_config=dataset_config)


def _stamp_vae_cache(file_item, cache_path, feats=None, *, incomplete=False):
    from safetensors.torch import save_file
    if feats is None:
        feats = {lv: torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    data = {f"vae_anchor_{lv}": f for lv, f in feats.items()}
    if not incomplete:
        data[CACHE_VERSION_KEY] = torch.ones(1)
    elif incomplete == "no_version":
        pass  # omit version key
    else:  # incomplete == "missing_level"
        del data["vae_anchor_mid"]
        data[CACHE_VERSION_KEY] = torch.ones(1)
    save_file(data, str(cache_path))
    file_item._vae_cache_path = str(cache_path)
    file_item.is_vae_anchor_cached = True


def test_lazy_read_delivers_features(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    feats = {lv: torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    _stamp_vae_cache(fi, tmp_path / "vae.safetensors", feats=feats)
    out = fi.get_vae_anchor_features()
    assert out is not None and set(out.keys()) == set(FEATURE_LEVELS)
    assert torch.equal(out["level_0"], feats["level_0"])


def test_cleanup_releases_keeps_metadata(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    _stamp_vae_cache(fi, tmp_path / "vae.safetensors")
    fi.get_vae_anchor_features()
    assert fi.vae_anchor_features is not None
    fi.cleanup_vae_anchor()
    assert fi.vae_anchor_features is None
    assert fi.is_vae_anchor_cached is True


def test_incomplete_cache_missing_level_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    _stamp_vae_cache(fi, tmp_path / "vae.safetensors", incomplete="missing_level")
    assert fi.get_vae_anchor_features() is None


def test_incomplete_cache_no_version_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    _stamp_vae_cache(fi, tmp_path / "vae.safetensors", incomplete="no_version")
    assert fi.get_vae_anchor_features() is None


# ---------------------------------------------------------------------------
# Batch feature stacking
# ---------------------------------------------------------------------------

def test_batch_stacks_vae_features(tmp_path):
    from torchvision import transforms
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path), resolution=32,
        vae_anchor_loss_weight=0.05, vae_anchor_loss_min_t=0.0, vae_anchor_loss_max_t=0.4,
    )
    fi = _make_file_item(image_path, cfg)
    feats = {lv: torch.randn(1, 16, 8, 8) for lv in FEATURE_LEVELS}
    _stamp_vae_cache(fi, tmp_path / "vae.safetensors", feats=feats)
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.vae_anchor_features is not None
    assert set(batch.vae_anchor_features.keys()) == set(FEATURE_LEVELS)
    assert batch.vae_anchor_features["level_0"].shape == (1, 16, 8, 8)
    assert batch.vae_anchor_loss_weight_list == [0.05]


# ---------------------------------------------------------------------------
# preflight_vae_anchor
# ---------------------------------------------------------------------------

def test_preflight_inert_when_absent():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_vae_anchor
    assert preflight_vae_anchor(None, [], None, False) is None


def test_preflight_dataset_only_activation():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_vae_anchor
    dc = SimpleNamespace(vae_anchor_loss_weight=0.1,
                         vae_anchor_loss_min_t=None, vae_anchor_loss_max_t=None)
    cfg = preflight_vae_anchor(None, [dc], 'krea2', False)
    assert cfg is not None and cfg.loss_weight == 0.0


def test_preflight_krea2_low_vram_rejected():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_vae_anchor
    cfg = VAEAnchorConfig(loss_weight=0.05)
    with pytest.raises(ValueError, match="low_vram"):
        preflight_vae_anchor(cfg, [], 'krea2', True)
