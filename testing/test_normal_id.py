"""Focused tests for the Phase 3 normal-anchor (Sapiens surface normals).

Covers: NormalIDConfig defaults + validation, DatasetConfig per-dataset
overrides, the lazy NormalCachingFileItemDTOMixin read/cleanup/reject paths,
cache_normal_gt stamping with a fake encoder, compute_normal_loss math +
differentiability, DataLoaderBatchDTO normal aggregation, and the
preflight_normal_id dataset-only activation. No Sapiens weights or GPU
required -- the perceptor is faked where needed.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, NormalIDConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.normal_id import cache_normal_gt, NORMAL_SIZE, NORMAL_CACHE_VERSION_KEY
from toolkit.normal_id_loss import compute_normal_loss


# ---------------------------------------------------------------------------
# NormalIDConfig defaults + validation
# ---------------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = NormalIDConfig()
    assert c.loss_weight == 0.0
    assert c.loss_min_t == 0.4
    assert c.loss_max_t == 0.8
    assert c.model_id == 'facebook/sapiens-normal-0.3b'
    assert c.grad_checkpoint is True
    assert c.preview_every == 100
    assert c.preview_only is False
    assert c.preview_max_keep == 500


def test_config_has_no_enabled_field():
    c = NormalIDConfig()
    assert not hasattr(c, 'enabled')


def test_partial_object_merge_preserves_supplied_and_fills_missing():
    c = NormalIDConfig(loss_weight=0.2, loss_min_t=0.1)
    assert c.loss_weight == 0.2
    assert c.loss_min_t == 0.1
    # unspecified fields fall back to safe defaults
    assert c.loss_max_t == 0.8
    assert c.model_id == 'facebook/sapiens-normal-0.3b'
    assert c.grad_checkpoint is True
    assert c.preview_every == 100


def test_negative_loss_min_t_raises():
    with pytest.raises(ValueError, match="loss_min_t"):
        NormalIDConfig(loss_min_t=-0.01)


def test_loss_max_t_above_one_raises():
    with pytest.raises(ValueError, match="loss_max_t"):
        NormalIDConfig(loss_max_t=1.01)


def test_loss_min_t_greater_than_loss_max_t_raises():
    with pytest.raises(ValueError, match="loss_min_t"):
        NormalIDConfig(loss_min_t=0.7, loss_max_t=0.3)


def test_valid_timestep_bounds_accepted():
    c = NormalIDConfig(loss_min_t=0.0, loss_max_t=1.0)
    assert (c.loss_min_t, c.loss_max_t) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# DatasetConfig per-dataset overrides default to None (inherit)
# ---------------------------------------------------------------------------

def test_dataset_normal_overrides_default_none():
    d = DatasetConfig(dataset_path=".", resolution=64)
    assert d.normal_loss_weight is None
    assert d.normal_loss_min_t is None
    assert d.normal_loss_max_t is None


def test_dataset_normal_overrides_accepted():
    d = DatasetConfig(
        dataset_path=".", resolution=64,
        normal_loss_weight=0.5, normal_loss_min_t=0.2, normal_loss_max_t=0.6,
    )
    assert d.normal_loss_weight == 0.5
    assert d.normal_loss_min_t == 0.2
    assert d.normal_loss_max_t == 0.6


# ---------------------------------------------------------------------------
# NormalCachingFileItemDTOMixin lazy read / cleanup / reject
# ---------------------------------------------------------------------------

def _make_image(path):
    Image.new("RGB", (32, 32), (100, 100, 100)).save(str(path))
    return str(path)


def _make_file_item(image_path, dataset_config):
    return FileItemDTO(path=image_path, dataset_config=dataset_config)


def _stamp_normal_cache(file_item, cache_path, tensor=None, *, garbage=False,
                        nan_data=False, missing_version=False):
    expected_key = "normal_gt"
    if garbage:
        with open(cache_path, "wb") as f:
            f.write(b"not a safetensors file")
        written = None
    else:
        written = tensor if tensor is not None else torch.randn(3, NORMAL_SIZE, NORMAL_SIZE)
        data = {expected_key: written.contiguous()}
        if not missing_version:
            data[NORMAL_CACHE_VERSION_KEY] = torch.ones(1)
        save_file(data, str(cache_path))
    file_item._normal_cache_path = str(cache_path)
    file_item._normal_cache_key = expected_key
    file_item.is_normal_cached = True
    return str(cache_path), expected_key, written


def test_lazy_read_delivers_normal_gt(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    expected = torch.randn(3, NORMAL_SIZE, NORMAL_SIZE)
    _stamp_normal_cache(fi, tmp_path / "normals.safetensors", tensor=expected)

    assert fi.normal_gt is None  # not yet materialized
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    assert torch.equal(fi.normal_gt, expected)


def test_cleanup_releases_tensor_keeps_metadata(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    expected = torch.randn(3, NORMAL_SIZE, NORMAL_SIZE)
    _stamp_normal_cache(fi, tmp_path / "normals.safetensors", tensor=expected)
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    assert fi.normal_gt is not None

    fi.cleanup_normal()
    assert fi.normal_gt is None  # released
    # metadata intact -> re-read works
    assert fi.is_normal_cached is True
    assert fi._normal_cache_path is not None
    fi2 = _make_file_item(image_path, cfg)
    fi2._normal_cache_path = fi._normal_cache_path
    fi2._normal_cache_key = fi._normal_cache_key
    fi2.is_normal_cached = True
    assert torch.equal(fi2.get_normal_gt(), expected)


def test_corrupt_cache_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    _stamp_normal_cache(fi, tmp_path / "normals.safetensors", garbage=True)
    assert fi.get_normal_gt() is None


def test_non_finite_cache_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    nan = torch.full((3, NORMAL_SIZE, NORMAL_SIZE), float("nan"))
    _stamp_normal_cache(fi, tmp_path / "normals.safetensors", tensor=nan)
    assert fi.get_normal_gt() is None


# ---------------------------------------------------------------------------
# cache_normal_gt stamping with a fake encoder (no Sapiens download)
# ---------------------------------------------------------------------------

class _FakeNormalEncoder:
    """Stand-in for DifferentiableNormalEncoder that returns a fixed normal map."""

    def __init__(self, normal_map):
        self._normal_map = normal_map

    def encode(self, pil_image):
        return self._normal_map.clone()


def test_cache_normal_gt_stamps_and_writes(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    fake_normal = torch.randn(3, NORMAL_SIZE, NORMAL_SIZE)
    encoder = _FakeNormalEncoder(fake_normal)
    perceptor_cfg = SimpleNamespace()

    cache_normal_gt([fi], perceptor_cfg, encoder=encoder)

    assert fi.is_normal_cached is True
    assert fi._normal_cache_key == "normal_gt"
    assert fi._normal_cache_path is not None
    assert fi.normal_gt is None  # resident tensor NOT held (lazy)
    assert os.path.exists(fi._normal_cache_path)
    # the written file is readable back through the mixin
    assert torch.equal(fi.get_normal_gt(), fake_normal.half())


def test_cache_normal_gt_hit_skips_recompute(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    first_normal = torch.randn(3, NORMAL_SIZE, NORMAL_SIZE)
    # pre-populate the cache file at the expected path
    cache_dir = os.path.join(os.path.dirname(image_path), "_normal_cache")
    os.makedirs(cache_dir, exist_ok=True)
    expected_path = os.path.join(
        cache_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_normals.safetensors"
    )
    save_file(
        {"normal_gt": first_normal.half(), NORMAL_CACHE_VERSION_KEY: torch.ones(1)},
        expected_path,
    )

    # a DIFFERENT fake normal proves the encoder was NOT called on a hit
    called = {"count": 0}

    def _encode(pil_image):
        called["count"] += 1
        return torch.zeros(3, NORMAL_SIZE, NORMAL_SIZE)

    encoder = SimpleNamespace(encode=_encode)
    cache_normal_gt([fi], SimpleNamespace(), encoder=encoder)

    assert called["count"] == 0  # cache hit -> no recompute
    assert fi._normal_cache_path == expected_path
    assert torch.equal(fi.get_normal_gt(), first_normal.half())


# ---------------------------------------------------------------------------
# compute_normal_loss math + differentiability
# ---------------------------------------------------------------------------

class _FakePerceptor(nn.Module):
    """Maps (B,3,H,W) -> (B,3,NORMAL_SIZE,NORMAL_SIZE) unit-ish normals.

    A fixed linear conv keeps the graph so we can assert gradients flow to
    the input pixels.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1, bias=True)

    def forward(self, pixels):
        out = self.proj(pixels)
        out = F.interpolate(out, size=(NORMAL_SIZE, NORMAL_SIZE), mode="bilinear",
                            align_corners=False)
        return out / (out.norm(dim=1, keepdim=True) + 1e-5)


def test_compute_normal_loss_returns_per_sample_and_finite():
    enc = _FakePerceptor()
    pixels = torch.rand(2, 3, 16, 16, requires_grad=True)
    gt = torch.randn(2, 3, NORMAL_SIZE, NORMAL_SIZE)
    cos_loss, l1_loss, gen_det, ref_det = compute_normal_loss(enc, pixels, gt)
    assert cos_loss.shape == (2,)
    assert l1_loss.shape == (2,)
    assert torch.isfinite(cos_loss).all()
    assert torch.isfinite(l1_loss).all()
    assert gen_det.shape == (2, 3, NORMAL_SIZE, NORMAL_SIZE)


def test_compute_normal_loss_gradient_flows_to_pixels():
    enc = _FakePerceptor()
    pixels = torch.rand(1, 3, 16, 16, requires_grad=True)
    gt = torch.randn(1, 3, NORMAL_SIZE, NORMAL_SIZE)
    cos_loss, l1_loss, _, _ = compute_normal_loss(enc, pixels, gt)
    loss = (cos_loss + l1_loss).sum()
    loss.backward()
    assert pixels.grad is not None
    assert torch.isfinite(pixels.grad).all()
    assert pixels.grad.abs().sum() > 0


def test_compute_normal_loss_identical_normals_near_zero():
    # when the perceptor output equals the GT, cos_loss ~ 0 and l1 ~ 0.
    class _IdentityEncoder(nn.Module):
        def __init__(self, fixed):
            super().__init__()
            self._fixed = fixed

        def forward(self, pixels):
            return self._fixed.expand(pixels.shape[0], -1, -1, -1)

    gt_unit = torch.randn(1, 3, NORMAL_SIZE, NORMAL_SIZE)
    gt_unit = gt_unit / (gt_unit.norm(dim=1, keepdim=True) + 1e-5)
    enc = _IdentityEncoder(gt_unit)
    pixels = torch.rand(1, 3, 16, 16)
    cos_loss, l1_loss, _, _ = compute_normal_loss(enc, pixels, gt_unit)
    assert cos_loss.item() < 1e-4
    assert l1_loss.item() < 1e-4


# ---------------------------------------------------------------------------
# DataLoaderBatchDTO normal aggregation
# ---------------------------------------------------------------------------

def test_batch_aggregates_normal_gt_and_scalars(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path), resolution=32,
        normal_loss_weight=0.3, normal_loss_min_t=0.1, normal_loss_max_t=0.5,
    )
    fi = _make_file_item(image_path, cfg)
    expected = torch.randn(3, NORMAL_SIZE, NORMAL_SIZE)
    _stamp_normal_cache(fi, tmp_path / "normals.safetensors", tensor=expected)
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))

    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.normal_gt_list is not None
    assert len(batch.normal_gt_list) == 1
    assert torch.equal(batch.normal_gt_list[0], expected)
    assert batch.normal_loss_weight_list == [0.3]
    assert batch.normal_loss_min_t_list == [0.1]
    assert batch.normal_loss_max_t_list == [0.5]


def test_batch_without_normal_gt_has_no_list(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    # normal_gt_list is only assigned when at least one item has a cached map
    assert not hasattr(batch, "normal_gt_list") or batch.normal_gt_list is None


# ---------------------------------------------------------------------------
# preflight_normal_id dataset-only activation
# ---------------------------------------------------------------------------

def test_preflight_inert_when_absent():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_normal_id
    assert preflight_normal_id(None, [], None, False) is None


def test_preflight_dataset_only_activation_builds_disabled_default():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_normal_id
    dc = SimpleNamespace(normal_loss_weight=0.5, normal_loss_min_t=None,
                         normal_loss_max_t=None)
    cfg = preflight_normal_id(None, [dc], 'krea2', False)
    assert cfg is not None
    assert cfg.loss_weight == 0.0  # disabled default, but caching still runs
    # dataset override is what activates it
    from extensions_built_in.sd_trainer.SDTrainer import normal_active_for_dataset
    assert normal_active_for_dataset(cfg, dc) is True


def test_preflight_inactive_config_returned_disabled():
    # A disabled process config (loss_weight == 0, no dataset activation) is
    # returned as-is so the trainer holds an inert config (mirrors depth).
    from extensions_built_in.sd_trainer.SDTrainer import preflight_normal_id
    cfg = NormalIDConfig(loss_weight=0.0)
    out = preflight_normal_id(cfg, [], 'krea2', False)
    assert out is not None
    assert out.loss_weight == 0.0


def test_preflight_krea2_low_vram_rejected():
    # Normal decodes x0 through the VAE with grad -> same tiled-decode hazard
    # as depth; reject krea2 + low_vram when normal is active.
    from extensions_built_in.sd_trainer.SDTrainer import preflight_normal_id
    cfg = NormalIDConfig(loss_weight=0.01)
    with pytest.raises(ValueError, match="low_vram"):
        preflight_normal_id(cfg, [], 'krea2', True)


def test_preflight_low_vram_ok_when_disabled():
    # disabled normal (loss_weight 0) does not trigger the low_vram guard
    from extensions_built_in.sd_trainer.SDTrainer import preflight_normal_id
    cfg = NormalIDConfig(loss_weight=0.0)
    out = preflight_normal_id(cfg, [], 'krea2', True)
    assert out is not None and out.loss_weight == 0.0


def test_preflight_low_vram_ok_non_krea():
    # non-krea arch with low_vram is fine (no tiled-Qwen-VAE decode hazard)
    from extensions_built_in.sd_trainer.SDTrainer import preflight_normal_id
    cfg = NormalIDConfig(loss_weight=0.01)
    out = preflight_normal_id(cfg, [], 'flux1', True)
    assert out is not None and out.loss_weight == 0.01
