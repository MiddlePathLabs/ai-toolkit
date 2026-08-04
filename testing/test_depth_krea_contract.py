"""Krea depth-GT caching contract tests (Task 5a).

Uses a FAKE Krea model that records calls -- no DA2 weights, no GPU. Enforces
the four load-bearing contracts of the depth-GT caching pipeline:

  1. The VAE round-trip callback routes every arch through sd.encode_images
     and sd.decode_latents, NEVER a direct vae.encode (a scalar
     scaling_factor mis-normalizes AutoencoderKL / Qwen VAEs).
  2. The round-trip returns finite pixels in [0, 1].
  3. Active Krea depth with low_vram: true raises at backend preflight.
  4. mask_source subject/body raise during Phase 2 (auto-masking is Phase 3).

Plus: the cache fingerprint varies with arch and vae_id (so the caching pass
supplying them matters), the caching function stamps the attributes Task 3's
DepthCachingFileItemDTOMixin reads and persists safetensors under the
fingerprint key, a cache hit skips recomputation, and depth-inactive configs
gate the whole pass off (no perceptor load, no file-item stamping).
"""
import os

import torch
from PIL import Image

from toolkit.config_modules import DepthConsistencyConfig
from toolkit.depth_consistency import build_depth_cache_fingerprint, depth_cache_key
from toolkit.depth_perceptor import cache_depth_gt
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer, preflight_depth_consistency


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class _FakeVAE:
    """A VAE whose .encode must NEVER be reached by the depth round-trip."""

    def __init__(self):
        self.config = None

    def parameters(self):
        # Residency-guard probe: yield a CPU tensor so the trainer's device
        # check is a no-op against the CPU fake.
        yield torch.zeros(1, device="cpu")

    def encode(self, *args, **kwargs):
        raise AssertionError(
            "Depth round-trip must NOT call vae.encode directly; "
            "it must route through sd.encode_images / sd.decode_latents."
        )


class _FakeModelConfig:
    def __init__(self, arch):
        self.arch = arch


class _FakeKreaSD:
    """Records encode_images / decode_latents calls; mirrors Krea signatures."""

    def __init__(self):
        self.vae = _FakeVAE()
        self.model_config = _FakeModelConfig("krea2")
        self.vae_torch_dtype = torch.float32
        self.encode_images_calls = 0
        self.decode_latents_calls = 0

    def encode_images(self, image_list, device=None, dtype=None):
        self.encode_images_calls += 1
        imgs = torch.stack(list(image_list))  # (B, 3, H, W)
        b, c, h, w = imgs.shape
        # emulate a downsampled latent (Krea uses 16 channels)
        return torch.zeros(b, 16, max(1, h // 8), max(1, w // 8))

    def decode_latents(self, latents, device=None, dtype=None):
        self.decode_latents_calls += 1
        b, _, h, w = latents.shape
        # nominal [-1, 1] pixel range; (0.5 + 1) * 0.5 = 0.75 after the callback
        return torch.full((b, 3, h * 8, w * 8), 0.5)


class _FakeTrainer:
    """Minimal stand-in so SDTrainer methods bind without a real trainer."""


class _FakeEncoder:
    """Stands in for DifferentiableDepthEncoder so no DA2 weights load."""

    def __init__(self):
        self.forward_calls = 0

    def __call__(self, arr):
        self.forward_calls += 1
        #DifferentiableDepthEncoder.forward returns (B, Hd, Wd); take [0] later.
        return torch.zeros(arr.shape[0], 8, 8)


class _StubFileItem:
    """Minimal file item with the transform + path attributes the cache reads."""

    def __init__(self, path):
        self.path = path
        self.scale_to_width = 64
        self.scale_to_height = 64
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = 64
        self.crop_height = 64
        self.flip_x = False
        self.flip_y = False
        self.is_depth_cached = False
        self.depth_gt = None
        self._depth_cache_path = None
        self._depth_cache_key = None


# ----------------------------------------------------------------------
# Contract 1 + 2: round-trip routing and output range
# ----------------------------------------------------------------------

def test_roundtrip_routes_through_encode_images_and_decode_latents():
    trainer = _FakeTrainer()
    sd = _FakeKreaSD()
    trainer.sd = sd
    trainer.device_torch = torch.device("cpu")

    pixels = torch.rand(1, 3, 64, 64)
    out = SDTrainer._depth_vae_roundtrip(trainer, pixels)

    assert sd.encode_images_calls == 1
    assert sd.decode_latents_calls == 1
    # vae.encode raising AssertionError would fail the test if reached.
    assert torch.isfinite(out).all()
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert out.shape[0] == 1 and out.shape[1] == 3


def test_roundtrip_output_is_finite_and_unit_clamped():
    trainer = _FakeTrainer()
    sd = _FakeKreaSD()
    trainer.sd = sd
    trainer.device_torch = torch.device("cpu")

    pixels = torch.rand(1, 3, 32, 40)
    out = SDTrainer._depth_vae_roundtrip(trainer, pixels)

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 - 1e-6
    assert float(out.max()) <= 1.0 + 1e-6


# ----------------------------------------------------------------------
# Contract 3: active Krea depth + low_vram raises at preflight
# ----------------------------------------------------------------------

def test_krea_depth_with_low_vram_raises():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    with pytest_raises(ValueError, match="low_vram"):
        preflight_depth_consistency(cfg, [], arch="krea2", low_vram=True)


def test_krea_depth_without_low_vram_does_not_raise():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    result = preflight_depth_consistency(cfg, [], arch="krea2", low_vram=False)
    assert result is cfg


# ----------------------------------------------------------------------
# Contract 4 (Phase 3): mask_source subject/body are now ALLOWED at preflight
# (auto-masking ships them). The cross-check that subject_mask is enabled runs
# in hook_before_train_loop; preflight itself no longer rejects them.
# ----------------------------------------------------------------------

def test_mask_source_subject_allowed_in_phase3():
    cfg = DepthConsistencyConfig(loss_weight=0.001, mask_source="subject")
    result = preflight_depth_consistency(cfg, [], arch="krea2", low_vram=False)
    assert result is cfg


def test_mask_source_body_allowed_in_phase3():
    cfg = DepthConsistencyConfig(loss_weight=0.001, mask_source="body")
    result = preflight_depth_consistency(cfg, [], arch="flux", low_vram=False)
    assert result is cfg


def test_mask_source_none_is_allowed():
    cfg = DepthConsistencyConfig(loss_weight=0.001, mask_source="none")
    result = preflight_depth_consistency(cfg, [], arch="flux", low_vram=False)
    assert result is cfg


# ----------------------------------------------------------------------
# Fingerprint sensitivity to arch and vae_id
# ----------------------------------------------------------------------

def _write_image(path):
    Image.new("RGB", (64, 64), (128, 64, 200)).save(path)


def test_fingerprint_varies_with_arch(tmp_path):
    p = tmp_path / "img.png"
    _write_image(str(p))
    fi = _StubFileItem(str(p))
    fp_krea = build_depth_cache_fingerprint(
        fi, DepthConsistencyConfig(), arch="krea2", vae_id="vae:id-1"
    )
    fp_flux = build_depth_cache_fingerprint(
        fi, DepthConsistencyConfig(), arch="flux", vae_id="vae:id-1"
    )
    assert fp_krea != fp_flux


def test_fingerprint_varies_with_vae_id(tmp_path):
    p = tmp_path / "img.png"
    _write_image(str(p))
    fi = _StubFileItem(str(p))
    fp_a = build_depth_cache_fingerprint(
        fi, DepthConsistencyConfig(), arch="krea2", vae_id="vae:id-a"
    )
    fp_b = build_depth_cache_fingerprint(
        fi, DepthConsistencyConfig(), arch="krea2", vae_id="vae:id-b"
    )
    assert fp_a != fp_b


# ----------------------------------------------------------------------
# Caching pass: stamps file items and writes safetensors under the key
# ----------------------------------------------------------------------

def test_cache_depth_gt_stamps_items_and_writes_safetensors(tmp_path):
    p = tmp_path / "img.png"
    _write_image(str(p))
    fi = _StubFileItem(str(p))
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    encoder = _FakeEncoder()

    cache_depth_gt(
        [fi], cfg,
        encoder=encoder, arch="krea2", vae_id="vae:id-x",
        device=torch.device("cpu"),
        roundtrip_fn=lambda arr: arr,
    )

    # the encoder ran exactly once (one miss)
    assert encoder.forward_calls == 1
    # Task 3's mixin reads these exact attributes
    assert fi.is_depth_cached is True
    assert fi.depth_gt is None
    assert fi._depth_cache_path is not None
    assert fi._depth_cache_key is not None
    # key + path match the fingerprint
    fingerprint = build_depth_cache_fingerprint(
        fi, cfg, arch="krea2", vae_id="vae:id-x"
    )
    assert fi._depth_cache_key == depth_cache_key(fingerprint)
    # safetensors actually written with that key
    assert os.path.exists(fi._depth_cache_path)
    from safetensors import safe_open
    with safe_open(fi._depth_cache_path, framework="pt", device="cpu") as f:
        assert fi._depth_cache_key in f.keys()


def test_cache_depth_gt_hit_skips_recompute(tmp_path):
    p = tmp_path / "img.png"
    _write_image(str(p))
    fi = _StubFileItem(str(p))
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    encoder = _FakeEncoder()

    roundtrip_calls = [0]

    def _roundtrip(arr):
        roundtrip_calls[0] += 1
        return arr

    # first pass: miss -> compute + write
    cache_depth_gt(
        [fi], cfg,
        encoder=encoder, arch="krea2", vae_id="vae:id-y",
        device=torch.device("cpu"), roundtrip_fn=_roundtrip,
    )
    assert encoder.forward_calls == 1
    assert roundtrip_calls[0] == 1

    # reset counters; same item, same fingerprint -> header hit, no recompute
    encoder.forward_calls = 0
    roundtrip_calls[0] = 0
    fi.depth_gt = None
    cache_depth_gt(
        [fi], cfg,
        encoder=encoder, arch="krea2", vae_id="vae:id-y",
        device=torch.device("cpu"), roundtrip_fn=_roundtrip,
    )
    assert encoder.forward_calls == 0
    assert roundtrip_calls[0] == 0
    # still stamped for the worker
    assert fi.is_depth_cached is True
    assert fi._depth_cache_key is not None


# ----------------------------------------------------------------------
# Inertness: depth-inactive config gates the whole pass off
# ----------------------------------------------------------------------

def test_depth_inactive_skips_caching():
    trainer = _FakeTrainer()
    trainer.dataset_configs = []

    trainer.depth_consistency_config = None
    assert SDTrainer._depth_should_cache(trainer) is False

    trainer.depth_consistency_config = DepthConsistencyConfig(loss_weight=0.0)
    assert SDTrainer._depth_should_cache(trainer) is False


def test_depth_active_signals_caching():
    trainer = _FakeTrainer()
    trainer.dataset_configs = []
    trainer.depth_consistency_config = DepthConsistencyConfig(loss_weight=0.001)
    assert SDTrainer._depth_should_cache(trainer) is True


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def pytest_raises(expected_exception, match=None):
    # local import keeps the module top light; pytest is the test runner
    import pytest
    return pytest.raises(expected_exception, match=match)
