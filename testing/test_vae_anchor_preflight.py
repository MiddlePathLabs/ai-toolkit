"""Backend preflight tests for the VAE-anchor trainer init.

Scope: the preflight rejections that can be tested without a real model --
Krea 2 low_vram (existing contract), and the Qwen-Image 2.1 additions:
low_vram forcing tiled decode, and the differentiable-decode pixel limit
checked against the exact bucket geometry (or the dataset resolution
fallback when no loader buckets are available).
"""
import pytest

from toolkit.config_modules import VAEAnchorConfig, DatasetConfig
from extensions_built_in.sd_trainer.SDTrainer import preflight_vae_anchor


def _ds(**kw):
    return DatasetConfig(**kw)


# ----------------------------------------------------------------------
# Inertness: anchor entirely absent -> complete no-op
# ----------------------------------------------------------------------

def test_no_config_no_active_dataset_returns_none():
    result = preflight_vae_anchor(None, [_ds()], arch='qwen_image_2', low_vram=True)
    assert result is None


def test_disabled_global_config_no_active_dataset_is_inert():
    cfg = VAEAnchorConfig(loss_weight=0.0)
    result = preflight_vae_anchor(cfg, [_ds()], arch='qwen_image_2', low_vram=True)
    assert result is cfg


def test_dataset_only_activation_constructs_disabled_default():
    datasets = [_ds(vae_anchor_loss_weight=0.1)]
    result = preflight_vae_anchor(None, datasets, arch='qwen_image_2', low_vram=False)
    assert result is not None
    assert result.loss_weight == 0.0


# ----------------------------------------------------------------------
# Existing Krea 2 contract is unchanged
# ----------------------------------------------------------------------

def test_krea2_low_vram_still_raises():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    with pytest.raises(ValueError, match="Krea 2"):
        preflight_vae_anchor(cfg, [_ds()], arch='krea2', low_vram=True)


def test_krea2_low_vram_false_is_fine():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    result = preflight_vae_anchor(cfg, [_ds()], arch='krea2', low_vram=False)
    assert result is cfg


# ----------------------------------------------------------------------
# Qwen-Image 2.1: low_vram forces tiled decode
# ----------------------------------------------------------------------

def test_qwen_low_vram_raises():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    with pytest.raises(ValueError, match="low_vram"):
        preflight_vae_anchor(cfg, [_ds()], arch='qwen_image_2', low_vram=True)


def test_qwen_low_vram_raises_via_dataset_only_activation():
    datasets = [_ds(vae_anchor_loss_weight=0.1)]
    with pytest.raises(ValueError, match="low_vram"):
        preflight_vae_anchor(None, datasets, arch='qwen_image_2', low_vram=True)


# ----------------------------------------------------------------------
# Qwen-Image 2.1: differentiable-decode pixel limit vs actual buckets
# ----------------------------------------------------------------------

LIMIT = 1024 * 1024


def test_qwen_bucket_above_limit_raises():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    with pytest.raises(ValueError, match="tile threshold"):
        preflight_vae_anchor(
            cfg, [_ds()], arch='qwen_image_2', low_vram=False,
            decode_pixel_limit=LIMIT, max_active_bucket_pixels=1536 * 1536,
        )


def test_qwen_bucket_at_limit_is_untiled_and_passes():
    # tiled decode triggers only ABOVE the threshold; exactly 1024x1024 is fine
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    result = preflight_vae_anchor(
        cfg, [_ds()], arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT,
    )
    assert result is cfg


def test_qwen_opted_out_dataset_is_not_checked():
    # a high-res dataset that explicitly zeroes the anchor is exempt; the
    # inheriting low-res one is fine, so no raise
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    datasets = [
        _ds(vae_anchor_loss_weight=0.0, resolution=2048),
        _ds(resolution=512),
    ]
    result = preflight_vae_anchor(
        cfg, datasets, arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT,
    )
    assert result is cfg


def test_qwen_inheriting_dataset_is_checked():
    # the opted-out dataset may stay high-res; the inheriting one cannot
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    datasets = [
        _ds(vae_anchor_loss_weight=0.0, resolution=2048),
        _ds(resolution=1536),
    ]
    with pytest.raises(ValueError, match="resolution 1536"):
        preflight_vae_anchor(
            cfg, datasets, arch='qwen_image_2', low_vram=False,
            decode_pixel_limit=LIMIT,
        )


# ----------------------------------------------------------------------
# Qwen-Image 2.1: resolution fallback when bucket geometry is unknown
# ----------------------------------------------------------------------

def test_qwen_resolution_fallback_raises_above_limit():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    with pytest.raises(ValueError, match="resolution 1536"):
        preflight_vae_anchor(
            cfg, [_ds(resolution=1536)], arch='qwen_image_2', low_vram=False,
            decode_pixel_limit=LIMIT,
        )


def test_qwen_resolution_fallback_passes_at_limit():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    result = preflight_vae_anchor(
        cfg, [_ds(resolution=1024)], arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT,
    )
    assert result is cfg


def test_qwen_limit_not_set_skips_resolution_check():
    # no pixel limit exposed (non-qwen wrapper or older call sites): the old
    # behavior -- no resolution restriction -- is preserved
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    result = preflight_vae_anchor(
        cfg, [_ds(resolution=2048)], arch='qwen_image_2', low_vram=False,
    )
    assert result is cfg


def test_flux_with_limit_set_is_unaffected():
    cfg = VAEAnchorConfig(loss_weight=0.05, vae_model_path='x.safetensors')
    result = preflight_vae_anchor(
        cfg, [_ds(resolution=2048)], arch='flux', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg
