"""Qwen-Image 2.1 differentiable-decode guard tests.

``_decode_rgba`` tiles the 16x VAE decode above 1 MP (and always under
low_vram). Tiled decode cannot carry the perceptual aux losses' gradient, so
when a decode actually needs gradient and would tile, the wrapper must raise
instead of silently tiling. No-grad sampling keeps the tiled fast path.

Exercised on a bare instance with stubbed VAE/model_config -- no checkpoint.
"""
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import (
    TILE_DECODE_ABOVE_PIXELS,
    QwenImage2Model,
)


def _model(low_vram=False):
    model = QwenImage2Model.__new__(QwenImage2Model)
    model.model_config = SimpleNamespace(low_vram=low_vram)
    model.vae_scale_factor = 16
    model.vae_device_torch = torch.device('cpu')
    model.vae_torch_dtype = torch.float32
    # enough of the VAE for the code below the guard to run
    model.vae = SimpleNamespace(
        device=torch.device('cpu'),
        to=lambda device: None,
        config=SimpleNamespace(
            z_dim=64,
            latents_mean=[0.0] * 64,
            latents_std=[1.0] * 64,
        ),
        decode=lambda latents: SimpleNamespace(
            sample=torch.zeros(
                latents.shape[0], 4, 1,
                latents.shape[-2] * 16, latents.shape[-1] * 16,
            )
        ),
        enable_tiling=lambda **kwargs: None,
        disable_tiling=lambda: None,
    )
    return model


def _latents(h, w, requires_grad=True):
    return torch.randn(1, 64, h, w, requires_grad=requires_grad)


def test_tiled_decode_with_grad_raises():
    # 128x128 latents -> 2048x2048 pixels, above the tile threshold
    assert 128 * 128 * 16 * 16 > TILE_DECODE_ABOVE_PIXELS
    with pytest.raises(ValueError, match="cannot carry gradient"):
        _model()._decode_rgba(_latents(128, 128))


def test_untiled_decode_with_grad_passes_the_guard():
    # 64x64 latents -> exactly 1024x1024 pixels: at the threshold, not above
    result = _model()._decode_rgba(_latents(64, 64))
    assert result.shape == (1, 4, 1024, 1024)


def test_tiled_decode_without_grad_keeps_fast_path():
    # sampling / validation decode under no_grad must still tile silently
    result = _model()._decode_rgba(_latents(128, 128, requires_grad=False))
    assert result.shape == (1, 4, 2048, 2048)


def test_grad_enabled_but_detached_latents_still_tile():
    # grad mode on but the tensor carries no graph (e.g. logging): allowed
    with torch.no_grad():
        detached = _latents(128, 128).detach()
    result = _model()._decode_rgba(detached)
    assert result.shape == (1, 4, 2048, 2048)


def test_low_vram_forces_tiling_even_below_threshold():
    with pytest.raises(ValueError, match="low_vram"):
        _model(low_vram=True)._decode_rgba(_latents(64, 64))
