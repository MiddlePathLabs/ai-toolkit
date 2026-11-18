"""Still-image passes through the Wan-style Qwen VAEs keep no temporal feature cache.

Port of Fizgig #369 (3d22809f): a single-frame encode/decode has no next frame that
could reuse the per-conv feature cache, so holding it only raises peak memory
(~2 GiB on an untiled 1024x1024 decode). A still must produce identical output with
the cache forced on (old behavior) vs off (fixed path), and a video must keep it.

Covers both fork implementations:
- toolkit.models.v2.vae.qwen_image.QwenImageVAE (krea2 / qwen_image / nucleus_image;
  fix = still-pass flag + clear_cache override over the installed diffusers base)
- extensions_built_in qwen_image_2 vendored AutoencoderKLQwenImage21 (direct guards)
"""
import importlib.util
import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.models.v2.vae.qwen_image import QwenImageVAE

_spec = importlib.util.spec_from_file_location(
    "qwen_image_2_vae_under_test",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "extensions_built_in",
        "diffusion_models",
        "qwen_image_2",
        "src",
        "vae.py",
    ),
)
_vendored = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _vendored
_spec.loader.exec_module(_vendored)
AutoencoderKLQwenImage21 = _vendored.AutoencoderKLQwenImage21


def _tiny(cls, **kwargs):
    torch.manual_seed(19)
    return cls(base_dim=4, z_dim=2, num_res_blocks=1, **kwargs).eval()


@pytest.fixture()
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _force_cache_list(vae, cache_attr, conv_num_attr):
    """A fresh cache list to force the old always-cached behavior.

    The vendored VAE keeps real lists on its attributes; the QwenImageVAE override
    holds them at None during a still pass, so a fresh list is built from the
    conv count clear_cache() recorded.
    """
    cache = getattr(vae, cache_attr)
    if cache is not None:
        return cache
    return [None] * getattr(vae, conv_num_attr)


def _assert_still_matches_cached(vae, direction, tiled, fresh_cache):
    vae.use_tiling = tiled
    vae.tile_sample_min_height = vae.tile_sample_min_width = 32
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = 24
    in_channels = getattr(vae.config, "in_channels", None) or getattr(
        vae.config, "input_channels", 4
    )

    if direction == "encode":
        module, cache_attr = vae.encoder, "_enc_feat_map"
        conv_num_attr = "_enc_conv_num"
        x = torch.randn(1, in_channels, 1, 64, 64)

        def run():
            return vae.encode(x).latent_dist.parameters

    else:
        module, cache_attr = vae.decoder, "_feat_map"
        conv_num_attr = "_conv_num"
        with torch.no_grad():
            z = vae.encode(torch.randn(1, in_channels, 1, 64, 64)).latent_dist.mode()

        def run():
            return vae.decode(z).sample

    forward = module.forward
    cached_bytes = []

    def cached(*args, **kwargs):
        kwargs["feat_cache"] = _force_cache_list(vae, cache_attr, conv_num_attr)
        result = forward(*args, **kwargs)
        cached_bytes.append(
            sum(t.numel() * t.element_size() for t in kwargs["feat_cache"] if torch.is_tensor(t))
        )
        return result

    with patch.object(module, "forward", side_effect=cached):
        with torch.no_grad():
            expected = run()
    with patch.object(module, "forward", wraps=forward) as calls:
        with torch.no_grad():
            actual = run()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if direction == "decode":  # pixels byte-identical after the usual to-image map
        to_px = lambda t: (t.float().add(1).div(2).clamp(0, 1) * 255).byte()
        assert torch.equal(to_px(actual), to_px(expected))
    assert cached_bytes and max(cached_bytes) > 0  # the forced run really cached
    assert all(c.kwargs["feat_cache"] is None for c in calls.call_args_list)
    assert (len(calls.call_args_list) > 1) == tiled


def _assert_video_keeps_cache(vae, in_channels, hw=24):
    vae.use_tiling = False
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = 8
    x = torch.randn(1, in_channels, 5, hw, hw)
    with patch.object(vae.encoder, "forward", wraps=vae.encoder.forward) as enc, patch.object(
        vae.decoder, "forward", wraps=vae.decoder.forward
    ) as dec:
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
            image = vae.decode(z).sample
    assert image.shape == x.shape
    for call in list(enc.call_args_list) + list(dec.call_args_list):
        assert call.kwargs["feat_cache"] is not None


@pytest.mark.parametrize("direction", ["encode", "decode"])
@pytest.mark.parametrize("tiled", [False, True])
def test_qwen_image_vae_still_matches_cached(single_thread, direction, tiled):
    """krea2-family VAE: still passes are cache-free and identical to cached ones."""
    vae = _tiny(QwenImageVAE)
    _assert_still_matches_cached(vae, direction, tiled, fresh_cache=True)


def test_qwen_image_vae_video_keeps_cache(single_thread):
    vae = _tiny(QwenImageVAE)
    with torch.no_grad():
        vae.encode(torch.randn(1, 3, 1, 64, 64))  # a still first: the flag must not leak
    _assert_video_keeps_cache(vae, in_channels=3)


@pytest.mark.parametrize("direction", ["encode", "decode"])
@pytest.mark.parametrize("tiled", [False, True])
def test_qwen_image_21_vae_still_matches_cached(single_thread, direction, tiled):
    """Vendored Qwen 2.1 VAE: still passes are cache-free and identical to cached ones."""
    vae = _tiny(AutoencoderKLQwenImage21)
    _assert_still_matches_cached(vae, direction, tiled, fresh_cache=False)


def test_qwen_image_21_vae_is_stills_only(single_thread):
    """The fork's 2.1 VAE replaces Wan's causal 3D convs with 2D image-specialized
    ones that refuse temporal caches, so video through it is unsupported by design;
    the still guards above are the only cache paths that matter for it."""
    vae = _tiny(AutoencoderKLQwenImage21)
    with pytest.raises(ValueError, match="image specialization"):
        with torch.no_grad():
            vae.encode(torch.randn(1, 4, 5, 64, 64))
