"""Cache-identity tests for build_depth_cache_fingerprint.

A cache hit requires source signature, full transform, perceptor (model_id +
input_size + blur), arch, and VAE identity to all match. Mutating any one
input independently must produce a different fingerprint (cache miss).
"""
import os
from types import SimpleNamespace

import pytest

from toolkit.depth_consistency import build_depth_cache_fingerprint, depth_cache_key


def _make_file_item(path, **transform_overrides):
    transform = dict(
        scale_to_width=512, scale_to_height=512,
        crop_x=0, crop_y=0, crop_width=512, crop_height=512,
        flip_x=False, flip_y=False,
    )
    transform.update(transform_overrides)
    return SimpleNamespace(path=str(path), **transform)


def _make_config(**overrides):
    cfg = dict(
        model_id='depth-anything/Depth-Anything-V2-Small-hf',
        input_size=518, pixel_blur_sigma=0.0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _fingerprint(path, *, transform=None, config=None, arch='krea2', vae_id='AutoencoderKLQwenImage::krea-vae'):
    fi = _make_file_item(path, **(transform or {}))
    cfg = _make_config(**(config or {}))
    return build_depth_cache_fingerprint(fi, cfg, arch=arch, vae_id=vae_id)


# ----------------------------------------------------------------------
# Hit: identical inputs -> identical fingerprint
# ----------------------------------------------------------------------

def test_identical_inputs_produce_identical_fingerprint(tmp_path):
    p = tmp_path / 'img.jpg'
    p.write_bytes(b'baseline-content')
    # two independently constructed file_items / configs over the same inputs
    assert _fingerprint(p) == _fingerprint(p)


def test_fingerprint_is_sha256_prefix_length(tmp_path):
    p = tmp_path / 'img.jpg'
    p.write_bytes(b'baseline-content')
    fp = _fingerprint(p)
    # schema:4, sha256[:20] -> 20 hex chars
    assert len(fp) == 20
    assert all(c in '0123456789abcdef' for c in fp)


# ----------------------------------------------------------------------
# Miss: mutating each input independently changes the fingerprint
# ----------------------------------------------------------------------

@pytest.mark.parametrize("label,kwargs", [
    ('scale_to_width', dict(transform=dict(scale_to_width=640))),
    ('scale_to_height', dict(transform=dict(scale_to_height=640))),
    ('crop_x', dict(transform=dict(crop_x=8))),
    ('crop_y', dict(transform=dict(crop_y=8))),
    ('crop_width', dict(transform=dict(crop_width=500))),
    ('crop_height', dict(transform=dict(crop_height=500))),
    ('flip_x', dict(transform=dict(flip_x=True))),
    ('flip_y', dict(transform=dict(flip_y=True))),
    ('model_id', dict(config=dict(model_id='depth-anything/Depth-Anything-V2-Large-hf'))),
    ('input_size', dict(config=dict(input_size=1024))),
    ('pixel_blur_sigma', dict(config=dict(pixel_blur_sigma=1.5))),
    ('arch', dict(arch='flux')),
    ('vae_id', dict(vae_id='AutoencoderKL::other-vae')),
])
def test_mutating_input_changes_fingerprint(tmp_path, label, kwargs):
    p = tmp_path / 'img.jpg'
    p.write_bytes(b'baseline-content')
    base = _fingerprint(p)
    assert _fingerprint(p, **kwargs) != base, f"{label} did not change fingerprint"


def test_source_size_change_changes_fingerprint(tmp_path):
    p = tmp_path / 'img.jpg'
    p.write_bytes(b'baseline')
    base = _fingerprint(p)
    p.write_bytes(b'baseline-with-more-bytes')  # different st_size, same path
    assert _fingerprint(p) != base


def test_source_mtime_change_changes_fingerprint(tmp_path):
    p = tmp_path / 'img.jpg'
    p.write_bytes(b'baseline')
    os.utime(str(p), ns=(1_000_000_000, 1_000_000_000))
    base = _fingerprint(p)
    # identical size, only mtime_ns moves
    os.utime(str(p), ns=(1_000_000_000, 9_999_999_999))
    assert _fingerprint(p) != base


# ----------------------------------------------------------------------
# Cache key helper
# ----------------------------------------------------------------------

def test_cache_key_wraps_fingerprint(tmp_path):
    assert depth_cache_key('abc123') == 'depth_gt_abc123'
    # a real fingerprint feeds the key, matching the safetensors tensor name
    p = tmp_path / 'img.jpg'
    p.write_bytes(b'baseline-content')
    fp = _fingerprint(p)
    assert depth_cache_key(fp) == f'depth_gt_{fp}'
