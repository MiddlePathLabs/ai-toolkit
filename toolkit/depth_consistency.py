"""Depth-consistency auxiliary loss via a frozen Depth-Anything-V2 perceptor.

License notice: the Depth-Anything-V2 HuggingFace weights
(depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf) are released under
CC-BY-NC-4.0 (non-commercial use only). Selecting the Krea 2 architecture does
NOT download a perceptor; the weights download lazily only when depth is
enabled (loss_weight > 0). See the model cards for full license terms.

This module currently exposes the cache-identity contract only. The DA2
forward pass and the scale-and-shift-invariant / multi-scale-gradient loss
math are added in a later task alongside the trainer integration.
"""
import hashlib
import json
import os


def build_depth_cache_fingerprint(file_item, config, *, arch: str, vae_id: str) -> str:
    """Build a 20-char sha256 fingerprint identifying one cached depth-GT.

    A cache hit requires ALL of: source signature (size + mtime_ns), the
    complete image transform (scale + crop + flip), the perceptor identity
    (model_id + input_size + pixel_blur_sigma), the model architecture, and
    the VAE identity to match. Changing any one forces a miss.

    The trainer supplies ``arch=self.sd.model_config.arch`` and a stable
    ``vae_id`` (VAE class FQN + ``self.sd.vae.config._name_or_path``, falling
    back to ``model.model_kwargs.vae_path``).
    """
    source_stat = os.stat(file_item.path)

    payload = {
        'schema': 4,
        'source': {
            'size': source_stat.st_size,
            'mtime_ns': source_stat.st_mtime_ns,
        },
        'transform': {
            'scale_to_width': getattr(file_item, 'scale_to_width', None),
            'scale_to_height': getattr(file_item, 'scale_to_height', None),
            'crop_x': getattr(file_item, 'crop_x', None),
            'crop_y': getattr(file_item, 'crop_y', None),
            'crop_width': getattr(file_item, 'crop_width', None),
            'crop_height': getattr(file_item, 'crop_height', None),
            'flip_x': bool(getattr(file_item, 'flip_x', False)),
            'flip_y': bool(getattr(file_item, 'flip_y', False)),
        },
        'perceptor': {
            'model_id': config.model_id,
            'input_size': int(config.input_size),
            'pixel_blur_sigma': float(config.pixel_blur_sigma),
        },
        'arch': arch,
        'vae_id': vae_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:20]


def depth_cache_key(fingerprint: str) -> str:
    """Wrap a fingerprint into the safetensors tensor key used in a cache file."""
    return f"depth_gt_{fingerprint}"
