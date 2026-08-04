"""Cross-VAE perceptual anchor using the Flux 2 VAE encoder.

A frozen Flux 2 VAE ENCODER (custom BFL AutoEncoder, loaded by file path from
``extensions_built_in/diffusion_models/flux2/src/autoencoder.py``) acts as a
perceptual discriminator: the live loss decodes the predicted x0 through the
training model's VAE, encodes those pixels with the Flux 2 encoder, and matches
the multi-scale features against cached GT via cosine similarity.

Adapted for this fork: the DECODE side uses ``self.sd.decode_latents`` (the
training model's own VAE path, correct for Krea 2's AutoencoderKLQwenImage)
rather than the source's manual ``vae.decode`` with scaling_factor/shift_factor
(which is incompatible with the Qwen VAE's latents_mean/latents_std). The Flux
2 VAE is used ONLY for the perceptual feature ENCODE.

Weights: HuggingFace ``ai-toolkit/flux2_vae`` / ``ae.safetensors`` (auto-download
via huggingface_hub). The ``einops`` dep is required by the flux2 autoencoder.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

FEATURE_LEVELS = ["level_0", "level_1", "level_2", "level_3", "mid"]
LEVEL_CHANNELS = {"level_0": 128, "level_1": 256, "level_2": 512, "level_3": 512, "mid": 512}
CACHE_VERSION_KEY = "vae_anchor_v1"


class VAEAnchorEncoder(nn.Module):
    """Frozen Flux 2 VAE encoder for the perceptual anchor loss.

    Loads only the encoder (+ bn) of the Flux 2 VAE. Forward hooks on the
    encoder's down/mid blocks capture multi-scale features for the loss.
    """

    def __init__(self, vae_path: str = ""):
        super().__init__()
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._encoder = None
        self._loaded = False
        self._vae_path = vae_path

    @staticmethod
    def _resolve_vae_path(vae_path: str) -> str:
        if vae_path and os.path.exists(vae_path):
            return vae_path
        from huggingface_hub import hf_hub_download
        print("  VAE anchor: downloading Flux 2 VAE from ai-toolkit/flux2_vae...")
        return hf_hub_download(repo_id="ai-toolkit/flux2_vae", filename="ae.safetensors")

    def load(self, device: torch.device, dtype: torch.dtype):
        if self._loaded:
            return
        self._vae_path = self._resolve_vae_path(self._vae_path)
        # Import the flux2 AutoEncoder by file path to avoid the heavy
        # extensions_built_in package __init__ chain.
        import importlib.util
        _ae_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "extensions_built_in", "diffusion_models", "flux2", "src", "autoencoder.py",
        )
        _spec = importlib.util.spec_from_file_location("flux2_autoencoder", _ae_path)
        _ae_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ae_mod)
        AutoEncoder = _ae_mod.AutoEncoder
        AutoEncoderParams = _ae_mod.AutoEncoderParams

        ae = AutoEncoder(AutoEncoderParams())
        state_dict = load_file(self._vae_path)
        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
        encoder_sd = (
            {k[len("encoder."):]: v for k, v in encoder_keys.items()}
            if encoder_keys else state_dict
        )
        ae.encoder.load_state_dict(encoder_sd, strict=False)
        bn_keys = {k: v for k, v in state_dict.items() if k.startswith("bn.")}
        if bn_keys:
            ae.bn.load_state_dict({k[len("bn."):]: v for k, v in bn_keys.items()}, strict=False)

        self._encoder = ae.encoder
        self._encoder.to(device=device, dtype=dtype).eval()
        self._encoder.requires_grad_(False)
        self._register_hooks()
        self._loaded = True

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()
        encoder = self._encoder

        def _hook(name):
            def fn(module, inp, out):
                self._features[name] = out
            return fn

        self._hooks.append(encoder.down[0].block[1].register_forward_hook(_hook("level_0")))
        self._hooks.append(encoder.down[1].block[1].register_forward_hook(_hook("level_1")))
        self._hooks.append(encoder.down[2].block[1].register_forward_hook(_hook("level_2")))
        self._hooks.append(encoder.down[3].block[1].register_forward_hook(_hook("level_3")))
        self._hooks.append(encoder.mid.block_2.register_forward_hook(_hook("mid")))

    def encode_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode pixels in [-1, 1] (B,3,H,W); return (final, features_dict).

        Gradients flow through the encoder to the input pixels.
        """
        assert self._loaded, "Call load() first"
        self._features.clear()
        enc_dtype = next(self._encoder.parameters()).dtype
        final = self._encoder(x.to(dtype=enc_dtype))
        features = {k: v for k, v in self._features.items()}
        return final, features

    @staticmethod
    def compute_loss(
        pred_features: Dict[str, torch.Tensor],
        ref_features: Dict[str, torch.Tensor],
        level_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Per-sample cosine feature loss across levels. Returns (B,) + per-level dict."""
        if level_weights is None:
            level_weights = {"level_0": 4.0, "level_1": 2.0, "level_2": 1.0, "level_3": 1.0, "mid": 1.0}
        device = next(iter(pred_features.values())).device
        batch_size = next(iter(pred_features.values())).shape[0]
        total = torch.zeros(batch_size, device=device)
        per_level: Dict[str, float] = {}
        n = 0
        for level in FEATURE_LEVELS:
            if level not in pred_features or level not in ref_features:
                continue
            pred = pred_features[level]
            ref = ref_features[level].to(pred.device, dtype=pred.dtype)
            if pred.shape[2:] != ref.shape[2:]:
                ref = F.interpolate(ref, size=pred.shape[2:], mode="bilinear", align_corners=False)
            pred_flat = pred.flatten(2)  # (B, C, N)
            ref_flat = ref.flatten(2)
            cos_sim = F.cosine_similarity(pred_flat, ref_flat, dim=1)  # (B, N)
            level_loss = (1.0 - cos_sim).mean(dim=1)  # (B,)
            total = total + level_weights.get(level, 1.0) * level_loss
            per_level[level] = float(level_loss.detach().mean().item())
            n += 1
        if n > 0:
            total = total / n
        return total, per_level

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()


def encode_reference_features(encoder: VAEAnchorEncoder, pil_image, target_size: int = 512) -> Dict[str, torch.Tensor]:
    """PIL -> dict of CPU fp16 feature tensors (cache path)."""
    import torchvision.transforms.functional as TF
    w, h = pil_image.size
    if min(w, h) != target_size:
        scale = target_size / min(w, h)
        new_w = max(8, (int(w * scale) // 8) * 8)
        new_h = max(8, (int(h * scale) // 8) * 8)
        pil_image = pil_image.resize((new_w, new_h))
    else:
        new_w = max(8, (w // 8) * 8)
        new_h = max(8, (h // 8) * 8)
        if new_w != w or new_h != h:
            pil_image = pil_image.resize((new_w, new_h))
    img = TF.to_tensor(pil_image).unsqueeze(0) * 2.0 - 1.0  # [0,1] -> [-1,1]
    device = encoder._encoder.conv_in.weight.device
    dtype = encoder._encoder.conv_in.weight.dtype
    img = img.to(device=device, dtype=dtype)
    with torch.no_grad():
        _, features = encoder.encode_with_features(img)
    return {k: v.cpu().half() for k, v in features.items()}


# ---------------------------------------------------------------------------
# GT-feature caching (stamps file items for lazy worker reads)
# ---------------------------------------------------------------------------

def _vae_cache_path(file_item) -> str:
    img_dir = os.path.dirname(file_item.path)
    cache_dir = os.path.join(img_dir, "_vae_anchor_cache")
    filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
    return os.path.join(cache_dir, f"{filename_no_ext}_vaeanchor.safetensors")


def _vae_cache_hit(cache_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    try:
        with safe_open(cache_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if CACHE_VERSION_KEY not in keys:
                return False
            return all(f"vae_anchor_{lv}" in keys for lv in FEATURE_LEVELS)
    except Exception:  # noqa: BLE001
        return False


def _atomic_save_file(save_data: dict, cache_path: str) -> None:
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".safetensors", dir=cache_dir)
    os.close(fd)
    try:
        save_file(save_data, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def cache_vae_anchor(
    file_items: List,
    config,
    *,
    encoder: VAEAnchorEncoder,
) -> None:
    """Stamp every file item with its vae-anchor cache metadata and persist GT features.

    GT features come from the raw source image (transform-independent). The 5
    per-level feature tensors are written under ``vae_anchor_<level>`` keys; the
    worker re-reads them lazily via the mixin.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    hits, misses = 0, 0
    for file_item in tqdm(file_items, desc="Caching GT vae-anchor features"):
        cache_path = _vae_cache_path(file_item)
        file_item._vae_cache_path = cache_path
        file_item.is_vae_anchor_cached = True
        file_item.vae_anchor_features = None

        if _vae_cache_hit(cache_path):
            hits += 1
            continue

        misses += 1
        pil_image = exif_transpose(Image.open(file_item.path)).convert("RGB")
        w, h = pil_image.size
        features = encode_reference_features(encoder, pil_image, target_size=min(w, h))
        save_data = {CACHE_VERSION_KEY: torch.ones(1)}
        for level, feat in features.items():
            save_data[f"vae_anchor_{level}"] = feat
        _atomic_save_file(save_data, cache_path)

    if hits or misses:
        print(f"  - GT vae-anchor cache: {hits} reused, {misses} computed.")
