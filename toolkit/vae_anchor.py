"""Cross-VAE perceptual anchor using a matching Flux 2 VAE encoder.

The repository's canonical ``toolkit.models.v2.vae.flux2_kl.AutoEncoder`` is
the only accepted backend. A caller must provide a local, licensed
``ae.safetensors`` checkpoint in the matching BFL layout through
``vae_model_path``. The anchor never downloads a checkpoint, substitutes the
training VAE, or accepts unchecked partial loading.

The live loss decodes the predicted x0 through the training model's VAE,
encodes those pixels with the separate frozen Flux 2 encoder, and matches
multi-scale features against cached GT via cosine similarity. The matching
encoder remains gradient-enabled with frozen parameters so the loss gradient
reaches generated pixels.
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
# These are properties of the BFL Flux 2 KL encoder in
# ``toolkit.models.v2.vae.flux2_kl``.  They are deliberately explicit: a
# different VAE can have the same broad API while producing incompatible
# features, and must not silently satisfy this auxiliary loss.
FLUX2_INPUT_CHANNELS = 3
FLUX2_MOMENT_CHANNELS = 64  # 2 * z_channels (32), before 2x2 packing/BN.
FLUX2_LATENT_CHANNELS = 128  # 2 * 2 pixel shuffle * z_channels.
CACHE_VERSION_KEY = "vae_anchor_v3_flux2_kl_contract"


class VAEAnchorEncoder(nn.Module):
    """Frozen matching Flux 2 VAE encoder for the perceptual anchor loss.

    The full canonical AutoEncoder is loaded strictly from the supplied
    checkpoint, then only its encoder is retained. Forward hooks on the
    encoder's down/mid blocks capture multi-scale features for the loss.
    """

    def __init__(self, vae_path: str = ""):
        super().__init__()
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._encoder = None
        self._loaded = False
        self._contract_validated = False
        self._vae_path = vae_path

    @staticmethod
    def _module_out_channels(module: nn.Module) -> Optional[int]:
        channels = getattr(module, "out_channels", None)
        if channels is None:
            conv = getattr(module, "conv2", None)
            channels = getattr(conv, "out_channels", None)
        return int(channels) if channels is not None else None

    @classmethod
    def _validate_encoder_contract(cls, encoder: nn.Module) -> None:
        """Reject a same-shaped-but-wrong VAE before hooks or cache setup.

        The canonical BFL encoder has four resolution levels, two residual
        blocks per level, and the channel progression 128/256/512/512.  The
        hooked outputs are the second residual block at each level plus
        ``mid.block_2``; they are pre-latent-normalization activations.
        """
        down = getattr(encoder, "down", None)
        if down is None or len(down) != 4:
            raise ValueError("Flux 2 encoder must expose exactly four down levels")
        for level, name in enumerate(("level_0", "level_1", "level_2", "level_3")):
            blocks = getattr(down[level], "block", None)
            if blocks is None or len(blocks) < 2:
                raise ValueError(f"Flux 2 encoder {name} must expose block[1]")
            actual = cls._module_out_channels(blocks[1])
            expected = LEVEL_CHANNELS[name]
            if actual != expected:
                raise ValueError(
                    f"Flux 2 encoder {name} channel contract is {expected}, got {actual}"
                )

        mid = getattr(encoder, "mid", None)
        mid_block = getattr(mid, "block_2", None)
        if mid_block is None or cls._module_out_channels(mid_block) != LEVEL_CHANNELS["mid"]:
            raise ValueError("Flux 2 encoder mid.block_2 must output 512 channels")

        conv_in = getattr(encoder, "conv_in", None)
        conv_out = getattr(encoder, "conv_out", None)
        quant_conv = getattr(encoder, "quant_conv", None)
        if getattr(conv_in, "in_channels", None) != FLUX2_INPUT_CHANNELS:
            raise ValueError("Flux 2 encoder input must be 3-channel RGB")
        if getattr(conv_out, "out_channels", None) != FLUX2_MOMENT_CHANNELS:
            raise ValueError("Flux 2 encoder must emit 64 latent moments")
        if getattr(quant_conv, "in_channels", None) != FLUX2_MOMENT_CHANNELS:
            raise ValueError("Flux 2 quant_conv input must be 64 latent moments")
        if getattr(quant_conv, "out_channels", None) != FLUX2_MOMENT_CHANNELS:
            raise ValueError("Flux 2 quant_conv output must be 64 latent moments")

    @classmethod
    def _validate_autoencoder_contract(cls, vae: nn.Module) -> None:
        cls._validate_encoder_contract(getattr(vae, "encoder", None))
        bn = getattr(vae, "bn", None)
        if (
            bn is None
            or getattr(bn, "num_features", None) != FLUX2_LATENT_CHANNELS
            or getattr(bn, "affine", True)
            or not getattr(bn, "track_running_stats", False)
        ):
            raise ValueError(
                "Flux 2 checkpoint must provide the canonical 128-channel "
                "non-affine BatchNorm latent normalization"
            )

    @staticmethod
    def _resolve_vae_path(vae_path: str) -> str:
        """Resolve only a local matching checkpoint; never download/fallback."""
        if not vae_path:
            raise RuntimeError(
                "VAE-anchor backend unavailable: provide vae_model_path pointing "
                "to a licensed local Flux 2 ae.safetensors checkpoint in the "
                "matching BFL layout. Automatic downloads and unrelated VAE "
                "substitutions are disabled."
            )
        path = os.path.abspath(os.path.expanduser(vae_path))
        if not os.path.isfile(path):
            raise RuntimeError(
                "VAE-anchor backend unavailable: matching Flux 2 checkpoint "
                f"not found at {vae_path!r}. Set vae_model_path to an existing "
                "licensed ae.safetensors file; no automatic download or VAE "
                "fallback is permitted."
            )
        if not path.lower().endswith(".safetensors"):
            raise RuntimeError(
                "VAE-anchor backend unavailable: vae_model_path must be a "
                "matching Flux 2 ae.safetensors checkpoint, not "
                f"{vae_path!r}."
            )
        return path

    def load(self, device: torch.device, dtype: torch.dtype):
        if self._loaded:
            return
        self._vae_path = self._resolve_vae_path(self._vae_path)
        # Reuse the exact BFL Flux2 implementation used by the model loaders.
        # AutoEncoder.load_model performs normal checkpoint/config shape sniffing
        # and strict state-dict loading; do not hand-load a partial encoder.
        from toolkit.models.v2.vae.flux2_kl import AutoEncoder

        try:
            vae = AutoEncoder.load_model(
                self._vae_path,
                dtype=torch.float32,
                device=None,
                use_comfy_weights=False,
            )
            # Validate the loaded module as well as its state dict.  Strict
            # loading only proves that keys fit this class; it does not prove
            # that a replacement VAE has the Flux 2 feature recipe.
            self._validate_autoencoder_contract(vae)
        except Exception as exc:
            raise RuntimeError(
                "VAE-anchor backend unavailable: the local checkpoint is not "
                "compatible with toolkit.models.v2.vae.flux2_kl.AutoEncoder "
                "or failed the strict Flux 2 shape/hook/normalization contract. "
                "Supply the matching licensed Flux 2 ae.safetensors checkpoint; "
                "unrelated VAEs are not supported."
            ) from exc

        if not hasattr(vae, "encoder"):
            raise RuntimeError(
                "VAE-anchor backend unavailable: matching Flux 2 checkpoint did "
                "not produce an encoder component."
            )
        self._encoder = vae.encoder
        # Keep the frozen estimator in its loaded precision/device, while
        # retaining autograd through its forward into generated pixels.
        self._encoder.to(device=device, dtype=dtype).eval()
        self._encoder.requires_grad_(False)
        del vae
        self._register_hooks()
        self._contract_validated = True
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

    def _validate_feature_outputs(self, features: Dict[str, torch.Tensor]) -> None:
        if set(features) != set(FEATURE_LEVELS):
            raise RuntimeError(
                "VAE-anchor backend unavailable: Flux 2 encoder hooks did not "
                f"capture exactly {FEATURE_LEVELS}; got {sorted(features)}."
            )
        for level in FEATURE_LEVELS:
            tensor = features[level]
            expected = LEVEL_CHANNELS[level]
            if tensor.ndim != 4 or tensor.shape[1] != expected:
                raise RuntimeError(
                    "VAE-anchor backend unavailable: Flux 2 feature "
                    f"{level} must be (B,{expected},H,W), got {tuple(tensor.shape)}."
                )

    def encode_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode pixels in [-1, 1] (B,3,H,W); return (final, features_dict).

        Gradients flow through the encoder to the input pixels.  The captured
        features are pre-latent-normalization activations, matching the
        canonical encoder blocks and the cached-reference path.
        """
        assert self._loaded, "Call load() first"
        self._features.clear()
        enc_dtype = next(self._encoder.parameters()).dtype
        final = self._encoder(x.to(dtype=enc_dtype))
        features = {k: v for k, v in self._features.items()}
        if self._contract_validated:
            self._validate_feature_outputs(features)
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
