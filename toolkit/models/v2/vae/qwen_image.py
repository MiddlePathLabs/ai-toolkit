import torch
from contextlib import contextmanager

from diffusers import AutoencoderKLQwenImage

from .._mixin import OstrisModelMixin


class QwenImageVAE(AutoencoderKLQwenImage, OstrisModelMixin):
    """The Qwen-Image (wan-style video) VAE, shared by the qwen_image family,
    nucleus_image and krea2."""

    aitk_subfolder = "vae"

    # A still (single-frame) pass has no next frame that could reuse the temporal
    # feature cache, yet every conv's full-resolution features are held in it for
    # nothing (~2 GiB of extra peak VRAM on an untiled 1024x1024 decode). While a
    # still pass runs, the two cache slots are held at None so every internal
    # feat_cache=self._feat_map passes None and the convs skip caching entirely;
    # videos keep the cache. Same behavior as ComfyUI and Fizgig #369. Implemented
    # here rather than by patching the installed diffusers the base class lives in.
    _still_frame_pass = False

    def clear_cache(self):
        super().clear_cache()
        if self._still_frame_pass:
            self._feat_map = None
            self._enc_feat_map = None

    @contextmanager
    def _no_frame_cache_for_stills(self, num_frames):
        self._still_frame_pass = num_frames == 1
        try:
            yield
        finally:
            self._still_frame_pass = False

    def _encode(self, x):
        with self._no_frame_cache_for_stills(x.shape[2]):
            return super()._encode(x)

    def _decode(self, z, return_dict=True):
        with self._no_frame_cache_for_stills(z.shape[2]):
            return super()._decode(z, return_dict=return_dict)


class QwenImageVAEHolderMixin:
    """BaseModel-side encode_images/decode_latents for models whose self.vae
    is the Qwen-Image VAE: it is a video VAE, so images ride in a single-frame
    dim and latents are normalized with the config's latents_mean/std."""

    # tile the decode when low_vram (decode only; encode stays untiled)
    vae_decode_tiled_on_low_vram = False

    def encode_images(self, image_list, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        images = images.unsqueeze(2)  # add the frame dim
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.squeeze(2)  # drop the frame dim
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents = latents.unsqueeze(2)  # add the frame dim

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        # full-resolution decode spikes VRAM; models opt in to tiling it
        tiled = self.vae_decode_tiled_on_low_vram and self.model_config.low_vram
        if tiled:
            self.vae.enable_tiling()
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()

        images = images.squeeze(2)  # drop the frame dim
        return images.to(device, dtype=dtype)
