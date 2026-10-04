"""Behavioral regressions for H3 conditioning, audio rows, and presence facts."""
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    MinimaxH3FastModel,
    MinimaxH3Model,
)
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.dto import DTO


def _channel_rows(frames: int) -> torch.Tensor:
    left = torch.arange(frames, dtype=torch.float32).unsqueeze(1)
    right = 1000.0 + torch.arange(frames, dtype=torch.float32).unsqueeze(1)
    return torch.cat([left, right], dim=0)


def test_fit_audio_rows_preserves_stereo_boundaries_for_adjacent_lengths():
    # H3 has 158 video latents -> 263 audio latents. Exercise trim, exact
    # fit, and pad so a flattened-row implementation cannot pass by accident.
    target_frames = 263
    for source_frames in (262, 263, 264):
        fitted = packing.fit_audio_rows(_channel_rows(source_frames), target_frames)
        assert fitted.shape == (2 * target_frames, 1)
        expected_left = torch.cat(
            [
                torch.arange(min(source_frames, target_frames), dtype=torch.float32),
                torch.zeros(max(target_frames - source_frames, 0)),
            ]
        )
        expected_right = torch.cat(
            [
                1000.0
                + torch.arange(min(source_frames, target_frames), dtype=torch.float32),
                torch.zeros(max(target_frames - source_frames, 0)),
            ]
        )
        assert torch.equal(fitted[:target_frames, 0], expected_left)
        assert torch.equal(fitted[target_frames:, 0], expected_right)


def test_condition_encoder_uses_fresh_cpu_seed_and_accepts_still_images():
    calls = []

    class RecordingVAE:
        dtype = torch.float16

        def encode(self, frames, **kwargs):
            calls.append((frames.shape, kwargs))
            return torch.randn(frames.shape, generator=kwargs["generator"])

    model = object.__new__(MinimaxH3Model)
    model.vae = SimpleNamespace(
        device=torch.device("cpu"),
        to=lambda device: None,
        video_vae=RecordingVAE(),
    )
    model.vae_device_torch = torch.device("cpu")
    model.vae_encode_fp32 = False
    image = torch.zeros(2, 3, 8, 8)

    torch.manual_seed(1)
    first = model.encode_condition_images(image)
    torch.manual_seed(999)
    second = model.encode_condition_images(image)

    assert torch.equal(first, second)
    assert first.dtype == torch.float32
    assert calls[0][0] == (2, 3, 1, 8, 8)
    assert calls[0][1]["sample"] is True
    assert calls[0][1]["fp16_round"] is True


def test_batch_audio_presence_keeps_later_real_item_when_first_is_silent():
    config = SimpleNamespace(
        num_frames=17,
        auto_frame_count=False,
        load_image_when_caching_latents=False,
        cache_tensors_to_disk=False,
    )

    class Item:
        def __init__(self, audio, cached=False, marker=None):
            self.dataset_config = config
            self.is_latent_cached = cached
            self.is_audio_only = False
            self.extra_values = []
            self.num_frames = 17
            self.tensor = torch.zeros(3, 4, 4)
            self.audio_data = audio
            self.audio_tensor = None
            self.loss_multiplier = 1.0
            self.is_reg = False
            self.prior_reg = False
            self.path = "fixture.png"
            self._latent = (
                DTO(torch.zeros(2, 1), audio_present=torch.tensor(marker))
                if cached
                else None
            )

        def get_latent(self):
            return self._latent

        def __getattr__(self, name):
            return None

    batch = DataLoaderBatchDTO(
        file_items=[Item(None), Item({"waveform": torch.ones(1, 2)})]
    )
    assert batch.audio_data[0] is None
    assert batch.audio_data[1] is not None
    assert batch.audio_present.tolist() == [False, True]
    cached = DataLoaderBatchDTO(
        file_items=[Item(None, cached=True, marker=0), Item(None, cached=True, marker=1)]
    )
    assert cached.audio_present.tolist() == [False, True]
    assert cached.latents.get("audio_present").tolist() == [0.0, 1.0]


def test_fast_video_shift_is_used_by_base_sigma_conversion():
    sigma = torch.tensor([0.2, 0.8])
    base = object.__new__(MinimaxH3Model)
    fast = object.__new__(MinimaxH3FastModel)
    base.video_sigma_shift = 12.0
    fast.video_sigma_shift = 10.0
    assert torch.allclose(
        base.timestep_to_base_sigma(packing.shift_sigma(sigma, 12.0)), sigma
    )
    assert torch.allclose(
        fast.timestep_to_base_sigma(packing.shift_sigma(sigma, 10.0)), sigma
    )


def test_mixed_audio_preparation_encodes_present_rows_once_and_masks_absent():
    calls = []
    model = object.__new__(MinimaxH3Model)
    model._silence_audio_rows = lambda a_lat: torch.zeros(2 * a_lat, 2)

    def encode_audio(items):
        calls.append(items)
        return torch.tensor(
            [[[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0],
              [20.0, 20.0], [21.0, 21.0], [22.0, 22.0], [23.0, 23.0]]]
        )

    model.encode_audio = encode_audio
    class Batch(SimpleNamespace):
        @property
        def audio_latents(self):
            return getattr(self, "_h3_audio_rows", None)

    batch = Batch(
        audio_data=[None, {"waveform": torch.ones(1, 2)}],
        audio_present=torch.tensor([False, True]),
        latents=None,
    )
    rows, target, mask, present, clean = model._prepare_audio_batch(
        batch,
        batch_size=2,
        a_lat=3,
        do_audio=True,
        sigma_a=torch.tensor([0.5, 0.5]),
        device=torch.device("cpu"),
    )
    shared_noise = batch._h3_audio_noise.clone()
    rows_again, target_again, mask_again, present_again, clean_again = (
        model._prepare_audio_batch(
            batch,
            batch_size=2,
            a_lat=3,
            do_audio=True,
            sigma_a=torch.tensor([0.25, 0.25]),
            device=torch.device("cpu"),
        )
    )
    assert len(calls) == 1 and len(calls[0]) == 1
    assert present.tolist() == [False, True]
    assert mask.tolist() == [False, True]
    assert torch.equal(clean[0], torch.zeros(6, 2))
    assert torch.equal(clean[1, :3], torch.tensor([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]))
    assert torch.equal(clean[1, 3:], torch.tensor([[20.0, 20.0], [21.0, 21.0], [22.0, 22.0]]))
    assert target.shape == clean.shape
    assert rows.shape == clean.shape
    assert torch.equal(clean_again, clean)
    assert torch.equal(batch._h3_audio_noise, shared_noise)
    assert target_again.shape == clean_again.shape
    assert rows_again.shape == clean_again.shape
    assert present_again.tolist() == [False, True]
    assert mask_again.tolist() == [False, True]
