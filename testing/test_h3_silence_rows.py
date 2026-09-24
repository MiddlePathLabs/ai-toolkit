"""No-soundtrack placeholder: VAE-encoded silence, packed like real audio rows."""
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src import packing


class _StubAudioVAE:
    HOP_LENGTH = 800
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def encode(self, waveform):
        self.calls.append(tuple(waveform.shape))
        b, _, samples = waveform.shape
        t = samples // self.HOP_LENGTH
        # a silent waveform maps to a non-zero latent, like the released VAE
        return waveform.new_full((b, 32, t), 0.25) + waveform.mean()


def _model(vae):
    model = object.__new__(MinimaxH3Model)
    model.vae = SimpleNamespace(audio_vae=vae)
    return model


def test_silence_rows_are_encoded_not_zero_and_packed_like_audio():
    vae = _StubAudioVAE()
    rows = _model(vae)._silence_audio_rows(5)
    assert rows.shape == (1, 5 * packing.AUDIO_CHANNELS, 32)
    assert torch.all(rows == 0.25)
    # one mono batch item per stereo channel, a_lat * hop samples of silence
    assert vae.calls == [(packing.AUDIO_CHANNELS, 1, 5 * 800)]


def test_silence_rows_are_cached_per_length():
    vae = _StubAudioVAE()
    model = _model(vae)
    a = model._silence_audio_rows(2)
    b = model._silence_audio_rows(2)
    model._silence_audio_rows(3)
    assert a is b
    assert len(vae.calls) == 2


def test_image_items_get_the_two_frame_placeholder():
    assert packing.audio_latent_num_frames(1) == 2
