"""Reference-video soundtracks follow the same timeline as their frames, and
H3 models never use the old LoKr key format."""
import pytest
import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    MinimaxH3Model,
    MinimaxH3Ref2VAModel,
)
from extensions_built_in.diffusion_models.minimax_h3.minimax_h3_image import (
    MinimaxH3ImageModel,
)
from extensions_built_in.diffusion_models.minimax_h3.src.ref_video_cache import (
    fit_ref_audio,
)
from toolkit.config_modules import ModelConfig

SR = 32000
FPS = 24


def _ramp(seconds: float) -> torch.Tensor:
    n = int(seconds * SR)
    return torch.linspace(0.0, 1.0, n).repeat(2, 1)


def test_trim_mode_keeps_real_time_and_cuts_the_tail():
    wave = _ramp(4.0)
    out = fit_ref_audio(wave, SR, num_frames=22, fps=FPS, trim_tail=True)
    assert out.shape[-1] == round(22 / FPS * SR)
    assert torch.equal(out, wave[:, : out.shape[-1]])


def test_untrimmed_mode_stretches_the_whole_soundtrack():
    # frames are spread over the whole 4 s clip, so the soundtrack must cover
    # the whole clip too, not just its first 22/24 s
    wave = _ramp(4.0)
    out = fit_ref_audio(wave, SR, num_frames=22, fps=FPS, trim_tail=False)
    assert out.shape == (2, round(22 / FPS * SR))
    assert float(out[0, 0]) == pytest.approx(0.0, abs=1e-3)
    assert float(out[0, -1]) == pytest.approx(1.0, abs=1e-3)
    assert float(out[0, out.shape[-1] // 2]) == pytest.approx(0.5, abs=1e-2)


def test_untrimmed_mode_leaves_a_matching_length_alone():
    wave = _ramp(22 / FPS)
    out = fit_ref_audio(wave, SR, num_frames=22, fps=FPS, trim_tail=False)
    assert out is wave


@pytest.mark.parametrize(
    "cls, arch",
    [
        (MinimaxH3Model, "minimax_h3"),
        (MinimaxH3Ref2VAModel, "minimax_h3_ref2va"),
        (MinimaxH3ImageModel, "minimax_h3_image"),
    ],
)
def test_h3_models_use_new_lokr_format(cls, arch):
    model = cls("cpu", ModelConfig(arch=arch, name_or_path="x"))
    assert model.use_old_lokr_format is False
