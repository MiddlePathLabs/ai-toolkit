"""H3 target/condition encodes upcast only the video VAE encoder, then restore it."""
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE


def _stub():
    vae = SimpleNamespace(
        encoder=torch.nn.Linear(4, 4).half(),
        quant_conv=torch.nn.Conv1d(4, 4, 1).half(),
        decoder=torch.nn.Linear(4, 4).half(),
    )
    vae._encoder_in_fp32 = MiniMaxH3VideoVAE._encoder_in_fp32.__get__(vae)
    return vae


def test_encoder_runs_fp32_and_is_restored():
    vae = _stub()
    with vae._encoder_in_fp32():
        assert vae.encoder.weight.dtype == torch.float32
        assert vae.quant_conv.weight.dtype == torch.float32
        assert vae.decoder.weight.dtype == torch.float16  # decoder untouched
    assert vae.encoder.weight.dtype == torch.float16
    assert vae.quant_conv.weight.dtype == torch.float16


def test_restore_is_lossless_and_survives_errors():
    vae = _stub()
    before = vae.encoder.weight.detach().clone()
    try:
        with vae._encoder_in_fp32():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert vae.encoder.weight.dtype == torch.float16
    assert torch.equal(vae.encoder.weight, before)


def test_fp32_storage_is_left_alone():
    vae = _stub()
    vae.encoder.float()
    vae.quant_conv.float()
    with vae._encoder_in_fp32():
        pass
    assert vae.encoder.weight.dtype == torch.float32
