"""D-OPSD teacher loss: musubi-tuner's magnitude/direction split of the MSE."""
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import ModelConfig
from toolkit.h3_dopsd import decomposed_teacher_loss, parse_dopsd_settings, validate_dopsd


def _pt(seed=0, shape=(2, 4, 3, 5, 6)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def test_defaults_equal_plain_mse():
    p, t = _pt()
    per_sample = decomposed_teacher_loss(p, t)
    mse = torch.nn.functional.mse_loss(p, t, reduction="none").reshape(2, -1).mean(1)
    assert torch.allclose(per_sample, mse, atol=1e-6)


def test_direction_gradient_is_rotational():
    p, t = _pt()
    p.requires_grad_(True)
    decomposed_teacher_loss(p, t, mag_weight=0.0).sum().backward()
    # pure direction: the gradient is orthogonal to p, per sample
    radial = (p.grad * p.detach()).reshape(2, -1).sum(1)
    assert torch.allclose(radial, torch.zeros(2), atol=1e-5)


def test_magnitude_optimum_is_the_teacher_norm_not_the_mean():
    # plain MSE against two equally likely teachers pulls the norm toward the
    # (shorter) mean; the split keeps it at the per-sample teacher norm
    t1 = torch.tensor([[1.0, 0.0]]).view(1, 2, 1)
    t2 = torch.tensor([[0.0, 1.0]]).view(1, 2, 1)
    p = (0.5 * (t1 + t2)).clone().requires_grad_(True)
    loss = decomposed_teacher_loss(p, t1) + decomposed_teacher_loss(p, t2)
    loss.sum().backward()
    # at |p| = 0.707 < 1 the split pushes the norm up (plain MSE would sit still)
    assert (p.grad * p.detach()).sum() < 0


def test_dc_weight_zero_ignores_a_colour_cast():
    p, t = _pt()
    cast = torch.randn(2, 4, 1, 1, 1)
    base = decomposed_teacher_loss(p, t, dc_weight=0.0)
    shifted = decomposed_teacher_loss(p + cast, t, dc_weight=0.0)
    assert torch.allclose(base, shifted, atol=1e-5)
    assert not torch.allclose(decomposed_teacher_loss(p + cast, t), decomposed_teacher_loss(p, t))


def test_per_sample_weights():
    p, t = _pt()
    mixed = decomposed_teacher_loss(p, t, torch.tensor([0.0, 1.0]), torch.tensor([0.3, 1.0]))
    assert mixed[1] == pytest.approx(decomposed_teacher_loss(p[1:], t[1:])[0].item(), abs=1e-6)
    assert mixed[0] == pytest.approx(
        decomposed_teacher_loss(p[:1], t[:1], 0.0, 0.3)[0].item(), abs=1e-6
    )


def test_trainer_swap_keeps_existing_weighting_and_shape():
    p, t = _pt()
    plain = torch.nn.functional.mse_loss(p, t, reduction="none")
    weighted = plain * torch.tensor([2.0, 0.5]).view(2, 1, 1, 1, 1)
    settings = SimpleNamespace(loss_mag_weight=1.0, loss_dc_weight=1.0)
    out = SDTrainer._decomposed_teacher_loss(weighted, p, t, settings)
    assert out.shape == weighted.shape
    assert out.mean().item() == pytest.approx(weighted.mean().item(), rel=1e-5)


def _cfg(**kw):
    return ModelConfig(name_or_path="x", arch="minimax_h3_ref2va", model_kwargs=kw)


def test_settings_parse_and_validate():
    s = parse_dopsd_settings(_cfg(dopsd=True, dopsd_loss_mag_weight=0.5, dopsd_loss_dc_weight=0.3))
    assert (s.loss_mag_weight, s.loss_dc_weight) == (0.5, 0.3)
    assert parse_dopsd_settings(_cfg(dopsd=True)).loss_mag_weight == 1.0
    with pytest.raises(ValueError):
        parse_dopsd_settings(_cfg(dopsd=True, dopsd_loss_dc_weight=-1))
    model = SimpleNamespace(get_base_model_version=lambda: "minimax_h3_ref2va")
    with pytest.raises(ValueError, match="loss_type"):
        validate_dopsd(s, arch="minimax_h3_ref2va", model=model,
                       train_config=SimpleNamespace(loss_type="mae", batch_size=1))
