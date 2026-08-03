from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig
from toolkit.lora_special import LoRAModule, LoRASpecialNetwork


class FakeNetwork:
    network_type = "lora"


def make_trainer(params, **train_kwargs):
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.params = params
    trainer.train_config = TrainConfig(**train_kwargs)
    trainer.step_num = 0
    trainer.optimizer = None
    return trainer


def test_train_config_defaults_keep_noising_disabled():
    config = TrainConfig()

    assert config.weight_noise.enabled is False
    assert config.weight_noise.mode == "relative"
    assert config.weight_noise.sigma == pytest.approx(0.00125)
    assert config.weight_noise.bound_norm is False
    assert config.weight_noise.log_every == 50
    assert config.gradient_noise.enabled is False
    assert config.gradient_noise.mode == "neelakantan"
    assert config.gradient_noise.sigma == pytest.approx(0.001)
    assert config.gradient_noise.eta == pytest.approx(0.01)
    assert config.gradient_noise.gamma == pytest.approx(0.55)
    assert config.gradient_noise.log_every == 50


def test_noising_config_rejects_unknown_modes():
    with pytest.raises(ValueError, match="weight noise mode"):
        TrainConfig(weight_noise={"mode": "unknown"})

    with pytest.raises(ValueError, match="gradient noise mode"):
        TrainConfig(gradient_noise={"mode": "unknown"})


def test_lora_module_tags_projection_parameters():
    module = LoRAModule(
        "test",
        torch.nn.Linear(4, 3),
        network=FakeNetwork(),
        lora_dim=2,
        use_bias=True,
    )

    assert module.lora_down.weight._is_lora is True
    assert module.lora_up.weight._is_lora is True
    assert module.lora_up.bias._is_lora is True


def test_network_registration_tags_every_adapter_parameter():
    class Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class UntaggedAdapter(torch.nn.Module):
        def __init__(self, name, _module, *args, **kwargs):
            super().__init__()
            self.lora_name = name
            self.lora_down = torch.nn.Linear(4, 2, bias=False)
            self.lora_up = torch.nn.Linear(2, 4, bias=False)
            self.extra = torch.nn.Parameter(torch.ones(1))

    network = LoRASpecialNetwork(
        text_encoder=[],
        unet=Transformer2DModel(),
        train_text_encoder=False,
        module_class=UntaggedAdapter,
    )

    assert network.unet_loras
    assert all(
        getattr(parameter, "_is_lora", False)
        for adapter in network.unet_loras
        for parameter in adapter.parameters()
    )


def test_gradient_noise_only_changes_tagged_gradients():
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.ones(8))
    tagged._is_lora = True
    untagged = torch.nn.Parameter(torch.ones(8))
    tagged.grad = torch.ones_like(tagged)
    untagged.grad = torch.ones_like(untagged)

    trainer = make_trainer(
        [tagged, untagged],
        gradient_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.5,
            "log_every": 1,
        },
    )
    before_untagged = untagged.grad.clone()

    trainer._inject_gradient_noise()

    assert not torch.equal(tagged.grad, torch.ones_like(tagged))
    assert torch.equal(untagged.grad, before_untagged)
    assert trainer._last_grad_noise_norm > 0
    assert trainer._last_grad_noise_snr > 0


@pytest.mark.parametrize(
    ("mode", "values"),
    [
        ("relative", {"sigma": 0.25}),
        ("neelakantan", {"eta": 0.25, "gamma": 0.55}),
    ],
)
def test_gradient_noise_modes_modify_tagged_gradient(mode, values):
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.ones(8))
    tagged._is_lora = True
    tagged.grad = torch.ones_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={"enabled": True, "mode": mode, "log_every": 1, **values},
    )

    trainer._inject_gradient_noise()

    assert not torch.equal(tagged.grad, torch.ones_like(tagged))
    assert trainer._last_grad_noise_norm > 0


def test_weight_noise_targets_tagged_parameters_and_bounds_norm():
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    tagged._is_lora = True
    untagged = torch.nn.Parameter(torch.ones(4))
    before_tagged = tagged.detach().clone()
    before_untagged = untagged.detach().clone()
    before_norm = before_tagged.norm()

    trainer = make_trainer(
        [tagged, untagged],
        weight_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.5,
            "bound_norm": True,
            "log_every": 1,
        },
    )

    trainer._inject_weight_noise()

    assert not torch.equal(tagged, before_tagged)
    assert torch.equal(untagged, before_untagged)
    assert tagged.detach().norm().item() == pytest.approx(before_norm.item(), rel=1e-5)
    assert trainer._last_weight_noise_norm > 0
    assert trainer._last_weight_norm == pytest.approx(before_norm.item())


def test_weight_noise_logs_nothing_when_cadence_is_zero():
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.ones(8))
    tagged._is_lora = True
    trainer = make_trainer(
        [tagged],
        weight_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.5,
            "log_every": 0,
        },
    )

    trainer._inject_weight_noise()

    assert getattr(trainer, "_last_weight_noise_norm", None) is None
    assert getattr(trainer, "_last_weight_norm", None) is None


def test_relative_weight_noise_leaves_zero_initialized_parameters_unchanged():
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.zeros(8))
    tagged._is_lora = True
    trainer = make_trainer(
        [tagged],
        weight_noise={
            "enabled": True,
            "mode": "relative",
            "sigma": 0.5,
            "log_every": 1,
        },
    )

    trainer._inject_weight_noise()

    assert torch.equal(tagged, torch.zeros_like(tagged))
    assert trainer._last_weight_noise_norm == 0.0


def test_gradient_noise_logs_nothing_when_cadence_is_zero():
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.ones(8))
    tagged._is_lora = True
    tagged.grad = torch.ones_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.5,
            "log_every": 0,
        },
    )

    trainer._inject_gradient_noise()

    assert getattr(trainer, "_last_grad_noise_norm", None) is None
    assert getattr(trainer, "_last_grad_noise_snr", None) is None


def test_relative_gradient_noise_leaves_zero_gradients_unchanged():
    torch.manual_seed(7)
    tagged = torch.nn.Parameter(torch.ones(8))
    tagged._is_lora = True
    tagged.grad = torch.zeros_like(tagged)
    trainer = make_trainer(
        [tagged],
        gradient_noise={
            "enabled": True,
            "mode": "relative",
            "sigma": 0.5,
            "log_every": 1,
        },
    )

    trainer._inject_gradient_noise()

    assert torch.equal(tagged.grad, torch.zeros_like(tagged))
    assert trainer._last_grad_noise_norm == 0.0
    assert trainer._last_grad_noise_snr == 0.0


def test_fisher_trace_uses_only_tagged_adam_state():
    tagged = torch.nn.Parameter(torch.ones(4))
    tagged._is_lora = True
    untagged = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([tagged, untagged], lr=0.1)
    tagged.grad = torch.ones_like(tagged)
    untagged.grad = torch.ones_like(untagged)
    optimizer.step()

    trainer = make_trainer([tagged, untagged])
    trainer.optimizer = optimizer
    trainer._record_fisher_trace()

    assert trainer._last_fisher_trace == pytest.approx(
        float(optimizer.state[tagged]["exp_avg_sq"].sum())
    )


class RecordingAccelerator:
    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(params, max_norm)


def make_hook_trainer(*, accumulating=False):
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter._is_lora = True
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.params = [parameter]
    trainer.train_config = TrainConfig(
        optimizer="adamw",
        weight_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.05,
            "log_every": 1,
        },
        gradient_noise={
            "enabled": True,
            "mode": "absolute",
            "sigma": 0.05,
            "log_every": 1,
        },
    )
    trainer.step_num = 0
    trainer.optimizer = torch.optim.AdamW([parameter], lr=0.01)
    trainer.accelerator = RecordingAccelerator()
    trainer.sd = SimpleNamespace(is_multistage=False)
    trainer.steps_this_boundary = 0
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.is_grad_accumulation_step = accumulating
    trainer.adapter = None
    trainer.ema = None
    trainer.embedding = None
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
    trainer.timer = lambda _name: nullcontext()
    trainer.end_of_training_loop = lambda: None

    def train_single_accumulation(_batch):
        parameter.grad = torch.ones_like(parameter)
        return torch.tensor(1.25)

    trainer.train_single_accumulation = train_single_accumulation
    return trainer, parameter


def test_hook_train_loop_emits_and_clears_all_noising_metrics():
    trainer, _parameter = make_hook_trainer()

    loss_dict = trainer.hook_train_loop(object())

    assert loss_dict["loss"] == pytest.approx(1.25)
    assert loss_dict["grad_noise_norm"] > 0
    assert loss_dict["grad_noise_snr"] > 0
    assert loss_dict["weight_noise_norm"] > 0
    assert loss_dict["weight_norm"] > 0
    assert loss_dict["fisher_trace"] > 0
    for metric_name in (
        "_last_grad_noise_norm",
        "_last_grad_noise_snr",
        "_last_weight_noise_norm",
        "_last_weight_norm",
        "_last_fisher_trace",
    ):
        assert getattr(trainer, metric_name, None) is None


def test_hook_train_loop_defers_noising_during_gradient_accumulation():
    trainer, parameter = make_hook_trainer(accumulating=True)
    before = parameter.detach().clone()

    loss_dict = trainer.hook_train_loop(object())

    assert torch.equal(parameter, before)
    assert list(loss_dict) == ["loss"]
