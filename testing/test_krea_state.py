"""Production regressions for Krea state, CFG/reference, and adapter semantics."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from extensions_built_in.diffusion_models.krea2.src import pipeline as krea_pipeline
from toolkit.lora_special import LoRAModule
from toolkit.models.base_model import BaseModel
from toolkit.network_mixins import ToolkitNetworkMixin


class _DeviceLinear(nn.Linear):
    @property
    def device(self):
        return self.weight.device


class _Network(ToolkitNetworkMixin):
    network_type = "lora"

    def __init__(self):
        super().__init__()
        self._modules_list = []

    def get_all_modules(self):
        return self._modules_list


def _lora_module(*, module_dropout=None):
    network = _Network()
    base = nn.Linear(3, 2, bias=False)
    base.weight.requires_grad_(False)
    module = LoRAModule(
        "test",
        base,
        lora_dim=2,
        alpha=2,
        module_dropout=module_dropout,
        network=network,
    )
    network._modules_list = [module]
    module.apply_to()
    with torch.no_grad():
        module.lora_down.weight.fill_(0.5)
        module.lora_up.weight.fill_(0.25)
    network.is_active = True
    network.multiplier = [1.0, 3.0]
    return network, module, base


def test_preview_state_restores_mixed_parameter_flags_and_gradients():
    model = _DeviceLinear(2, 2, bias=False)
    model.weight.requires_grad_(False)
    model.bias = nn.Parameter(torch.ones(2), requires_grad=True)

    sd = BaseModel.__new__(BaseModel)
    sd.model = model
    sd.vae = None
    sd.text_encoder = None
    sd.adapter = None
    sd.refiner_unet = None

    sd.save_device_state()
    model.weight.requires_grad_(True)
    model.bias.requires_grad_(False)
    sd.set_device_state(sd.device_state)

    assert model.weight.requires_grad is False
    assert model.bias.requires_grad is True
    out = model(torch.ones(1, 2)).sum()
    out.backward()
    assert model.weight.grad is None
    assert model.bias.grad is not None


def test_legacy_boolean_device_state_remains_supported():
    model = _DeviceLinear(2, 2, bias=False)
    sd = BaseModel.__new__(BaseModel)
    sd.model = model
    sd.vae = None
    sd.text_encoder = None
    sd.adapter = None
    sd.refiner_unet = None

    sd.set_device_state(
        {
            "unet": {
                "training": True,
                "device": torch.device("cpu"),
                "requires_grad": False,
            }
        }
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_strengths_expand_cfg_semantics_before_token_flattening():
    network, module, base = _lora_module()
    x = torch.ones(12, 3)
    base_out = module.org_forward(x)

    with network.semantic_batch(cfg_branches=2):
        out = module(x)

    delta = out - base_out
    per_row = delta[:, 0]
    # B=2 strengths [1, 3], branch-major CFG, three tokens per row.
    expected = torch.tensor([1, 1, 1, 3, 3, 3, 1, 1, 1, 3, 3, 3], dtype=per_row.dtype)
    assert torch.allclose(per_row / per_row[0], expected)


def test_token_expansion_keeps_sample_order_without_cfg_context():
    network, module, base = _lora_module()
    x = torch.ones(6, 3)
    base_out = module.org_forward(x)
    out = module(x)
    delta = out - base_out
    per_row = delta[:, 0]
    expected = torch.tensor([1, 1, 1, 3, 3, 3], dtype=per_row.dtype)
    assert torch.allclose(per_row / per_row[0], expected)


def test_module_dropout_returns_base_and_replays_checkpoint_rng():
    network, module, base = _lora_module(module_dropout=1.0)
    x = torch.ones(2, 3, requires_grad=True)

    expected = base(x)
    out = module(x)
    assert torch.allclose(out, expected)
    assert out.shape == expected.shape

    module.module_dropout = 0.5
    torch.manual_seed(123)
    first = checkpoint(module, x, use_reentrant=False)
    first.sum().backward()
    first_grads = [p.grad.detach().clone() if p.grad is not None else None for p in module.parameters()]

    for parameter in module.parameters():
        parameter.grad = None
    x.grad = None
    torch.manual_seed(123)
    second = checkpoint(module, x, use_reentrant=False)
    second.sum().backward()
    second_grads = [p.grad.detach().clone() if p.grad is not None else None for p in module.parameters()]

    assert torch.equal(first, second)
    for first_grad, second_grad in zip(first_grads, second_grads):
        if first_grad is None:
            assert second_grad is None
        else:
            assert torch.equal(first_grad, second_grad)
    assert base.weight.grad is None


def test_component_load_kwargs_forwards_holder_precision_exclusions():
    class _PrecisionModel(BaseModel):
        def get_quantization_exclude_modules(self):
            return ["first", "last*"]

    sd = _PrecisionModel.__new__(_PrecisionModel)
    sd.device_torch = torch.device("cpu")
    sd.te_device_torch = torch.device("cpu")
    sd.vae_device_torch = torch.device("cpu")
    sd.torch_dtype = torch.float32
    sd.vae_torch_dtype = torch.float32
    sd.model_config = SimpleNamespace(
        quantize=True,
        qtype="qfloat8",
        accuracy_recovery_adapter=None,
        layer_offloading=False,
        low_vram=False,
        model_kwargs={},
        quantize_te=False,
        qtype_te="qfloat8",
        layer_offloading_text_encoder_percent=0.0,
        layer_offloading_transformer_percent=0.0,
    )

    kwargs = sd.component_load_kwargs("transformer")
    assert kwargs["exclude_quant_modules"] == ["first", "last*"]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("legacy_controls", [False, True])
def test_krea_batched_cfg_reference_predictions_and_gradients(batch_size, legacy_controls):
    from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model

    class ReferenceDependentDiT(torch.nn.Module):
        config = SimpleNamespace(patch=2, txtlayers=1)

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        @property
        def device(self):
            return self.scale.device

        def forward(self, img, t, mask, reflen, **kwargs):
            reference_values = img[:, -reflen:].mean(dim=-1)
            reference_mask = mask[:, -reflen:].float()
            reference_mean = (
                (reference_values * reference_mask).sum(dim=1)
                / reference_mask.sum(dim=1)
            )
            return (
                img[:, :-reflen] * self.scale
                + reference_mean[:, None, None]
                + t[:, None, None]
            )

    model = object.__new__(Krea2Model)
    model.model = ReferenceDependentDiT()
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.vae_scale_factor = 1
    model.patch_size = 2
    model.is_edit = True
    model.kv_cache = False
    model.model_config = SimpleNamespace(model_kwargs={})
    model.encode_images = lambda images, **kwargs: torch.cat((images, images[:, :1]), dim=1)
    controls = [[torch.full((3, 4, 2), 0.25)]]
    reference_means = [-0.5]
    if batch_size == 2:
        controls.append([torch.full((3, 2, 4), 0.75), torch.ones(3, 4, 2)])
        reference_means.append(0.75)
    batch = SimpleNamespace(
        file_items=[object() for _ in range(batch_size)],
        control_tensor_list=None if legacy_controls else controls,
        control_tensor=(
            torch.stack([torch.full((3, 4, 2), value) for value in [0.25, 0.75][:batch_size]])
            if legacy_controls else None
        ),
    )
    if legacy_controls and batch_size == 2:
        reference_means[1] = 0.5
    latents = torch.zeros(2 * batch_size, 4, 4, 4, requires_grad=True)
    timesteps = torch.tensor([100.0, 200.0][:batch_size])
    prediction = model.get_noise_prediction(
        latents,
        timesteps,
        SimpleNamespace(text_embeds=[torch.zeros(1, 8) for _ in range(2 * batch_size)]),
        batch=batch,
    )
    expected = (torch.tensor(reference_means) + timesteps / 1000).repeat(2)
    torch.testing.assert_close(prediction, expected[:, None, None, None].expand_as(prediction))
    prediction.sum().backward()
    torch.testing.assert_close(latents.grad, torch.full_like(latents, 2.0))


def test_krea_reference_batch_must_match_latent_batch():
    with pytest.raises(ValueError, match="must match"):
        krea_pipeline.predict_velocity(
            SimpleNamespace(config=SimpleNamespace(patch=2, txtlayers=1)),
            torch.zeros(2, 4, 4, 4),
            torch.ones(2),
            torch.zeros(1, 1, 8),
            torch.ones(1, 1),
            ref_latents=[[]],
        )
