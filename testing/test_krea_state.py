"""Production regressions for Krea state, CFG/reference, and adapter semantics."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from extensions_built_in.diffusion_models.krea2.src import pipeline as krea_pipeline

import toolkit.models.base_model as base_model_module
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


@pytest.mark.parametrize("strengths", ([2.0], [2.0, 2.0]))
def test_equal_cfg_strengths_match_legacy_batch_expansion(strengths):
    network, module, _ = _lora_module()
    network.multiplier = list(strengths)
    x = torch.ones(len(strengths) * 2 * 3, 3)

    legacy = module(x)
    with network.semantic_batch(cfg_branches=2):
        cfg = module(x)

    torch.testing.assert_close(cfg, legacy)


def test_network_mixin_starts_inactive_for_early_lycoris_reads():
    network = _Network()

    assert network.is_active is False


def test_production_cfg_prediction_matches_independent_strength_and_gradient_oracle(
    monkeypatch,
):
    from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model

    network = _Network()
    base = nn.Linear(3, 1, bias=False)
    base.weight.requires_grad_(False)
    adapter = LoRAModule(
        "text",
        base,
        lora_dim=2,
        alpha=2,
        network=network,
    )
    network._modules_list = [adapter]
    adapter.apply_to()
    with torch.no_grad():
        adapter.lora_down.weight.fill_(0.5)
        adapter.lora_up.weight.fill_(0.25)
    network.is_active = True

    class TextConditionedDiT(nn.Module):
        config = SimpleNamespace(patch=2, txtlayers=1)

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.0))
            self.gradient_checkpointing = True

        @property
        def device(self):
            return self.scale.device

        @property
        def dtype(self):
            return self.scale.dtype

        def _predict(self, img, context):
            b, text_len = context.shape[:2]
            text = adapter(context.reshape(-1, 3)).reshape(b, text_len, 1)
            return img * self.scale + text.mean(dim=1, keepdim=True)

        def forward(self, img, context, t, **kwargs):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                checkpoint_context = getattr(
                    self, "_aitk_semantic_checkpoint_context_fn", None
                )
                checkpoint_kwargs = {"use_reentrant": False}
                if checkpoint_context is not None:
                    checkpoint_kwargs["context_fn"] = checkpoint_context
                return checkpoint(
                    self._predict, img, context, **checkpoint_kwargs
                )
            return self._predict(img, context)

    class PreparedModel(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        @property
        def config(self):
            return self.module.config

        @property
        def device(self):
            return self.module.device

        @property
        def dtype(self):
            return self.module.dtype

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    transformer = TextConditionedDiT()
    prepared_model = PreparedModel(transformer)
    monkeypatch.setattr(
        base_model_module,
        "unwrap_model",
        lambda candidate: (
            candidate.module
            if isinstance(candidate, PreparedModel)
            else candidate
        ),
    )
    model = object.__new__(Krea2Model)
    model.model = prepared_model
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.is_edit = False
    model.kv_cache = False
    model.model_config = SimpleNamespace(model_kwargs={})
    model.network = network

    conditional = AdvancedPromptEmbeds(
        text_embeds=[
            torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
            torch.tensor([[1.2, 1.3, 1.4], [1.5, 1.6, 1.7]]),
        ]
    )
    unconditional = AdvancedPromptEmbeds(
        text_embeds=[
            torch.tensor([[-0.2, -0.3, -0.4], [-0.5, -0.6, -0.7]]),
            torch.tensor([[-1.2, -1.3, -1.4], [-1.5, -1.6, -1.7]]),
        ]
    )
    timesteps = torch.tensor([100.0, 200.0])
    latents = torch.randn(2, 4, 2, 2)

    network.multiplier = [1.0, 3.0]
    joint_prediction = model.predict_noise(
        latents=latents,
        conditional_embeddings=conditional,
        unconditional_embeddings=unconditional,
        timestep=timesteps,
        guidance_scale=1.25,
        is_input_scaled=True,
    )
    joint_prediction.sum().backward()
    joint_adapter_grads = [
        parameter.grad.detach().clone()
        for parameter in (adapter.lora_down.weight, adapter.lora_up.weight)
    ]
    joint_scale_grad = transformer.scale.grad.detach().clone()

    for parameter in transformer.parameters():
        parameter.grad = None
    adapter.zero_grad(set_to_none=True)
    independent_predictions = []
    for index, strength in enumerate((1.0, 3.0)):
        sample_latents = latents[index:index + 1]
        network.multiplier = [strength]
        independent_predictions.append(
            model.predict_noise(
                latents=sample_latents,
                conditional_embeddings=AdvancedPromptEmbeds(
                    text_embeds=[conditional.text_embeds[index]]
                ),
                unconditional_embeddings=AdvancedPromptEmbeds(
                    text_embeds=[unconditional.text_embeds[index]]
                ),
                timestep=timesteps[index:index + 1],
                guidance_scale=1.25,
                is_input_scaled=True,
            )
        )
        independent_predictions[-1].sum().backward()
    independent_prediction = torch.cat(independent_predictions, dim=0)

    torch.testing.assert_close(joint_prediction, independent_prediction)
    for joint_grad, parameter in zip(
        joint_adapter_grads, (adapter.lora_down.weight, adapter.lora_up.weight)
    ):
        torch.testing.assert_close(joint_grad, parameter.grad)
    torch.testing.assert_close(joint_scale_grad, transformer.scale.grad)
    assert network._cfg_branches is None
    assert not hasattr(prepared_model, "_aitk_semantic_checkpoint_context_fn")
    assert not hasattr(transformer, "_aitk_semantic_checkpoint_context_fn")

    def fail_prediction(*_args, **_kwargs):
        raise RuntimeError("sentinel prediction failure")

    monkeypatch.setattr(transformer, "forward", fail_prediction)
    network.multiplier = [1.0, 3.0]
    with pytest.raises(RuntimeError, match="sentinel prediction failure"):
        model.predict_noise(
            latents=latents,
            conditional_embeddings=conditional,
            unconditional_embeddings=unconditional,
            timestep=timesteps,
            guidance_scale=1.25,
            is_input_scaled=True,
        )
    assert network._cfg_branches is None
    assert not hasattr(prepared_model, "_aitk_semantic_checkpoint_context_fn")
    assert not hasattr(transformer, "_aitk_semantic_checkpoint_context_fn")




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
