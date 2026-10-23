"""CPU-only Fibo reference batching through BaseModel.predict_noise; no weights."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchvision.transforms.functional as TF

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.models.base_model import BaseModel


def _load_fibo_model():
    root = Path(__file__).resolve().parents[1]
    fibo_dir = root / "extensions_built_in/diffusion_models/fibo2"
    reference_helpers = runpy.run_path(str(fibo_dir / "src/references.py"))
    namespace = {
        "torch": torch,
        "TF": TF,
        "F": torch.nn.functional,
        "BaseModel": BaseModel,
        **{name: reference_helpers[name] for name in ("drop_repeats", "prepare_references")},
    }
    source = ast.parse((fibo_dir / "fibo2.py").read_text(encoding="utf-8"))
    model_class = next(
        node for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "Fibo2Model"
    )
    names = {
        "match_target_res", "_normalize_control_images", "_encode_crops",
        "_batch_ref_latents", "get_noise_prediction",
    }
    model_class.body = [
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    pipeline = ast.parse((fibo_dir / "src/pipeline.py").read_text(encoding="utf-8"))
    stack_embeds = next(
        node for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "stack_text_embeds"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            stack_embeds,
            model_class,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(fibo_dir / "fibo2.py"), "exec"),
        namespace,
    )
    return namespace["Fibo2Model"]


# Load real methods without the diffusion-model registry's optional model imports.
Fibo2Model = _load_fibo_model()


class _RecordingTransformer(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.float32

    def forward(self, latents, timestep, embeds, mask, ref_latents):
        self.inputs = SimpleNamespace(
            latents=latents,
            timestep=timestep,
            embeds=embeds,
            ref_latents=ref_latents,
        )
        return torch.zeros_like(latents)


@pytest.fixture
def model():
    model = Fibo2Model.__new__(Fibo2Model)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.model_config = SimpleNamespace(model_kwargs={})
    model.model = _RecordingTransformer()
    model.encoded_images = []

    def encode_images(images, device, dtype):
        assert device == torch.device("cpu")
        assert dtype == torch.float32
        model.encoded_images.extend(images)
        return images[0][:1, :1, :1].unsqueeze(0)

    model.encode_images = encode_images
    return model


def _batch(control_tensor_list=None, control_tensor=None):
    return SimpleNamespace(
        file_items=[object(), object()],
        control_tensor_list=control_tensor_list,
        control_tensor=control_tensor,
    )


def _predict(model, batch, cfg):
    conditional = AdvancedPromptEmbeds(
        text_embeds=[torch.full((1, 1), value) for value in (1.0, 2.0)]
    )
    unconditional = AdvancedPromptEmbeds(
        text_embeds=[torch.full((1, 1), value) for value in (-2.0, -1.0)]
    )
    return model.predict_noise(
        latents=torch.zeros(2, 1, 4, 4),
        conditional_embeddings=conditional,
        unconditional_embeddings=unconditional if cfg else None,
        timestep=torch.tensor([100.0, 200.0]),
        guidance_scale=3.0,
        is_input_scaled=True,
        batch=batch,
    )


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("control_format", ["list", "tensor"])
def test_reference_batches_follow_original_sample_and_cfg_order(
    model, cfg, singleton, control_format
):
    references = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
    if singleton:
        references = references[:1]
    if control_format == "list":
        batch = _batch(control_tensor_list=[[image] for image in references])
    else:
        batch = _batch(control_tensor=references)

    result = _predict(model, batch, cfg)

    expected = [-1.0, -1.0] if singleton else [-1.0, 1.0]
    if cfg:
        expected *= 2
    inputs = model.model.inputs
    torch.testing.assert_close(inputs.ref_latents[0].flatten(), torch.tensor(expected))
    torch.testing.assert_close(
        inputs.embeds.flatten(),
        torch.tensor([-2.0, -1.0, 1.0, 2.0] if cfg else [1.0, 2.0]),
    )
    torch.testing.assert_close(
        inputs.timestep,
        torch.tensor([100.0, 200.0] * (2 if cfg else 1)),
    )
    assert inputs.latents.shape[0] == (4 if cfg else 2)
    assert result.shape == (2, 1, 4, 4)
    assert len(model.encoded_images) == 2


def test_multiple_references_encode_only_original_samples_for_cfg(model):
    black, white = torch.zeros(3, 8, 8), torch.ones(3, 8, 8)
    references = torch.stack([torch.stack([black, white]), torch.stack([white, black])])

    _predict(model, _batch(control_tensor=references), cfg=True)

    first_ref, second_ref = model.model.inputs.ref_latents
    torch.testing.assert_close(first_ref.flatten(), torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    torch.testing.assert_close(second_ref.flatten(), torch.tensor([1.0, -1.0, 1.0, -1.0]))
    assert len(model.encoded_images) == 4


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("references", [None, [[], []]])
def test_batches_without_references_remain_unconditioned(model, cfg, references):
    _predict(model, _batch(control_tensor_list=references), cfg)

    assert model.model.inputs.ref_latents is None
    assert not model.encoded_images


def test_reference_set_count_is_validated_against_original_batch(model):
    batch = _batch(control_tensor_list=[[torch.zeros(3, 8, 8)]] * 3)

    with pytest.raises(ValueError, match="got 3 control image sets for 2 prompts"):
        _predict(model, batch, cfg=True)

    assert not model.encoded_images


@pytest.mark.parametrize("mismatch", ["count", "size"])
def test_incompatible_reference_shapes_still_raise(model, mismatch):
    black, white = torch.zeros(3, 8, 8), torch.ones(3, 8, 8)
    references = [[black], [white, black] if mismatch == "count" else [white]]
    if mismatch == "size":
        model.encode_images = lambda images, **kwargs: torch.zeros(
            1, 1, 1, 2 if images[0][0, 0, 0].item() < 0 else 3
        )

    with pytest.raises(ValueError, match="same number and size of references"):
        _predict(model, _batch(control_tensor_list=references), cfg=True)


def test_latent_batch_must_match_original_or_cfg_batch(model):
    batch = _batch(control_tensor_list=[[torch.zeros(3, 8, 8)]] * 2)

    with pytest.raises(ValueError, match="expected the original batch size or twice"):
        model._batch_ref_latents(batch, 6, (64, 64))

    assert not model.encoded_images
