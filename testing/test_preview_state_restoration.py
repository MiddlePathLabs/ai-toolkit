"""Preview failures unwind real generation methods without model downloads."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from toolkit.sample_step_hook import install_sample_step_hooks

ROOT = Path(__file__).resolve().parents[1]


def _methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.returns = None
        for argument in method.args.posonlyargs + method.args.args + method.args.kwonlyargs:
            argument.annotation = None
    extracted = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(extracted), str(ROOT / path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _Module(torch.nn.Module):
    def __init__(self, fail, name):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self._device = "cpu"
        self.fail = fail
        self.name = name
        self.text_model = SimpleNamespace(final_layer_norm=SimpleNamespace(weight=self.weight))
        self.conv_in = SimpleNamespace(weight=self.weight)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self.weight.dtype

    def to(self, device=None, dtype=None):
        if device is not None:
            self._device = str(device)
        if dtype is not None:
            self.weight.data = self.weight.data.to(dtype=dtype)
        self.fail(self.name + "_move")
        return self


class _Delta:
    def __init__(self, fail, index):
        self.value = 0.0
        self.fail = fail
        self.index = index

    def merge_in(self, weight):
        self.fail("merge_" + str(self.index))
        self.value += weight

    def merge_out(self, weight):
        self.value -= weight


class _Scheduler:
    def step(self, *args, **kwargs):
        return SimpleNamespace(prev_sample=torch.zeros(1))


@pytest.fixture
def preview(monkeypatch):
    cuda_state = [torch.tensor([11], dtype=torch.uint8), torch.tensor([22], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in cuda_state])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: cuda_state.__setitem__(slice(None), states))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: cuda_state[0].fill_(seed % 256))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: [state.fill_(seed % 256) for state in cuda_state])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    failure = SimpleNamespace(stage=None, error=None, consumed=False)

    def fail(stage):
        if failure.stage == stage and not failure.consumed:
            failure.consumed = True
            torch.rand(1)
            torch.cuda.manual_seed_all(77)
            raise failure.error

    namespace = {"torch": torch, "copy": copy}
    model_source = ROOT / "toolkit/stable_diffusion_model.py"
    error_class = next(
        node for node in ast.parse(model_source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "PreviewStateRestorationError"
    )
    exec(compile(ast.Module(body=[error_class], type_ignores=[]), str(model_source), "exec"), namespace)
    network_methods = _methods(
        "toolkit/network_mixins.py", "ToolkitNetworkMixin", {"merge_in", "merge_out"}, namespace,
    )

    class Network(torch.nn.Module):
        merge_in = network_methods["merge_in"]
        merge_out = network_methods["merge_out"]

        def __init__(self):
            super().__init__()
            self.network_type = "lora"
            self.multiplier = 0.35
            self.is_active = False
            self.is_merged_in = False
            self.can_merge_in = True
            self.deltas = [_Delta(fail, index) for index in range(3)]

        def get_all_modules(self):
            return self.deltas

        def __enter__(self):
            self.is_active = True

        def __exit__(self, *_args):
            self.is_active = False

    class Assistant(_Module):
        def __init__(self):
            super().__init__(fail, "assistant")
            self.is_active = False
            self.multiplier = -1.0

        def force_to(self, device, dtype):
            self.to(device, dtype)

    class Pipeline:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.scheduler = namespace["scheduler"]
            self.vae = namespace["model"].vae
            self.text_encoder_2 = None
            self.tokenizer_2 = None
            fail("pipeline_setup")

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            self.scheduler.step()
            fail("pipeline")
            return SimpleNamespace(images=[object()])

    empty = {name: {} for name in ("vae", "unet", "text_encoder", "adapter", "refiner_unet")}
    adapter_types = {name: type(name, (), {}) for name in (
        "T2IAdapter", "ControlNetModel", "ReferenceAdapter", "CustomAdapter", "IPAdapter",
        "ClipVisionAdapter", "LlamaModel", "T5EncoderModel", "UMT5EncoderModel", "Gemma2Model", "Qwen2Model",
    )}
    namespace.update(adapter_types)
    namespace.update({
        "unwrap_model": lambda model: model,
        "network_base_is_quantized": lambda network: False,
        "flush": lambda: None,
        "print_acc": lambda *_args: None,
        "tqdm": lambda values, **kwargs: values,
        "empty_preset": empty,
        "StableDiffusionPipeline": Pipeline,
        "FluxPipeline": Pipeline,
        "StableDiffusionXLImg2ImgPipeline": lambda **kwargs: fail("refiner_setup"),
    })
    methods = _methods(
        "toolkit/stable_diffusion_model.py", "StableDiffusion",
        {"generate_images", "save_device_state", "restore_device_state", "set_device_state", "set_device_state_preset"},
        namespace,
    )
    Model = type("Model", (), methods)
    model = Model()
    namespace["model"] = model
    namespace["scheduler"] = _Scheduler()
    model.network = Network()
    model.assistant_lora = Assistant()
    model.invert_assistant_lora = True
    model.model_config = SimpleNamespace(assistant_lora_path="assistant", inference_lora_path=None, use_flux_cfg=False)
    model.vae = _Module(fail, "vae")
    model.unet = _Module(fail, "unet")
    model.text_encoder = _Module(fail, "encoder")
    model.adapter = None
    model.refiner_unet = None
    model.noise_scheduler = namespace["scheduler"]
    model.pipeline = None
    model.device_state = {"outer_snapshot": True}
    model.device_torch = "cuda:0"
    model.vae_device_torch = "cuda:0"
    model.te_device_torch = "cuda:0"
    model.torch_dtype = torch.float32
    model.tokenizer = [None, None]
    model.sample_prompts_cache = None
    model.decorator = None
    model._sample_step_index = 31
    model.sample_step_hook = lambda *_args: None
    for name in ("is_xl", "is_v3", "is_pixart", "is_lumina2", "is_auraflow"):
        setattr(model, name, False)
    model.is_flux = True
    model.text_encoder = [model.text_encoder, _Module(fail, "encoder_2")]
    embeds = SimpleNamespace(text_embeds=torch.zeros(1), pooled_embeds=torch.zeros(1))
    embeds.to = lambda *args, **kwargs: embeds

    def encode(*args, **kwargs):
        fail("encode")
        return embeds

    model.encode_prompt = encode
    model._after_sample_image = lambda *_args: fail("after_image")
    config = SimpleNamespace(
        network_multiplier=0.8, adapter_image_path=None, seed=42, prompt="a", prompt_2="a",
        negative_prompt="", negative_prompt_2="", extra_values=[], refiner_start_at=1.0,
        height=32, width=32, num_inference_steps=1, guidance_scale=1.0, latents=None,
        post_process_embeddings=lambda *_args: None,
        save_image_atomic=lambda *_args: fail("save"), log_image=lambda *_args: fail("log"),
    )
    return model, config, failure, cuda_state, Pipeline, namespace


@pytest.mark.parametrize("error_type", [torch.cuda.OutOfMemoryError, ValueError])
@pytest.mark.parametrize("stage", [
    "assistant_move", "unet_move", "merge_0", "merge_1", "vae_move", "encoder_move",
    "pipeline_setup", "refiner_setup", "encode", "pipeline", "save", "log", "after_image",
])
def test_preview_restores_state_after_setup_and_loop_failures(preview, error_type, stage):
    model, config, failure, cuda_state, _pipeline, namespace = preview
    if stage == "refiner_setup":
        model.refiner_unet = _Module(lambda _stage: None, "refiner")
    original_device_state = model.device_state
    rng_state = torch.get_rng_state().clone()
    initial_cuda = [state.clone() for state in cuda_state]
    failure.stage = stage
    failure.error = error_type("injected preview failure")

    with pytest.raises(error_type) as caught:
        model.generate_images([config])

    assert caught.value is failure.error
    assert not model.assistant_lora.is_active
    assert model.assistant_lora.multiplier == -1.0
    assert model.assistant_lora.device == "cpu"
    assert model.assistant_lora.dtype == torch.float64
    assert not model.network.is_merged_in
    assert all(delta.value == pytest.approx(0.0) for delta in model.network.deltas)
    assert model.network.multiplier == 0.35
    assert model.network.training and not model.network.is_active
    assert model.unet.training and model.unet.device == "cpu"
    assert model.unet.weight.requires_grad and model.unet.dtype == torch.float64
    assert all(encoder.training and encoder.device == "cpu" and encoder.weight.requires_grad for encoder in model.text_encoder)
    assert model.vae.training and model.vae.device == "cpu"
    assert model.device_state is original_device_state
    assert model._sample_step_index == 31
    assert "step" not in namespace["scheduler"].__dict__
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert all(torch.equal(actual, expected) for actual, expected in zip(cuda_state, initial_cuda))


@pytest.mark.parametrize("inference", [False, True])
def test_success_restores_original_eval_state_and_preexisting_merge(preview, inference):
    model, config, _failure, _cuda, _pipeline, _namespace = preview
    model.network.eval()
    model.network.merge_in(config.network_multiplier)
    if inference:
        model.model_config.assistant_lora_path = None
        model.model_config.inference_lora_path = "inference"
    model.generate_images([config])
    assert not model.network.training
    assert model.network.is_merged_in
    assert all(delta.value == pytest.approx(config.network_multiplier) for delta in model.network.deltas)
    assert model.network.multiplier == 0.35
    assert not model.assistant_lora.is_active


def test_hook_installation_restores_prior_overrides_and_partial_setup():
    scheduler = _Scheduler()
    previous = lambda *_args, **kwargs: SimpleNamespace(prev_sample=torch.ones(1))
    scheduler.step = previous
    holder = SimpleNamespace(sample_step_hook=lambda *_args: None, noise_scheduler=scheduler)
    unwrap = install_sample_step_hooks(holder, SimpleNamespace(scheduler=scheduler))
    assert scheduler.step is not previous
    unwrap()
    assert scheduler.step is previous

    class RejectingScheduler(_Scheduler):
        def __setattr__(self, name, value):
            if name == "step":
                raise RuntimeError("hook installation failed")
            super().__setattr__(name, value)

    holder.noise_scheduler = RejectingScheduler()
    with pytest.raises(RuntimeError, match="hook installation failed"):
        install_sample_step_hooks(holder, SimpleNamespace(scheduler=scheduler))
    assert scheduler.step is previous


def test_merge_out_failure_retains_only_pending_modules(preview):
    model, config, _failure, _cuda, _pipeline, _namespace = preview
    network = model.network
    network.merge_in(config.network_multiplier)
    pending = network.deltas[1]
    original = pending.merge_out
    pending.merge_out = lambda weight: (_ for _ in ()).throw(ValueError("merge out failed"))
    with pytest.raises(ValueError, match="merge out failed"):
        network.merge_out(config.network_multiplier)
    assert network.is_merged_in
    assert network.deltas[2].value == pytest.approx(0.0)
    pending.merge_out = original
    network.merge_out(config.network_multiplier)
    assert not network.is_merged_in
    assert all(delta.value == pytest.approx(0.0) for delta in network.deltas)


def test_custom_adapter_prompt_state_is_restored_when_encoding_fails(preview):
    model, config, failure, _cuda, _pipeline, namespace = preview

    class Adapter(_Module, namespace["CustomAdapter"]):
        control_lora = None
        is_unconditional_run = True
        num_images = 7

        def condition_prompt(self, prompt, **kwargs):
            return prompt

    model.adapter = Adapter(lambda _stage: None, "adapter")
    failure.stage = "encode"
    failure.error = ValueError("encoding failed")
    with pytest.raises(ValueError, match="encoding failed"):
        model.generate_images([config])
    assert model.adapter.is_unconditional_run
    assert model.adapter.num_images == 7
    assert model.adapter.training and model.adapter.device == "cpu"
    assert model.adapter.weight.requires_grad


def test_reference_resources_and_remaining_state_unwind_even_if_cleanup_raises(preview):
    model, config, failure, _cuda, _pipeline, namespace = preview
    original_scheduler = object()

    class Adapter(_Module, namespace["ReferenceAdapter"]):
        noise_scheduler = original_scheduler
        cleared = False

        def clear_memory(self):
            self.cleared = True
            raise RuntimeError("reference cleanup failed")

    model.adapter = Adapter(lambda _stage: None, "adapter")
    failure.stage = "encode"
    failure.error = ValueError("encoding failed")
    rng_state = torch.get_rng_state().clone()
    with pytest.raises(namespace["PreviewStateRestorationError"], match="state restoration failed") as caught:
        model.generate_images([config])
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "reference cleanup failed"
    assert model.adapter.cleared
    assert model.adapter.noise_scheduler is original_scheduler
    assert model.adapter.training and model.adapter.device == "cpu"
    assert not model.network.is_merged_in
    assert not model.assistant_lora.is_active
    assert model.unet.device == "cpu"
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert "step" not in namespace["scheduler"].__dict__


def test_full_module_bias_allocation_failure_does_not_publish_weight():
    namespace = {"torch": torch, "_dequantize_if_needed": lambda value: value}
    method = _methods(
        "toolkit/lora_special.py", "FullModule", {"merge_in"}, namespace,
    )["merge_in"]
    base = torch.nn.Linear(1, 1)
    original_weight = base.weight.detach().clone()
    original_bias = base.bias.detach().clone()

    class FailingBias:
        def float(self):
            raise torch.cuda.OutOfMemoryError("bias allocation failed")

    module = SimpleNamespace(
        can_merge_in=True, diff=torch.ones_like(base.weight), diff_b=FailingBias(),
        org_module=[base], weight_is_quantized=False,
    )
    with pytest.raises(torch.cuda.OutOfMemoryError, match="bias allocation failed"):
        method(module, merge_weight=0.8)
    assert torch.equal(base.weight, original_weight)
    assert torch.equal(base.bias, original_bias)


@pytest.mark.parametrize("cleanup_stage", ["merge_out", "assistant", "device"])
def test_cleanup_oom_is_fatal_and_remaining_cleanup_still_runs(preview, cleanup_stage):
    model, config, failure, _cuda, _pipeline, namespace = preview
    failure.stage = "encode"
    failure.error = torch.cuda.OutOfMemoryError("generation out of memory")
    cleanup_error = torch.cuda.OutOfMemoryError("cleanup out of memory")

    def fail_cleanup(*args, **kwargs):
        raise cleanup_error

    if cleanup_stage == "merge_out":
        model.network.merge_out = fail_cleanup
    elif cleanup_stage == "assistant":
        original_force_to = model.assistant_lora.force_to

        def force_to(device, dtype):
            if str(device) == "cpu":
                raise cleanup_error
            return original_force_to(device, dtype)

        model.assistant_lora.force_to = force_to
    else:
        model.restore_device_state = fail_cleanup
    rng_state = torch.get_rng_state().clone()
    previous_snapshot = model.device_state

    with pytest.raises(namespace["PreviewStateRestorationError"]) as caught:
        model.generate_images([config])

    assert caught.value.__cause__ is cleanup_error
    assert "out of memory" not in str(caught.value).lower()
    assert model.network.training and model.network.multiplier == 0.35
    assert not model.network.is_active
    assert not model.assistant_lora.is_active
    assert model.device_state is previous_snapshot
    assert model._sample_step_index == 31
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert "step" not in namespace["scheduler"].__dict__
