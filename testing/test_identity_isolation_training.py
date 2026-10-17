"""Synthetic CPU integration: production prior/student paths and optimizer windows.

No model weights, assets, downloads, training datasets or inference are required.
"""

from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from testing.test_trainer_lifecycle import _batch, _trainer_for_real_hook
from toolkit.config_modules import TrainConfig
from toolkit.identity_isolation import IsolationConfig
from toolkit.identity_isolation_training import (
    dop_selected,
    isolation_batch_category,
    validate_isolation_optimizer,
    validate_isolation_process,
    validate_isolation_training,
)
from toolkit.optimizer_runtime import OptimizerRuntimeAdapter
from toolkit.optimizers.rose import Rose
from toolkit.prompt_utils import PromptEmbeds


def _config(**overrides):
    raw = {"preset": "dop_only", "dop_interval": 2, **overrides}
    return IsolationConfig(raw)


def _category(batch, category):
    batch.file_items = [
        SimpleNamespace(dataset_config=SimpleNamespace(isolation_category=category))
        for _ in batch.file_items
    ]
    return batch


class _PredictionModel:
    def __init__(self, network):
        self.network = network
        self.unet = torch.nn.Linear(1, 1, bias=False)
        self.unet.weight.requires_grad_(False)
        self.unet.weight.data.fill_(2.0)
        self.lora = torch.nn.Parameter(torch.tensor(0.5))
        self.calls = []
        self.fail = False

    def predict_noise(self, **kwargs):
        self.calls.append(
            (
                self.network.is_active,
                torch.is_grad_enabled(),
                self.unet.training,
                kwargs,
            )
        )
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        value = kwargs["latents"] * self.unet.weight.reshape(1)
        value = (
            value
            + kwargs["conditional_embeddings"].text_embeds.sum()
            + kwargs["marker"]
        )
        return value + self.lora if self.network.is_active else value


def _prediction_trainer(isolation=True):
    trainer = object.__new__(SDTrainer)
    trainer.network = SimpleNamespace(is_active=True)
    trainer.sd = _PredictionModel(trainer.network)
    trainer.train_config = TrainConfig(dtype="float32")
    trainer.device_torch = torch.device("cpu")
    trainer.adapter = None
    trainer.embedding = None
    trainer.isolation_config = _config() if isolation else None
    trainer.additional_logs = {}
    return trainer


def _paired_pass(trainer):
    noisy = torch.tensor([[1.0, 2.0]], requires_grad=True)
    embeds = PromptEmbeds(torch.tensor([[[3.0]]], requires_grad=True))
    timesteps = torch.tensor([500.0])
    noise = torch.tensor([[0.2, -0.3]])
    batch = SimpleNamespace(file_items=[object()])
    kwargs = {"marker": torch.tensor(0.25)}
    teacher = trainer.get_prior_prediction(
        noisy_latents=noisy,
        conditional_embeds=embeds,
        match_adapter_assist=False,
        network_weight_list=[1.0],
        timesteps=timesteps,
        pred_kwargs=kwargs,
        batch=batch,
        noise=noise,
    )
    student, loss = trainer._preservation_prediction_loss(
        noisy_latents=noisy,
        timesteps=timesteps,
        preservation_embeds=embeds,
        unconditional_embeds=None,
        batch=batch,
        pred_kwargs=kwargs,
        prior_pred=teacher,
        multiplier=2.0,
    )
    return teacher, student, loss


def test_real_prior_and_student_match_inputs_and_lora_gradient_state():
    trainer = _prediction_trainer()
    teacher, student, loss = _paired_pass(trainer)
    assert not teacher.requires_grad
    assert student.requires_grad
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert trainer.sd.lora.grad.item() == pytest.approx(2.0)
    assert trainer.sd.unet.weight.grad is None
    reference, adapted = trainer.sd.calls
    assert reference[:3] == (False, False, False)
    assert adapted[:2] == (True, True)
    for key in ("latents", "timestep", "marker"):
        assert torch.equal(reference[3][key], adapted[3][key])
    assert torch.equal(
        reference[3]["conditional_embeddings"].text_embeds,
        adapted[3]["conditional_embeddings"].text_embeds,
    )
    for key in (
        "guidance_scale",
        "guidance_embedding_scale",
        "rescale_cfg",
        "detach_unconditional",
        "bypass_guidance_embedding",
    ):
        assert reference[3][key] == adapted[3][key]
    assert trainer.network.is_active
    assert trainer.sd.unet.training
    assert trainer.additional_logs["isolation/preservation_raw"] == pytest.approx(0.25)
    assert trainer.additional_logs["isolation/preservation_weighted"] == pytest.approx(
        0.5
    )
    assert trainer.additional_logs["isolation/preservation_pass_count"] == 1


@pytest.mark.parametrize("training,active", [(True, True), (False, False)])
def test_reference_exception_restores_network_and_unet(training, active):
    trainer = _prediction_trainer()
    trainer.sd.unet.train(training)
    trainer.network.is_active = active
    trainer.sd.fail = True
    with pytest.raises(RuntimeError, match="synthetic forward"):
        _paired_pass(trainer)
    assert trainer.sd.unet.training is training
    assert trainer.network.is_active is active
    assert trainer.sd.lora.grad is None


def test_legacy_preservation_uses_same_production_loss_without_isolation_logs():
    trainer = _prediction_trainer(isolation=False)
    _, _, loss = _paired_pass(trainer)
    assert loss.item() == pytest.approx(0.5)
    assert trainer.additional_logs == {"loss/preservation": pytest.approx(0.5)}
    assert all(dop_selected(None, step, True) for step in range(5))
    assert not dop_selected(None, 0, False)


def test_global_preservation_caption_and_legacy_dataset_replacement():
    trainer = _prediction_trainer()
    trainer.isolation_config = _config(
        subjects=[
            {"folder_path": "one", "trigger": "alice", "class_name": "woman"},
            {"folder_path": "two", "trigger": "bob", "class_name": "man"},
        ],
        fused_captions=True,
    )
    items = [SimpleNamespace(trigger_word="alice")]
    assert trainer._preservation_prompts(["alicewoman and bob"], items) == [
        "a woman and a man"
    ]
    trainer.isolation_config = None
    trainer.trigger_word = "global"
    trainer.train_config.diff_output_preservation_class = "person"
    assert trainer._preservation_prompts(["alice and bob"], items) == ["person and bob"]


def test_cached_path_requires_per_item_conditioning_but_legacy_keeps_fallback():
    trainer = _prediction_trainer()
    trainer.cached_dop_class_embeds = PromptEmbeds(torch.ones(1, 1, 1))
    batch = SimpleNamespace(file_items=[object(), object()], dop_prompt_embeds=None)
    with pytest.raises(ValueError, match="per-item"):
        trainer._cached_preservation_embeds(batch, torch.float32)
    batch.dop_prompt_embeds = PromptEmbeds(torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="one preservation conditioning"):
        trainer._cached_preservation_embeds(batch, torch.float32)
    batch.dop_prompt_embeds = PromptEmbeds(torch.ones(2, 1, 1, requires_grad=True))
    cached = trainer._cached_preservation_embeds(batch, torch.float32)
    assert cached.text_embeds.shape[0] == 2
    assert not cached.text_embeds.requires_grad
    trainer.isolation_config = None
    batch.dop_prompt_embeds = None
    assert (
        trainer._cached_preservation_embeds(batch, torch.float32)
        is trainer.cached_dop_class_embeds
    )


class _SkipOnce:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.step_was_skipped = False
        self.skip_next = True

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.step_was_skipped = self.skip_next
        self.skip_next = False
        if not self.step_was_skipped:
            self.optimizer.step()


def test_real_hook_stride_skip_resume_and_no_fake_skipped_loss():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SkipOnce(torch.optim.AdamW(model.parameters(), lr=0.01))
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.additional_logs = {}
    trainer.isolation_config = _config()
    trainer.train_config.diff_output_preservation = True
    batch = _category(_batch([1.0], [1.0]), "normal")
    selected = []
    original = trainer.train_single_accumulation

    def train(item, accum_scale=1.0):
        selected.append(trainer._isolation_dop_selected)
        return original(item, accum_scale)

    trainer.train_single_accumulation = train
    trainer.hook_train_loop(batch)
    assert trainer.completed_update_id == 0
    assert trainer.lr_scheduler.calls == 0
    trainer.hook_train_loop(batch)
    assert trainer.completed_update_id == 1
    trainer.additional_logs["isolation/preservation_raw"] = 9.0
    trainer.additional_logs["isolation/preservation_weighted"] = 18.0
    trainer.hook_train_loop(batch)
    assert "isolation/preservation_raw" not in trainer.additional_logs
    assert "isolation/preservation_weighted" not in trainer.additional_logs
    assert trainer.additional_logs["isolation/preservation_pass_count"] == 0
    trainer.hook_train_loop(batch)
    assert selected == [True, True, False, True]
    # Restoring the checkpoint's successful-update counter restores its phase.
    trainer.completed_update_id = 1
    trainer._prepare_isolation_window([batch], window_start=True)
    assert not trainer._isolation_dop_selected
    trainer.completed_update_id = 2
    trainer._prepare_isolation_window([batch], window_start=False)
    assert not trainer._isolation_dop_selected
    trainer._prepare_isolation_window([batch], window_start=True)
    assert trainer._isolation_dop_selected


def test_production_preservation_loss_uses_real_hook_sample_normalization():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.isolation_config = _config()
    trainer.train_config.diff_output_preservation = True
    trainer.additional_logs = {}
    trainer.predict_noise = lambda **kwargs: model(kwargs["noisy_latents"])
    batches = [
        _category(_batch([1.0], [0.0]), "normal"),
        _category(_batch([2.0, 3.0, 4.0], [0.0] * 3), "normal"),
    ]

    def train(batch, accum_scale=1.0):
        _, loss = trainer._preservation_prediction_loss(
            noisy_latents=batch.x,
            timesteps=torch.ones(len(batch.x)),
            preservation_embeds=None,
            unconditional_embeds=None,
            batch=batch,
            pred_kwargs={},
            prior_pred=torch.zeros_like(batch.x),
            multiplier=2.0,
        )
        trainer._last_objective_eligible = True
        (loss * accum_scale).backward()
        return loss.detach()

    trainer.train_single_accumulation = train
    trainer.hook_train_loop(batches)
    assert model.weight.grad.item() == pytest.approx(30.0)
    assert trainer.additional_logs["isolation/preservation_raw"] == pytest.approx(7.5)
    assert trainer.additional_logs["isolation/preservation_weighted"] == pytest.approx(
        15.0
    )
    assert trainer.additional_logs["isolation/preservation_pass_count"] == 2


@pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, Rose])
def test_composite_runtime_actual_step_group_ratio_and_scheduler_restoration(
    optimizer_cls,
):
    params = [
        torch.nn.Parameter(torch.tensor([1.0, -1.0])),
        torch.nn.Parameter(torch.tensor([2.0, -2.0])),
    ]
    groups = [{"params": [params[0]], "lr": 0.01}, {"params": [params[1]], "lr": 0.003}]
    optimizer = optimizer_cls(groups, lr=0.01, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    runtime = OptimizerRuntimeAdapter.inspect(optimizer)
    config = _config(composite_lr_multiplier=2.0)
    validate_isolation_optimizer(config, optimizer, runtime)
    trainer = SimpleNamespace(
        optimizer=optimizer,
        optimizer_runtime=runtime,
        isolation_config=config,
        _isolation_window_category="composite",
        train_config=TrainConfig(),
        sd=SimpleNamespace(),
        category_stop=None,
        modality_router=None,
        additional_logs={},
        _remember_routing_active_params=lambda _: None,
    )
    for parameter in params:
        parameter.grad = torch.tensor([1.0, -2.0])
    before = [parameter.detach().clone() for parameter in params]
    baseline_params = [torch.nn.Parameter(value.clone()) for value in before]
    baseline = optimizer_cls(
        [
            {"params": [baseline_params[0]], "lr": 0.01},
            {"params": [baseline_params[1]], "lr": 0.003},
        ],
        lr=0.01,
        weight_decay=0.0,
    )
    for parameter in baseline_params:
        parameter.grad = torch.tensor([1.0, -2.0])
    baseline.step()
    assert BaseSDTrainProcess._open_optimizer_runtime_window(trainer, phase="step")
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.02, 0.006]
    )
    try:
        optimizer.step()
    finally:
        runtime.end_window(optimizer)
    assert all(not torch.equal(old, new) for old, new in zip(before, params))
    for old, scaled, normal in zip(before, params, baseline_params):
        assert torch.allclose(scaled - old, 2.0 * (normal - old), atol=2e-7)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.01, 0.003]
    )
    assert trainer.additional_logs["isolation/effective_lr_group_0"] == pytest.approx(
        0.02
    )
    assert trainer.additional_logs["isolation/lr_multiplier"] == 2.0
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.005, 0.0015]
    )
    assert [group["initial_lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.01, 0.003]
    )
    assert runtime._window_open is False
    assert not hasattr(optimizer, "_runtime_step_scale")


def test_mixed_category_rejected_by_actual_hook_before_backward():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.isolation_config = _config()
    trainer.additional_logs = {}
    trainer.train_config.diff_output_preservation = True
    batches = [
        _category(_batch([1.0], [0.0]), "normal"),
        _category(_batch([1.0], [0.0]), "composite"),
    ]
    with pytest.raises(ValueError, match="homogeneous"):
        trainer.hook_train_loop(batches)
    assert model.weight.grad is None
    with pytest.raises(ValueError, match="homogeneous"):
        isolation_batch_category(
            trainer.isolation_config, [batches[1]], previous="normal"
        )


def test_unsupported_actual_optimizer_rejected_even_if_label_says_adamw():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    runtime = OptimizerRuntimeAdapter.inspect(optimizer, optimizer_type="adamw")
    with pytest.raises(ValueError, match="actual torch AdamW or toolkit Rose"):
        validate_isolation_optimizer(_config(), optimizer, runtime)
    validate_isolation_optimizer(None, optimizer, runtime)
    actual = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    fused = OptimizerRuntimeAdapter.inspect(actual)
    fused.update_phase = "backward"
    with pytest.raises(ValueError, match="fused-backward"):
        validate_isolation_optimizer(_config(), actual, fused)


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_text_encoder", True),
        ("loss_type", "mean_flow"),
        ("do_guidance_loss", True),
        ("inverted_mask_prior", True),
        ("correct_pred_norm", True),
        ("blank_prompt_preservation", True),
    ],
)
def test_unsafe_reference_objectives_fail_explicitly(field, value):
    train = TrainConfig(diff_output_preservation=True)
    setattr(train, field, value)
    trainer = SimpleNamespace(
        isolation_config=_config(),
        train_config=train,
        model_config=SimpleNamespace(arch="krea2", model_kwargs={}),
        network_config=SimpleNamespace(type="lora"),
        config={},
    )
    with pytest.raises(ValueError, match="identity_isolation"):
        validate_isolation_training(trainer)


def test_composite_accumulation_restriction_and_fusion_only_legacy_dop():
    config = _config(
        dop_enabled=False,
        fused_captions=True,
        subjects=[
            {"folder_path": "one", "trigger": "alice", "class_name": "woman"},
        ],
    )
    assert all(dop_selected(config, update, True) for update in range(4))
    trainer = SimpleNamespace(
        isolation_config=_config(
            composites=True,
            stranger_asset_dir="assets",
            subjects=[
                {"folder_path": "one", "trigger": "alice", "class_name": "woman"},
            ],
        ),
        train_config=TrainConfig(gradient_accumulation=2),
        model_config=SimpleNamespace(arch="krea2", model_kwargs={}),
        network_config=SimpleNamespace(type="lora"),
        config={},
    )
    with pytest.raises(ValueError, match="gradient_accumulation=1"):
        validate_isolation_training(trainer)


def test_first_and_regular_preview_prompts_are_fused_before_cache_encoding():
    from toolkit.config_modules import SampleConfig

    trainer = object.__new__(SDTrainer)
    trainer.train_config = TrainConfig()
    trainer.isolation_config = _config(
        fused_captions=True,
        subjects=[
            {"folder_path": "one", "trigger": "alice", "class_name": "woman"},
        ],
    )
    trainer.sample_config = SampleConfig(prompts=["alice in a coat"])
    trainer.first_sample_config = SampleConfig(prompts=["alice smiling"])
    trainer.save_root = "."
    trainer.trigger_word = None
    encoded = []

    def encode(prompt):
        encoded.append(prompt)
        return PromptEmbeds(torch.ones(1, 1, 1))

    trainer.sd = SimpleNamespace(
        encode_control_in_text_embeddings=False, encode_prompt=encode
    )
    trainer._static_embed_cache_path = lambda *_: None
    trainer._load_static_embed = lambda _: None
    trainer._save_static_embed = lambda *_: None
    trainer.cache_sample_prompts()
    assert "alicewoman in a coat" in encoded
    assert "alicewoman smiling" in encoded
    assert len(trainer.sd.sample_prompts_cache) == 1
    assert len(trainer._isolation_first_sample_cache) == 1
    regular = trainer.sd.sample_prompts_cache
    first = trainer._isolation_first_sample_cache

    def sample(**_kwargs):
        assert trainer.sd.sample_prompts_cache is first
        raise RuntimeError("sample failure")

    trainer._sample = sample
    with pytest.raises(RuntimeError, match="sample failure"):
        trainer.sample(is_first=True)
    assert trainer.sd.sample_prompts_cache is regular


@pytest.mark.parametrize(
    "model",
    [
        {"arch": "krea2:turbo"},
        {"arch": "krea2:o_edit"},
        {"arch": "krea2", "name_or_path": "krea/Krea-2-Turbo"},
    ],
)
def test_non_raw_declared_roles_are_rejected(model):
    trainer = SimpleNamespace(
        isolation_config=_config(),
        train_config=TrainConfig(),
        model_config=SimpleNamespace(arch="krea2", model_kwargs={}),
        network_config=SimpleNamespace(type="lora"),
        config={"model": model},
    )
    with pytest.raises(ValueError, match="RAW"):
        validate_isolation_training(trainer)


def test_process_preflight_shares_guards_without_preparing_assets_or_mutation():
    process = {
        "model": {"arch": "krea2", "name_or_path": "local-declared-raw"},
        "network": {"type": "lora"},
        "train": {"optimizer": "adamw"},
        "datasets": [{"folder_path": "does-not-exist"}],
    }
    validate_isolation_process(process, _config())
    assert process["train"] == {"optimizer": "adamw"}
    process["model"]["arch"] = "krea2:turbo"
    with pytest.raises(ValueError, match="RAW"):
        validate_isolation_process(process, _config())
    process["model"]["arch"] = "krea2"
    process["train"]["optimizer"] = "adamw8bit"
    with pytest.raises(ValueError, match="configured optimizer"):
        validate_isolation_process(process, _config())
    process["train"]["optimizer"] = "adamw"
    process["datasets"][0]["is_reg"] = True
    with pytest.raises(ValueError, match="is_reg/prior_reg"):
        validate_isolation_process(process, _config())
    validate_isolation_process({}, None)
