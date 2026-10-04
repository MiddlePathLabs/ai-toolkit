from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


def test_visual_focus_mask_uses_independent_fp32_row_denominators():
    values = torch.tensor([[1.0, 3.0, 99.0], [2.0, 4.0, 6.0]], dtype=torch.float16)
    weights = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float16)
    result, valid = SDTrainer._per_sample_weighted_mean(values, weights)
    assert torch.equal(valid, torch.tensor([True, True]))
    assert result.tolist() == pytest.approx([1.0, 4.0])
def test_subject_region_amplitude_is_not_normalized_away():
    values = torch.tensor([[2.0, 4.0, 100.0]])
    effective_weights = torch.tensor([[1.0, 0.1, 0.0]])
    focus_weights = torch.tensor([[1.0, 1.0, 0.0]])
    result, valid = SDTrainer._per_sample_weighted_mean(
        values, effective_weights, focus_weights
    )
    assert torch.equal(valid, torch.tensor([True]))
    assert result.item() == pytest.approx(1.2)



def test_empty_visual_row_is_not_a_disconnected_nan_objective():
    values = torch.tensor([[float("nan"), 1.0], [0.0, 0.0]])
    weights = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    result, valid = SDTrainer._per_sample_weighted_mean(values, weights)
    assert torch.equal(valid, torch.tensor([False, True]))
    assert result.tolist() == pytest.approx([0.0, 0.0])


def test_joint_audio_presence_and_sample_weight_are_applied_once():
    trainer = SimpleNamespace(
        device_torch=torch.device("cpu"),
        train_config=SimpleNamespace(per_image_adaptive_lr_mode="loss"),
    )
    pred = torch.tensor([[1.0, 3.0], [10.0, 10.0]])
    target = torch.zeros_like(pred)
    rows, eligible = SDTrainer._audio_loss_per_sample(
        trainer,
        pred,
        target,
        torch.tensor([True, False]),
        torch.tensor([2.0, 7.0]),
    )
    assert torch.equal(eligible, torch.tensor([True, False]))
    assert rows.tolist() == pytest.approx([10.0, 0.0])


def test_eligible_nonfinite_audio_fails_causally():
    trainer = SimpleNamespace(
        device_torch=torch.device("cpu"),
        train_config=SimpleNamespace(per_image_adaptive_lr_mode="loss"),
    )
    with pytest.raises(FloatingPointError, match="audio objective"):
        SDTrainer._audio_loss_per_sample(
            trainer,
            torch.tensor([[float("nan")]]),
            torch.zeros(1, 1),
            torch.tensor([True]),
            torch.tensor([1.0]),
        )


class _Timer:
    def __call__(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def start(self, _name):
        return None

    def stop(self, _name):
        return None


class _Accelerator:
    def __init__(self, optimizer=None, *, scaler=None, base_optimizer=None):
        self.unscale_calls = 0
        self.clip_calls = 0
        self.optimizer = optimizer
        self.scaler = scaler
        self.base_optimizer = base_optimizer

    def unscale_gradients(self, _optimizer):
        self.unscale_calls += 1
        if self.scaler is not None:
            self.scaler.unscale_(self.base_optimizer)

    def clip_grad_norm_(self, parameters, max_norm):
        self.clip_calls += 1
        # Accelerate's clipping primitive owns the one AMP unscale.
        self.unscale_gradients(self.optimizer)
        return torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

    def backward(self, loss):
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()


class _Clock:
    def __init__(self):
        self.calls = 0

    def step(self, *_args):
        self.calls += 1

    def update(self):
        self.calls += 1


def _trainer_for_real_hook(
    model, optimizer, *, accumulation_steps=1, scaler=None, base_optimizer=None
):
    trainer = object.__new__(SDTrainer)
    trainer.device_torch = torch.device("cpu")
    trainer.optimizer = optimizer
    trainer.params = list(model.parameters())
    trainer.accelerator = _Accelerator(
        optimizer, scaler=scaler, base_optimizer=base_optimizer
    )
    trainer.timer = _Timer()
    trainer.train_config = SimpleNamespace(
        gradient_accumulation_steps=accumulation_steps,
        optimizer="sgd",
        max_grad_norm=1e9,
        per_image_adaptive_lr_mode="loss",
    )
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.sd = SimpleNamespace(is_multistage=False)
    trainer.embedding = None
    trainer.adapter = None
    trainer.ema = _Clock()
    trainer.lr_scheduler = _Clock()
    trainer.completed_update_id = 0
    trainer.steps_this_boundary = 0
    trainer.current_boundary_index = 0
    trainer.is_grad_accumulation_step = False
    trainer._last_optimizer_update_success = False
    trainer._last_optimizer_update_skipped = False
    trainer._optimizer_window_active = False
    trainer._window_sample_total = 0
    trainer._window_has_objective = False
    trainer._post_step_active_ids = None
    trainer._post_step_active_resolved = False
    trainer.modality_router = None
    trainer.optimizer_runtime = None
    trainer._inject_gradient_noise = lambda: None
    trainer._record_fisher_trace = lambda: None
    trainer._inject_weight_noise = lambda: None
    trainer._last_batches = []

    def train_single(batch, accum_scale=1.0):
        trainer._last_objective_eligible = bool(batch.eligible)
        loss = ((model(batch.x) - batch.y) ** 2).mean()
        if batch.eligible:
            trainer.accelerator.backward(loss * accum_scale)
            if getattr(trainer, "_force_inf_grad", False):
                for parameter in model.parameters():
                    parameter.grad.fill_(float("inf"))
        return loss.detach()

    trainer.train_single_accumulation = train_single
    return trainer


def _batch(x, y, *, eligible=True):
    x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    return SimpleNamespace(
        file_items=[object() for _ in range(x.shape[0])],
        latents=x,
        x=x,
        y=y,
        eligible=eligible,
    )


def test_fractional_masks_preserve_mass_and_exclude_nonfinite_gradient():
    values = torch.tensor([[4.0, float("nan")]], requires_grad=True)
    result, valid = SDTrainer._per_sample_weighted_mean(
        values, torch.tensor([[0.25, 0.0]])
    )
    assert torch.equal(valid, torch.tensor([True]))
    assert result.item() == pytest.approx(4.0)
    result.sum().backward()
    assert values.grad.tolist() == [[1.0, 0.0]]


def test_absent_audio_is_index_excluded_before_subtraction():
    trainer = SimpleNamespace(
        device_torch=torch.device("cpu"),
        train_config=SimpleNamespace(per_image_adaptive_lr_mode="loss"),
    )
    pred = torch.tensor([[2.0], [float("nan")]], requires_grad=True)
    target = torch.zeros_like(pred)
    rows, eligible = SDTrainer._audio_loss_per_sample(
        trainer,
        pred,
        target,
        torch.tensor([True, False]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(eligible, torch.tensor([True, False]))
    rows.sum().backward()
    assert pred.grad.tolist() == [[4.0], [0.0]]


def test_real_sdtrainer_hook_uses_sample_weighted_update_and_update_clocks():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _trainer_for_real_hook(model, optimizer)
    batches = [
        _batch([1.0], [1.0]),
        _batch([2.0, 3.0, 4.0], [0.0, 1.0, 2.0]),
        _batch([5.0, 6.0], [1.0, 1.0]),
    ]
    all_x = torch.cat([item.x for item in batches])
    all_y = torch.cat([item.y for item in batches])
    expected_model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(expected_model.weight, 0.0)
    expected_optimizer = torch.optim.SGD(expected_model.parameters(), lr=0.1)
    expected_optimizer.zero_grad()
    (((expected_model(all_x) - all_y) ** 2).mean()).backward()
    expected_optimizer.step()

    trainer.hook_train_loop(batches)
    assert torch.allclose(model.weight, expected_model.weight)
    assert trainer.completed_update_id == 1
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema.calls == 1
    assert trainer.accelerator.unscale_calls == 1
    assert trainer.accelerator.clip_calls == 1

def test_real_sdtrainer_group_clip_unscales_once_with_native_group_norm():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.params = [{"params": list(model.parameters())}]
    trainer.hook_train_loop([_batch([1.0], [1.0])])
    assert trainer.accelerator.unscale_calls == 1
    assert trainer.accelerator.clip_calls == 0
    assert trainer.completed_update_id == 1



def test_real_sdtrainer_tail_flush_updates_once_without_replaying_input():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _trainer_for_real_hook(model, optimizer, accumulation_steps=2)
    batch = _batch([1.0, 2.0, 3.0], [1.0, 0.0, 1.0])
    trainer.is_grad_accumulation_step = True
    trainer.hook_train_loop([batch])
    assert trainer.completed_update_id == 0
    trainer.is_grad_accumulation_step = False
    trainer.hook_train_loop([])
    assert trainer.completed_update_id == 1
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema.calls == 1


def test_real_sdtrainer_empty_and_numeric_zero_have_distinct_clock_semantics():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.hook_train_loop([_batch([1.0], [1.0], eligible=False)])
    assert trainer.completed_update_id == 0
    assert trainer.lr_scheduler.calls == 0
    trainer.hook_train_loop([_batch([0.0], [0.0], eligible=True)])
    assert trainer.completed_update_id == 1
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema.calls == 1


def test_real_sdtrainer_amp_skip_discards_window_and_next_step_succeeds(monkeypatch):
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer
    from accelerate.state import AcceleratorState, PartialState

    monkeypatch.setattr(AcceleratorState, "_shared_state", {})
    monkeypatch.setattr(PartialState, "_shared_state", {})

    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.0)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Build the real Accelerate wrapper around a real CPU GradScaler. The
    # wrapper sets step_was_skipped after scaler.step(), not before it.
    Accelerator(cpu=True)
    scaler = torch.amp.GradScaler("cpu")
    optimizer = AcceleratedOptimizer(base_optimizer, scaler=scaler)
    trainer = _trainer_for_real_hook(
        model, optimizer, scaler=scaler, base_optimizer=base_optimizer
    )
    before = model.weight.detach().clone()
    trainer._force_inf_grad = True
    trainer.hook_train_loop([_batch([1.0], [1.0])])
    assert optimizer.step_was_skipped is True
    assert torch.equal(model.weight, before)
    assert trainer.completed_update_id == 0
    assert trainer.lr_scheduler.calls == 0
    assert trainer.ema.calls == 0

    trainer._force_inf_grad = False
    trainer.hook_train_loop([_batch([1.0], [1.0])])
    assert optimizer.step_was_skipped is False
    assert not torch.equal(model.weight, before)
    assert trainer.completed_update_id == 1
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema.calls == 1

def test_real_sdtrainer_hook_accepts_adaptive_adamw_cpu_update():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, eps=3e-4)
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.hook_train_loop([_batch([1.0, 2.0], [1.0, 0.0])])
    assert trainer.completed_update_id == 1
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema.calls == 1
    assert torch.isfinite(model.weight).all()


def test_real_sdtrainer_hook_oom_does_not_advance_update_clocks():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _trainer_for_real_hook(model, optimizer)

    def raise_oom(_batch, accum_scale=1.0):
        raise torch.cuda.OutOfMemoryError("synthetic CPU lifecycle OOM")

    trainer.train_single_accumulation = raise_oom
    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer.hook_train_loop([_batch([1.0], [1.0])])
    assert trainer.completed_update_id == 0
    assert trainer.lr_scheduler.calls == 0
    assert trainer.ema.calls == 0
def test_completed_update_snapshot_exposes_boundary_clock():
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    host = SimpleNamespace(step_num=17, epoch_num=3, completed_update_id=5)
    snapshot = BaseSDTrainProcess.get_completed_update_snapshot(host)
    assert snapshot == {
        "step": 17,
        "epoch": 3,
        "completed_update_id": 5,
    }


# ---------------------------------------------------------------------------
# F10 / F13 / F14 acceptance oracles (independent arithmetic, no self-compare)
# ---------------------------------------------------------------------------

class _LossConfig:
    """Train config whose unset gates are inert (falsy) instead of crashing."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def __getattr__(self, name):
        return False


def _loss_trainer(config_overrides=None, sd=None):
    trainer = object.__new__(SDTrainer)
    trainer.device_torch = torch.device("cpu")
    trainer.train_config = _LossConfig(
        loss_type="mse",
        loss_target="noise",
        inverted_mask_prior_multiplier=0.5,
        audio_loss_multiplier=1.0,
        max_loss=None,
        **(config_overrides or {}),
    )
    # get_loss_target keeps the main target arithmetic out of scope here.
    trainer.sd = sd if sd is not None else SimpleNamespace(
        is_flow_matching=True,
        get_loss_target=lambda noise, batch, timesteps: noise,
        scale_loss=lambda loss: loss,
    )
    trainer.loss_watch = None
    trainer.additional_logs = {}
    trainer.adapter = None
    trainer.embedding = None
    trainer.ema = None
    trainer.step_num = 0
    trainer._adaptive_lr_window_steps = 1
    for attr in (
        "depth_consistency_config", "normal_config", "body_proportion_config",
        "face_id_config", "body_shape_config", "vae_anchor_config",
        "subject_mask_config", "adapter_config", "network_config", "dfe",
    ):
        setattr(trainer, attr, None)
    return trainer


def test_f10_prior_masks_use_each_samples_own_region_not_row_zero():
    """F10: disjoint per-item prior masks select their own complement; a
    fully-masked (zero-complement) item contributes exactly zero instead of
    inheriting sample 0's region."""
    H = W = 8
    # protected regions: sample 0 top half, sample 1 EVERYWHERE (empty complement)
    mask = torch.zeros(2, 1, H, W)
    mask[0, :, :4, :] = 1.0
    mask[1, :, :, :] = 1.0
    trainer = _loss_trainer({"inverted_mask_prior": True})
    batch = SimpleNamespace(
        get_is_reg_list=lambda: [False, False],
        loss_multiplier_list=[1.0, 1.0],
        mask_tensor=mask,
        latents=torch.zeros(2, 1, H, W),
    )
    noise_pred = torch.zeros(2, 1, H, W, requires_grad=True)
    prior_pred = torch.full((2, 1, H, W), 2.0, requires_grad=True)
    noise = torch.zeros(2, 1, H, W)
    loss = trainer.calculate_loss(
        noise_pred,
        noise,
        torch.zeros(2, 1, H, W),
        torch.tensor([500.0, 500.0]),
        batch,
        prior_pred=prior_pred,
    )
    # independent expectation: visual target == noise -> 0 everywhere.
    # prior term = 0.5 * mean over own complement of (pred - prior)^2
    #   sample 0 complement = rows 4:8 -> 0.5 * 4.0 = 2.0
    #   sample 1 complement empty   -> 0.0
    # batch mean -> (2.0 + 0.0) / 2 = 1.0
    # a row-0 broadcast would give sample 1 sample 0's region -> 2.0.
    assert float(loss.detach()) == pytest.approx(1.0, abs=1e-6)
    loss.backward()
    assert prior_pred.grad is not None
    assert prior_pred.grad[0, :, :4, :].abs().sum().item() == pytest.approx(0.0, abs=1e-7)
    assert prior_pred.grad[0, :, 4:, :].abs().sum().item() > 0.0
    assert prior_pred.grad[1].abs().sum().item() == pytest.approx(0.0, abs=1e-7)


def _audio_guidance_trainer(uncond_audio, guidance_scale=2.0):
    from toolkit.dto import DTO
    from toolkit.prompt_utils import PromptEmbeds

    trainer = _loss_trainer(
        {
            "do_guidance_loss": True,
            "do_guidance_loss_cfg_zero": True,
            "guidance_loss_sigma_min": 0.0,
            "guidance_loss_schedule": None,
        }
    )
    trainer._guidance_loss_target_batch = guidance_scale
    trainer.unconditional_embeds = PromptEmbeds(torch.zeros(1, 2, 3))

    def fake_predict_noise(**_kwargs):
        return DTO(torch.zeros(1), audio=uncond_audio.clone())

    trainer.predict_noise = fake_predict_noise
    return trainer


def _expected_cfg_zero_audio_loss(pred, a_target, uncond, scale, multiplier, dtype_bits):
    """Independent FP64 oracle for the CFG-Zero guided audio objective."""
    import torch as _t

    total = 0.0
    B = a_target.shape[0]
    for i in range(B):
        a = a_target[i].double().reshape(-1)
        u = uncond[i].double().reshape(-1)
        p = pred[i].double().reshape(-1)
        s = float((a * u).sum()) / max(float((u * u).sum()), 1e-8)
        u_prime = u * s
        blended = u_prime + scale * (a - u_prime)
        total += float(((p - blended) ** 2).mean()) * float(multiplier[i])
    return total / B


@pytest.mark.parametrize(
    "case",
    [
        ("normal", torch.float16),
        ("zero_uncond", torch.float16),
        ("tiny_uncond", torch.float32),
        ("large_uncond", torch.float16),
    ],
)
def test_f13_cfg_zero_ratio_matches_fp64_oracle_and_stays_finite(case):
    """F13: the CFG-Zero dot/norm ratio is computed in FP32 and matches an
    independent FP64 oracle for zero/tiny/large unconditionals without any
    disconnected-zero substitution."""
    name, dtype = case
    torch.manual_seed(7)
    B, C, T = 2, 4, 16
    pred = torch.randn(B, C, T, dtype=dtype)
    a_target = torch.randn(B, C, T, dtype=dtype)
    uncond = torch.randn(B, C, T, dtype=dtype)
    if name == "zero_uncond":
        uncond = torch.zeros_like(uncond)
    elif name == "tiny_uncond":
        uncond = uncond * 1e-20
    elif name == "large_uncond":
        uncond = uncond * 1e4
    trainer = _audio_guidance_trainer(uncond)
    loss = trainer._loss_for_audio_only_batch(
        audio_pred=pred,
        audio_target=a_target.clone(),
        audio_mask=torch.ones(B, dtype=torch.bool),
        audio_sigma=None,
        noisy_latents=torch.zeros(B, 1, 2, 2),
        timesteps=torch.full((B,), 500.0),
        batch=SimpleNamespace(),
        additional_loss=0.0,
        loss_multiplier=torch.ones(B),
    )
    assert torch.isfinite(loss)
    expected = _expected_cfg_zero_audio_loss(
        pred.float(), a_target.float(), uncond.float(), 2.0, torch.ones(B), dtype
    )
    assert float(loss.detach()) == pytest.approx(expected, rel=3e-2, abs=3e-3)


def test_f14_base_sigma_is_exact_across_grids_and_boundaries():
    """F14: base sigma resolution is exact FP32 for any timestep grid (no
    stale lookup state across grid changes) and delegates to the model's
    active-shift inversion."""
    trainer = object.__new__(SDTrainer)
    trainer.sd = SimpleNamespace()  # no timestep_to_base_sigma -> t/1000
    linear = torch.tensor([0.0, 250.0, 500.0, 1000.0])
    assert SDTrainer._base_sigma(trainer, linear).tolist() == [0.0, 0.25, 0.5, 1.0]
    # a different grid (sigmoid-distributed timesteps) resolves on the same
    # call path with no cross-grid staleness
    sig = torch.tensor([0.001, 0.02, 0.5, 0.98, 0.999])
    assert SDTrainer._base_sigma(trainer, sig * 1000.0).tolist() == pytest.approx(sig.tolist(), abs=1e-7)
    # fp16 timesteps are resolved in FP32 (values fp16 would destroy)
    ts16 = torch.tensor([0.1, 333.3], dtype=torch.float16)
    out = SDTrainer._base_sigma(trainer, ts16)
    assert out.dtype == torch.float32
    # expected from the fp16-quantized timestep, resolved in FP32
    assert out.tolist() == pytest.approx([float(ts16[0].item()) / 1000.0, float(ts16[1].item()) / 1000.0], abs=1e-7)
    # boundaries
    assert SDTrainer._base_sigma(trainer, torch.tensor([0.0, 1000.0])).tolist() == [0.0, 1.0]

    # the real H3 inversion uses the ACTIVE model shift (Fast ships 10, not 12)
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
        MinimaxH3Model,
    )

    for shift in (12.0, 10.0):
        holder = SimpleNamespace(video_sigma_shift=shift)
        sig_v = torch.tensor([0.126, 0.5, 0.68, 1.0])
        base = MinimaxH3Model.timestep_to_base_sigma(holder, sig_v.clone())
        expect = [float(s / (shift - (shift - 1.0) * s)) for s in sig_v.tolist()]
        assert base.tolist() == pytest.approx(expect, rel=1e-6)


def test_real_hook_injects_gradient_noise_after_clip_before_update():
    model = torch.nn.Linear(1, 1, bias=True)
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.1)
    model.weight._is_lora = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    trainer = _trainer_for_real_hook(model, optimizer)
    trainer.train_config.max_grad_norm = 0.05
    trainer.train_config.gradient_noise = SimpleNamespace(
        enabled=True, mode="absolute", sigma=0.1, log_every=1,
    )
    trainer.step_num = 0
    trainer._inject_gradient_noise = SDTrainer._inject_gradient_noise.__get__(trainer)
    clipped_gradient = 1.2 * 0.05 / (2.0 * 1.2 ** 2) ** 0.5
    noise = torch.randn((1, 1), generator=torch.Generator().manual_seed(173)) * 0.1
    torch.manual_seed(173)
    result = trainer.hook_train_loop([_batch([1.0], [0.0])])
    assert result["optimizer_update"] == 1.0
    torch.testing.assert_close(
        model.weight, torch.tensor([[0.5]]) - 0.2 * (clipped_gradient + noise),
        rtol=1e-5, atol=1e-7,
    )
    assert model.bias.item() == pytest.approx(0.1 - 0.2 * clipped_gradient, abs=1e-7)
    assert result["grad_noise_norm"] == pytest.approx(abs(noise.item()))


@pytest.mark.parametrize(
    ("window_active", "update_success"),
    [(True, True), (False, False)],
    ids=["open-window", "closed-failed-update"],
)
def test_manual_checkpoint_waits_for_successful_closed_optimizer_boundary(window_active, update_success):
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pending = []
    trainer = SimpleNamespace(
        is_ui_trainer=True,
        _optimizer_window_active=window_active,
        _last_optimizer_update_success=update_success,
        optimizer=optimizer,
        should_save=lambda: True,
        update_db_key=lambda *_args: pending.append("cleared"),
        save=lambda _step: pending.append("saved"),
    )
    DiffusionTrainer.maybe_save(trainer)
    assert pending == []
    assert model.weight.grad.item() == 3.0
