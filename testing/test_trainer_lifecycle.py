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
