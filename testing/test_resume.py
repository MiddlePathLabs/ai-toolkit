from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.data_loader import StatefulRandomSampler
from toolkit.ema import ExponentialMovingAverage


def _host(tmp_path, model, optimizer, scheduler, loader, *, resume_mode):
    host = object.__new__(BaseSDTrainProcess)
    host.accelerator = SimpleNamespace(
        is_main_process=True,
        is_local_main_process=True,
        num_processes=1,
    )
    host.device_torch = torch.device("cpu")
    host.model_config = SimpleNamespace(compile=False, block_compile=False)
    host.train_config = SimpleNamespace(
        gradient_accumulation=1,
        gradient_accumulation_steps=1,
        ema_config=SimpleNamespace(
            use_ema=True,
            ema_decay=0.8,
            warmup=False,
            use_feedback=False,
            feedback_rate=0.0,
            param_multiplier=1.0,
            save_raw_weights=False,
        ),
        merge_network_on_save=False,
    )
    host.config = {
        "train": {
            "resume_mode": resume_mode,
            "gradient_accumulation": 1,
            "gradient_accumulation_steps": 1,
        }
    }
    host.resume_mode = resume_mode
    host.save_root = str(tmp_path)
    host._resume_state_path = str(tmp_path / "training_state.pt")
    host._resume_state = None
    host._resume_state_restored = False
    host._resume_snapshot_step_override = None
    host._optimizer_window_active = False
    host.dataset_configs = []
    host.optimizer = optimizer
    host.lr_scheduler = scheduler
    host.data_loader = loader
    host.data_loader_reg = None
    host.ema = ExponentialMovingAverage(
        list(model.parameters()), decay=0.8, use_num_updates=False
    )
    host.sd = SimpleNamespace()
    host.step_num = 0
    host.epoch_num = 0
    host.completed_update_id = 0
    host.loss_watch = None
    host.category_stop = None
    host.meta = {}
    host.job = SimpleNamespace(name="resume_cpu_trajectory")
    host.adapter = None
    host.snr_gos = None
    host.is_fine_tuning = False
    host.network = None
    host.network_config = None
    host._save_live_weights = lambda step_num, save_meta, prefix="": str(
        tmp_path / f"weights{prefix}{step_num}.safetensors"
    )
    host.update_training_metadata = lambda: None
    host.clean_up_saves = lambda: None
    host.post_save_hook = lambda _path: None
    return host


def _step(host, model, loader_iterator, x, y, noise):
    try:
        batch_x, batch_y = next(loader_iterator)
    except StopIteration:
        loader_iterator = iter(host.data_loader)
        batch_x, batch_y = next(loader_iterator)
    effective_target = batch_y + noise * 0.1
    loss = 0.5 * (model(batch_x) - effective_target).square().mean()
    loss.backward()
    host.optimizer.step()
    host.optimizer.zero_grad()
    host.lr_scheduler.step()
    host.ema.update()
    host.completed_update_id += 1
    host.step_num = host.completed_update_id
    return loader_iterator, (float(batch_x.item()), float(effective_target.item()))


def _oracle(initial_weight, samples, lr=0.2, momentum=0.5, decay=0.9, ema_decay=0.8):
    weight = float(initial_weight)
    velocity = 0.0
    ema = weight
    learning_rate = lr
    for x, target in samples:
        gradient = (weight * x - target) * x
        velocity = momentum * velocity + gradient
        weight -= learning_rate * velocity
        learning_rate *= decay
        ema = ema_decay * ema + (1.0 - ema_decay) * weight
    return weight, velocity, learning_rate, ema


def _new_run(tmp_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    targets = torch.tensor([[1.5], [2.5], [3.5], [4.5], [5.5], [6.5]])
    dataset = TensorDataset(values, targets)
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.set_state(torch.random.get_rng_state())
    sampler = StatefulRandomSampler(dataset, sampler_generator)
    iterator_generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, generator=iterator_generator
    )
    return model, optimizer, scheduler, loader


def test_interrupted_resume_matches_uninterrupted_cpu_oracle(tmp_path):
    seed = 7331
    total_updates = 6

    model_full, opt_full, sched_full, loader_full = _new_run(tmp_path / "full", seed)
    host_full = _host(tmp_path / "full", model_full, opt_full, sched_full, loader_full, resume_mode="exact")
    iterator_full = iter(loader_full)
    samples = []
    for _ in range(total_updates):
        noise = torch.rand(()) + np.random.rand()
        iterator_full, sample = _step(host_full, model_full, iterator_full, None, None, noise)
        samples.append(sample)

    model_part, opt_part, sched_part, loader_part = _new_run(tmp_path / "part", seed)
    host_part = _host(tmp_path / "part", model_part, opt_part, sched_part, loader_part, resume_mode="exact")
    iterator_part = iter(loader_part)
    for _ in range(3):
        noise = torch.rand(()) + np.random.rand()
        iterator_part, _ = _step(host_part, model_part, iterator_part, None, None, noise)
    BaseSDTrainProcess.save(host_part, step=host_part.step_num)

    # Prove that restore uses the persisted input RNG rather than whatever the
    # interrupted process happened to consume after its checkpoint.
    _ = torch.rand(17)
    model_resume, opt_resume, sched_resume, loader_resume = _new_run(tmp_path / "part", seed + 99)
    host_resume = _host(
        tmp_path / "part",
        model_resume,
        opt_resume,
        sched_resume,
        loader_resume,
        resume_mode="exact",
    )
    host_resume._load_resume_state_if_present()
    host_resume._validate_resume_runtime_support()
    BaseSDTrainProcess._restore_resume_optimizer_state(host_resume)
    BaseSDTrainProcess._restore_resume_scheduler_state(host_resume)
    BaseSDTrainProcess.setup_ema(host_resume)
    BaseSDTrainProcess._restore_resume_loader_state(host_resume)
    iterator_resume = iter(loader_resume)
    for _ in range(3):
        noise = torch.rand(()) + np.random.rand()
        iterator_resume, _ = _step(
            host_resume, model_resume, iterator_resume, None, None, noise
        )

    oracle_weight, oracle_velocity, oracle_lr, oracle_ema = _oracle(0.5, samples)
    torch.testing.assert_close(
        model_full.weight, torch.tensor([[oracle_weight]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(model_resume.weight, model_full.weight, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model_resume.weight, torch.tensor([[oracle_weight]]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        opt_resume.state[model_resume.weight]["momentum_buffer"],
        torch.tensor([[oracle_velocity]]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert sched_resume.get_last_lr()[0] == pytest.approx(oracle_lr)
    torch.testing.assert_close(
        host_resume.ema.shadow_params[0], torch.tensor([[oracle_ema]]), rtol=1e-6, atol=1e-6
    )
    assert host_resume.completed_update_id == total_updates
    assert host_resume.ema.num_updates is None


# ---------------------------------------------------------------------------
# Atomic raw-state write + resume-mode gate semantics oracles
# ---------------------------------------------------------------------------

def _raw_state(**overrides):
    state = {
        "kind": "ai_toolkit.raw_training_state",
        "schema": 1,
        "recipe_identity": "RID",
        "determinism": {"supported": True, "reasons": []},
        "snapshot": {"step": 3, "epoch": 0, "completed_update_id": 3},
        "params": [torch.zeros(1)],
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_epoch": 3},
        "ema": None,
        "rng": None,
        "data": {"train": None, "reg": None},
    }
    state.update(overrides)
    return state


def test_atomic_training_state_write_is_all_or_nothing(tmp_path, monkeypatch):
    target = tmp_path / "training_state.pt"
    holder = SimpleNamespace(save_root=str(tmp_path), _resume_state_path=str(target))

    # clean save lands only at the final path, no temp litter
    BaseSDTrainProcess._atomic_save_training_state(holder, _raw_state())
    assert target.exists()
    assert torch.load(str(target), weights_only=False)["kind"] == "ai_toolkit.raw_training_state"
    assert list(tmp_path.glob(".training_state_*")) == []
    good_bytes = target.read_bytes()

    # interrupted write (partial bytes then raise) leaves the previous
    # complete file untouched and no partial replacement behind
    real_save = torch.save

    def crashing_save(obj, path, *args, **kwargs):
        with open(path, "wb") as fh:
            fh.write(b"PK\x03\x04partial")
        raise RuntimeError("simulated crash mid-save")

    monkeypatch.setattr(torch, "save", crashing_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        BaseSDTrainProcess._atomic_save_training_state(holder, _raw_state())
    monkeypatch.setattr(torch, "save", real_save)
    assert target.read_bytes() == good_bytes
    assert list(tmp_path.glob(".training_state_*")) == []

    # a corrupted/partial file on disk is rejected fail-closed, never resumed
    corrupt = tmp_path / "corrupt_state.pt"
    corrupt.write_bytes(b"PK\x03\x04truncated")
    loader = SimpleNamespace(resume_mode="auto", _resume_state_path=str(corrupt), _resume_state=None)
    with pytest.raises(ValueError, match="weights_only"):
        BaseSDTrainProcess._load_resume_state_if_present(loader)


def _gate_holder(tmp_path, resume_mode, *, state=None, supported=True):
    if supported:
        dataset_configs = [SimpleNamespace(num_workers=0, buckets=False)]
        device_torch = torch.device("cpu")
    else:
        dataset_configs = [
            SimpleNamespace(num_workers=2, buckets=True),
            SimpleNamespace(num_workers=2, buckets=True),
        ]
        device_torch = torch.device("cuda")
    holder = SimpleNamespace(
        resume_mode=resume_mode,
        _resume_state_path=str(tmp_path / "training_state.pt"),
        _resume_state=state,
        accelerator=SimpleNamespace(num_processes=1),
        device_torch=device_torch,
        model_config=SimpleNamespace(compile=False, block_compile=False),
        train_config=SimpleNamespace(gradient_accumulation=1, gradient_accumulation_steps=1),
        dataset_configs=dataset_configs,
        _resume_recipe_identity=lambda: "RID",
        optimizer=None,
        lr_scheduler=None,
        step_num=0,
        start_step=0,
        epoch_num=0,
        completed_update_id=0,
    )
    holder._deterministic_resume_supported = (
        lambda: BaseSDTrainProcess._deterministic_resume_supported(holder)
    )
    return holder


def test_resume_mode_gate_semantics(tmp_path):
    # fresh run: auto with no state file loads nothing and validates clean
    fresh = _gate_holder(tmp_path, "auto", supported=True)
    BaseSDTrainProcess._load_resume_state_if_present(fresh)
    assert fresh._resume_state is None
    BaseSDTrainProcess._validate_resume_runtime_support(fresh)  # no raise

    # auto/exact with a complete state on an unsupported (GPU/worker/bucket)
    # environment refuse rather than silently approximating a resume
    state = _raw_state()
    strict = _gate_holder(tmp_path, "auto", state=state, supported=False)
    with pytest.raises(ValueError, match="exact replay is unsupported"):
        BaseSDTrainProcess._validate_resume_runtime_support(strict)

    exact_unsupported = _gate_holder(tmp_path, "exact", state=state, supported=False)
    with pytest.raises(ValueError, match="single-process CPU"):
        BaseSDTrainProcess._validate_resume_runtime_support(exact_unsupported)

    # exact on the supported deterministic scope with matching identity admits
    exact_ok = _gate_holder(tmp_path, "exact", state=state, supported=True)
    BaseSDTrainProcess._validate_resume_runtime_support(exact_ok)  # no raise

    # identity mismatch is refused even in the supported scope
    other = _gate_holder(tmp_path, "exact", state=_raw_state(recipe_identity="OTHER"), supported=True)
    with pytest.raises(ValueError, match="recipe/config identity differs"):
        BaseSDTrainProcess._validate_resume_runtime_support(other)

    # weights_only never loads raw state and never gates on determinism
    warm = _gate_holder(tmp_path, "weights_only", supported=False)
    warm._resume_state_path = str(tmp_path / "absent.pt")  # also nothing on disk
    BaseSDTrainProcess._load_resume_state_if_present(warm)
    assert warm._resume_state is None
    BaseSDTrainProcess._validate_resume_runtime_support(warm)  # no raise

    # a saved-but-unsupported state is equally ignored under weights_only
    warm_state = _gate_holder(tmp_path, "weights_only", supported=False)
    p = tmp_path / "training_state.pt"
    torch.save(_raw_state(), str(p))
    warm_state._resume_state_path = str(p)
    BaseSDTrainProcess._load_resume_state_if_present(warm_state)
    assert warm_state._resume_state is None
    # and restoration is a no-op: counters/optimizer stay untouched (a warm
    # start, never a resume)
    BaseSDTrainProcess._restore_resume_optimizer_state(warm_state)
    assert warm_state.step_num == 0 and warm_state.completed_update_id == 0

    # incomplete and unknown-schema states are explicit errors, not resumes
    p2 = tmp_path / "incomplete.pt"
    torch.save({"kind": "ai_toolkit.raw_training_state", "schema": 1}, str(p2))
    broken = _gate_holder(tmp_path, "auto")
    broken._resume_state_path = str(p2)
    with pytest.raises(ValueError, match="incomplete"):
        BaseSDTrainProcess._load_resume_state_if_present(broken)

    p3 = tmp_path / "badschema.pt"
    torch.save(_raw_state(schema=99), str(p3))
    bad = _gate_holder(tmp_path, "auto")
    bad._resume_state_path = str(p3)
    with pytest.raises(ValueError, match="schema"):
        BaseSDTrainProcess._load_resume_state_if_present(bad)
