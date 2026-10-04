import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from safetensors.torch import load_file, save_file

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.data_loader import StatefulRandomSampler
from toolkit.ema import ExponentialMovingAverage

class _LinearNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.linear = model
        self.multiplier = 1.0

    def save_weights(self, path, dtype, metadata, extra_state_dict=None):
        save_file(
            {key: value.detach().cpu().to(dtype) for key, value in self.state_dict().items()},
            path,
            metadata=metadata,
        )

    def load_weights(self, path):
        self.load_state_dict(load_file(path))
        return {}



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
    host.train_config.start_step = None
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
    host._resume_exact = False
    host._resume_next_step = None
    host._optimizer_window_active = False
    host.dataset_configs = []
    host.optimizer = optimizer
    host.lr_scheduler = scheduler
    host.data_loader = loader
    host.data_loader_reg = None
    host.ema = ExponentialMovingAverage(
        list(model.parameters()), decay=0.8, use_num_updates=False
    )
    host.sd = SimpleNamespace(unet=model)
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
    host.network = _LinearNetwork(model)
    host.network_config = None
    host.named_lora = False
    host.embedding = None
    host.decorator = None
    host.save_config = SimpleNamespace(dtype="float32")

    def update_metadata():
        host.meta["training_info"] = host.get_training_info()

    host.update_training_metadata = update_metadata
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


def _new_run(tmp_path, seed, *, device="cpu", constant_inputs=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    if constant_inputs:
        values = torch.ones(6, 1)
        targets = torch.full((6, 1), 1.5)
    else:
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
        "optimizer": {"state": {}, "param_groups": [{"params": [0]}]},
        "scheduler": {"last_epoch": 3},
        "ema": None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": None,
        },
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

    # GPU, workers, buckets, compile and accumulation relax replay, not raw
    # optimizer continuation or checkpoint identity.
    state = _raw_state()
    for mode in ("auto", "continue"):
        continuation = _gate_holder(tmp_path, mode, state=state, supported=False)
        continuation.model_config.compile = True
        continuation.train_config.gradient_accumulation_steps = 3
        BaseSDTrainProcess._validate_resume_runtime_support(continuation)
        assert not continuation._resume_exact

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
    for mode in ("auto", "continue"):
        fresh_distributed = _gate_holder(tmp_path, mode, state=None)
        fresh_distributed._resume_state = None
        fresh_distributed.accelerator.num_processes = 2
        with pytest.raises(ValueError, match="single-process"):
            BaseSDTrainProcess._validate_resume_runtime_support(fresh_distributed)
        distributed = _gate_holder(tmp_path, mode, state=state)
        distributed.accelerator.num_processes = 2
        with pytest.raises(ValueError, match="distributed/multi-process"):
            BaseSDTrainProcess._validate_resume_runtime_support(distributed)
    weights_only_distributed = _gate_holder(tmp_path, "weights_only", state=None, supported=False)
    weights_only_distributed.accelerator.num_processes = 2
    BaseSDTrainProcess._validate_resume_runtime_support(weights_only_distributed)

    saved_gpu = _gate_holder(
        tmp_path, "exact", state=_raw_state(determinism={"supported": False, "reasons": ["CUDA"]})
    )
    with pytest.raises(ValueError, match="exact replay is unsupported"):
        BaseSDTrainProcess._validate_resume_runtime_support(saved_gpu)


    # weights_only must not ignore and later replace an existing trajectory
    warm_state = _gate_holder(tmp_path, "weights_only", supported=False)
    p = tmp_path / "training_state.pt"
    torch.save(_raw_state(), str(p))
    preserved = p.read_bytes()
    warm_state._resume_state_path = str(p)
    with pytest.raises(ValueError, match="new output folder"):
        BaseSDTrainProcess._load_resume_state_if_present(warm_state)
    assert warm_state._resume_state is None
    assert p.read_bytes() == preserved
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


class _LoopTimer:
    def __call__(self, _name):
        return nullcontext()

    def start(self, _name):
        pass

    def stop(self, _name):
        pass


def _training_batch(items):
    x, y = zip(*items)
    return SimpleNamespace(
        file_items=[object() for _ in items],
        latents=torch.stack(x),
        x=torch.stack(x),
        y=torch.stack(y),
    )


def _loop_host(tmp_path, model, optimizer, scheduler, loader, *, resume_mode="exact"):
    host = _host(tmp_path, model, optimizer, scheduler, loader, resume_mode=resume_mode)
    host.train_config.steps = 6
    host.train_config.optimizer = "sgd"
    host.train_config.max_grad_norm = 1e9
    host.train_config.per_image_adaptive_lr_mode = "loss"
    host.train_config.do_paramiter_swapping = False
    host.train_config.do_random_cfg = False
    host.train_config.free_u = False
    host.train_config.disable_sampling = True
    host.train_config.validation_config = None
    host.device_torch = next(model.parameters()).device
    host.sd.is_multistage = False
    host.params = list(model.parameters())
    host.modules_being_trained = []
    host.save_config.save_every = 2
    host.sample_config = SimpleNamespace(sample_every=0, sample_start_step=0)
    host.logging_config = SimpleNamespace(log_every=0)
    host.logger = SimpleNamespace(commit=lambda **_kwargs: None)
    host.timer = _LoopTimer()
    host.progress_bar = None
    host.torch_profiler = None
    host.performance_log_every = 0
    host.start_step = 0
    host.grad_accumulation_step = 1
    host.steps_this_boundary = 0
    host.current_boundary_index = 0
    host.num_consecutive_oom = 0
    host.optimizer_runtime = None
    host.modality_router = None
    host._post_step_active_ids = None
    host._post_step_active_resolved = False
    host._adaptive_lr_members = {}
    host._category_window_kinds = set()
    host._repair_pending_text_embeddings = lambda *_args: None
    host.ensure_params_requires_grad = lambda **_kwargs: None
    host._inject_gradient_noise = lambda: None
    host._inject_weight_noise = lambda: None
    host._record_fisher_trace = lambda: None
    host._last_optimizer_update_success = False
    host._last_optimizer_update_skipped = False
    host.end_step_hook = lambda: None
    host.accelerator.accumulate = lambda _modules: nullcontext()
    host.accelerator.wait_for_everyone = lambda: None
    host.accelerator.clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    host.hook_train_loop = SDTrainer.hook_train_loop.__get__(host)

    def train_single(batch, accum_scale=1.0):
        x = batch.x.to(host.device_torch)
        target = batch.y.to(host.device_torch) + (torch.rand((), device=host.device_torch) + np.random.rand()) * 0.1
        loss = 0.5 * (model(x) - target).square().mean()
        host._last_objective_eligible = True
        (loss * accum_scale).backward()
        return loss.detach()

    host.train_single_accumulation = train_single
    loader.collate_fn = _training_batch
    host.end_of_training_loop = lambda: None
    return host


def _run_loop(host):
    host._run_training_loop(host.optimizer, host.data_loader, None, iter(host.data_loader), None)


def _restore(host):
    host._load_resume_state_if_present()
    host._validate_resume_runtime_support()
    host._restore_resume_optimizer_state()
    host._restore_resume_scheduler_state()
    host.setup_ema()
    host._restore_resume_loader_state()


@pytest.mark.parametrize("resume_mode", ["auto", "exact"])
def test_production_interval_save_restores_next_input_and_final_cursor(tmp_path, resume_mode):
    seed = 491
    full_model, full_opt, full_scheduler, full_loader = _new_run(tmp_path / "full_loop", seed)
    full = _loop_host(tmp_path / "full_loop", full_model, full_opt, full_scheduler, full_loader, resume_mode=resume_mode)
    _run_loop(full)

    part_model, part_opt, part_scheduler, part_loader = _new_run(tmp_path / "part_loop", seed)
    part = _loop_host(tmp_path / "part_loop", part_model, part_opt, part_scheduler, part_loader, resume_mode=resume_mode)

    class Interrupted(Exception):
        pass

    def interrupt_after_interval():
        if part.step_num == 3:
            raise Interrupted()

    part.end_step_hook = interrupt_after_interval
    with pytest.raises(Interrupted):
        _run_loop(part)
    state = torch.load(part._resume_state_path, weights_only=False)
    assert state["snapshot"] == {"step": 3, "epoch": 0, "completed_update_id": 3}
    export = tmp_path / "part_loop" / "resume_cpu_trajectory_ema_000000002.safetensors"
    from toolkit.metadata import load_metadata_from_safetensors
    assert load_metadata_from_safetensors(str(export))["training_info"]["step"] == 3

    resumed_model, resumed_opt, resumed_scheduler, resumed_loader = _new_run(tmp_path / "part_loop", seed + 99)
    resumed = _loop_host(tmp_path / "part_loop", resumed_model, resumed_opt, resumed_scheduler, resumed_loader, resume_mode=resume_mode)
    _restore(resumed)
    _run_loop(resumed)
    assert resumed.step_num == resumed.completed_update_id == full.completed_update_id == 6
    torch.testing.assert_close(resumed_model.weight, full_model.weight, rtol=0, atol=0)
    torch.testing.assert_close(
        resumed_opt.state[resumed_model.weight]["momentum_buffer"],
        full_opt.state[full_model.weight]["momentum_buffer"], rtol=0, atol=0,
    )
    torch.testing.assert_close(resumed.ema.shadow_params[0], full.ema.shadow_params[0], rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == full_scheduler.state_dict()
    resumed.save(step=6)
    resumed.save()
    assert torch.load(resumed._resume_state_path, weights_only=False)["snapshot"]["step"] == 6


def test_weights_only_loads_physical_export_without_training_metadata(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    model, optimizer, scheduler, loader = _new_run(source, 81)
    previous = _host(source, model, optimizer, scheduler, loader, resume_mode="exact")
    previous.train_config.ema_config.warmup = True
    previous.setup_ema()
    iterator = iter(loader)
    for _ in range(2):
        iterator, _ = _step(previous, model, iterator, None, None, 0.0)
    previous.step_num = previous.completed_update_id = 200
    previous.epoch_num = 9
    previous.ema.num_updates = 200
    previous.save(step=200)
    destination.mkdir()
    for export in source.glob("*.safetensors"):
        (destination / export.name).write_bytes(export.read_bytes())
    assert not (destination / "training_state.pt").exists()
    new_model, new_optimizer, new_scheduler, new_loader = _new_run(destination, 92)
    warm = _host(destination, new_model, new_optimizer, new_scheduler, new_loader, resume_mode="weights_only")
    warm.start_step = 0
    warm.train_config.ema_config.warmup = True
    before_rng = torch.random.get_rng_state().clone()
    before_sampler = new_loader.sampler.state_dict()
    warm._load_resume_state_if_present()
    selected = warm.get_latest_save_path()
    warm.load_weights(selected)
    warm.setup_ema()
    assert warm.step_num == warm.start_step == warm.epoch_num == warm.completed_update_id == 0
    assert warm._resume_state is None
    assert not hasattr(warm, "_resume_tread_seed")
    assert new_optimizer.state == {}
    assert new_scheduler.last_epoch == 0
    assert warm.ema.num_updates == 0
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert before_sampler["cursor"] == new_loader.sampler.cursor

    # The full-model LoRM path used to import the same inference clocks directly.
    full_path = destination / "resume_cpu_trajectory.safetensors"
    save_file(
        model.state_dict(), str(full_path),
        metadata={"training_info": json.dumps({"step": 200, "epoch": 9, "tread_seed": 987})},
    )
    warm.device = "cpu"
    warm.get_latest_save_path = lambda: str(full_path)
    warm.load_lorm()
    assert warm.step_num == warm.start_step == warm.epoch_num == warm.completed_update_id == 0
    assert not hasattr(warm, "_resume_tread_seed")


@pytest.mark.parametrize("mode", ["auto", "continue", "exact", "weights_only"])
def test_train_config_accepts_final_resume_modes(mode):
    assert TrainConfig(resume_mode=mode).resume_mode == mode


def test_raw_restore_ignores_unmatched_inference_and_checks_parameter_state(tmp_path):
    model, optimizer, scheduler, loader = _new_run(tmp_path, 19)
    previous = _host(tmp_path, model, optimizer, scheduler, loader, resume_mode="exact")
    _step(previous, model, iter(loader), None, None, 0.0)
    previous.save(step=1)
    # A newer inference export must not seed a model paired with older momentum.
    save_file(
        {"linear.weight": torch.full_like(model.weight, 123.0)},
        str(tmp_path / "resume_cpu_trajectory_ema.safetensors"),
        metadata={"training_info": json.dumps({"step": 999})},
    )
    new_model, new_opt, new_scheduler, new_loader = _new_run(tmp_path, 29)
    resumed = _host(tmp_path, new_model, new_opt, new_scheduler, new_loader, resume_mode="continue")
    _restore(resumed)
    assert resumed.get_latest_save_path() is None
    torch.testing.assert_close(new_model.weight, model.weight)
    assert resumed.step_num == resumed.completed_update_id == 1

    resumed._resume_state["params"][0] = torch.zeros(2, 1)
    untouched = new_model.weight.detach().clone()
    with pytest.raises(ValueError, match="shape/dtype"):
        resumed._restore_resume_optimizer_state()
    torch.testing.assert_close(new_model.weight, untouched)


def test_continue_restores_global_rng_restarts_traversal_and_checks_membership(tmp_path):
    model, optimizer, scheduler, loader = _new_run(tmp_path, 11)
    previous = _host(tmp_path, model, optimizer, scheduler, loader, resume_mode="exact")
    iterator = iter(loader)
    _step(previous, model, iterator, None, None, 0.0)
    previous.save(step=1)
    saved_rng = torch.load(previous._resume_state_path, weights_only=False)["rng"]
    model2, optimizer2, scheduler2, loader2 = _new_run(tmp_path, 12)
    resumed = _host(tmp_path, model2, optimizer2, scheduler2, loader2, resume_mode="continue")
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    _restore(resumed)
    assert resumed._resume_state["data"]["train"]["sampler"]["cursor"] == 1
    assert loader2.sampler.cursor == 0
    python_probe = random.Random()
    python_probe.setstate(saved_rng["python"])
    assert random.random() == python_probe.random()
    numpy_probe = np.random.RandomState()
    numpy_probe.set_state(saved_rng["numpy"])
    assert np.random.rand() == numpy_probe.rand()
    torch_probe = torch.Generator()
    torch_probe.set_state(saved_rng["torch"].cpu())
    assert torch.rand((), generator=torch_probe).item() == torch.rand(()).item()
    resumed._resume_state["data"]["train"]["datasets"][0]["file_ids"] = [("missing.png", False, False)]
    with pytest.raises(ValueError, match="dataset membership"):
        BaseSDTrainProcess._restore_resume_loader_state(resumed)


def test_production_interval_waits_for_complete_accumulation_boundary(tmp_path):
    model, optimizer, scheduler, loader = _new_run(tmp_path, 517)
    host = _loop_host(tmp_path, model, optimizer, scheduler, loader, resume_mode="continue")
    host.train_config.gradient_accumulation_steps = 3
    _run_loop(host)
    state = torch.load(host._resume_state_path, weights_only=False)
    assert state["snapshot"] == {"step": 6, "epoch": 0, "completed_update_id": 2}
    assert not host._optimizer_window_active
    assert host.completed_update_id == 2
    assert (tmp_path / "resume_cpu_trajectory_ema_000000002.safetensors").exists()
    assert (tmp_path / "resume_cpu_trajectory_ema_000000005.safetensors").exists()
    assert not (tmp_path / "resume_cpu_trajectory_ema_000000004.safetensors").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("snapshot", {"step": 2, "epoch": 0, "completed_update_id": 3}),
        ("params", ["not a tensor"]),
        ("optimizer", {"state": {99: {}}, "param_groups": [{"params": [0]}]}),
        ("rng", None),
        ("data", {"train": None}),
    ],
)
def test_raw_checkpoint_invalid_components_fail_closed(tmp_path, field, value):
    path = tmp_path / "training_state.pt"
    torch.save(_raw_state(**{field: value}), path)
    host = _gate_holder(tmp_path, "continue")
    with pytest.raises(ValueError, match="Raw checkpoint state is invalid"):
        BaseSDTrainProcess._load_resume_state_if_present(host)
    assert host._resume_state is None


def test_recipe_identity_excludes_only_resume_mode(tmp_path):
    model, optimizer, scheduler, loader = _new_run(tmp_path, 713)
    host = _host(tmp_path, model, optimizer, scheduler, loader, resume_mode="auto")
    identity = host._resume_recipe_identity()
    host.config["train"]["resume_mode"] = "continue"
    assert host._resume_recipe_identity() == identity
    host.config["train"]["gradient_accumulation"] = 2
    assert host._resume_recipe_identity() != identity


@pytest.mark.parametrize("mode", ["auto", "continue", "exact"])
def test_existing_inference_checkpoint_requires_complete_raw_state(tmp_path, mode):
    model, optimizer, scheduler, loader = _new_run(tmp_path, 809)
    previous = _host(tmp_path, model, optimizer, scheduler, loader, resume_mode="weights_only")
    previous.save(step=200)
    (tmp_path / "training_state.pt").unlink()
    next_model, next_optimizer, next_scheduler, next_loader = _new_run(tmp_path, 810)
    resumed = _host(tmp_path, next_model, next_optimizer, next_scheduler, next_loader, resume_mode=mode)
    original = next_model.weight.detach().clone()
    with pytest.raises(ValueError, match="without a complete raw"):
        resumed.get_latest_save_path()
    torch.testing.assert_close(next_model.weight, original)
    assert next_optimizer.state == {}


def test_fresh_auto_pretrained_lora_is_an_initial_weight_source_not_resume(tmp_path):
    source_model, source_optimizer, source_scheduler, source_loader = _new_run(tmp_path / "source", 91)
    source = _host(
        tmp_path / "source", source_model, source_optimizer, source_scheduler, source_loader,
        resume_mode="weights_only",
    )
    source.step_num = source.completed_update_id = 200
    source.save(step=200)
    checkpoint = tmp_path / "source" / "resume_cpu_trajectory_ema_000000200.safetensors"
    model, optimizer, scheduler, loader = _new_run(tmp_path / "fresh", 101)
    fresh = _host(tmp_path / "fresh", model, optimizer, scheduler, loader, resume_mode="auto")
    fresh.network_config = SimpleNamespace(pretrained_lora_path=str(checkpoint))
    selected = fresh.get_latest_save_path()
    fresh.load_weights(selected)
    assert selected == str(checkpoint)
    assert fresh.step_num == fresh.completed_update_id == 0
    assert optimizer.state == {}


@pytest.mark.parametrize("mode", ["auto", "continue"])
def test_single_process_cannot_continue_multirank_raw_checkpoint(tmp_path, mode):
    state = _raw_state(determinism={"supported": False, "reasons": ["distributed/multi-process execution"]})
    host = _gate_holder(tmp_path, mode, state=state)
    with pytest.raises(ValueError, match="distributed/multi-process"):
        BaseSDTrainProcess._validate_resume_runtime_support(host)


@pytest.mark.parametrize("mode", ["auto", "continue"])
def test_recipe_mismatch_rejects_nonexact_continuation(tmp_path, mode):
    host = _gate_holder(tmp_path, mode, state=_raw_state(recipe_identity="OTHER"))
    with pytest.raises(ValueError, match="recipe/config identity differs"):
        BaseSDTrainProcess._validate_resume_runtime_support(host)


def test_weights_only_can_save_its_own_fresh_trajectory_but_not_replace_another(tmp_path):
    model, optimizer, scheduler, loader = _new_run(tmp_path / "fresh", 17)
    fresh = _host(tmp_path / "fresh", model, optimizer, scheduler, loader, resume_mode="weights_only")
    fresh._load_resume_state_if_present()
    fresh.save(step=1)
    fresh.step_num = fresh.completed_update_id = 2
    fresh.save(step=2)
    assert torch.load(fresh._resume_state_path, weights_only=False)["snapshot"]["step"] == 2

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    raw_path = occupied / "training_state.pt"
    torch.save(_raw_state(snapshot={"step": 9, "epoch": 1, "completed_update_id": 9}), raw_path)
    preserved = raw_path.read_bytes()
    reset_model, reset_optimizer, reset_scheduler, reset_loader = _new_run(occupied, 18)
    reset = _host(occupied, reset_model, reset_optimizer, reset_scheduler, reset_loader, resume_mode="weights_only")
    reset._weights_only_owns_raw_state = False
    with pytest.raises(ValueError, match="new output folder"):
        reset.save(step=1)
    assert raw_path.read_bytes() == preserved
    assert reset_optimizer.state == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA continuation regression requires a GPU")
def test_cuda_auto_continuation_matches_uninterrupted_constant_inputs(tmp_path):
    seed = 491
    device = "cuda"
    full_model, full_opt, full_scheduler, full_loader = _new_run(
        tmp_path / "full_loop", seed, device=device, constant_inputs=True,
    )
    full = _loop_host(tmp_path / "full_loop", full_model, full_opt, full_scheduler, full_loader, resume_mode="auto")
    _run_loop(full)

    part_model, part_opt, part_scheduler, part_loader = _new_run(
        tmp_path / "part_loop", seed, device=device, constant_inputs=True,
    )
    part = _loop_host(tmp_path / "part_loop", part_model, part_opt, part_scheduler, part_loader, resume_mode="auto")

    class Interrupted(Exception):
        pass

    def interrupt_after_interval():
        if part.step_num == 3:
            raise Interrupted()

    part.end_step_hook = interrupt_after_interval
    with pytest.raises(Interrupted):
        _run_loop(part)
    state = torch.load(part._resume_state_path, weights_only=False)
    assert state["snapshot"] == {"step": 3, "epoch": 0, "completed_update_id": 3}

    resumed_model, resumed_opt, resumed_scheduler, resumed_loader = _new_run(
        tmp_path / "part_loop", seed + 99, device=device, constant_inputs=True,
    )
    resumed = _loop_host(
        tmp_path / "part_loop", resumed_model, resumed_opt, resumed_scheduler, resumed_loader, resume_mode="auto",
    )
    _restore(resumed)
    assert not resumed._resume_exact
    assert resumed_loader.sampler.cursor == 0
    assert state["data"]["train"]["sampler"]["cursor"] != 0
    _run_loop(resumed)
    assert resumed_model.weight.device.type == "cuda"
    momentum = resumed_opt.state[resumed_model.weight]["momentum_buffer"]
    assert momentum.device.type == "cuda"
    assert resumed.step_num == resumed.completed_update_id == full.completed_update_id == 6
    torch.testing.assert_close(resumed_model.weight, full_model.weight, rtol=0, atol=0)
    torch.testing.assert_close(momentum, full_opt.state[full_model.weight]["momentum_buffer"], rtol=0, atol=0)
    torch.testing.assert_close(resumed.ema.shadow_params[0], full.ema.shadow_params[0], rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == full_scheduler.state_dict()
