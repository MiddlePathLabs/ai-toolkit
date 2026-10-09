"""Exercise production OOM recovery and telemetry flush without model imports."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _class_node(relative_path, name):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _compile_function(name, body, arguments, namespace):
    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=body, decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<training telemetry regression>", "exec"), namespace)
    return namespace[name]


@pytest.fixture
def telemetry_runtime():
    trainer_node = _class_node("extensions_built_in/sd_trainer/SDTrainer.py", "SDTrainer")
    methods = {node.name: node for node in trainer_node.body if isinstance(node, ast.FunctionDef)}
    selected = {
        "_bin_update", "_bin_finalize", "_canonical_telemetry_payload",
        "_drain_loss_metrics", "hook_train_loop_oom",
    }
    namespace = {"Optional": Optional}
    trainer_class = ast.ClassDef(
        name="SDTrainer", bases=[], keywords=[],
        body=[methods[name] for name in methods if name in selected], decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[trainer_class], type_ignores=[]))
    exec(compile(module, "<SDTrainer telemetry>", "exec"), namespace)
    trainer_type = namespace["SDTrainer"]

    # Load the dependency-free buffer itself, not the trainer package imports.
    exec(compile(
        (ROOT / "extensions_built_in/sd_trainer/metric_buffer.py").read_text(encoding="utf-8"),
        "metric_buffer.py", "exec",
    ), namespace)

    flush_body = methods["hook_train_loop"].body
    flush_start = next(
        index for index, node in enumerate(flush_body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "loss_dict" and node.value.func.attr == "update"
    )
    trainer_type.flush_telemetry = _compile_function(
        "flush_telemetry", flush_body[flush_start:], ["self", "loss_dict"], namespace,
    )

    base_node = _class_node("jobs/process/BaseSDTrainProcess.py", "BaseSDTrainProcess")
    # Preserve the actual exception classification and complete OOM recovery block.
    attempt = next(
        node for node in ast.walk(base_node)
        if isinstance(node, ast.Try)
        and any(isinstance(child, ast.Attribute) and child.attr == "hook_train_loop"
                for child in ast.walk(node))
        and any(isinstance(handler.type, ast.Attribute) and handler.type.attr == "OutOfMemoryError"
                for handler in node.handlers)
    )
    recovery = next(
        node for node in ast.walk(base_node)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == "did_oom"
    )

    class OutOfMemoryError(RuntimeError):
        pass

    namespace.update(
        torch=SimpleNamespace(cuda=SimpleNamespace(
            OutOfMemoryError=OutOfMemoryError, ipc_collect=lambda: None,
        )),
        print_acc=lambda *_args: None,
        clear_optimizer_gradients=lambda optimizer: optimizer.clear_calls.append(True),
    )
    initialize = ast.parse("did_oom = False\nloss_dict = None").body
    run_attempt = _compile_function(
        "run_attempt", initialize + [attempt, recovery],
        ["self", "optimizer", "batch_list"], namespace,
    )
    trainer = trainer_type()
    trainer._metric_buffer = namespace["MetricBuffer"]()
    trainer._id_sim_bins = {}
    trainer._depth_loss_bins = {}
    trainer._bsh_sim_bins = {}
    trainer._diffusion_loss_bins = {}
    trainer.logger = SimpleNamespace(payloads=[])
    trainer.logger.log = trainer.logger.payloads.append
    trainer.end_of_training_loop = lambda: None
    trainer.accelerator = SimpleNamespace(accumulate=lambda _modules: nullcontext())
    trainer.modules_being_trained = []
    trainer.optimizer_runtime = None
    trainer.network = None
    trainer.num_consecutive_oom = 0
    trainer._reset_routing_window = lambda: None
    optimizer = SimpleNamespace(clear_calls=[], zero_calls=[])
    optimizer.zero_grad = lambda **kwargs: optimizer.zero_calls.append(kwargs)
    trainer.optimizer = optimizer
    return trainer, run_attempt, OutOfMemoryError


def _append_metrics(trainer, value, *, weight=1):
    trainer._metric_buffer.add_scalar("id_sim", value, weight=weight)
    trainer._metric_buffer.add_per_sample("id_sim", value, sample_tag="pending")
    for attribute, prefix in (
        ("_id_sim_bins", "id_sim"),
        ("_depth_loss_bins", "depth_loss"),
        ("_bsh_sim_bins", "bsh_sim"),
        ("_diffusion_loss_bins", "diffusion_loss"),
    ):
        trainer._bin_update(getattr(trainer, attribute), f"{prefix}_t40", value)
    trainer._last_id_sim = value


@pytest.mark.parametrize("error_kind", ["typed", "runtime"])
def test_training_oom_discards_pending_metrics_before_next_success(telemetry_runtime, error_kind):
    trainer, run_attempt, oom_type = telemetry_runtime
    _append_metrics(trainer, 99.0)
    trainer._last_identity_loss = 77.0
    trainer._last_normal_loss = 55.0
    trainer._last_lr_effective = 33.0
    trainer._grad_cos_acc = {"n": 1, "cos": 99.0}
    trainer._last_grad_norm_diffusion = 99.0
    trainer._last_grad_norm_depth = 99.0
    trainer._last_grad_cos_diff_depth = 99.0
    trainer._optimizer_window_active = True
    trainer._window_sample_total = 8
    trainer._window_has_objective = True
    trainer._adaptive_lr_members = {"discarded": 1.0}
    error = oom_type("allocation failed") if error_kind == "typed" else RuntimeError("CUDA out of memory")

    def aborted(_batch_list):
        _append_metrics(trainer, 101.0)
        raise error

    trainer.hook_train_loop = aborted
    run_attempt(trainer, trainer.optimizer, [])
    assert trainer.logger.payloads == []
    assert trainer._metric_buffer.flush_scalars() == {}
    assert trainer._metric_buffer.flush_per_sample() == {}
    assert trainer._last_identity_loss is None
    assert trainer._last_normal_loss is None
    assert trainer._last_lr_effective is None
    assert trainer._grad_cos_acc is None
    assert trainer._last_grad_norm_diffusion is None
    assert trainer._last_grad_norm_depth is None
    assert trainer._last_grad_cos_diff_depth is None
    assert not trainer._optimizer_window_active
    assert trainer._window_sample_total == 0
    assert not trainer._window_has_objective
    assert trainer._adaptive_lr_members == {}
    assert trainer.optimizer.zero_calls == [{"set_to_none": True}]
    assert trainer.optimizer.clear_calls == [True]

    def success(_batch_list):
        _append_metrics(trainer, 0.25)
        return trainer.flush_telemetry({"loss": 1.0})

    trainer.hook_train_loop = success
    run_attempt(trainer, trainer.optimizer, [])
    assert trainer.num_consecutive_oom == 0
    assert trainer.logger.payloads == [{
        "identity/sim": 0.25,
        "identity/sim/t40": 0.25,
        "depth/loss/t40": 0.25,
        "body_shape/sim/t40": 0.25,
        "diffusion/loss/t40": 0.25,
    }]
    assert trainer._drain_loss_metrics() == {}


def test_normal_flush_preserves_weighted_means_and_drains_legacy_metrics(telemetry_runtime):
    trainer, _, _ = telemetry_runtime
    _append_metrics(trainer, 0.2)
    _append_metrics(trainer, 0.8, weight=3)
    trainer._last_identity_loss = 0.3
    trainer._last_normal_loss = 0.4
    result = trainer.flush_telemetry({"loss": 1.0})
    assert result == {"loss": 1.0, "id_sim": 0.8, "identity_loss": 0.3, "normal_loss": 0.4}
    payload = trainer.logger.payloads[0]
    assert payload["identity/sim"] == pytest.approx(0.65)
    assert payload["identity/loss_raw"] == 0.3
    for key in ("identity/sim/t40", "depth/loss/t40", "body_shape/sim/t40", "diffusion/loss/t40"):
        assert payload[key] == pytest.approx(0.5)
    assert trainer._metric_buffer.flush_scalars() == {}
    assert trainer._drain_loss_metrics() == {}
    assert all(getattr(trainer, name) == {} for name in (
        "_id_sim_bins", "_depth_loss_bins", "_bsh_sim_bins", "_diffusion_loss_bins",
    ))


def test_unexpected_training_exception_propagates(telemetry_runtime):
    trainer, run_attempt, _ = telemetry_runtime
    error = RuntimeError("shape mismatch")

    def failed(_batch_list):
        raise error

    trainer.hook_train_loop = failed
    with pytest.raises(RuntimeError, match="shape mismatch") as raised:
        run_attempt(trainer, trainer.optimizer, [])
    assert raised.value is error
    assert trainer.optimizer.zero_calls == []
    assert trainer.logger.payloads == []
