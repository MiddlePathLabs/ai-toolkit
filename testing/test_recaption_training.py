"""Exercise recaption training hooks without importing the GPU training stack."""
import ast
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from toolkit import recaption as rc
from toolkit.loss_watch import PerImageAdaptiveLR


SOURCE = Path(__file__).resolve().parents[1] / "jobs/process/BaseSDTrainProcess.py"


class _RngState:
    def cpu(self):
        return self

    def clone(self):
        return self


def _training_hooks():
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    trainer = next(node for node in source.body if isinstance(node, ast.ClassDef)
                   and node.name == "BaseSDTrainProcess")
    names = {
        "_capture_training_state", "_restore_resume_loader_state", "_setup_recaption",
        "_recaption_items", "_recaption_on_window", "_recaption_poll", "_recaption_close",
    }
    methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in names]
    run = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
               and node.name == "_run_training_loop")
    boundary = next(node for node in ast.walk(run) if isinstance(node, ast.If)
                    and any(isinstance(part, ast.Attribute) and part.attr == "_adaptive_lr_window_steps"
                            for part in ast.walk(node.test)))
    # Execute the real boundary block, including its multiplier-write ordering.
    methods.append(ast.FunctionDef(
        name="_window_boundary",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="dataloader")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[boundary], decorator_list=[],
    ))
    namespace = {
        "copy": copy, "json": json, "os": os, "random": random, "print_acc": lambda *_: None,
        "get_dataloader_datasets": lambda loader: loader,
        "get_dataloader_state": lambda loader: {"schema": 1, "num_workers": 0, "datasets": []},
        "read_tread_seed": lambda _: None,
        "np": SimpleNamespace(random=SimpleNamespace(get_state=lambda: None, set_state=lambda _: None)),
        "torch": SimpleNamespace(
            random=SimpleNamespace(get_rng_state=_RngState, set_rng_state=lambda _: None),
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    }
    extracted = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(extracted), str(SOURCE), "exec"), namespace)
    return type("TrainingHooks", (), {name: namespace[name] for name in names | {"_window_boundary"}})


@pytest.fixture
def host(tmp_path):
    trainer = _training_hooks()()
    trainer.loss_watch = PerImageAdaptiveLR(log_fn=lambda *_: None)
    trainer.recaption = None
    trainer._recaption_peer = False
    trainer._recaption_peer_seq = 0
    trainer._resume_state = None
    trainer._resume_exact = False
    trainer.train_config = SimpleNamespace(
        per_image_adaptive_lr_recaption=SimpleNamespace(enabled=True),
        per_image_adaptive_lr_stats_only=False, unload_text_encoder=False, steps=100,
    )
    trainer.is_caching_text_embeddings = False
    trainer.accelerator = SimpleNamespace(is_main_process=True, num_processes=2)
    trainer.save_root = str(tmp_path)
    trainer.optimizer = SimpleNamespace(param_groups=[], state_dict=lambda: {})
    trainer.lr_scheduler = SimpleNamespace(state_dict=lambda: {})
    trainer._optimizer_window_active = False
    trainer.ema = None
    trainer.category_stop = None
    trainer.data_loader = [SimpleNamespace(file_list=[
        SimpleNamespace(path="before", loss_multiplier=0.5),
        SimpleNamespace(path="after", loss_multiplier=0.5),
    ])]
    trainer.data_loader_reg = None
    trainer.config = {}
    trainer.sd = None
    trainer.step_num = 8
    trainer._adaptive_lr_window_steps = 2
    trainer._deterministic_resume_supported = lambda: (True, [])
    trainer._resume_recipe_identity = lambda: {}
    trainer.get_completed_update_snapshot = lambda: {"step": 8}
    trainer._resume_parameter_identity = lambda: []
    trainer._resume_component_type = lambda *_, **__: "test"
    trainer._cpu_state = copy.deepcopy
    return trainer


@pytest.mark.parametrize("failure", ["has_results", "summary", "close", "handle_reset"])
def test_shutdown_always_closes_original_manager_and_preserves_cursor(host, failure):
    calls = []

    def has_results():
        if failure == "has_results":
            raise RuntimeError("poll probe failed")
        if failure == "handle_reset":
            host.recaption = None
        return False

    def summary(_):
        calls.append("summary")
        if failure == "summary":
            raise RuntimeError("summary failed")

    def close():
        calls.append("close")
        if failure == "close":
            raise RuntimeError("close failed")

    manager = SimpleNamespace(reset_seq=7, has_results=has_results, summary=summary, close=close)
    host.recaption = manager
    host._recaption_close()
    assert calls.count("close") == 1
    assert host.recaption is None
    assert host._recaption_peer_seq == 7
    # Final checkpoint saves happen after shutdown, when the manager is gone.
    assert host._capture_training_state()["recaption_reset_seq"] == 7
    host._recaption_close()
    assert calls.count("close") == 1


def _state_file(host, resets):
    path = Path(host.save_root) / rc.STATE_FILENAME
    path.write_text(json.dumps({"items": {}, "reset_seq": resets[-1][0], "resets": resets}),
                    encoding="utf-8")


@pytest.mark.parametrize("is_main", [False, True])
def test_resume_replays_only_post_checkpoint_resets_on_every_rank(host, monkeypatch, is_main):
    watcher = host.loss_watch
    for key in ("before", "after"):
        watcher._mult[key] = 0.5
        watcher._restored_residuals[key] = {3: 0.4, 4: 0.3}
        watcher._confirmed_stuck.add(key)
    # The main manager has already applied two resets, but both images have
    # accumulated new history since then. That history belongs to this checkpoint.
    host.recaption = SimpleNamespace(reset_seq=2)
    checkpoint = host._capture_training_state()
    assert checkpoint["recaption_reset_seq"] == 2
    _state_file(host, [[1, "before"], [2, "before"], [3, "after"]])
    host.recaption = None
    host._resume_state = checkpoint
    host.loss_watch = PerImageAdaptiveLR(log_fn=lambda *_: None)
    host._restore_resume_loader_state()
    assert host._recaption_peer_seq == 2
    host.accelerator.is_main_process = is_main

    def manager_factory(*_, **kwargs):
        state = json.loads((Path(kwargs["save_root"]) / rc.STATE_FILENAME).read_text(encoding="utf-8"))
        return SimpleNamespace(reset_seq=state["reset_seq"], on_window=lambda *_: None)

    monkeypatch.setattr(rc, "RecaptionManager", manager_factory)
    host._setup_recaption(host.data_loader)
    assert host._recaption_peer_seq == 3
    assert host.loss_watch._restored_residuals["before"] == {3: 0.4, 4: 0.3}
    assert "before" in host.loss_watch._confirmed_stuck
    assert host.loss_watch.multiplier("before") == 0.5
    assert "after" not in host.loss_watch._restored_residuals
    assert "after" not in host.loss_watch._confirmed_stuck
    assert [item.loss_multiplier for item in host.data_loader[0].file_list] == [0.5, 1.0]
    if is_main:
        assert host.recaption.reset_seq == 3
    # A later window must not replay these resets a second time.
    host.loss_watch._restored_residuals["after"] = {5: 0.2}
    host._recaption_on_window()
    assert host.loss_watch._restored_residuals["after"] == {5: 0.2}


def test_peer_reset_is_visible_in_same_boundary_multiplier_writes(host):
    host._recaption_peer = True
    host.loss_watch._mult["after"] = 0.2
    host.loss_watch._restored_residuals["after"] = {1: 0.3}
    host.loss_watch.epoch_boundary = lambda _: None
    _state_file(host, [[1, "after"]])
    host._window_boundary(host.data_loader)
    assert host.data_loader[0].file_list[1].loss_multiplier == 1.0
    assert "after" not in host.loss_watch._restored_residuals
    assert host._recaption_peer_seq == 1


def test_stats_only_boundary_does_not_write_training_multipliers(host):
    host.train_config.per_image_adaptive_lr_stats_only = True
    host.loss_watch.epoch_boundary = lambda _: None
    host._window_boundary(host.data_loader)
    assert [item.loss_multiplier for item in host.data_loader[0].file_list] == [0.5, 0.5]


def test_cursor_is_not_restored_without_matching_watcher_state(host):
    checkpoint = host._capture_training_state()
    checkpoint["loss_watch"] = None
    checkpoint["recaption_reset_seq"] = 19
    host._resume_state = checkpoint
    host._restore_resume_loader_state()
    assert host._recaption_peer_seq == 0


def test_legacy_resume_preserves_history_and_warns_about_ambiguous_resets(host):
    host.loss_watch._restored_residuals["before"] = {3: 0.4}
    checkpoint = host._capture_training_state()
    checkpoint.pop("recaption_reset_seq")
    _state_file(host, [[1, "before"], [2, "after"]])
    messages = []
    host._restore_resume_loader_state.__func__.__globals__["print_acc"] = messages.append
    host._resume_state = checkpoint
    host._restore_resume_loader_state()
    assert host._recaption_peer_seq == 2
    assert host.loss_watch._restored_residuals["before"] == {3: 0.4}
    assert any("legacy checkpoint has no reset cursor" in message for message in messages)
    assert any("Post-checkpoint caption changes cannot" in message for message in messages)
    host._recaption_peer = True
    host._recaption_on_window()
    assert host.loss_watch._restored_residuals["before"] == {3: 0.4}


def test_main_manager_never_reuses_checkpoint_reset_sequences(host, monkeypatch):
    host._recaption_peer_seq = 4
    _state_file(host, [[1, "before"], [2, "after"]])
    monkeypatch.setattr(rc, "RecaptionManager", lambda *_, **__: SimpleNamespace(reset_seq=2))
    host._setup_recaption(host.data_loader)
    assert host.recaption.reset_seq == 4
    assert host._capture_training_state()["recaption_reset_seq"] == 4
