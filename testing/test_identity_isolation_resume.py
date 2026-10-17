"""Raw-checkpoint phase and content-identity checks with a synthetic CPU adapter."""

import copy
from pathlib import Path

import pytest
import torch

from testing.test_identity_isolation_data import _process
from testing.test_resume import _host, _new_run, _step
from toolkit.identity_isolation_data import configure_isolation
from toolkit.identity_isolation_training import dop_selected


def _resume_host(path, prepared, isolation, seed):
    model, _, _, loader = _new_run(path, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    host = _host(path, model, optimizer, scheduler, loader, resume_mode="exact")
    host.config = copy.deepcopy(prepared)
    host.isolation_config = isolation
    host.train_config.diff_output_preservation = True
    return host, model, loader


def test_raw_checkpoint_restores_isolation_phase_sampler_and_rejects_changed_source(
    tmp_path,
):
    raw = _process(tmp_path, counts=(1,), dop_enabled=True, dop_interval=3)
    raw["train"]["resume_mode"] = "exact"
    prepared = copy.deepcopy(raw)
    isolation = configure_isolation(prepared)
    trajectory = tmp_path / "trajectory"
    first, model, loader = _resume_host(trajectory, prepared, isolation, 731)
    iterator = iter(loader)
    for _ in range(2):
        iterator, _ = _step(first, model, iterator, None, None, torch.tensor(0.0))
    first.save(step=first.step_num)
    expected_next = next(iterator)
    expected_average = first.optimizer.state[model.weight]["exp_avg"].clone()

    resumed, restored_model, restored_loader = _resume_host(
        trajectory, prepared, isolation, 999
    )
    resumed._load_resume_state_if_present()
    resumed._validate_resume_runtime_support()
    resumed._restore_resume_optimizer_state()
    resumed._restore_resume_scheduler_state()
    resumed.setup_ema()
    resumed._restore_resume_loader_state()
    assert resumed.completed_update_id == 2
    assert not dop_selected(isolation, resumed.completed_update_id, True)
    actual_next = next(iter(restored_loader))
    for actual, expected in zip(actual_next, expected_next):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        resumed.optimizer.state[restored_model.weight]["exp_avg"],
        expected_average,
        rtol=0,
        atol=0,
    )
    assert resumed.lr_scheduler.get_last_lr() == first.lr_scheduler.get_last_lr()
    assert dop_selected(isolation, resumed.completed_update_id + 1, True)

    source = Path(raw["datasets"][0]["folder_path"]) / "0.txt"
    source.write_text("person0 in a changed scene", encoding="utf-8")
    changed = copy.deepcopy(raw)
    changed_isolation = configure_isolation(changed)
    incompatible, _, _ = _resume_host(trajectory, changed, changed_isolation, 731)
    incompatible._load_resume_state_if_present()
    with pytest.raises(ValueError):
        incompatible._validate_resume_runtime_support()
