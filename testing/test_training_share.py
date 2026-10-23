import json
import os
import time
from types import SimpleNamespace

import pytest
import torch

from toolkit.config_modules import ShareConfig, TrainConfig
from toolkit.training_share import (
    ShareProtocolError,
    ShareState,
    ShareTimeout,
    _load_meta,
    compute_delta,
    merge,
    next_frac,
    split_for_round,
    state_dict_fp32,
    validate_share_setup,
)


def _keys(n: int) -> list:
    return [f"E:/ds/img_{i:03d}.jpg" for i in range(n)]


def _net(seed: int) -> torch.nn.Linear:
    torch.manual_seed(seed)
    return torch.nn.Linear(6, 6)


# --------------------------------------------------------------------- split


def test_split_deterministic_and_role_complementary():
    keys = _keys(26)
    a = split_for_round(42, 1, keys, 0.5)
    b = split_for_round(42, 1, keys, 0.5)
    assert a == b
    lead_side = set(keys) - a
    assert a | lead_side == set(keys)
    assert not (a & lead_side)
    assert len(a) == 13 and len(lead_side) == 13


def test_split_redraws_per_round():
    keys = _keys(26)
    r1 = split_for_round(42, 1, keys, 0.5)
    r2 = split_for_round(42, 2, keys, 0.5)
    assert r1 != r2
    # size stays the clamp for a fixed N and frac
    assert len(r1) == len(r2) == 13
    # a different seed is a different partition
    assert split_for_round(7, 1, keys, 0.5) != r1


@pytest.mark.parametrize("n,frac,expected", [(26, 0.5, 13), (25, 0.5, 12), (10, 0.4, 4), (9, 0.9, 8), (9, 0.1, 1)])
def test_split_size_clamps(n, frac, expected):
    assert len(split_for_round(42, 3, _keys(n), frac)) == expected


# ---------------------------------------------------------------- delta math


def _sd(**tensors):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in tensors.items()}


def test_delta_merge_roundtrip():
    start = _sd(a=[[1.0, 2.0]], b=[[3.0]])
    lead_end = _sd(a=[[1.5, 2.5]], b=[[3.25]])
    helper_end = _sd(a=[[0.5, 1.0]], b=[[2.0]])
    d_lead = compute_delta(lead_end, start)
    d_helper = compute_delta(helper_end, start)
    # merged on the lead: its live weights plus the helper's change
    merged = merge(lead_end, d_helper)
    want_a = lead_end["a"] + helper_end["a"] - start["a"]
    want_b = lead_end["b"] + helper_end["b"] - start["b"]
    assert torch.allclose(merged["a"], want_a)
    assert torch.allclose(merged["b"], want_b)
    # order-independent: merging both deltas into start is the same point
    both = merge(merge(start, d_lead), d_helper)
    assert torch.allclose(both["a"], want_a)
    assert torch.allclose(both["b"], want_b)


def test_merge_passes_through_keys_without_delta():
    cur = _sd(a=[[1.0]], b=[[2.0]])
    out = merge(cur, {"a": torch.tensor([[0.5]])})
    assert torch.allclose(out["a"], torch.tensor([[1.5]]))
    assert torch.allclose(out["b"], cur["b"])


# ------------------------------------------------------- protocol over files


def _states(tmp_path, n=10, frac=0.4, seed=42):
    """A bound lead/helper pair sharing one dir over n items (helper gets 4, lead 6)."""
    keys = _keys(n)
    lead = ShareState("lead", str(tmp_path / "share"), frac, seed, "launch-1", log=lambda m: None)
    helper = ShareState("helper", str(tmp_path / "share"), frac, seed, "launch-1", log=lambda m: None)
    lead.set_keys(keys)
    helper.set_keys(keys)
    lead.bind(_net(1))
    helper.bind(_net(2))
    return lead, helper


def test_lead_helper_roundtrip_over_files(tmp_path):
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    # both sides now sit on the lead's round-1 start
    start = {k: v.clone() for k, v in lead._current_sd().items()}
    for k in start:
        assert torch.allclose(helper._current_sd()[k], start[k])

    # each side "trains" its share: nudge its own weights, step its target count.
    # driven helper-first because the lead's close blocks on the helper's delta
    # (in production the sides run concurrently)
    with torch.no_grad():
        lead._network.weight += 0.25
        helper._network.weight -= 0.5
    for _ in range(helper.my_target_steps(1)):
        helper.hook_end_of_step(step=0, is_save_step=False, is_sample_step=False, is_last_step=False)
    assert helper.round_no == 2 and helper._waiting
    for _ in range(lead.my_target_steps(1) - 1):
        lead.hook_end_of_step(step=0, is_save_step=False, is_sample_step=False, is_last_step=False)
    assert lead.round_no == 1  # one step of its share left: the round stays open
    lead.hook_end_of_step(step=0, is_save_step=False, is_sample_step=False, is_last_step=False)
    assert lead.round_no == 2
    assert not os.path.exists(os.path.join(str(tmp_path / "share"), "round_000001"))
    merged = lead._current_sd()
    want = start["weight"] + 0.25 - 0.5
    assert torch.allclose(merged["weight"], want)
    # round 2 starts exactly at the merged point
    from safetensors.torch import load_file

    start2 = load_file(os.path.join(str(tmp_path / "share"), "round_000002", "start.safetensors"))
    assert torch.allclose(start2["weight"], want)

    # the helper adopts the merged point at the top of its next step
    helper.hook_step_top()
    assert not helper._waiting
    assert torch.allclose(helper._current_sd()["weight"], want)

    # final round: helper posts its delta, lead merges and finishes
    with torch.no_grad():
        helper._network.weight += 1.0
        lead._network.weight += 2.0
    for _ in range(helper.my_target_steps(2)):
        helper.hook_end_of_step(step=0, is_save_step=False, is_sample_step=False, is_last_step=False)
    for _ in range(lead.my_target_steps(2) - 1):
        lead.hook_end_of_step(step=0, is_save_step=False, is_sample_step=False, is_last_step=False)
    lead.hook_end_of_step(step=0, is_save_step=False, is_sample_step=False, is_last_step=True)
    assert not lead.running
    assert os.path.exists(os.path.join(str(tmp_path / "share"), "final.json"))
    final_want = want + 2.0 + 1.0
    assert torch.allclose(lead._current_sd()["weight"], final_want)

    # a helper with budget left is released by final.json
    helper._begin_round(3)
    helper._waiting = True
    helper.hook_step_top()
    assert helper.stop_loop and not helper.running


def test_save_step_forces_early_round_close(tmp_path):
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    with torch.no_grad():
        helper._network.weight += 0.5
    for _ in range(helper.my_target_steps(1)):
        helper.hook_end_of_step(0, False, False, False)
    # the lead closes mid-share at a save step: checkpoint weights hold both halves
    lead.hook_end_of_step(step=0, is_save_step=True, is_sample_step=False, is_last_step=False)
    assert lead.round_no == 2
    round_start_w = helper._start_sd["weight"]
    assert torch.allclose(lead._current_sd()["weight"], round_start_w + 0.5)


def test_hold_folds_the_next_save_step_into_the_round(tmp_path):
    # saves fire one step index past a round boundary (round ends after L
    # steps at index L-1; save at index L): hold keeps the round open one more
    # step so the save coincides with the round close instead of opening a
    # 1-step round (which would also desync round numbering between the sides)
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    for _ in range(helper.my_target_steps(1)):
        helper.hook_end_of_step(0, False, False, False)
    target = lead.my_target_steps(1)
    for _ in range(target - 1):
        lead.hook_end_of_step(0, False, False, False, hold=False)
    # target reached, but the next step saves: hold the round open
    lead.hook_end_of_step(0, False, False, False, hold=True)
    assert lead.round_no == 1 and lead.steps_this_round == target
    # the save step itself then closes the round (no hold signal needed)
    lead.hook_end_of_step(0, True, False, False, hold=False)
    assert lead.round_no == 2 and lead.steps_this_round == 0


def test_owns_matches_role_split():
    keys = _keys(10)
    lead = ShareState("lead", "x", 0.4, 42, "l", log=lambda m: None)
    helper = ShareState("helper", "x", 0.4, 42, "l", log=lambda m: None)
    lead.set_keys(keys)
    helper.set_keys(keys)
    for st in (lead, helper):
        st._begin_round(1)
        st.running = True
    helper_keys = split_for_round(42, 1, keys, 0.4)
    for k in keys:
        assert helper.owns([k]) == (k in helper_keys)
        assert lead.owns([k]) == (k not in helper_keys)
    # a batch mixes roles only when it spans both sides
    a, b = sorted(helper_keys)[0], sorted(set(keys) - helper_keys)[0]
    assert not lead.owns([a, b]) and not helper.owns([a, b])


def test_stale_share_dir_refused(tmp_path):
    lead, _ = _states(tmp_path)
    os.makedirs(os.path.join(str(tmp_path / "share"), "round_000001"))
    with open(os.path.join(str(tmp_path / "share"), "round_000001", "job.json"), "w") as f:
        f.write("{}")
    with pytest.raises(ShareProtocolError):
        lead.hook_begin()


def test_dataset_mismatch_refused(tmp_path):
    lead, helper = _states(tmp_path)
    helper.set_keys(_keys(11))  # one item more than the lead sees
    lead.hook_begin()
    with pytest.raises(ShareProtocolError):
        helper.hook_begin()


def test_helper_budget_shortfall_refused_at_handshake(tmp_path):
    # the reviewer's scenario: N=4, frac=0.75 (helper n=3, lead L=1), both sides
    # 6 steps. The helper funds 2 rounds, the lead needs 6 -- without the guard
    # the helper exits after round 2 and the lead hangs on a delta until timeout.
    lead, helper = _states(tmp_path, n=4, frac=0.75)
    lead.total_steps = 6
    helper.total_steps = 6
    lead.hook_begin()
    with pytest.raises(ShareProtocolError, match="helper budget too small.*= 18"):
        helper.hook_begin()


def test_helper_budget_funding_every_round_passes(tmp_path):
    lead, helper = _states(tmp_path, n=4, frac=0.75)
    lead.total_steps = 6
    helper.total_steps = 18  # 6 rounds x n=3
    lead.hook_begin()
    helper.hook_begin()
    assert helper.running


def test_missing_delta_times_out(tmp_path):
    lead, _ = _states(tmp_path)
    lead.timeout_s = 0.5
    lead.hook_begin()
    lead._begin_round(1)  # already begun; keep the round open
    lead.steps_this_round = lead.my_target_steps(1)
    with pytest.raises(ShareTimeout):
        lead.hook_end_of_step(0, False, False, False)


def test_state_dict_fp32_copies_off_device():
    net = _net(3).to(torch.float64)
    sd = state_dict_fp32(net)
    assert sd["weight"].dtype == torch.float32
    assert sd["weight"].device.type == "cpu"
    sd["weight"] += 1.0
    assert not torch.allclose(sd["weight"], net.state_dict()["weight"].float())


# -------------------------------------------------------------------- config


def test_share_config_disabled_by_default_and_validates_when_enabled():
    assert not ShareConfig().enabled
    ok = ShareConfig(enabled=True, role="lead", dir="E:/share", frac=0.5)
    assert ok.enabled and ok.role == "lead" and ok.frac == 0.5
    assert ShareConfig(enabled=True, role="helper", dir="E:/share", frac=0.5, seed=7, timeout_s=600.0).seed == 7
    for bad in (
        {"role": "chauffeur"},
        {"dir": None},
        {"frac": 0.0},
        {"frac": 1.0},
        {"timeout_s": 0},
    ):
        kwargs = {"enabled": True, "dir": "E:/share"}
        kwargs.update(bad)
        with pytest.raises(ValueError):
            ShareConfig(**kwargs)
    # disabled configs skip validation entirely
    ShareConfig(role="chauffeur", frac=5.0)


def _share_cfg(**over):
    base = {"enabled": True, "role": "lead", "dir": "E:/share", "frac": 0.5}
    base.update(over)
    return ShareConfig(**base)


def test_validate_share_setup_rejects_incompatible_recipes():
    with pytest.raises(ValueError, match="batch_size"):
        validate_share_setup(_share_cfg(), TrainConfig(batch_size=2), has_reg_datasets=False)
    with pytest.raises(ValueError, match="gradient_accumulation"):
        validate_share_setup(_share_cfg(), TrainConfig(gradient_accumulation=2), has_reg_datasets=False)
    with pytest.raises(ValueError, match="EMA"):
        validate_share_setup(
            _share_cfg(), TrainConfig(ema_config={"use_ema": True}), has_reg_datasets=False
        )
    with pytest.raises(ValueError, match="datasets_reg"):
        validate_share_setup(_share_cfg(), TrainConfig(), has_reg_datasets=True)
    with pytest.raises(ValueError, match="at least 4"):
        validate_share_setup(_share_cfg(), TrainConfig(), has_reg_datasets=False, n_items=3)
    # the golden recipe's shape passes, and None/off is a no-op
    validate_share_setup(_share_cfg(), TrainConfig(batch_size=1, gradient_accumulation=1), has_reg_datasets=False, n_items=26)
    validate_share_setup(None, TrainConfig(batch_size=4), has_reg_datasets=True)


# ------------------------------------------------------------------ adaptive


def test_next_frac_gives_the_slower_card_the_smaller_share():
    # helper 2x slower per step -> helper share shrinks toward t_lead/(t_l+t_h)
    f = next_frac(t_lead=1.0, t_helper=2.0, current=0.5, center=0.5)
    assert f == pytest.approx(0.5 * 0.5 + 0.5 * (1.0 / 3.0))  # smoothed toward 0.333
    # repeated application converges into the band, never beyond it
    frac = 0.5
    for _ in range(50):
        frac = next_frac(1.0, 4.0, frac, 0.5)
    assert frac == pytest.approx(0.35, abs=0.01)  # band floor = 0.5 - 0.15
    # lead slower -> helper share grows, clamped to the band ceiling
    frac = 0.5
    for _ in range(50):
        frac = next_frac(4.0, 1.0, frac, 0.5)
    assert frac == pytest.approx(0.65, abs=0.01)
    # degenerate timings leave the share alone
    assert next_frac(0.0, 2.0, 0.5, 0.5) == 0.5
    assert next_frac(1.0, 0.0, 0.5, 0.5) == 0.5


def test_adaptive_roundtrip_over_files(tmp_path):
    # helper reports steps/seconds in delta metadata; lead adapts frac and
    # publishes it (plus its LRs) in the next round's job; helper adopts both
    lead = ShareState("lead", str(tmp_path / "s"), 0.5, 42, "l", log=lambda m: None, adaptive=True)
    helper = ShareState("helper", str(tmp_path / "s"), 0.5, 42, "l", log=lambda m: None, adaptive=True)
    for st in (lead, helper):
        st.set_keys(_keys(10))
    fake_opt = torch.optim.SGD(torch.nn.Linear(4, 4).parameters(), lr=0.1)
    lead.bind(_net(1), fake_opt)
    helper.bind(_net(2), torch.optim.SGD(torch.nn.Linear(4, 4).parameters(), lr=0.9))

    lead.hook_begin()
    helper.hook_begin()
    assert helper._optimizer.param_groups[0]["lr"] == pytest.approx(0.1)  # lead's round-1 LR adopted

    # round 1: helper "trains" slowly relative to the lead
    with torch.no_grad():
        helper._network.weight += 0.5
    helper._round_t0 = time.time() - 4.0  # pretend the round took 4s
    for _ in range(helper.my_target_steps(1)):
        helper.hook_end_of_step(0, False, False, False)
    delta_meta = _load_meta(os.path.join(str(tmp_path / "s"), "round_000001", "delta.safetensors"))
    assert int(delta_meta["steps"]) == 5 and float(delta_meta["seconds"]) >= 3.9

    lead._round_t0 = time.time() - 1.0  # the lead's same round took 1s
    for _ in range(lead.my_target_steps(1) - 1):
        lead.hook_end_of_step(0, False, False, False)
    lead.hook_end_of_step(0, False, False, False)
    assert lead.round_no == 2
    assert lead.frac < 0.5  # helper slower (4s for 5 steps vs 1s for 5) -> smaller helper share
    job2 = json.load(open(os.path.join(str(tmp_path / "s"), "round_000002", "job.json")))
    assert abs(job2["frac"] - lead.frac) < 1e-9 and job2["adaptive"] is True
    assert isinstance(job2["lrs"], list) and len(job2["lrs"]) == len(fake_opt.param_groups)

    helper.hook_step_top()
    assert helper.frac == pytest.approx(lead.frac)
    assert helper._optimizer.param_groups[0]["lr"] == pytest.approx(job2["lrs"][0])
    # the helper's round-2 target moved with the adopted frac
    assert helper.my_target_steps(2) == len(split_for_round(42, 2, _keys(10), helper.frac))


def test_fixed_frac_rejects_job_frac_drift(tmp_path):
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    # a job frac that differs from this side's config is a protocol error
    # unless adaptive is on
    job_path = os.path.join(str(tmp_path / "share"), "round_000002", "job.json")
    os.makedirs(os.path.dirname(job_path))
    with open(job_path, "w") as f:
        json.dump({"round": 2, "frac": 0.9, "seed": 42, "n_items": len(_keys(10)), "steps": 0}, f)
    helper._begin_round(2)
    helper._waiting = True
    with pytest.raises(ShareProtocolError, match="frac mismatch"):
        helper.hook_step_top()
