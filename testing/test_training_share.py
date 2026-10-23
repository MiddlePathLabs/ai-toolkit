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
    adaptive_helper_budget,
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
    lead.catch_up = False   # fail-fast path (catch_up default absorbs instead)
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
    # the per-iteration accumulation window is a separate knob and equally
    # incompatible: a partial window cannot be exchanged as a round delta
    for bad_steps in (4, -1):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            validate_share_setup(
                _share_cfg(), TrainConfig(gradient_accumulation_steps=bad_steps), has_reg_datasets=False
            )
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


@pytest.mark.parametrize(
    "n,center,lead_steps,save_every",
    [(100, 0.1, 900, 0), (100, 0.5, 300, 50), (4, 0.5, 20, 2),
     (100, 0.9, 225, 0), (4, 0.5, 8, 1)],
)
def test_adaptive_budget_funds_complete_helper_shares(
    tmp_path, monkeypatch, n, center, lead_steps, save_every,
):
    import toolkit.training_share as ts

    monkeypatch.setattr(ts.time, "time", lambda: 1000.0)
    lead, helper = _states(tmp_path, n=n, frac=center)
    lead.adaptive = helper.adaptive = True
    lead.catch_up = helper.catch_up = False
    lead.timeout_s = helper.timeout_s = 0
    lead.total_steps = lead_steps
    lead.hook_begin()
    helper.hook_begin()
    allowance = adaptive_helper_budget(n, helper.lead_steps, center)
    start = lead._current_sd()["weight"]
    helper_steps = 0
    prepared_round = 0
    for step in range(lead_steps):
        if prepared_round != lead.round_no:
            prepared_round = lead.round_no
            assert helper.round_no == lead.round_no and not helper._waiting
            for _ in range(helper.my_target_steps(helper.round_no)):
                assert helper_steps < allowance, "helper exhausted before completing its assigned share"
                with torch.no_grad():
                    helper._network.weight.add_(1)
                helper_steps += 1
                helper._round_t0 = 999.9
                helper.hook_end_of_step(
                    helper_steps - 1, False, False, helper_steps == allowance,
                )
        lead._round_t0 = 900.0
        old_round = lead.round_no
        lead.hook_end_of_step(
            step, bool(save_every and step and step % save_every == 0), False,
            step + 1 == lead_steps,
            hold=bool(save_every and step + 1 < lead_steps and (step + 1) % save_every == 0),
        )
        if old_round != lead.round_no or not lead.running:
            helper.hook_step_top()
    assert helper.stop_loop and not helper.running
    assert torch.allclose(lead._current_sd()["weight"], start + helper_steps)


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


# ------------------------------------------------------------------ catch-up


def test_catch_up_absorbs_a_missed_round(tmp_path):
    lead, _helper = _states(tmp_path)
    lead.catch_up = True
    lead.timeout_s = 0.3
    lead.hook_begin()
    for _ in range(lead.my_target_steps(1)):
        lead.hook_end_of_step(0, False, False, False)
    # the round closed on a miss instead of raising
    assert lead.round_no == 2 and lead._consecutive_missed == 1
    owed = split_for_round(42, 1, _keys(10), 0.4)
    assert lead._owed == owed                       # the helper's round-1 items
    # the next job excludes them so the helper stays off them
    job2 = json.load(open(os.path.join(str(tmp_path / "share"), "round_000002", "job.json")))
    assert job2["excluded"] == sorted(owed)
    # and the lead's round-2 share grew to cover them
    helper2 = split_for_round(42, 2, _keys(10), 0.4)
    assert lead.my_target_steps(2) == 10 - len(helper2 - owed)
    # owed items are lead-owned this round no matter where the redrawn split
    # places them (and the helper, symmetrically, excludes them)
    for k in owed:
        assert lead.owns([k])


def test_catch_up_disabled_still_raises(tmp_path):
    lead, _helper = _states(tmp_path)
    lead.catch_up = False
    lead.timeout_s = 0.3
    lead.hook_begin()
    with pytest.raises(ShareTimeout):
        for _ in range(lead.my_target_steps(1)):
            lead.hook_end_of_step(0, False, False, False)


def test_solo_streak_skips_the_deadline_and_rejoins_on_a_delta(tmp_path):
    from safetensors.torch import save_file

    lead, _helper = _states(tmp_path)
    lead.catch_up = True
    lead.timeout_s = 0.3
    lead.hook_begin()
    for _round in range(3):  # rounds 1-3 all miss
        for _ in range(lead.my_target_steps(lead.round_no)):
            lead.hook_end_of_step(0, False, False, False)
    assert lead._consecutive_missed == 3
    # the 4th miss must not sit out a deadline: close is immediate
    lead.timeout_s = 9999.0
    t0 = time.time()
    for _ in range(lead.my_target_steps(lead.round_no)):
        lead.hook_end_of_step(0, False, False, False)
    assert time.time() - t0 < 2.0
    assert lead._consecutive_missed == 4
    # a delta that shows up is consumed and the streak resets
    rd = os.path.join(str(tmp_path / "share"), f"round_{lead.round_no:06d}")
    save_file(
        {k: torch.zeros_like(v) for k, v in lead._start_sd.items()},
        os.path.join(rd, "delta.safetensors"),
        metadata={"round": str(lead.round_no), "steps": "4", "seconds": "10"},
    )
    for _ in range(lead.my_target_steps(lead.round_no)):
        lead.hook_end_of_step(0, False, False, False)
    assert lead._consecutive_missed == 0 and not lead._owed


def test_helper_excludes_owed_items_and_zero_deltas_an_empty_round(tmp_path):
    from safetensors.torch import save_file

    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    keys = _keys(10)
    excluded = sorted(split_for_round(42, 2, keys, 0.4))  # everything the helper would own
    rd = os.path.join(str(tmp_path / "share"), "round_000002")
    os.makedirs(rd)
    save_file(lead._current_sd(), os.path.join(rd, "start.safetensors"))
    with open(os.path.join(rd, "job.json"), "w") as f:
        json.dump({"round": 2, "frac": 0.4, "seed": 42, "n_items": 10, "steps": 0,
                   "adaptive": False, "excluded": excluded}, f)
    helper._begin_round(2)
    helper._waiting = True
    # partial exclusion shrinks the share and flips ownership of excluded items
    helper._excluded = {excluded[0]}
    assert not helper.owns([excluded[0]])
    assert helper.my_target_steps(2) == len(split_for_round(42, 2, keys, 0.4)) - 1
    helper._excluded = set(excluded)
    # round 3 will never come in this test: final.json releases the helper's
    # wait after it posts round 2's zero delta
    with open(os.path.join(str(tmp_path / "share"), "final.json"), "w") as f:
        json.dump({"launch": "x"}, f)
    helper.hook_step_top()
    # nothing to train: a zero delta is posted and the helper moved on
    assert helper.round_no == 3 and helper.stop_loop and not helper.running
    assert os.path.exists(os.path.join(rd, "delta.safetensors"))


def test_write_delta_discards_a_retired_round(tmp_path):
    # a round the lead already pruned can never consume our delta (the lead
    # only reads its CURRENT round), so a late delta is discarded quietly and
    # must not resurrect an orphan round dir
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    import shutil

    rd = os.path.join(str(tmp_path / "share"), "round_000001")
    shutil.rmtree(rd, ignore_errors=True)
    helper.write_delta(1, compute_delta(helper._current_sd(), helper._start_sd))
    assert not os.path.exists(rd)

    # a live round still takes the delta
    lead2, helper2 = _states(tmp_path / "b")
    lead2.hook_begin()
    helper2.hook_begin()
    helper2.write_delta(1, compute_delta(helper2._current_sd(), helper2._start_sd))
    assert os.path.exists(os.path.join(str(tmp_path / "b"), "share", "round_000001", "delta.safetensors"))


@pytest.mark.parametrize("phase", ["serialize", "rename"])
def test_write_delta_tolerates_the_lead_pruning_mid_write(tmp_path, monkeypatch, phase):
    import safetensors
    import safetensors.torch
    import toolkit.training_share as ts

    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    rd = os.path.join(helper.share_dir, "round_000001")
    failures = []

    if phase == "serialize":
        original = safetensors.torch.save_file

        def pruned_mid_write(*args, **kwargs):
            lead.prune_round(1)
            try:
                return original(*args, **kwargs)
            except safetensors.SafetensorError as exc:
                failures.append(exc)
                raise

        monkeypatch.setattr(safetensors.torch, "save_file", pruned_mid_write)
    else:
        original = ts._replace

        def pruned_mid_write(path):
            lead.prune_round(1)
            try:
                return original(path)
            except FileNotFoundError as exc:
                failures.append(exc)
                raise

        monkeypatch.setattr(ts, "_replace", pruned_mid_write)

    helper.write_delta(1, compute_delta(helper._current_sd(), helper._start_sd))
    assert len(failures) == 1
    assert not os.path.exists(rd)


@pytest.mark.parametrize("phase", ["serialize", "rename"])
def test_write_delta_propagates_live_round_failures(tmp_path, monkeypatch, phase):
    from safetensors import SafetensorError
    import toolkit.training_share as ts

    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    rd = os.path.join(helper.share_dir, "round_000001")
    path = os.path.join(rd, "delta.safetensors")
    if phase == "serialize":
        os.mkdir(path + ".tmp")
        error = SafetensorError
    else:
        original = ts._replace

        def missing_temporary_file(path):
            os.remove(path + ".tmp")
            original(path)

        monkeypatch.setattr(ts, "_replace", missing_temporary_file)
        error = FileNotFoundError

    with pytest.raises(error):
        helper.write_delta(1, compute_delta(helper._current_sd(), helper._start_sd))
    assert os.path.isfile(os.path.join(rd, "job.json"))
    assert os.path.isfile(os.path.join(rd, "start.safetensors"))
    assert not os.path.exists(path)


@pytest.mark.parametrize("outcome", ["newer", "final", "timeout"])
def test_wait_start_recovers_from_an_orphan_job(tmp_path, monkeypatch, outcome):
    import toolkit.training_share as ts

    lead, helper = _states(tmp_path)
    lead.adaptive = helper.adaptive = True
    lead.frac = 0.6
    lead.total_steps = 12
    lead._optimizer = SimpleNamespace(param_groups=[{"lr": 0.2}])
    helper._optimizer = SimpleNamespace(param_groups=[{"lr": 0.9}])
    lead._owed = set(lead.keys)
    lead.hook_begin()
    orphan = os.path.join(helper.share_dir, "round_000001")
    os.remove(os.path.join(orphan, "start.safetensors"))
    helper.timeout_s = 1.0
    clock = [0.0]
    retries = []
    monkeypatch.setattr(ts.time, "time", lambda: clock[0])

    def advance(seconds):
        retries.append(seconds)
        assert len(retries) <= 4, "orphan start retries bypassed the deadline"
        assert helper.frac == 0.4
        assert helper._optimizer.param_groups[0]["lr"] == 0.9
        assert not helper._budget_checked and helper.lead_steps == 0
        assert not helper._excluded
        clock[0] += seconds
        if outcome == "newer":
            lead.frac = 0.5
            lead._optimizer.param_groups[0]["lr"] = 0.3
            lead._owed.clear()
            lead.open_round(2, lead._current_sd())
        elif outcome == "final":
            lead.write_final()

    monkeypatch.setattr(ts.time, "sleep", advance)
    if outcome == "timeout":
        with pytest.raises(ShareTimeout, match="no start"):
            helper.wait_start(1)
        assert clock[0] == 1.0
    else:
        start, job = helper.wait_start(1)
        if outcome == "newer":
            assert helper.round_no == 2 and job["round"] == 2
            for key, value in lead._current_sd().items():
                assert torch.equal(start[key], value)
            assert helper.frac == 0.5
            assert helper._optimizer.param_groups[0]["lr"] == 0.3
            assert helper._budget_checked and helper.lead_steps == 12
            assert not helper._excluded
        else:
            assert start is None and job is None
    assert retries
    assert os.path.isfile(os.path.join(orphan, "job.json"))
    assert not os.path.exists(os.path.join(orphan, "start.safetensors"))
    if outcome != "newer":
        assert helper.round_no == 0 and helper.frac == 0.4
        assert helper._optimizer.param_groups[0]["lr"] == 0.9
        assert not helper._budget_checked and helper.lead_steps == 0
        assert not helper._excluded


def test_late_initial_helper_zeroes_empty_round_then_rejoins_and_finishes(tmp_path, monkeypatch):
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    start = lead._current_sd()
    lead.prune_round(1)
    lead._owed = set(lead.helper_keys(2))
    lead.open_round(2, start)
    lead._begin_round(2)
    helper.hook_begin()
    rd2 = os.path.join(helper.share_dir, "round_000002")
    assert helper.round_no == 2 and helper.running and helper._waiting
    assert helper.my_target_steps(2) == 0
    assert not any(helper.owns([key]) for key in helper.keys)
    assert not os.path.exists(os.path.join(rd2, "delta.safetensors"))
    with torch.no_grad():
        lead._network.weight += 0.25
    published = []
    original = helper.write_delta

    def complete_empty_round(round_no, delta, seconds=0.0):
        original(round_no, delta, seconds)
        if round_no == 2:
            saved_delta, meta = lead._read_delta(2)
            published.append((saved_delta, meta))
            for _ in range(lead.my_target_steps(2)):
                lead.hook_end_of_step(0, False, False, False)

    monkeypatch.setattr(helper, "write_delta", complete_empty_round)
    helper.hook_step_top()
    assert len(published) == 1
    delta, meta = published[0]
    assert meta["steps"] == 0 and meta["seconds"] == 0
    assert all(torch.count_nonzero(value).item() == 0 for value in delta.values())
    assert not os.path.exists(rd2)
    assert helper.round_no == lead.round_no == 3
    assert helper.running and not helper._waiting and not helper.stop_loop
    assert helper.my_target_steps(3) > 0
    assert torch.allclose(helper._current_sd()["weight"], start["weight"] + 0.25)
    assert torch.equal(helper._current_sd()["bias"], start["bias"])

    with torch.no_grad():
        lead._network.weight += 0.125
        helper._network.weight -= 0.5
    for _ in range(helper.my_target_steps(3)):
        helper.hook_end_of_step(0, False, False, False)
    for step in range(lead.my_target_steps(3)):
        lead.hook_end_of_step(0, False, False, step == lead.my_target_steps(3) - 1)
    assert torch.allclose(lead._current_sd()["weight"], start["weight"] - 0.125)
    assert torch.equal(lead._current_sd()["bias"], start["bias"])
    assert os.path.isfile(os.path.join(helper.share_dir, "final.json"))
    helper.hook_step_top()
    assert helper.stop_loop and not helper.running


def test_late_helper_skips_retired_rounds_to_rejoin(tmp_path):
    # reviewer's scenario: the lead misses rounds 1-3 (helper presumed slow)
    # and advances to round 4, pruning 1-3. A helper still finishing round 1
    # must not wait for the pruned round 2 -- it skips to the lead's live
    # round and rejoins there instead of dying on ShareTimeout.
    from safetensors.torch import load_file

    lead, helper = _states(tmp_path)
    lead.catch_up = True
    lead.timeout_s = 0.3
    helper.timeout_s = 5.0
    lead.hook_begin()
    helper.hook_begin()
    for _round in range(3):  # the lead closes rounds 1-3 on misses, solo at 3
        for _ in range(lead.my_target_steps(lead.round_no)):
            lead.hook_end_of_step(0, False, False, False)
    assert lead.round_no == 4 and lead._consecutive_missed == 3
    assert not os.path.exists(os.path.join(str(tmp_path / "share"), "round_000001"))

    # the helper finishes round 1: the delta is discarded (round retired) and
    # its next wait skips straight to the live round 4
    for _ in range(helper.my_target_steps(1)):
        helper.hook_end_of_step(0, False, False, False)
    assert helper.round_no == 2 and helper._waiting
    t0 = time.time()
    helper.hook_step_top()
    assert helper.round_no == 4 and not helper._waiting
    assert time.time() - t0 < 4.0  # no ShareTimeout sit-out for rounds 2-3
    start4 = load_file(os.path.join(str(tmp_path / "share"), "round_000004", "start.safetensors"))
    assert torch.allclose(helper._current_sd()["weight"], start4["weight"])
    # the discarded delta did not resurrect the pruned round
    assert not os.path.exists(os.path.join(str(tmp_path / "share"), "round_000001"))


def test_helper_save_cadence_never_closes_or_holds_its_round(tmp_path):
    # only the lead's save/sample cadence is snapped to round boundaries; a
    # helper-local save step used to force its rounds closed early and
    # under-train its share every round. The protocol must ignore it.
    lead, helper = _states(tmp_path)
    lead.hook_begin()
    helper.hook_begin()
    target = helper.my_target_steps(1)
    for _ in range(target - 1):
        helper.hook_end_of_step(0, is_save_step=True, is_sample_step=True, is_last_step=False)
        assert helper.round_no == 1
    # at the target with a next-step save pending, hold is ignored too
    helper.hook_end_of_step(0, False, False, False, hold=True)
    assert helper.round_no == 2 and helper.steps_this_round == 0


def test_stale_check_tolerates_tmp_debris_but_not_protocol_files(tmp_path):
    lead, _ = _states(tmp_path)
    rd = os.path.join(str(tmp_path / "share"), "round_000001")
    os.makedirs(rd)
    with open(os.path.join(rd, "delta.safetensors.tmp"), "w") as f:
        f.write("debris from a crashed write")
    lead.hook_begin()  # leftover tmp alone is not a foreign-delta hazard

    lead2, _ = _states(tmp_path / "b")
    rd2 = os.path.join(str(tmp_path / "b"), "share", "round_000001")
    os.makedirs(rd2)
    with open(os.path.join(rd2, "delta.safetensors"), "w") as f:
        f.write("stale foreign delta")
    with pytest.raises(ShareProtocolError):
        lead2.hook_begin()
