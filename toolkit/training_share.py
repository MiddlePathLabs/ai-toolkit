"""Two-GPU training share: LoRA delta-combine over a shared folder (slice 1).

Ported in spirit from Fizgig v8.9 "Helpers" (training share). Two processes
train the same LoRA on disjoint halves of the same dataset and merge their
weight changes every round:

  * the LEAD (saves checkpoints, renders samples) and a HELPER each start a
    round from an identical snapshot of the LoRA weights;
  * each side trains only its own share of the items that round (the split is
    redrawn per round from a hash of seed, round number and item path, so
    both sides compute it independently and identically);
  * at the round's end the helper writes its weight DELTA and the lead merges
    its own live weights with that delta; both continue from the merged point.

Slice 1 scope: one helper, fixed frac, both processes on one machine sharing a
folder (identical dataset paths). A missed delta raises (the round dir is kept
for diagnosis); degrade-to-solo and helper catch-up are out of scope. Mid-round
resume of a share run is not supported -- restart the arm.

Protocol files under ShareConfig.dir:

  round_%06d/start.safetensors  lead    fp32 network state dict both sides start the round from
  round_%06d/job.json           lead    {round, frac, seed, launch, n_items}
  round_%06d/delta.safetensors  helper  fp32 weight change over the round
  final.json                    lead    written after the last merge; a waiting helper exits on it

The share dir must not already contain round_000001 when the lead starts (a
stale dir from an earlier run would silently merge foreign deltas).

Step budgets keep a share run comparable to a solo run at the same total image
count: with combined target T and helper share frac, lead trains round(T*frac)
steps and helper T - that, each with its own LR schedule over its own budget.
For any fixed frac, each side's schedule position at combined image k is
cos(pi*k/T) -- exactly the solo trajectory.
"""

import hashlib
import json
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Set

START_FILE = "start.safetensors"
DELTA_FILE = "delta.safetensors"
JOB_FILE = "job.json"
FINAL_FILE = "final.json"

MIN_DEADLINE_S = 300.0


class ShareTimeout(RuntimeError):
    """The other side did not produce its round artifact within the deadline."""


class ShareProtocolError(RuntimeError):
    """The share dir state or a received artifact is not what this side expects."""


def validate_share_setup(share_config, train_config, has_reg_datasets: bool, n_items: int = 0) -> None:
    """Cross-config fail-fasts for a share run. Config-object level: no GPU, no files.

    ``share_config`` may be None (share off). ``n_items`` is checked when the
    dataset size is already known (0 = not known yet).
    """
    if share_config is None or not getattr(share_config, 'enabled', False):
        return
    if getattr(train_config, 'batch_size', 1) != 1:
        raise ValueError("training share requires train.batch_size 1 (one item per step on both sides)")
    if getattr(train_config, 'gradient_accumulation', 1) != 1:
        raise ValueError("training share requires train.gradient_accumulation 1 (a batch must not span the split)")
    ema = getattr(train_config, 'ema_config', None)
    if ema is not None and getattr(ema, 'use_ema', False):
        raise ValueError("training share requires EMA off (slice 1 does not merge EMA shadows)")
    if has_reg_datasets:
        raise ValueError("training share does not support datasets_reg (reg alternation breaks step parity)")
    if n_items and n_items < 4:
        raise ValueError(f"training share needs at least 4 items to split, dataset has {n_items}")


def split_for_round(seed: int, round_no: int, keys: Iterable[str], frac: float) -> Set[str]:
    """The HELPER's item keys for a round.

    Deterministic on both sides with zero communication: order every key by
    md5(seed:round:key) and give the helper the first ``frac`` fraction of
    them. ``frac`` is the helper's share of items; the lead owns the rest.
    Both sides always get at least one item.
    """
    ordered = sorted(
        (str(k) for k in keys),
        key=lambda k: hashlib.md5(f"{int(seed)}:{int(round_no)}:{k}".encode()).hexdigest(),
    )
    n = round(len(ordered) * float(frac))
    if len(ordered) >= 2:
        n = min(max(n, 1), len(ordered) - 1)
    return set(ordered[:n])


def collect_item_paths(dataloader) -> List[str]:
    """Every dataset item path a dataloader can yield, for the split keys.

    Handles the shapes ``get_dataloader_from_datasets`` can produce: the
    dataset directly, a list of datasets, or a concat/pooled wrapper exposing
    ``.datasets``. Paths are the FileItemDTO path strings both processes see.
    """
    out: List[str] = []
    stack = [getattr(dataloader, 'dataset', None)]
    while stack:
        d = stack.pop()
        if d is None:
            continue
        if isinstance(d, (list, tuple)):
            stack.extend(d)
        elif hasattr(d, 'file_list'):
            out.extend(str(fi.path) for fi in d.file_list)
        elif hasattr(d, 'datasets'):
            stack.extend(list(d.datasets))
    return sorted(set(out))


def state_dict_fp32(network) -> Dict[str, "torch.Tensor"]:
    """A detached CPU fp32 copy of every entry of ``network.state_dict()``."""
    return {k: v.detach().float().cpu().clone() for k, v in network.state_dict().items()}


def compute_delta(cur: Dict[str, "torch.Tensor"], start: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """``cur - start`` over the shared keys, fp32 on CPU."""
    return {k: cur[k].float() - start[k].float() for k in cur if k in start}


def merge(current: Dict[str, "torch.Tensor"], delta: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """``current + delta`` per key; keys without a delta pass through unchanged."""
    return {k: (v.float() + delta[k].float()) if k in delta else v for k, v in current.items()}


def _replace(path: str) -> None:
    # Windows can hold the destination open for a moment (indexer, AV); retry like the trainer's config writes.
    for attempt in range(20):
        try:
            os.replace(path + ".tmp", path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _write_json(path: str, data: dict) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    _replace(path)


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_sd(path: str, sd: Dict[str, "torch.Tensor"], metadata: Optional[dict] = None) -> None:
    from safetensors.torch import save_file

    meta = {k: str(v) for k, v in (metadata or {}).items()}
    save_file({k: v.contiguous() for k, v in sd.items()}, path + ".tmp", metadata=meta)
    _replace(path)


def _load_sd(path: str) -> Dict[str, "torch.Tensor"]:
    from safetensors.torch import load_file

    return load_file(path)


def _load_meta(path: str) -> dict:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        return dict(f.metadata() or {})


def next_frac(t_lead: float, t_helper: float, current: float, center: float,
              band: float = 0.15, smooth: float = 0.5) -> float:
    """Adaptive share: the helper's fraction for the next round.

    ``t_*`` are per-step times; equal wall-clock per round needs
    n/L = t_lead/t_helper, i.e. frac = t_lead / (t_lead + t_helper) -- the
    SLOWER card gets the SMALLER share. Smoothed toward the measurement and
    clamped to a band around the configured center so plan budgets stay
    meaningful.
    """
    if t_lead <= 0 or t_helper <= 0:
        return current
    raw = t_lead / (t_lead + t_helper)
    smoothed = (1.0 - smooth) * current + smooth * raw
    return min(max(smoothed, center - band, 0.25), center + band, 0.75)

    return load_file(path)


def round_dir(share_dir: str, round_no: int) -> str:
    return os.path.join(share_dir, f"round_{int(round_no):06d}")


def wait_for_file(path: str, deadline_s: float, on_wait: Optional[Callable[[], None]] = None) -> bool:
    """Block until ``path`` exists. True if it appeared, False past the deadline."""
    announced = False
    deadline = time.time() + max(float(deadline_s), 0.0)
    while not os.path.exists(path):
        if time.time() >= deadline:
            return False
        if on_wait is not None and not announced:
            on_wait()
            announced = True
        time.sleep(0.5)
    return True


class ShareState:
    """Per-process round bookkeeping and file protocol for one share run.

    The trainer touches this through four points: :meth:`hook_begin` (before
    the training loop), :meth:`owns` (per batch fetch), :meth:`hook_step_top`
    (top of each step; the helper blocks here between rounds) and
    :meth:`hook_end_of_step` (after each completed step; rounds close here,
    before the loop's save/sample blocks run).
    """

    def __init__(
        self, role: str, share_dir: str, frac: float, seed: int, launch_id: str,
        timeout_s: float = 900.0, log: Callable[[str], None] = print,
        total_steps: int = 0, adaptive: bool = False, frac_center: Optional[float] = None,
    ):
        if role not in ("lead", "helper"):
            raise ValueError(f"share role must be 'lead' or 'helper', got {role!r}")
        self.role = role
        self.share_dir = share_dir
        self.frac = float(frac)                 # current round's helper share
        self.frac_center = float(frac_center if frac_center is not None else frac)
        self.adaptive = bool(adaptive)
        self.seed = int(seed)
        self.launch_id = str(launch_id)
        self.timeout_s = float(timeout_s)
        self.total_steps = int(total_steps)   # this side's train.steps budget
        self._log = log

        self.keys: List[str] = []
        self.running = False
        self.stop_loop = False            # helper: final.json seen, leave the step loop
        self.round_no = 0
        self.steps_this_round = 0
        self.delta_written = False
        self._round_t0: Optional[float] = None
        self._last_round_seconds: Optional[float] = None
        self._start_sd: Optional[dict] = None   # this side's round-start snapshot
        self._helper_keys_cache: Optional[Set[str]] = None
        self._cache_round = -1
        self._cache_frac = -1.0
        self._network = None
        self._optimizer = None
        self._waiting = False
        self._budget_checked = False
        self.combined_done = 0                 # images trained across both sides

    # ------------------------------------------------------------------ setup

    def bind(self, network, optimizer=None) -> None:
        self._network = network
        self._optimizer = optimizer

    def set_keys(self, keys: Iterable[str]) -> None:
        self.keys = sorted({str(k) for k in keys})

    # ------------------------------------------------------------------ split

    def helper_keys(self, round_no: int) -> Set[str]:
        if (self._helper_keys_cache is None or round_no != self._cache_round
                or self.frac != self._cache_frac):
            self._helper_keys_cache = split_for_round(self.seed, round_no, self.keys, self.frac)
            self._cache_round = round_no
            self._cache_frac = self.frac
        return self._helper_keys_cache

    def owns(self, paths: Iterable[str]) -> bool:
        """True when every path in a fetched batch belongs to this side this round."""
        if not self.running:
            return True
        helper = self.helper_keys(self.round_no)
        for p in paths:
            is_helpers = str(p) in helper
            if is_helpers != (self.role == "helper"):
                return False
        return True

    def my_target_steps(self, round_no: int) -> int:
        """How many optimizer steps this side trains in the round (its item share)."""
        n_helper = len(self.helper_keys(round_no))
        return n_helper if self.role == "helper" else max(len(self.keys) - n_helper, 1)

    # ------------------------------------------------------------------ files

    def open_round(self, round_no: int, start_sd: Dict) -> None:
        rd = round_dir(self.share_dir, round_no)
        if os.path.exists(os.path.join(rd, JOB_FILE)):
            raise ShareProtocolError(
                f"{rd} already has a job file; a stale share dir would merge foreign deltas. "
                f"Delete {self.share_dir} and restart both sides."
            )
        os.makedirs(rd, exist_ok=True)
        _save_sd(os.path.join(rd, START_FILE), start_sd)
        job = {
            "round": int(round_no), "frac": self.frac, "seed": self.seed,
            "launch": self.launch_id, "n_items": len(self.keys),
            "steps": self.total_steps, "adaptive": self.adaptive,
            "combined": self.combined_done,
        }
        if self.adaptive and self._optimizer is not None:
            # per-round LR handoff: the helper holds the lead's rate for the
            # round instead of stepping its own schedule (whose step count no
            # longer tracks plan when frac adapts)
            job["lrs"] = [float(g["lr"]) for g in self._optimizer.param_groups]
        _write_json(os.path.join(rd, JOB_FILE), job)

    def wait_delta(self, round_no: int):
        """The helper's delta plus its metadata dict ({steps, seconds})."""
        path = os.path.join(round_dir(self.share_dir, round_no), DELTA_FILE)
        if not wait_for_file(path, self.deadline_s(), on_wait=lambda: self._log(
            f"[share] round {round_no}: waiting for the helper's delta"
        )):
            raise ShareTimeout(
                f"[share] round {round_no}: no delta from the helper within {self.deadline_s():.0f}s "
                f"({path}); the round dir is preserved for diagnosis"
            )
        job = _read_json(os.path.join(round_dir(self.share_dir, round_no), JOB_FILE))
        if int(job.get("round", -1)) != int(round_no):
            raise ShareProtocolError(f"[share] round {round_no}: job.json is from another round: {job}")
        meta = _load_meta(path)
        meta = {k: (float(v) if k in ("steps", "seconds") else v) for k, v in meta.items()}
        return _load_sd(path), meta

    def write_delta(self, round_no: int, delta: Dict, seconds: float = 0.0) -> None:
        _save_sd(
            os.path.join(round_dir(self.share_dir, round_no), DELTA_FILE), delta,
            metadata={"round": int(round_no), "steps": int(self.steps_this_round), "seconds": round(seconds, 1)},
        )

    def wait_start(self, round_no: int):
        """Block for the lead's round start. Returns (state_dict, job), or (None, None) when the lead finished.

        Under adaptive sharing the helper ADOPTS the round's frac (and the
        lead's LRs) from the job file; the config frac is only the round-1
        value. ``final.json`` is watched for while waiting, so a helper that
        still has step budget left is released promptly when the lead
        finishes instead of sitting out its whole deadline.
        """
        job_path = os.path.join(round_dir(self.share_dir, round_no), JOB_FILE)
        final_path = os.path.join(self.share_dir, FINAL_FILE)
        deadline = time.time() + self.deadline_s()
        announced = False
        while True:
            if os.path.exists(job_path):
                job = _read_json(job_path)
                if int(job.get("round", -1)) != int(round_no):
                    raise ShareProtocolError(f"[share] round {round_no}: job.json is from another round: {job}")
                if int(job.get("n_items", -1)) != len(self.keys):
                    raise ShareProtocolError(
                        f"[share] dataset mismatch: the lead trains {job.get('n_items')} items, "
                        f"this side sees {len(self.keys)}"
                    )
                if int(job.get("seed", self.seed)) != self.seed:
                    raise ShareProtocolError(f"[share] seed mismatch with the lead: {job}")
                job_frac = float(job.get("frac", self.frac))
                if self.adaptive:
                    if not 0.0 < job_frac < 1.0:
                        raise ShareProtocolError(f"[share] round {round_no}: bad adaptive frac in job: {job}")
                    self.frac = job_frac
                elif abs(job_frac - self.frac) > 1e-9:
                    raise ShareProtocolError(f"[share] frac mismatch with the lead: {job}")
                if not self._budget_checked:
                    lead_steps = int(job.get("steps") or 0)
                    if self.total_steps and lead_steps and not self.adaptive:
                        # the helper must outlive every lead round: rounds x its share.
                        # A helper that runs out first leaves the lead waiting on a
                        # delta that can never arrive (it just times out later).
                        # (adaptive runs skip this: the helper's loop budget is
                        # inflated by the trainer and it stops on final.json)
                        n = len(self.helper_keys(1))
                        lead_share = max(len(self.keys) - n, 1)
                        rounds_needed = -(-lead_steps // lead_share)  # ceil
                        needed = rounds_needed * n
                        if self.total_steps < needed:
                            raise ShareProtocolError(
                                f"[share] helper budget too small: the lead's {lead_steps} steps at "
                                f"{lead_share}/round need {rounds_needed} rounds x {n} helper steps = {needed}; "
                                f"this helper has {self.total_steps}. Set helper train.steps to {needed}."
                            )
                    self._budget_checked = True
                lrs = job.get("lrs") if self.adaptive else None
                if lrs and self._optimizer is not None:
                    groups = self._optimizer.param_groups
                    if len(lrs) == len(groups):
                        for g, lr in zip(groups, lrs):
                            g["lr"] = float(lr)
                return _load_sd(os.path.join(round_dir(self.share_dir, round_no), START_FILE)), job
            if os.path.exists(final_path):
                return None, None
            if time.time() >= deadline:
                raise ShareTimeout(
                    f"[share] round {round_no}: no start from the lead within {self.deadline_s():.0f}s"
                )
            if not announced:
                self._log(f"[share] round {round_no}: waiting for the lead's start")
                announced = True
            time.sleep(0.5)

    def prune_round(self, round_no: int) -> None:
        import shutil

        shutil.rmtree(round_dir(self.share_dir, round_no), ignore_errors=True)

    def write_final(self) -> None:
        _write_json(os.path.join(self.share_dir, FINAL_FILE), {"launch": self.launch_id, "at": time.time()})

    def deadline_s(self) -> float:
        base = MIN_DEADLINE_S if self._last_round_seconds is None else max(MIN_DEADLINE_S, 3.0 * self._last_round_seconds)
        return min(base, self.timeout_s)

    # --------------------------------------------------------------- helpers

    def _current_sd(self) -> Dict:
        return state_dict_fp32(self._network)

    def _load_into_network(self, sd: Dict) -> None:
        result = self._network.load_state_dict(sd, strict=False)
        unexpected = getattr(result, "unexpected_keys", None)
        if unexpected:
            raise ShareProtocolError(f"[share] reload left unexpected keys: {list(unexpected)[:5]}")

    def _begin_round(self, round_no: int) -> None:
        self.round_no = int(round_no)
        self.steps_this_round = 0
        self.delta_written = False
        self._round_t0 = time.time()

    # ------------------------------------------------------------ hook points

    def hook_begin(self) -> None:
        """Pre-loop. Lead opens round 1 from its current weights; helper adopts the lead's."""
        os.makedirs(self.share_dir, exist_ok=True)
        if self.role == "lead":
            if os.path.exists(round_dir(self.share_dir, 1)):
                raise ShareProtocolError(
                    f"[share] {round_dir(self.share_dir, 1)} already exists; a stale share dir would "
                    f"merge foreign deltas. Delete {self.share_dir} and restart both sides."
                )
            start = self._current_sd()
            self.open_round(1, start)
            self._start_sd = start
            self._begin_round(1)
            self.running = True
            self._log(f"[share] round 1: lead with {len(self.keys)} items, helper share {self.frac:.0%}")
        else:
            start, _job = self.wait_start(1)
            if start is None:
                raise ShareProtocolError("[share] the lead finished before round 1 started")
            self._load_into_network(start)
            self._start_sd = start
            self._begin_round(1)
            self.running = True
            self._log(f"[share] round 1: helper with {len(self.keys)} items, helper share {self.frac:.0%}")

    def hook_step_top(self) -> None:
        """Top of a step. The helper blocks here between rounds until the next start (or final)."""
        if not self.running or self.role != "helper":
            return
        if not self._waiting:
            return
        start, _job = self.wait_start(self.round_no)
        if start is None:
            self._log("[share] the lead finished; stopping")
            self.running = False
            self.stop_loop = True
            return
        self._load_into_network(start)
        self._start_sd = start
        self._waiting = False
        self._round_t0 = time.time()
        self._log(f"[share] round {self.round_no}: from the lead's merged LoRA")

    def hook_end_of_step(
        self, step: int, is_save_step: bool, is_sample_step: bool, is_last_step: bool,
        hold: bool = False,
    ) -> None:
        """After a completed training step, before the loop's save/sample blocks.

        Closes the round when this side's share of it is trained, or when a
        save/sample/final step forces it, so saved checkpoints and samples
        always hold both halves. ``hold`` is the trainer saying the NEXT step
        saves or samples: the round then stays open one more step so the save
        lands on the round's own close instead of one step into the next round
        (saves fire at step indices % every == 0, round ends after L steps --
        inherently one step apart). Held only on the exact target crossing so
        a save_every of 1 cannot hold forever.
        """
        if not self.running:
            return
        self.steps_this_round += 1
        target = self.my_target_steps(self.round_no)
        target_hit = self.steps_this_round >= target
        forced = is_save_step or is_sample_step or is_last_step
        if not (target_hit or forced):
            return
        if target_hit and not forced and hold and self.steps_this_round == target:
            return
        took = time.time() - (self._round_t0 or time.time())
        if self.role == "lead":
            t_wait0 = time.time()
            delta, meta = self.wait_delta(self.round_no)
            waited = time.time() - t_wait0
            merged = merge(self._current_sd(), delta)
            self._load_into_network(merged)
            self.prune_round(self.round_no)
            self._last_round_seconds = took + waited
            helper_steps = int(meta.get("steps") or 0)
            self.combined_done += self.steps_this_round + helper_steps
            self._log(
                f"[share] round {self.round_no}: combined ({self.steps_this_round}/{target} steps here, "
                f"round {took:.0f}s, waited {waited:.0f}s)"
            )
            if (self.adaptive and self.steps_this_round and helper_steps
                    and float(meta.get("seconds") or 0) > 0):
                # equal wall-clock per round needs the slower card on the
                # smaller share; re-measure every round, smoothed and banded
                t_lead = took / self.steps_this_round
                t_helper = float(meta["seconds"]) / helper_steps
                new_frac = next_frac(t_lead, t_helper, self.frac, self.frac_center)
                if abs(new_frac - self.frac) > 1e-4:
                    self._log(
                        f"[share] frac {self.frac:.3f} -> {new_frac:.3f} "
                        f"(lead {t_lead:.2f} s/it, helper {t_helper:.2f} s/it)"
                    )
                self.frac = new_frac
            if is_last_step:
                self.write_final()
                self.running = False
                return
            nxt = self.round_no + 1
            self.open_round(nxt, self._current_sd())
            self._begin_round(nxt)
        else:
            if not self.delta_written:
                delta = compute_delta(self._current_sd(), self._start_sd)
                self.write_delta(self.round_no, delta, seconds=took)
                self.delta_written = True
                self._last_round_seconds = took
            if is_last_step:
                # budget exhausted; the lead's final.json releases us if we are still waiting
                self.running = False
                self.stop_loop = True
                return
            self._begin_round(self.round_no + 1)
            self._waiting = True
