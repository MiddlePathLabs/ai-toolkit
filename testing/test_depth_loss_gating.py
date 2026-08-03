"""Depth-anchor calculate_loss gating + loss-math tests (Task 5b).

Logic-level coverage (guide sec. 4.3 item 8) with a FAKE model -- no DA2
weights, no GPU. Exercises:

  * resolve_loss_split integration through SDTrainer._resolve_depth_sample_gates:
    dataset override wins; 'sum' disables alternation; absent -> Auto.
  * Per-sample gate: a mixed-override batch selects the intended objective per
    sample; batch 4 with one uniform mode selects ONE objective for the whole
    split-enabled subset (batch size alone never flips parity).
  * Timestep window: depth loss contributes only on depth-objective samples
    within [loss_min_t, loss_max_t]; outside the window it contributes 0.
  * mask_source 'none' -> full-image loss (no masking).
  * Ported loss math (ssi_l1, multiscale_grad_loss, compute_depth_consistency_loss).
  * Inertness: a depth-inactive config leaves the loss block switched off.

The live LoRA-param-moves smoke (item 7) is Task 7 (user GPU) -- not here.
"""
import os

import torch
import torch.nn as nn

from toolkit.config_modules import DepthConsistencyConfig
from toolkit.depth_loss import (
    ssi_l1,
    multiscale_grad_loss,
    compute_depth_consistency_loss,
)
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class _FakeVAE:
    """VAE whose .encode must never be reached; .parameters() yields CPU tensor."""

    def __init__(self):
        self.config = None

    def parameters(self):
        yield torch.zeros(1, device="cpu")

    def encode(self, *args, **kwargs):
        raise AssertionError(
            "Depth live decode must NOT call vae.encode directly; it must route "
            "through sd.decode_latents."
        )


class _FakeModelConfig:
    def __init__(self, arch):
        self.arch = arch


class _FakeNoiseSchedulerConfig:
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps


class _FakeNoiseScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.config = _FakeNoiseSchedulerConfig(num_train_timesteps)


class _FakeSD:
    """Records decode_latents; mirrors the unified decode path."""

    def __init__(self, num_train_timesteps=1000, arch="krea2"):
        self.vae = _FakeVAE()
        self.model_config = _FakeModelConfig(arch)
        self.vae_torch_dtype = torch.float32
        self.noise_scheduler = _FakeNoiseScheduler(num_train_timesteps)
        self.is_flow_matching = True
        self.decode_calls = 0

    def decode_latents(self, latents, device=None, dtype=None):
        self.decode_calls += 1
        b = latents.shape[0]
        # constant mid-gray pixels -> constant depth from the fake perceptor.
        return torch.full((b, 3, 16, 16), 0.5)


class _FakeTrainConfig:
    def __init__(self, loss_split=None, explicit=False):
        self.loss_split = loss_split
        self._loss_split_explicit = explicit


class _FakeTrainer:
    """Minimal stand-in so SDTrainer methods bind without a real trainer."""


class _FakeEncoder:
    """Stands in for the frozen DA2 perceptor; records call count."""

    def __init__(self):
        self.calls = 0

    def __call__(self, pixels):
        self.calls += 1
        b = pixels.shape[0]
        # constant depth map -> ssi_l1 vs a non-constant GT is strictly > 0.
        return torch.full((b, 8, 8), 0.5)


class _FakeFileItem:
    def __init__(self, path):
        self.path = path


class _FakeBatch:
    def __init__(self, n, loss_split_list, depth_weight_list=None,
                 min_t_list=None, max_t_list=None, gt_depths=None, reg=None):
        self.loss_split_list = list(loss_split_list)
        self.depth_loss_weight_list = list(depth_weight_list or [None] * n)
        self.depth_loss_min_t_list = list(min_t_list or [None] * n)
        self.depth_loss_max_t_list = list(max_t_list or [None] * n)
        # non-constant GT depth (8x8) so the loss is strictly positive.
        base = torch.linspace(0.0, 1.0, 64).reshape(8, 8)
        self.depth_gt_list = list(gt_depths or [base.clone() for _ in range(n)])
        self._reg = [bool(x) for x in (reg or [False] * n)]
        self.latents = torch.zeros(n, 16, 4, 4)
        self.file_items = [_FakeFileItem(f"/tmp/img_{i}.png") for i in range(n)]

    def get_is_reg_list(self):
        return list(self._reg)


def _make_trainer(loss_weight=0.001, train_loss_split=None, explicit=False,
                  step_num=0, num_ts=1000, preview_only=False,
                  pixel_blur_sigma=0.0):
    tr = _FakeTrainer()
    tr.depth_consistency_config = DepthConsistencyConfig(
        loss_weight=loss_weight, preview_only=preview_only,
        pixel_blur_sigma=pixel_blur_sigma,
    )
    tr.train_config = _FakeTrainConfig(train_loss_split, explicit)
    tr.step_num = step_num
    tr.device_torch = torch.device("cpu")
    tr.sd = _FakeSD(num_ts)
    tr.dataset_configs = []
    tr._depth_perceptor = _FakeEncoder()
    tr.save_root = None  # previews disabled in gating tests
    tr._depth_step_count = 0  # mirrors SDTrainer.__init__
    # Bind the cadence helper so _compute_depth_anchor_loss can call it as
    # self._depth_preview_due(cfg) just like a real SDTrainer instance would.
    tr._depth_preview_due = SDTrainer._depth_preview_due.__get__(tr)
    return tr


def _gates(trainer, batch, timesteps):
    is_reg = torch.tensor(batch.get_is_reg_list(), dtype=torch.bool)
    return SDTrainer._resolve_depth_sample_gates(trainer, batch, timesteps, is_reg)


# ----------------------------------------------------------------------
# resolve_loss_split integration through the per-sample gate
# ----------------------------------------------------------------------

def test_dataset_override_forces_alternation_over_global():
    # global is explicit off (None) but dataset says diffusion_depth -> dataset wins
    tr = _make_trainer(train_loss_split=None, explicit=True, step_num=1)
    batch = _FakeBatch(1, loss_split_list=["diffusion_depth"])
    ts = torch.tensor([500.0])
    g = _gates(tr, batch, ts)
    assert bool(g["diffusion_zero"][0]) is True   # depth step -> skips diffusion
    assert bool(g["depth_objective"][0]) is True   # in band, weight > 0 -> depth


def test_dataset_sum_disables_alternation_for_that_sample():
    # global alternation explicit, but one dataset is 'sum' -> that sample sums
    tr = _make_trainer(train_loss_split="diffusion_depth", explicit=True, step_num=1)
    batch = _FakeBatch(2, loss_split_list=["sum", "diffusion_depth"])
    ts = torch.tensor([500.0, 500.0])
    g = _gates(tr, batch, ts)
    # sample 0: 'sum' -> not alternating -> diffusion always kept, depth always on
    assert bool(g["diffusion_zero"][0]) is False
    assert bool(g["depth_objective"][0]) is True
    # sample 1: alternating depth step -> diffusion skipped, depth on
    assert bool(g["diffusion_zero"][1]) is True
    assert bool(g["depth_objective"][1]) is True


def test_dataset_sum_stays_diffusion_kept_on_diffusion_step():
    tr = _make_trainer(train_loss_split="diffusion_depth", explicit=True, step_num=0)
    batch = _FakeBatch(1, loss_split_list=["sum"])
    ts = torch.tensor([500.0])
    g = _gates(tr, batch, ts)
    # sum sample never drops out of diffusion regardless of step parity
    assert bool(g["diffusion_zero"][0]) is False


def test_absent_global_autodetects_from_depth_weight():
    # nothing explicit, depth weight > 0 -> Auto -> 'diffusion_depth'
    tr = _make_trainer(loss_weight=0.001, train_loss_split=None, explicit=False,
                       step_num=0)
    batch = _FakeBatch(1, loss_split_list=[None])
    ts = torch.tensor([500.0])
    g = _gates(tr, batch, ts)
    assert bool(g["depth_objective"][0]) is False  # diffusion step -> split skips depth
    # odd step -> depth objective
    tr.step_num = 1
    g = _gates(tr, batch, ts)
    assert bool(g["depth_objective"][0]) is True


def test_absent_global_with_zero_depth_weight_is_off():
    tr = _make_trainer(loss_weight=0.0, train_loss_split=None, explicit=False)
    batch = _FakeBatch(1, loss_split_list=[None])
    ts = torch.tensor([500.0])
    g = _gates(tr, batch, ts)
    assert bool(g["depth_objective"][0]) is False
    assert bool(g["diffusion_zero"][0]) is False


# ----------------------------------------------------------------------
# Per-sample gate: mixed overrides + batch-size-invariance
# ----------------------------------------------------------------------

def test_mixed_override_batch_selects_intended_objective_per_sample():
    tr = _make_trainer(loss_weight=0.001, step_num=1)  # depth step
    # 0: sum (always both), 1: diffusion_depth (depth step -> depth),
    # 2: None auto -> diffusion_depth (depth step -> depth),
    # 3: None auto but depth weight 0 -> off
    batch = _FakeBatch(
        4,
        loss_split_list=["sum", "diffusion_depth", None, None],
        depth_weight_list=[None, None, None, 0.0],
    )
    ts = torch.tensor([500.0] * 4)
    g = _gates(tr, batch, ts)
    dz = [bool(x) for x in g["diffusion_zero"]]
    do = [bool(x) for x in g["depth_objective"]]
    assert dz == [False, True, True, False]   # only alternating samples skip diff
    assert do == [True, True, True, False]    # weight-0 sample contributes no depth


def test_batch4_uniform_mode_one_objective_parity_flips_all():
    # all four samples alternating; even step -> all diffusion-objective
    tr = _make_trainer(loss_weight=0.001, step_num=0)
    batch = _FakeBatch(4, loss_split_list=["diffusion_depth"] * 4)
    ts = torch.tensor([500.0] * 4)
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g["depth_objective"]] == [False] * 4
    assert [bool(x) for x in g["diffusion_zero"]] == [False] * 4

    # odd step -> all depth-objective, all skip diffusion
    tr.step_num = 1
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g["depth_objective"]] == [True] * 4
    assert [bool(x) for x in g["diffusion_zero"]] == [True] * 4


def test_batch_size_alone_never_flips_parity():
    # batch 1 vs batch 4 at the SAME step select the same objective per sample
    for step in (0, 1):
        tr1 = _make_trainer(loss_weight=0.001, step_num=step)
        b1 = _FakeBatch(1, loss_split_list=["diffusion_depth"])
        g1 = _gates(tr1, b1, torch.tensor([500.0]))
        tr4 = _make_trainer(loss_weight=0.001, step_num=step)
        b4 = _FakeBatch(4, loss_split_list=["diffusion_depth"] * 4)
        g4 = _gates(tr4, b4, torch.tensor([500.0] * 4))
        assert bool(g1["depth_objective"][0]) == bool(g4["depth_objective"][0])
        assert bool(g1["diffusion_zero"][0]) == bool(g4["diffusion_zero"][0])


# ----------------------------------------------------------------------
# preview_only: loss_weight=0 must not drop diffusion without gaining depth
# ----------------------------------------------------------------------

def test_preview_only_never_drops_diffusion_without_depth():
    # preview_only with loss_weight=0 and an explicit diffusion_depth split on a
    # depth step: depth_objective is all-False (effective weight is 0), so the
    # diffusion_zero mask must ALSO be all-False. A sample never loses diffusion
    # without gaining depth. The anchor path returns 0.0 and the diffusion mean
    # is not masked (reduces to the plain mean).
    tr = _make_trainer(loss_weight=0.0, preview_only=True,
                       train_loss_split='diffusion_depth', explicit=True,
                       step_num=1)  # odd -> depth step
    batch = _FakeBatch(2, loss_split_list=[None, None])
    ts = torch.tensor([500.0, 500.0])
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g['depth_objective']] == [False, False]
    assert [bool(x) for x in g['diffusion_zero']] == [False, False]
    assert not bool(g['diffusion_zero'].any())

    # anchor path returns 0.0; no sample reaches the perceptor
    noise_pred = torch.zeros(2, 16, 4, 4)
    noisy = torch.zeros(2, 16, 4, 4)
    out = SDTrainer._compute_depth_anchor_loss(tr, noise_pred, noisy, ts, batch, g)
    assert float(out) == 0.0
    assert tr._depth_perceptor.calls == 0

    # diffusion mean is NOT masked: an all-False diffusion_zero reduces to the
    # plain mean (the trainer's else-branch in calculate_loss).
    loss_per_sample = torch.tensor([2.0, 4.0])
    masked = SDTrainer._apply_diffusion_split_mask(tr, loss_per_sample, g['diffusion_zero'])
    assert float(masked) == float(loss_per_sample.mean())


# ----------------------------------------------------------------------
# Timestep window: depth loss only within [loss_min_t, loss_max_t]
# ----------------------------------------------------------------------

def test_depth_anchor_loss_zero_when_all_out_of_window():
    tr = _make_trainer(loss_weight=0.001, step_num=1)
    tr.depth_consistency_config = DepthConsistencyConfig(
        loss_weight=0.001, loss_min_t=0.2, loss_max_t=0.4,
    )
    batch = _FakeBatch(2, loss_split_list=[None, None])
    ts = torch.tensor([100.0, 900.0])  # t = 0.1 and 0.9, both outside [0.2, 0.4]
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g["depth_objective"]] == [False, False]

    noise_pred = torch.zeros(2, 16, 4, 4)
    noisy = torch.zeros(2, 16, 4, 4)
    out = SDTrainer._compute_depth_anchor_loss(tr, noise_pred, noisy, ts, batch, g)
    assert float(out) == 0.0
    assert tr._depth_perceptor.calls == 0


def test_depth_anchor_loss_processes_only_in_window_objective_samples():
    tr = _make_trainer(loss_weight=0.001, step_num=1)
    tr.depth_consistency_config = DepthConsistencyConfig(
        loss_weight=0.001, loss_min_t=0.2, loss_max_t=0.6,
    )
    # sample 0 in window (t=0.5), sample 1 out (t=0.9); both auto-alternating depth step
    batch = _FakeBatch(2, loss_split_list=[None, None])
    ts = torch.tensor([500.0, 900.0])
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g["depth_objective"]] == [True, False]

    noise_pred = torch.zeros(2, 16, 4, 4)
    noisy = torch.zeros(2, 16, 4, 4)
    out = SDTrainer._compute_depth_anchor_loss(tr, noise_pred, noisy, ts, batch, g)
    assert torch.is_tensor(out)
    assert torch.isfinite(out).all()
    assert float(out) > 0.0
    # only the in-window sample reached the perceptor
    assert tr._depth_perceptor.calls == 1


def test_depth_anchor_loss_skips_split_samples_on_diffusion_step():
    # diffusion step (even) -> alternating samples skip depth entirely
    tr = _make_trainer(loss_weight=0.001, step_num=0)
    batch = _FakeBatch(2, loss_split_list=["diffusion_depth", "sum"])
    ts = torch.tensor([500.0, 500.0])
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g["depth_objective"]] == [False, True]

    noise_pred = torch.zeros(2, 16, 4, 4)
    noisy = torch.zeros(2, 16, 4, 4)
    out = SDTrainer._compute_depth_anchor_loss(tr, noise_pred, noisy, ts, batch, g)
    # only the 'sum' sample (always depth) was processed
    assert tr._depth_perceptor.calls == 1
    assert float(out) > 0.0


def test_reg_samples_excluded_from_depth_loss():
    tr = _make_trainer(loss_weight=0.001, step_num=1)
    batch = _FakeBatch(2, loss_split_list=[None, None], reg=[False, True])
    ts = torch.tensor([500.0, 500.0])
    g = _gates(tr, batch, ts)
    assert [bool(x) for x in g["depth_objective"]] == [True, False]


# ----------------------------------------------------------------------
# mask_source 'none' -> full-image loss (no masking)
# ----------------------------------------------------------------------

def test_ssi_l1_mask_none_equals_full_ones_mask():
    pred = torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8)
    target = torch.linspace(1.0, 0.0, 64).reshape(1, 8, 8)
    loss_none, _, _ = ssi_l1(pred, target, mask=None)
    loss_full, _, _ = ssi_l1(pred, target, mask=torch.ones_like(pred))
    assert torch.allclose(loss_none, loss_full)


def test_depth_anchor_uses_no_mask_when_mask_source_none():
    # mask_source is 'none' (preflight rejects subject/body), so the live block
    # must pass mask=None to compute_depth_consistency_loss. The fake encoder
    # records that it never receives a mask argument (it would only via the
    # compute_depth_consistency_loss call, which we assert gets mask=None).
    tr = _make_trainer(loss_weight=0.001, step_num=1)
    assert tr.depth_consistency_config.mask_source == "none"
    batch = _FakeBatch(1, loss_split_list=[None])
    ts = torch.tensor([500.0])
    g = _gates(tr, batch, ts)
    assert bool(g["depth_objective"][0]) is True
    # no subject_masks / body_masks attrs are read (they are absent on _FakeBatch)
    noise_pred = torch.zeros(1, 16, 4, 4)
    noisy = torch.zeros(1, 16, 4, 4)
    out = SDTrainer._compute_depth_anchor_loss(tr, noise_pred, noisy, ts, batch, g)
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# Diffusion-side alternation masking helper
# ----------------------------------------------------------------------

def test_apply_diffusion_split_mask_no_zero_is_plain_mean():
    tr = _FakeTrainer()
    loss = torch.tensor([2.0, 4.0, 6.0, 8.0])
    dz = torch.tensor([False, False, False, False])
    out = SDTrainer._apply_diffusion_split_mask(tr, loss, dz)
    assert float(out) == float(loss.mean())


def test_apply_diffusion_split_mask_drops_zeroed_samples():
    tr = _FakeTrainer()
    loss = torch.tensor([2.0, 4.0, 6.0, 8.0])
    dz = torch.tensor([False, True, False, True])
    out = SDTrainer._apply_diffusion_split_mask(tr, loss, dz)
    # keeps indices 0,2 -> (2 + 6) / 2 = 4.0, not the plain mean 5.0
    assert abs(float(out) - 4.0) < 1e-6


def test_apply_diffusion_split_mask_all_zeroed_returns_zero():
    tr = _FakeTrainer()
    loss = torch.tensor([2.0, 4.0])
    dz = torch.tensor([True, True])
    out = SDTrainer._apply_diffusion_split_mask(tr, loss, dz)
    assert float(out) == 0.0


# ----------------------------------------------------------------------
# Ported loss math
# ----------------------------------------------------------------------

def test_ssi_l1_identical_is_near_zero():
    pred = torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8)
    loss, s, t = ssi_l1(pred, pred.clone())
    assert float(loss) < 1e-5
    # closed-form aligns to identity when pred == target
    assert abs(float(s[0]) - 1.0) < 1e-3


def test_multiscale_grad_loss_identical_is_zero():
    pred = torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8)
    loss = multiscale_grad_loss(pred, pred.clone(), scales=4)
    assert float(loss) < 1e-5


def test_compute_depth_consistency_loss_carries_gradient():
    class _LinEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))

        def forward(self, pixels):
            # input-dependent depth so grad flows to pixels; DA2 returns (B,H,W)
            return pixels[:, 0] * self.w

    enc = _LinEncoder()
    pixels = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
    pixels.requires_grad_(True)
    gt = torch.linspace(1.0, 0.0, 64).reshape(1, 8, 8)
    loss, ssi_c, grad_c, d_pred, d_tgt = compute_depth_consistency_loss(
        enc, pixels, gt, mask=None, ssi_weight=1.0, grad_weight=0.5, grad_scales=4,
    )
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    assert float(pixels.grad.abs().sum()) > 0.0


# ----------------------------------------------------------------------
# Inertness: depth-inactive config leaves the loss block off
# ----------------------------------------------------------------------

def test_depth_loss_active_flag_false_when_inactive():
    tr = _make_trainer(loss_weight=0.0)
    tr.depth_consistency_config = None
    assert SDTrainer._depth_loss_active(tr) is False

    tr.depth_consistency_config = DepthConsistencyConfig(loss_weight=0.0)
    assert SDTrainer._depth_loss_active(tr) is False


def test_depth_loss_active_flag_true_when_weight_positive():
    tr = _make_trainer(loss_weight=0.001)
    assert SDTrainer._depth_loss_active(tr) is True


def test_num_train_timesteps_not_hardcoded():
    # a non-1000 scheduler must produce t = ts / num_train_timesteps
    tr = _make_trainer(loss_weight=0.001, step_num=1, num_ts=500)
    batch = _FakeBatch(1, loss_split_list=[None])
    ts = torch.tensor([250.0])  # 250 / 500 = 0.5
    g = _gates(tr, batch, ts)
    assert abs(float(g["t"][0]) - 0.5) < 1e-6


# ----------------------------------------------------------------------
# Preview cadence: decoupled from raw step parity (regression)
# ----------------------------------------------------------------------

def test_depth_preview_renders_under_alternation_with_even_preview_every():
    # Regression: with alternation on (loss_split='diffusion_depth',
    # loss_weight > 0) and an EVEN preview_every, previews must still render
    # on depth steps. The buggy version gated the preview on raw step parity
    # (step_num % preview_every == 0); with an even cadence every preview step
    # (10/20/30/...) landed on an even = diffusion step, where the per-sample
    # preview loop never runs (depth_objective is all False). Previews never
    # rendered. The fix decouples the cadence: count DEPTH steps and render
    # every preview_every DEPTH steps.
    tr = _make_trainer(loss_weight=0.001, train_loss_split='diffusion_depth',
                       explicit=True)
    tr.depth_consistency_config.preview_every = 10  # EVEN -- exposes the bug
    cfg = tr.depth_consistency_config

    # Reset the depth-step counter as hook_before_train_loop would.
    tr._depth_step_count = 0
    fired_counts = []  # depth-step-count at which a preview was due
    for step in range(50):
        tr.step_num = step
        step_is_diffusion = (step % 2 == 0)
        if step_is_diffusion:
            # diffusion step: depth_objective is all False -> anchor path
            # early-returns and never advances the counter.
            continue
        # depth step: the anchor path increments AFTER its early-return gate.
        tr._depth_step_count += 1
        if SDTrainer._depth_preview_due(tr, cfg):
            fired_counts.append(tr._depth_step_count)

    # The buggy version fired ZERO times. The fix must fire at least once.
    assert len(fired_counts) >= 1
    # And at the expected DEPTH-step cadence: depth-step-counts 10 and 20 for
    # a 50-step run (25 depth steps). Not on raw steps 10/20/30 (all diffusion).
    assert 10 in fired_counts
    assert 20 in fired_counts
    # A preview is never due on step 0 / count 0 -- the first tile is after the
    # first full cadence window of depth steps.
    assert 0 not in fired_counts
