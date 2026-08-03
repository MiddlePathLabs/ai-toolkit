"""Tests for loss-split resolution and loss_split config validation.

Covers:
  - resolve_loss_split() precedence: per-dataset > explicit global > autodetect
  - DatasetConfig.loss_split validation: None / 'diffusion_depth' / 'sum'
  - TrainConfig.loss_split validation + _loss_split_explicit (absent vs explicit-null)
  - batch size does not change step-parity objective selection
"""
import pytest

from toolkit.loss_split import resolve_loss_split
from toolkit.config_modules import DatasetConfig, TrainConfig


# ----------------------------------------------------------------------
# resolve_loss_split — per-dataset wins
# ----------------------------------------------------------------------

def test_per_dataset_force_on_overrides_explicit_global_off():
    # per-dataset 'diffusion_depth' wins even when global is explicit None (off)
    assert resolve_loss_split(
        ds_value='diffusion_depth',
        global_value=None,
        global_explicit=True,
        effective_depth_weight=0.0,
    ) == 'diffusion_depth'


def test_per_dataset_sum_force_off_overrides_explicit_global_on():
    # per-dataset 'sum' wins even when global is force-on
    assert resolve_loss_split(
        ds_value='sum',
        global_value='diffusion_depth',
        global_explicit=True,
        effective_depth_weight=1.0,
    ) is None


def test_per_dataset_sum_overrides_autodetect_on():
    # per-dataset 'sum' wins over autodetect (depth weight > 0)
    assert resolve_loss_split(
        ds_value='sum',
        global_value=None,
        global_explicit=False,
        effective_depth_weight=0.5,
    ) is None


def test_per_dataset_force_on_overrides_autodetect_off():
    # per-dataset 'diffusion_depth' wins over autodetect off (weight 0)
    assert resolve_loss_split(
        ds_value='diffusion_depth',
        global_value=None,
        global_explicit=False,
        effective_depth_weight=0.0,
    ) == 'diffusion_depth'


# ----------------------------------------------------------------------
# resolve_loss_split — explicit global (per-dataset unset)
# ----------------------------------------------------------------------

def test_explicit_global_force_on_with_per_dataset_unset():
    assert resolve_loss_split(
        ds_value=None,
        global_value='diffusion_depth',
        global_explicit=True,
        effective_depth_weight=0.0,
    ) == 'diffusion_depth'


def test_explicit_global_force_off_with_per_dataset_unset():
    # explicit global None is force-off even when depth is active
    assert resolve_loss_split(
        ds_value=None,
        global_value=None,
        global_explicit=True,
        effective_depth_weight=1.0,
    ) is None


# ----------------------------------------------------------------------
# resolve_loss_split — autodetect (per-dataset unset, global not explicit)
# ----------------------------------------------------------------------

def test_autodetect_on_when_depth_weight_positive():
    assert resolve_loss_split(
        ds_value=None,
        global_value=None,
        global_explicit=False,
        effective_depth_weight=0.005,
    ) == 'diffusion_depth'


def test_autodetect_off_when_depth_weight_zero():
    assert resolve_loss_split(
        ds_value=None,
        global_value=None,
        global_explicit=False,
        effective_depth_weight=0.0,
    ) is None


def test_autodetect_boundary_strictly_greater_than_zero():
    # exactly 0 is off; any positive value (even 1e-9) turns alternation on
    assert resolve_loss_split(
        ds_value=None, global_value=None, global_explicit=False,
        effective_depth_weight=0.0,
    ) is None
    assert resolve_loss_split(
        ds_value=None, global_value=None, global_explicit=False,
        effective_depth_weight=1e-9,
    ) == 'diffusion_depth'


# ----------------------------------------------------------------------
# DatasetConfig.loss_split validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, 'diffusion_depth', 'sum'])
def test_dataset_config_loss_split_accepts_valid(value):
    d = DatasetConfig(folder_path='dummy', loss_split=value)
    assert d.loss_split == value


def test_dataset_config_loss_split_default_is_none():
    assert DatasetConfig(folder_path='dummy').loss_split is None


@pytest.mark.parametrize("bad", ['off', 'on', 'true', 'diffusion', '', 'sum_diffusion', 'SUM'])
def test_dataset_config_loss_split_rejects_unknown(bad):
    with pytest.raises(ValueError, match="loss_split"):
        DatasetConfig(folder_path='dummy', loss_split=bad)


# ----------------------------------------------------------------------
# TrainConfig.loss_split explicitness tracking + validation
# ----------------------------------------------------------------------

def test_train_config_loss_split_absent_is_auto():
    t = TrainConfig()
    assert t.loss_split is None
    # absent key must remain Auto, not be treated as explicit off
    assert t._loss_split_explicit is False


def test_train_config_loss_split_explicit_none_is_off():
    t = TrainConfig(loss_split=None)
    assert t.loss_split is None
    # explicit None is force-off, distinguishable from absent
    assert t._loss_split_explicit is True


def test_train_config_loss_split_explicit_diffusion_depth():
    t = TrainConfig(loss_split='diffusion_depth')
    assert t.loss_split == 'diffusion_depth'
    assert t._loss_split_explicit is True


@pytest.mark.parametrize("bad", ['off', 'sum', 'true', 'diffusion', '', 'SUM'])
def test_train_config_loss_split_rejects_unknown(bad):
    with pytest.raises(ValueError, match="loss_split"):
        TrainConfig(loss_split=bad)


# ----------------------------------------------------------------------
# Batch size does not change step-parity objective selection.
# The resolver returns a per-sample *mode*; the trainer then gates the
# active objective on self.step_num parity alone (guide §0.4). This test
# locks the invariant that batch size is not an input to that selection.
# ----------------------------------------------------------------------

def _step_objective(mode, step_num):
    """Model of the trainer's step-parity selection (Task 5 implements it).

    'diffusion_depth' alternates objectives on odd step parity; any other
    mode (None = off) sums both objectives every step (diffusion primary).
    """
    if mode == 'diffusion_depth':
        return 'depth' if (step_num % 2 == 1) else 'diffusion'
    return 'diffusion'


def test_batch_size_does_not_change_step_parity_selection():
    mode = resolve_loss_split(
        ds_value=None, global_value=None, global_explicit=False,
        effective_depth_weight=0.005,
    )
    assert mode == 'diffusion_depth'
    baseline = [_step_objective(mode, s) for s in range(10)]
    # the objective sequence is a pure function of (mode, step_num); changing
    # the batch size must not alter which objective is active on a given step.
    for batch_size in (1, 2, 4, 8, 16):
        assert [_step_objective(mode, s) for s in range(10)] == baseline, (
            f"step-parity selection changed at batch_size={batch_size}"
        )


def test_batch_size_invariance_holds_when_split_off():
    mode = resolve_loss_split(
        ds_value='sum', global_value='diffusion_depth', global_explicit=True,
        effective_depth_weight=0.005,
    )
    assert mode is None
    baseline = [_step_objective(mode, s) for s in range(10)]
    for batch_size in (1, 4, 8):
        assert [_step_objective(mode, s) for s in range(10)] == baseline
    # when split is off, depth never becomes the sole objective
    assert all(o == 'diffusion' for o in baseline)
