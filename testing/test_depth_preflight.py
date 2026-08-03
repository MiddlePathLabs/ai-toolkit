"""Backend preflight tests for the depth-anchor trainer init.

Scope (Task 4): the preflight rejections that can be tested without a real
model -- random_crop / random_scale / non-empty augments rejection on a
depth-active dataset, dataset-only activation constructing a disabled default
DepthConsistencyConfig, and the inert no-op when depth is entirely absent.

The low_vram / mask_source raise paths are exercised through the trainer with a
fake Krea model in Task 5's test_depth_krea_contract.py and are intentionally
not covered here.
"""
import pytest

from toolkit.config_modules import DepthConsistencyConfig, DatasetConfig
from extensions_built_in.sd_trainer.SDTrainer import preflight_depth_consistency


def _ds(**kw):
    return DatasetConfig(**kw)


# ----------------------------------------------------------------------
# Inertness: depth entirely absent -> complete no-op
# ----------------------------------------------------------------------

def test_no_config_no_depth_dataset_returns_none():
    # random_crop on a non-depth dataset must NOT raise when depth is off.
    datasets = [_ds(random_crop=True)]
    result = preflight_depth_consistency(None, datasets, arch='flux', low_vram=False)
    assert result is None


def test_disabled_global_config_no_depth_dataset_is_inert():
    # loss_weight=0, no preview, no depth-active dataset -> no preflight raises,
    # even when a dataset uses random transforms (existing trainer behavior kept).
    cfg = DepthConsistencyConfig(loss_weight=0.0)
    datasets = [_ds(random_crop=True, random_scale=True, augments=['blur'])]
    result = preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
    assert result is cfg
    assert result.loss_weight == 0.0


# ----------------------------------------------------------------------
# Dataset-only activation constructs a disabled default config
# ----------------------------------------------------------------------

def test_dataset_only_activation_creates_disabled_default():
    # No process config, but a dataset sets depth_loss_weight > 0 -> the helper
    # must construct a disabled default rather than silently ignoring it.
    datasets = [_ds(depth_loss_weight=0.5)]
    result = preflight_depth_consistency(None, datasets, arch='flux', low_vram=False)
    assert result is not None
    assert result.loss_weight == 0.0
    assert result.mask_source == 'none'


def test_dataset_only_activation_still_runs_preflight():
    # dataset-only activation enables depth -> random_crop must still be rejected.
    datasets = [_ds(depth_loss_weight=0.5, random_crop=True)]
    with pytest.raises(ValueError, match="random_crop"):
        preflight_depth_consistency(None, datasets, arch='flux', low_vram=False)


def test_dataset_only_activation_with_global_disabled_runs_preflight():
    # global loss_weight=0 + depth-active dataset -> depth active via dataset,
    # so random_scale on that dataset must be rejected.
    cfg = DepthConsistencyConfig(loss_weight=0.0)
    datasets = [_ds(depth_loss_weight=0.5, random_scale=True)]
    with pytest.raises(ValueError, match="random_scale"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


# ----------------------------------------------------------------------
# Preflight rejections on depth-active datasets
# ----------------------------------------------------------------------

def test_random_crop_rejected_when_global_depth_active():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(depth_loss_weight=0.5, random_crop=True)]
    with pytest.raises(ValueError, match="random_crop"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_random_scale_rejected_when_global_depth_active():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(depth_loss_weight=0.5, random_scale=True)]
    with pytest.raises(ValueError, match="random_scale"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_augments_rejected_when_global_depth_active():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(depth_loss_weight=0.5, augments=['blur', 'noise'])]
    with pytest.raises(ValueError, match="augments"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_non_depth_dataset_not_rejected_when_depth_active():
    # A non-depth dataset (depth_loss_weight unset) must NOT trigger preflight
    # even when global depth is on; it has no GT depth cache to desync.
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(), _ds(random_scale=True)]
    result = preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
    assert result is cfg


def test_mixed_depth_and_non_depth_datasets_allowed():
    # Inverse coverage: a clean depth-active dataset plus a separate non-depth
    # dataset using augments/random_scale is a legitimate mixed config and must
    # NOT raise (the non-depth dataset has no GT depth cache to desync).
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [
        _ds(depth_loss_weight=0.5, resolution=512),
        _ds(augments=['blur'], random_scale=True),
    ]
    result = preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
    assert result is cfg


def test_empty_augments_list_allowed():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(depth_loss_weight=0.5, augments=[])]
    result = preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
    assert result is cfg


# ----------------------------------------------------------------------
# Allowed: deterministic bucket transforms + recorded flips
# ----------------------------------------------------------------------

def test_fixed_bucket_transforms_and_flips_allowed():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(depth_loss_weight=0.5, flip_x=True, flip_y=True, resolution=512)]
    result = preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
    assert result is cfg


def test_preview_only_activates_preflight():
    # preview_only with loss_weight=0 still activates depth -> preflight runs
    # and rejects random_crop on a depth-active dataset.
    cfg = DepthConsistencyConfig(loss_weight=0.0, preview_only=True)
    datasets = [_ds(depth_loss_weight=0.5, random_crop=True)]
    with pytest.raises(ValueError, match="random_crop"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
