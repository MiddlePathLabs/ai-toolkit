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
    datasets = [_ds(random_crop=True)]
    with pytest.raises(ValueError, match="random_crop"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_random_scale_rejected_when_global_depth_active():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(random_scale=True)]
    with pytest.raises(ValueError, match="random_scale"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_augments_rejected_when_global_depth_active():
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(augments=['blur', 'noise'])]
    with pytest.raises(ValueError, match="augments"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_explicit_zero_override_is_inactive_when_global_depth_is_active():
    # An explicit zero overrides the positive process weight, so this dataset
    # does not receive depth GT or loss and its random transform is irrelevant.
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [_ds(depth_loss_weight=0.0, random_scale=True)]
    result = preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)
    assert result is cfg


def test_mixed_depth_and_non_depth_datasets_allowed():
    # A clean inherited depth-active dataset plus a dataset that explicitly
    # overrides the weight to zero may use non-deterministic transforms.
    cfg = DepthConsistencyConfig(loss_weight=0.001)
    datasets = [
        _ds(resolution=512),
        _ds(depth_loss_weight=0.0, augments=['blur'], random_scale=True),
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
    # preview_only applies to inherited datasets even with zero loss weight,
    # because their GT depth still has to match the training transform.
    cfg = DepthConsistencyConfig(loss_weight=0.0, preview_only=True)
    datasets = [_ds(random_crop=True)]
    with pytest.raises(ValueError, match="random_crop"):
        preflight_depth_consistency(cfg, datasets, arch='flux', low_vram=False)


def test_preview_only_activates_preflight_independent_of_dataset_depth_active():
    # Isolates the preview_only activation branch: the dataset is clean and NOT
    # depth-active (depth_loss_weight unset -> _dataset_depth_active is False)
    # and loss_weight is 0. Depth is active only because preview_only=True, so
    # the Krea 2 low_vram guard must still raise. If preview_only were not wired
    # into _depth_active, preflight would return the config without raising.
    cfg = DepthConsistencyConfig(loss_weight=0.0, preview_only=True)
    datasets = [_ds()]  # clean, not depth-active
    with pytest.raises(ValueError, match="low_vram"):
        preflight_depth_consistency(cfg, datasets, arch='krea2', low_vram=True)
