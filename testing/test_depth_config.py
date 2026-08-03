"""Config-level tests for DepthConsistencyConfig.

Scope is config-level only (preflight rejection of random_crop/scale/augments
and dataset-only activation belong to Task 4). Covers safe defaults, the
no-enabled-field invariant, partial-object merge, timestep/mask/input_size
validation, and the absent-vs-explicit loss_split distinction.
"""
import pytest

from toolkit.config_modules import DepthConsistencyConfig, TrainConfig


# ----------------------------------------------------------------------
# Safe defaults (override the source fork's enabling defaults)
# ----------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = DepthConsistencyConfig()
    assert c.loss_weight == 0.0
    assert c.loss_min_t == 0.0
    assert c.loss_max_t == 1.0
    assert c.model_id == 'depth-anything/Depth-Anything-V2-Small-hf'
    assert c.input_size == 518
    assert c.pixel_blur_sigma == 0.0
    assert c.ssi_weight == 1.0
    assert c.grad_weight == 0.5
    assert c.grad_scales == 4
    assert c.mask_source == 'none'
    assert c.grad_checkpoint is True
    assert c.preview_every == 100
    assert c.preview_only is False
    assert c.preview_max_keep == 500


def test_config_has_no_enabled_field():
    # Enable by loss_weight > 0; there is no `enabled` attribute.
    c = DepthConsistencyConfig()
    assert not hasattr(c, 'enabled')


def test_partial_object_merge_preserves_supplied_and_fills_missing():
    # mirrors the UI migration: a partial saved object keeps its supplied
    # values and is filled with the safe defaults for everything else.
    c = DepthConsistencyConfig(loss_weight=0.5, input_size=1024, mask_source='subject')
    assert c.loss_weight == 0.5
    assert c.input_size == 1024
    assert c.mask_source == 'subject'
    # unspecified fields fall back to safe defaults
    assert c.model_id == 'depth-anything/Depth-Anything-V2-Small-hf'
    assert c.loss_min_t == 0.0
    assert c.loss_max_t == 1.0
    assert c.pixel_blur_sigma == 0.0
    assert c.ssi_weight == 1.0
    assert c.grad_weight == 0.5
    assert c.grad_scales == 4
    assert c.grad_checkpoint is True
    assert c.preview_every == 100
    assert c.preview_only is False
    assert c.preview_max_keep == 500


# ----------------------------------------------------------------------
# Timestep bounds validation
# ----------------------------------------------------------------------

def test_negative_loss_min_t_raises():
    with pytest.raises(ValueError, match="loss_min_t"):
        DepthConsistencyConfig(loss_min_t=-0.01)


def test_loss_max_t_above_one_raises():
    with pytest.raises(ValueError, match="loss_max_t"):
        DepthConsistencyConfig(loss_max_t=1.01)


def test_loss_min_t_greater_than_loss_max_t_raises():
    with pytest.raises(ValueError, match="loss_min_t"):
        DepthConsistencyConfig(loss_min_t=0.6, loss_max_t=0.4)


def test_valid_timestep_bounds_accepted():
    c = DepthConsistencyConfig(loss_min_t=0.0, loss_max_t=1.0)
    assert (c.loss_min_t, c.loss_max_t) == (0.0, 1.0)
    c2 = DepthConsistencyConfig(loss_min_t=0.2, loss_max_t=0.8)
    assert (c2.loss_min_t, c2.loss_max_t) == (0.2, 0.8)


# ----------------------------------------------------------------------
# mask_source enum validation (subject/body accepted at config level;
# they are rejected later by backend preflight until Phase 3)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("value", ['none', 'subject', 'body'])
def test_valid_mask_source_accepted(value):
    assert DepthConsistencyConfig(mask_source=value).mask_source == value


@pytest.mark.parametrize("bad", ['full', 'person', '', 'NONE', 'random', None])
def test_invalid_mask_source_raises(bad):
    with pytest.raises(ValueError, match="mask_source"):
        DepthConsistencyConfig(mask_source=bad)


# ----------------------------------------------------------------------
# input_size validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -518])
def test_non_positive_input_size_raises(bad):
    with pytest.raises(ValueError, match="input_size"):
        DepthConsistencyConfig(input_size=bad)


def test_positive_input_size_accepted():
    assert DepthConsistencyConfig(input_size=1024).input_size == 1024


# ----------------------------------------------------------------------
# Absent train.loss_split stays Auto (not off)
# ----------------------------------------------------------------------

def test_absent_train_loss_split_is_auto_not_off():
    # absent key must remain distinguishable from explicit null so the
    # resolver autodetects rather than forcing split off.
    t = TrainConfig()
    assert t.loss_split is None
    assert t._loss_split_explicit is False
