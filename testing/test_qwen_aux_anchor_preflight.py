"""Qwen-Image 2.1 preflight tests for the non-VAE perceptual aux anchors.

Scope: the shared differentiable-decode guard
(``_preflight_qwen_differentiable_decode``) as wired into the depth, normal,
body-proportion, face-identity and body-shape preflights -- low_vram forcing
tiled decode, the pixel limit against bucket geometry (or the dataset
resolution fallback), reg-dataset exemption, preview-only exemption, and the
unchanged Krea 2 / other-arch contracts. The VAE anchor's own coverage lives
in ``test_vae_anchor_preflight.py``.
"""
import pytest

from toolkit.config_modules import (
    BodyProportionConfig,
    BodyShapeConfig,
    DatasetConfig,
    DepthConsistencyConfig,
    FaceIDConfig,
    NormalIDConfig,
)
from extensions_built_in.sd_trainer.SDTrainer import (
    preflight_body_proportion,
    preflight_body_shape,
    preflight_depth_consistency,
    preflight_face_id,
    preflight_normal_id,
)

LIMIT = 1024 * 1024


def _ds(**kw):
    return DatasetConfig(**kw)


def _make_active():
    """One (label, preflight, enabled-config factory, dataset weight key) per
    anchor, each configured with a small positive weight."""
    return [
        pytest.param(
            preflight_depth_consistency,
            lambda: DepthConsistencyConfig(loss_weight=0.05, mask_source='none'),
            'depth_loss_weight',
            id='depth',
        ),
        pytest.param(
            preflight_normal_id,
            lambda: NormalIDConfig(loss_weight=0.05),
            'normal_loss_weight',
            id='normal',
        ),
        pytest.param(
            preflight_body_proportion,
            lambda: BodyProportionConfig(loss_weight=0.05),
            'body_proportion_loss_weight',
            id='body_proportion',
        ),
        pytest.param(
            preflight_face_id,
            lambda: FaceIDConfig(identity_loss_weight=0.05),
            'identity_loss_weight',
            id='face_identity',
        ),
        pytest.param(
            preflight_body_shape,
            lambda: BodyShapeConfig(loss_weight=0.05),
            'body_shape_loss_weight',
            id='body_shape',
        ),
    ]


# ----------------------------------------------------------------------
# Inertness: anchor entirely absent -> complete no-op, no qwen guards fire
# ----------------------------------------------------------------------

@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_no_config_no_active_dataset_returns_none(preflight, make_cfg, weight_key):
    assert preflight(None, [_ds()], arch='qwen_image_2', low_vram=True) is None


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_disabled_global_config_no_active_dataset_is_inert(preflight, make_cfg, weight_key):
    cfg = make_cfg()
    for attr in ('loss_weight', 'identity_loss_weight'):
        if hasattr(cfg, attr):
            setattr(cfg, attr, 0.0)
    result = preflight(cfg, [_ds()], arch='qwen_image_2', low_vram=True)
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_dataset_only_activation_builds_disabled_default(preflight, make_cfg, weight_key):
    datasets = [_ds(**{weight_key: 0.1})]
    with pytest.raises(ValueError, match="low_vram"):
        preflight(None, datasets, arch='qwen_image_2', low_vram=True)


# ----------------------------------------------------------------------
# Existing Krea 2 contract is unchanged (and no qwen guards leak to krea2)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_krea2_low_vram_still_raises(preflight, make_cfg, weight_key):
    with pytest.raises(ValueError, match="Krea 2"):
        preflight(make_cfg(), [_ds()], arch='krea2', low_vram=True)


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_krea2_low_vram_false_with_qwen_limit_params_is_fine(preflight, make_cfg, weight_key):
    # the new kwargs must stay inert on non-qwen archs even when populated
    cfg = make_cfg()
    result = preflight(
        cfg, [_ds(resolution=2048)], arch='krea2', low_vram=False,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg


# ----------------------------------------------------------------------
# Qwen-Image 2.1: low_vram forces tiled decode
# ----------------------------------------------------------------------

@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_low_vram_raises(preflight, make_cfg, weight_key):
    with pytest.raises(ValueError, match="low_vram"):
        preflight(make_cfg(), [_ds()], arch='qwen_image_2', low_vram=True)


# ----------------------------------------------------------------------
# Qwen-Image 2.1: differentiable-decode pixel limit vs actual buckets
# ----------------------------------------------------------------------

@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_bucket_above_limit_raises(preflight, make_cfg, weight_key):
    with pytest.raises(ValueError, match="tile threshold"):
        preflight(
            make_cfg(), [_ds()], arch='qwen_image_2', low_vram=False,
            decode_pixel_limit=LIMIT, max_active_bucket_pixels=1536 * 1536,
        )


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_bucket_at_limit_is_untiled_and_passes(preflight, make_cfg, weight_key):
    cfg = make_cfg()
    result = preflight(
        cfg, [_ds()], arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT,
    )
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_opted_out_dataset_is_not_checked(preflight, make_cfg, weight_key):
    cfg = make_cfg()
    datasets = [
        _ds(resolution=2048, **{weight_key: 0.0}),
        _ds(resolution=512),
    ]
    result = preflight(
        cfg, datasets, arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT,
    )
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_inheriting_dataset_is_checked(preflight, make_cfg, weight_key):
    datasets = [
        _ds(resolution=2048, **{weight_key: 0.0}),
        _ds(resolution=1536),
    ]
    with pytest.raises(ValueError, match="resolution 1536"):
        preflight(
            make_cfg(), datasets, arch='qwen_image_2', low_vram=False,
            decode_pixel_limit=LIMIT,
        )


# ----------------------------------------------------------------------
# Qwen-Image 2.1: resolution fallback when bucket geometry is unknown
# ----------------------------------------------------------------------

@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_resolution_fallback_raises_above_limit(preflight, make_cfg, weight_key):
    with pytest.raises(ValueError, match="resolution 1536"):
        preflight(
            make_cfg(), [_ds(resolution=1536)], arch='qwen_image_2', low_vram=False,
            decode_pixel_limit=LIMIT,
        )


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_resolution_fallback_passes_at_limit(preflight, make_cfg, weight_key):
    cfg = make_cfg()
    result = preflight(
        cfg, [_ds(resolution=1024)], arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT,
    )
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_resolution_list_uses_largest_entry(preflight, make_cfg, weight_key):
    # bucket-resolution lists (the common UI form) are checked at their
    # largest entry: [512, 768, 1536] must raise, [512, 768, 1024] must pass
    with pytest.raises(ValueError, match="resolution 1536"):
        preflight(
            make_cfg(), [_ds(resolution=[512, 768, 1536])],
            arch='qwen_image_2', low_vram=False, decode_pixel_limit=LIMIT,
        )
    cfg = make_cfg()
    result = preflight(
        cfg, [_ds(resolution=[512, 768, 1024])], arch='qwen_image_2',
        low_vram=False, decode_pixel_limit=LIMIT,
    )
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_reg_dataset_above_limit_is_not_checked(preflight, make_cfg, weight_key):
    # reg rows are skipped by every live anchor loss before the decode, so a
    # high-res reg set must not trip the pixel limit even when it inherits
    # the global weight
    cfg = make_cfg()
    datasets = [
        _ds(resolution=1024),
        _ds(is_reg=True, resolution=2048),
    ]
    result = preflight(
        cfg, datasets, arch='qwen_image_2', low_vram=False,
        decode_pixel_limit=LIMIT,
    )
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_global_weight_overridden_to_zero_everywhere_is_inert(preflight, make_cfg, weight_key):
    # a positive process weight overridden to zero on every non-reg dataset
    # never reaches the live decode (per-sample gates skip weight<=0), so
    # neither the low_vram rejection nor the pixel limit may fire
    cfg = make_cfg()
    datasets = [_ds(resolution=2048, **{weight_key: 0.0})]
    result = preflight(
        cfg, datasets, arch='qwen_image_2', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_reg_only_dataset_override_on_low_vram_is_inert(preflight, make_cfg, weight_key):
    # reg rows never reach the differentiable decode, so a reg-only dataset
    # activation cannot arm the qwen guards -- matching the VAE anchor's
    # reg-only-inert contract
    datasets = [
        _ds(resolution=512),
        _ds(is_reg=True, resolution=2048, **{weight_key: 0.1}),
    ]
    result = preflight(
        None, datasets, arch='qwen_image_2', low_vram=True,
        decode_pixel_limit=LIMIT,
    )
    assert result is not None


@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_qwen_limit_not_set_skips_resolution_check(preflight, make_cfg, weight_key):
    # no pixel limit exposed (non-qwen wrapper or older call sites): the old
    # behavior -- no resolution restriction -- is preserved
    cfg = make_cfg()
    result = preflight(
        cfg, [_ds(resolution=2048)], arch='qwen_image_2', low_vram=False,
    )
    assert result is cfg


# ----------------------------------------------------------------------
# Preview-only runs never decode under grad: exempt from the qwen guards
# ----------------------------------------------------------------------

def test_qwen_depth_preview_only_skips_guards():
    cfg = DepthConsistencyConfig(loss_weight=0.0, preview_only=True, mask_source='none')
    result = preflight_depth_consistency(
        cfg, [_ds(resolution=2048)], arch='qwen_image_2', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg


def test_qwen_normal_preview_only_skips_guards():
    cfg = NormalIDConfig(loss_weight=0.0, preview_only=True)
    result = preflight_normal_id(
        cfg, [_ds(resolution=2048)], arch='qwen_image_2', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg


def test_qwen_depth_preview_only_with_dataset_weight_skips_guards():
    # preview_only is process-wide and ANDed into depth_objective
    # (`in_band & (eff_w > 0) & (not preview_only)`), so even a positive
    # dataset weight contributes no gradient: every sample takes the no-grad
    # preview path and the differentiable-decode guards must not fire
    cfg = DepthConsistencyConfig(loss_weight=0.0, preview_only=True, mask_source='none')
    result = preflight_depth_consistency(
        cfg, [_ds(depth_loss_weight=0.1, resolution=2048)],
        arch='qwen_image_2', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg


def test_qwen_normal_preview_only_with_dataset_weight_skips_guards():
    # same contract for normal: the live decode runs under
    # set_grad_enabled(not preview_only) and the loss returns 0.0
    cfg = NormalIDConfig(loss_weight=0.0, preview_only=True)
    result = preflight_normal_id(
        cfg, [_ds(normal_loss_weight=0.1, resolution=2048)],
        arch='qwen_image_2', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg


# ----------------------------------------------------------------------
# Other archs are untouched
# ----------------------------------------------------------------------

@pytest.mark.parametrize('preflight,make_cfg,weight_key', _make_active())
def test_flux_with_limit_set_is_unaffected(preflight, make_cfg, weight_key):
    cfg = make_cfg()
    result = preflight(
        cfg, [_ds(resolution=2048)], arch='flux', low_vram=True,
        decode_pixel_limit=LIMIT, max_active_bucket_pixels=LIMIT * 4,
    )
    assert result is cfg
