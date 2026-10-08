"""Video-item guard for the image-only perceptual features.

The guard lives at the top of SDTrainer.hook_before_train_loop, before any
preflight or GT cache pass (those ``Image.open`` every file item and would
crash on a video first). These tests pin the two decision helpers so the
guard's semantics stay honest without a full trainer.
"""
from types import SimpleNamespace

from extensions_built_in.sd_trainer.SDTrainer import (
    dataloader_has_video_items,
    enabled_image_only_perceptual_features,
)


def _trainer(**flags):
    ns = SimpleNamespace(
        subject_mask_config=None,
        _depth_loss_active=lambda: flags.get('depth', False),
        _normal_loss_active=lambda: flags.get('normal', False),
        _body_proportion_loss_active=lambda: flags.get('body_proportion', False),
        _face_identity_loss_active=lambda: flags.get('face_identity', False),
        _body_shape_loss_active=lambda: flags.get('body_shape', False),
        _vae_anchor_loss_active=lambda: flags.get('vae_anchor', False),
    )
    if flags.get('subject_mask'):
        ns.subject_mask_config = SimpleNamespace(enabled=True)
    return ns


class _Loader:
    def __init__(self, *paths):
        self.dataset = SimpleNamespace(
            file_list=[SimpleNamespace(path=p) for p in paths]
        )


# ----------------------------------------------------------------------
# enabled_image_only_perceptual_features
# ----------------------------------------------------------------------

def test_no_features_enabled_is_empty():
    assert enabled_image_only_perceptual_features(_trainer()) == []


def test_feature_list_reflects_config_activation():
    names = enabled_image_only_perceptual_features(
        _trainer(subject_mask=True, depth=True, vae_anchor=True)
    )
    assert names == ['subject_mask', 'depth_consistency', 'vae_anchor']


def test_anchor_names_exclude_subject_mask_for_5d_warning():
    # the 5D-latent warning lists anchors only (subject_mask fails with its
    # own error instead of silently deactivating)
    names = [
        f for f in enabled_image_only_perceptual_features(
            _trainer(subject_mask=True, normal=True)
        ) if f != 'subject_mask'
    ]
    assert names == ['normal']


# ----------------------------------------------------------------------
# dataloader_has_video_items
# ----------------------------------------------------------------------

def test_video_detected_in_either_loader():
    assert dataloader_has_video_items((_Loader('/a/x.png'), _Loader('/a/y.mp4')))


def test_image_only_loaders_are_clean():
    assert not dataloader_has_video_items((_Loader('/a/x.jpg', '/a/y.webp'),))
    assert not dataloader_has_video_items((_Loader('/a/x.PNG'), None))


def test_video_extension_matching_is_case_insensitive():
    assert dataloader_has_video_items((_Loader('/a/x.MOV'),))
    assert dataloader_has_video_items((_Loader('/a/x.WebM'),))


def test_none_loaders_are_clean():
    assert not dataloader_has_video_items((None, None))
