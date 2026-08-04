"""Focused tests for the Phase 3 auto-masking (subject_mask) integration.

Covers: SubjectMaskConfig defaults, DatasetConfig per-dataset overrides, the
region-weight helpers (_build_subject_mask_weight, _build_body_restrict_mask)
bound to a fake trainer with synthetic masks, and the dataloader mask stacking.
No YOLO/SAM2/SegFormer models or GPU required -- the extractor is not exercised
here (covered by a separate acceptance run).
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, SubjectMaskConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = SubjectMaskConfig()
    assert c.enabled is False
    assert c.yolo_ckpt == 'yolo11n.pt'
    assert c.sam_size == 'small'
    assert c.cache_resolution == 256
    assert c.body_close_radius == 2
    assert c.background_loss_weight is None
    assert c.clothing_loss_weight is None
    assert c.body_loss_weight is None
    assert c.perceptual_restrict_to_body is False


def test_partial_object_merge():
    c = SubjectMaskConfig(enabled=True, background_loss_weight=0.5, body_loss_weight=2.0)
    assert c.enabled is True
    assert c.background_loss_weight == 0.5
    assert c.body_loss_weight == 2.0


def test_dataset_overrides_default_none():
    d = DatasetConfig(dataset_path=".", resolution=64)
    assert d.background_loss_weight is None
    assert d.clothing_loss_weight is None
    assert d.body_loss_weight is None
    assert d.perceptual_restrict_to_body is None


# ---------------------------------------------------------------------------
# Fake trainer + synthetic masks for the loss-weight helpers
# ---------------------------------------------------------------------------

def _fake_trainer(subject_mask_config, device=torch.device("cpu")):
    """A minimal stand-in exposing only what the two helpers read."""
    t = SimpleNamespace()
    t.subject_mask_config = subject_mask_config
    t.device_torch = device
    # bind the REAL unbound methods so the prod code path is exercised
    t._build_subject_mask_weight = SDTrainer._build_subject_mask_weight.__get__(t)
    t._build_body_restrict_mask = SDTrainer._build_body_restrict_mask.__get__(t)
    return t


def _batch_with_masks(B, Hc=8, Wc=8, per_item_weights=None, restrict=None):
    """Build a fake batch with stacked bool masks + per-item weight lists."""
    person = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
    body = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
    clothing = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
    # put a person in the left half, body in the top-left quarter
    person[:, :, :, :Wc // 2] = True
    body[:, :, :Hc // 2, :Wc // 2] = True
    clothing[:, :, :Hc // 2, Wc // 4:Wc // 2] = True
    per_item_weights = per_item_weights or {}
    batch = SimpleNamespace(
        subject_masks=person, body_masks=body, clothing_masks=clothing,
        background_loss_weight_list=per_item_weights.get('bg', [None] * B),
        clothing_loss_weight_list=per_item_weights.get('cl', [None] * B),
        body_loss_weight_list=per_item_weights.get('bd', [None] * B),
        perceptual_restrict_to_body_list=restrict or [None] * B,
        file_items=[SimpleNamespace() for _ in range(B)],
    )
    return batch


def test_build_subject_mask_weight_disabled_returns_none():
    t = _fake_trainer(SubjectMaskConfig())  # enabled=False
    assert t._build_subject_mask_weight(_batch_with_masks(2), (2, 16, 8, 8)) is None


def test_build_subject_mask_weight_enabled_but_all_weights_none_returns_none():
    t = _fake_trainer(SubjectMaskConfig(enabled=True))
    assert t._build_subject_mask_weight(_batch_with_masks(2), (2, 16, 8, 8)) is None


def test_build_subject_mask_weight_bg_zero_reduces_outside_person():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, background_loss_weight=0.0))
    w = t._build_subject_mask_weight(_batch_with_masks(1), (1, 16, 8, 8))
    assert w is not None
    assert w.shape == (1, 16, 8, 8)
    # outside the person region (right half) the weight should be ~0
    assert float(w[0, 0, 0, 6]) < 1e-5
    # inside the person region (left half) it should be ~1
    assert abs(float(w[0, 0, 0, 1]) - 1.0) < 1e-5


def test_build_subject_mask_weight_body_weight_boosts_body():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, body_loss_weight=2.0))
    w = t._build_subject_mask_weight(_batch_with_masks(1), (1, 16, 8, 8))
    assert w is not None
    # inside body (top-left) weight is 2.0; outside is 1.0
    assert abs(float(w[0, 0, 0, 0]) - 2.0) < 1e-5
    assert abs(float(w[0, 0, 6, 6]) - 1.0) < 1e-5


def test_build_subject_mask_weight_per_dataset_override_wins():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, body_loss_weight=2.0))
    batch = _batch_with_masks(1, per_item_weights={'bd': [3.0]})
    w = t._build_subject_mask_weight(batch, (1, 16, 8, 8))
    assert abs(float(w[0, 0, 0, 0]) - 3.0) < 1e-5


def test_build_body_restrict_mask_disabled_returns_none():
    t = _fake_trainer(SubjectMaskConfig())  # enabled=False
    assert t._build_body_restrict_mask(_batch_with_masks(2), (2, 8, 8)) is None


def test_build_body_restrict_mask_no_opt_in_returns_none():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, perceptual_restrict_to_body=False))
    assert t._build_body_restrict_mask(_batch_with_masks(2), (2, 8, 8)) is None


def test_build_body_restrict_mask_global_opt_in():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, perceptual_restrict_to_body=True))
    m = t._build_body_restrict_mask(_batch_with_masks(2), (2, 8, 8))
    assert m is not None
    assert m.shape == (2, 8, 8)
    # inside body (top-left) = 1.0, outside = 0.0
    assert abs(float(m[0, 0, 0]) - 1.0) < 1e-5
    assert abs(float(m[0, 6, 6]) - 0.0) < 1e-5


def test_build_body_restrict_mask_per_item_opt_in_leaves_others_one():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, perceptual_restrict_to_body=False))
    batch = _batch_with_masks(2, restrict=[True, False])
    m = t._build_body_restrict_mask(batch, (2, 8, 8))
    assert m is not None
    # item 0 opted in -> restricted (0 outside body); item 1 did not -> all ones
    assert abs(float(m[0, 6, 6]) - 0.0) < 1e-5
    assert abs(float(m[1, 6, 6]) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Dataloader mask stacking
# ---------------------------------------------------------------------------

def _make_image(path):
    from PIL import Image
    Image.new("RGB", (32, 32), (100, 100, 100)).save(str(path))
    return str(path)


def test_batch_stacks_subject_masks(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path), resolution=32,
        background_loss_weight=0.5, body_loss_weight=2.0, perceptual_restrict_to_body=True,
    )
    fi = FileItemDTO(path=image_path, dataset_config=cfg)
    fi.subject_mask = torch.ones(16, 16, dtype=torch.bool)
    fi.body_mask = torch.ones(16, 16, dtype=torch.bool)
    fi.clothing_mask = torch.zeros(16, 16, dtype=torch.bool)
    from torchvision import transforms
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.subject_masks.shape == (1, 1, 16, 16)
    assert batch.body_masks.shape == (1, 1, 16, 16)
    assert bool(batch.subject_masks.all()) is True
    assert bool(batch.clothing_masks.any()) is False
    assert batch.background_loss_weight_list == [0.5]
    assert batch.body_loss_weight_list == [2.0]
    assert batch.perceptual_restrict_to_body_list == [True]


def test_batch_without_masks_has_none(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = FileItemDTO(path=image_path, dataset_config=cfg)
    from torchvision import transforms
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert getattr(batch, 'subject_masks', None) is None
    assert getattr(batch, 'body_masks', None) is None
