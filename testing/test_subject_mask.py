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

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, SubjectMaskConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.subject_mask import _resolve_subject_match


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
    # identity-matching additions: fully inert by default
    assert c.subject_identity_dataset is None
    assert c.match_threshold == 0.40
    assert c.match_margin == 0.10
    assert c.subject_person_loss_weight is None
    assert c.others_person_loss_weight is None


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
    assert d.subject_person_loss_weight is None
    assert d.others_person_loss_weight is None


# ---------------------------------------------------------------------------
# Identity-match decision core (pure, model-free)
# ---------------------------------------------------------------------------

IMG_W = IMG_H = 100


def _isolate_stub(mask_slice=(10, 40, 10, 40), fail=False):
    calls = []

    def fn(person_box):
        calls.append(person_box)
        if fail:
            return None
        m = np.zeros((IMG_H, IMG_W), dtype=bool)
        y1, y2, x1, x2 = mask_slice
        m[y1:y2, x1:x2] = True
        return m

    fn.calls = calls
    return fn


def test_single_face_with_two_person_boxes_isolates():
    # A faceless companion (facing away / occluded / cropped) is still a rival
    # person: one detected face + two person boxes must NOT claim the whole
    # person parse as the subject.
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7],
        boxes=[[5, 0, 45, 100], [55, 0, 95, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is not None, "single face + 2 person boxes must isolate"
    assert len(iso.calls) == 1
    assert iso.calls[0] == [5, 0, 45, 100]  # box containing the face


def test_single_face_single_person_box_returns_whole_person():
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7],
        boxes=[[5, 0, 45, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is None
    assert iso.calls == []  # no isolation needed


def test_single_face_no_boxes_returns_whole_person():
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7], boxes=[],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is None


def test_single_face_with_split_detection_boxes_merges_to_whole_person():
    # Close-up pathology: YOLO emitted a near-duplicate second detection of
    # the same person. Duplicates are not rival people — merge them and treat
    # the single face as unambiguous.
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([30.0, 20.0, 60.0, 60.0])], cos=[0.7],
        boxes=[[5, 0, 40, 100], [8, 4, 44, 96]],  # near-duplicates (IoU ~0.59)
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is None, "duplicate detections of one person must merge"
    assert iso.calls == []


def test_single_face_with_nested_fragment_box_merges_to_whole_person():
    # Small duplicate detection nested inside the full person box.
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7],
        boxes=[[5, 0, 45, 100], [15, 30, 35, 70]],  # fragment fully inside main box
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is None
    assert iso.calls == []


def test_merged_box_is_used_for_isolation_when_persons_are_disjoint():
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7],
        boxes=[[5, 0, 40, 100], [8, 4, 44, 96], [55, 0, 95, 100]],  # near-dup pair + rival
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is not None
    assert iso.calls == [[5, 0, 44, 100]]  # union of the overlapping pair


def test_patchy_silhouette_falls_back_to_rectangle():
    # Regression origin: a spurious rival detection on a close-up led
    # to a near-full-frame SAM prompt whose patchy silhouette misses the
    # subject's own face. The padded rectangle around the containing person
    # box covers the face, so isolation degrades to the rectangle — never to
    # whole-person: a rival exists, and region=None would label the rival's
    # pixels as the subject with an empty others mask.
    iso = _isolate_stub(mask_slice=(80, 100, 80, 100))  # far from the face
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7],
        boxes=[[0, 0, 60, 100], [55, 0, 95, 100]],  # disjoint rivals, no merge
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is not None
    assert r["isolation"] == "rectangle"
    assert r["region"][30, 30]          # subject's face is inside the region
    assert not r["region"][50, 80]      # rival (x >= 65) stays outside


def test_isolation_gate_double_failure_unmatches():
    # Both the SAM silhouette AND the rectangle fallback miss the matched
    # face -> the match fails so both identity weights no-op for the image.
    # It must NOT return matched=True/region=None: a rival exists and the
    # whole person parse would assign the rival's pixels to the subject.
    iso = _isolate_stub(mask_slice=(80, 100, 80, 100))
    r = _resolve_subject_match(
        face_boxes=[np.array([10.0, 10.0, 90.0, 90.0]),   # huge subject face
                    np.array([60.0, 10.0, 80.0, 50.0])],  # rival face
        cos=[0.7, 0.1],
        # tiny box containing only the subject face's center: both the patchy
        # SAM mask and its padded rectangle miss >50% of the big face box
        boxes=[[50, 45, 55, 55]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is False
    assert r["region"] is None
    assert r["isolation"] == "failed"
    assert r["face_cover"] < 0.5


def test_below_threshold_unmatched():
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.2],
        boxes=[[5, 0, 45, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=_isolate_stub())
    assert r["matched"] is False


def test_two_faces_below_margin_unmatched():
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0]), np.array([60.0, 10.0, 80.0, 50.0])],
        cos=[0.6, 0.55],
        boxes=[[5, 0, 45, 100], [55, 0, 95, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=_isolate_stub())
    assert r["matched"] is False
    assert r["face_count"] == 2


def test_two_faces_with_margin_isolates():
    iso = _isolate_stub()
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0]), np.array([60.0, 10.0, 80.0, 50.0])],
        cos=[0.7, 0.1],
        boxes=[[5, 0, 45, 100], [55, 0, 95, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=iso)
    assert r["matched"] is True
    assert r["region"] is not None
    assert iso.calls == [[5, 0, 45, 100]]


def test_isolation_failure_falls_back_to_rectangle():
    # isolate_fn returns None -> padded person-box rectangle region
    r = _resolve_subject_match(
        face_boxes=[np.array([20.0, 10.0, 40.0, 50.0])], cos=[0.7],
        boxes=[[10, 5, 40, 60], [55, 0, 95, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1,
        isolate_fn=_isolate_stub(fail=True))
    assert r["matched"] is True
    region = r["region"]
    assert region is not None and region.any()
    assert region[30, 25] and not region[0, 0] and not region[99, 99]


def test_no_faces_unmatched():
    r = _resolve_subject_match(
        face_boxes=[], cos=[], boxes=[[5, 0, 45, 100], [55, 0, 95, 100]],
        img_w=IMG_W, img_h=IMG_H, threshold=0.4, margin=0.1, isolate_fn=_isolate_stub())
    assert r["matched"] is False
    assert r["face_count"] == 0


# ---------------------------------------------------------------------------
# Preflight identity-weight detection (global AND per-dataset)
# ---------------------------------------------------------------------------

def _dataset_with_file_items(**dataset_kwargs):
    cfg = DatasetConfig(dataset_path=".", resolution=64, **dataset_kwargs)
    fi = SimpleNamespace(dataset_config=cfg)
    return SimpleNamespace(file_list=[fi])


def test_identity_weights_present_global_only():
    from extensions_built_in.sd_trainer.SDTrainer import _sm_identity_weights_present
    smc = SubjectMaskConfig(enabled=True, others_person_loss_weight=0.5)
    assert _sm_identity_weights_present(smc, []) is True


def test_identity_weights_present_per_dataset_only():
    # Global weights unset, one dataset carries the override -> the preflight
    # must still require subject_identity_dataset, or the loss helper would
    # silently disable the weight for the entire run.
    from extensions_built_in.sd_trainer.SDTrainer import _sm_identity_weights_present
    smc = SubjectMaskConfig(enabled=True)
    ds = _dataset_with_file_items(subject_person_loss_weight=0.2)
    assert _sm_identity_weights_present(smc, [SimpleNamespace(file_list=[]), ds]) is True


def test_identity_weights_present_none():
    from extensions_built_in.sd_trainer.SDTrainer import _sm_identity_weights_present
    smc = SubjectMaskConfig(enabled=True)
    ds = _dataset_with_file_items()
    assert _sm_identity_weights_present(smc, [ds]) is False


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


def _batch_with_masks(B, Hc=8, Wc=8, per_item_weights=None, restrict=None,
                      with_identity_masks=False):
    """Build a fake batch with stacked bool masks + per-item weight lists."""
    person = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
    body = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
    clothing = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
    # put a person in the left half, body in the top-left quarter
    person[:, :, :, :Wc // 2] = True
    body[:, :, :Hc // 2, :Wc // 2] = True
    clothing[:, :, :Hc // 2, Wc // 4:Wc // 2] = True
    per_item_weights = per_item_weights or {}
    # identity-matched split (only stacked when requested): subject = left
    # quarter, others = remainder of the person region
    subject_person = others_person = None
    if with_identity_masks:
        subject_person = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
        others_person = torch.zeros(B, 1, Hc, Wc, dtype=torch.bool)
        subject_person[:, :, :, :Wc // 4] = True
        others_person[:, :, :, Wc // 4:Wc // 2] = True
    batch = SimpleNamespace(
        subject_masks=person, body_masks=body, clothing_masks=clothing,
        subject_person_masks=subject_person, others_person_masks=others_person,
        background_loss_weight_list=per_item_weights.get('bg', [None] * B),
        clothing_loss_weight_list=per_item_weights.get('cl', [None] * B),
        body_loss_weight_list=per_item_weights.get('bd', [None] * B),
        subject_person_loss_weight_list=per_item_weights.get('sp', [None] * B),
        others_person_loss_weight_list=per_item_weights.get('op', [None] * B),
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


def test_build_subject_mask_weight_subject_person_downweights():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, subject_person_loss_weight=0.1))
    batch = _batch_with_masks(1, with_identity_masks=True)
    w = t._build_subject_mask_weight(batch, (1, 16, 8, 8))
    assert w is not None
    # inside the subject region (left quarter) weight is 0.1
    assert abs(float(w[0, 0, 4, 1]) - 0.1) < 1e-5
    # inside the others region (middle) and outside the person (right) it stays 1.0
    assert abs(float(w[0, 0, 4, 3]) - 1.0) < 1e-5
    assert abs(float(w[0, 0, 4, 6]) - 1.0) < 1e-5


def test_build_subject_mask_weight_others_person_downweights():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, others_person_loss_weight=0.25))
    batch = _batch_with_masks(1, with_identity_masks=True)
    w = t._build_subject_mask_weight(batch, (1, 16, 8, 8))
    assert w is not None
    # inside the others region (middle) weight is 0.25; subject + outside stay 1.0
    assert abs(float(w[0, 0, 4, 3]) - 0.25) < 1e-5
    assert abs(float(w[0, 0, 4, 1]) - 1.0) < 1e-5
    assert abs(float(w[0, 0, 4, 6]) - 1.0) < 1e-5


def test_build_subject_mask_weight_identity_weights_without_masks_return_none():
    # Weights set but no identity masks cached (e.g. subject_identity_dataset
    # not configured) -> no weight map at all, not a full-image rescale.
    t = _fake_trainer(SubjectMaskConfig(enabled=True, subject_person_loss_weight=0.1))
    assert t._build_subject_mask_weight(_batch_with_masks(2), (2, 16, 8, 8)) is None


def test_build_subject_mask_weight_identity_per_dataset_override_wins():
    t = _fake_trainer(SubjectMaskConfig(enabled=True, subject_person_loss_weight=0.1))
    batch = _batch_with_masks(2, with_identity_masks=True,
                              per_item_weights={'sp': [0.5, None]})
    w = t._build_subject_mask_weight(batch, (2, 16, 8, 8))
    assert abs(float(w[0, 0, 4, 1]) - 0.5) < 1e-5
    assert abs(float(w[1, 0, 4, 1]) - 0.1) < 1e-5  # falls back to global


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
        subject_person_loss_weight=0.1, others_person_loss_weight=None,
    )
    fi = FileItemDTO(path=image_path, dataset_config=cfg)
    fi.subject_mask = torch.ones(16, 16, dtype=torch.bool)
    fi.body_mask = torch.ones(16, 16, dtype=torch.bool)
    fi.clothing_mask = torch.zeros(16, 16, dtype=torch.bool)
    fi.subject_person_mask = torch.ones(16, 16, dtype=torch.bool)
    fi.others_person_mask = torch.zeros(16, 16, dtype=torch.bool)
    from torchvision import transforms
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.subject_masks.shape == (1, 1, 16, 16)
    assert batch.body_masks.shape == (1, 1, 16, 16)
    assert bool(batch.subject_masks.all()) is True
    assert bool(batch.clothing_masks.any()) is False
    assert batch.subject_person_masks.shape == (1, 1, 16, 16)
    assert bool(batch.subject_person_masks.all()) is True
    assert bool(batch.others_person_masks.any()) is False
    assert batch.background_loss_weight_list == [0.5]
    assert batch.body_loss_weight_list == [2.0]
    assert batch.perceptual_restrict_to_body_list == [True]
    assert batch.subject_person_loss_weight_list == [0.1]
    assert batch.others_person_loss_weight_list == [None]


def test_batch_without_masks_has_none(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = FileItemDTO(path=image_path, dataset_config=cfg)
    from torchvision import transforms
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert getattr(batch, 'subject_masks', None) is None
    assert getattr(batch, 'body_masks', None) is None
    assert getattr(batch, 'subject_person_masks', None) is None
    assert getattr(batch, 'others_person_masks', None) is None
