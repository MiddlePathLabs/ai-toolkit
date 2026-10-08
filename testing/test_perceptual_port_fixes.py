"""Port-parity fixes vs ai-toolkit-perceptual (2026-10-08 audit).

* body-shape: live crop matches the cached-GT crop (full frame).
* identity: cached face bbox follows the dataloader's flip/scale/crop.
* config: perceptual-fork face_id layout is translated, not silently dropped.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.admission import (
    RULE_DEPTH_FORK_DEFAULTS,
    RULE_LEGACY_FACE_ID,
    RULE_LEGACY_FACE_ID_CONFLICT,
    RULE_UNSUPPORTED_PERCEPTUAL_KEYS,
    collect_admission_diagnostics,
)
from toolkit.body_shape import DifferentiableBodyShapeEncoder
from toolkit.config_modules import FaceIDConfig
from toolkit.face_id_loss import UNKNOWN_FRAME, map_face_bbox_to_training_frame
from toolkit.perceptual_config_compat import (
    SUPPORTED_FACE_ID_KEYS,
    normalize_perceptual_config,
)


# ---------------------------------------------------------------------------
# Body shape: GT crop == live crop
# ---------------------------------------------------------------------------

class _RecordingBodyShape(DifferentiableBodyShapeEncoder):
    """Real crop/resize/normalize code paths; backbone replaced by a recorder."""

    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer('img_mean', torch.tensor([0.406, 0.457, 0.48]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1))
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.seen = []

    def _backbone(self, x):
        self.seen.append(x.detach().clone())
        return x

    def _predict_betas(self, features):
        return torch.zeros(features.shape[0], 10)


def _portrait():
    from PIL import Image
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(96, 64, 3), dtype=np.uint8)  # H=96, W=64
    return Image.fromarray(arr), torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def test_body_shape_full_frame_forward_matches_cached_encode():
    enc = _RecordingBodyShape()
    pil, pixels = _portrait()
    enc.encode(pil)  # cache path: person_bbox=None
    enc(pixels, person_bboxes=[[0.0, 0.0, 64.0, 96.0]])  # trainer path after fix
    gt_input, live_input = enc.seen
    assert torch.allclose(gt_input, live_input, atol=1e-5)


def test_body_shape_unboxed_forward_is_a_different_crop():
    # Documents the pre-fix mismatch: forward(None) center-squares a portrait.
    enc = _RecordingBodyShape()
    pil, pixels = _portrait()
    enc.encode(pil)
    enc(pixels)
    gt_input, live_input = enc.seen
    assert not torch.allclose(gt_input, live_input, atol=1e-3)


# ---------------------------------------------------------------------------
# Identity: face bbox follows dataloader geometry
# ---------------------------------------------------------------------------

def _item(**kw):
    defaults = dict(
        width=1000, height=500, flip_x=False, flip_y=False,
        scale_to_width=1000, scale_to_height=500,
        crop_x=0, crop_y=0, crop_width=1000, crop_height=500,
        dataset_config=SimpleNamespace(buckets=True, random_crop=False),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _bbox(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_identity_bbox_no_transform_is_identity():
    out = map_face_bbox_to_training_frame(_bbox(0.1, 0.2, 0.3, 0.4), _item())
    assert out == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_identity_bbox_follows_horizontal_flip():
    out = map_face_bbox_to_training_frame(_bbox(0.1, 0.2, 0.3, 0.4), _item(flip_x=True))
    assert out == pytest.approx([0.7, 0.2, 0.9, 0.4])


def test_identity_bbox_follows_bucket_scale_and_crop():
    # 1000x500 -> 512x256, then crop x 128..384 (256 wide).
    item = _item(scale_to_width=512, scale_to_height=256, crop_x=128, crop_width=256, crop_height=256)
    # face at x 0.4..0.6 of the source -> 204.8..307.2 scaled -> 76.8..179.2 in crop
    out = map_face_bbox_to_training_frame(_bbox(0.4, 0.2, 0.6, 0.4), item)
    assert out == pytest.approx([76.8 / 256, 0.2, 179.2 / 256, 0.4])


def test_identity_bbox_cropped_out_returns_none():
    item = _item(scale_to_width=512, scale_to_height=256, crop_x=256, crop_width=256, crop_height=256)
    assert map_face_bbox_to_training_frame(_bbox(0.05, 0.2, 0.2, 0.4), item) is None


def test_identity_bbox_partially_cropped_is_clamped():
    item = _item(scale_to_width=512, scale_to_height=256, crop_x=128, crop_width=256, crop_height=256)
    out = map_face_bbox_to_training_frame(_bbox(0.2, 0.2, 0.3, 0.4), item)  # 102.4..153.6 scaled
    assert out == pytest.approx([0.0, 0.2, (153.6 - 128) / 256, 0.4])


def test_identity_bbox_nonbucket_center_crop():
    item = _item(dataset_config=SimpleNamespace(buckets=False, random_crop=False))
    # 1000x500 center square = x 250..750
    out = map_face_bbox_to_training_frame(_bbox(0.4, 0.2, 0.6, 0.4), item)
    assert out == pytest.approx([(400 - 250) / 500, 0.2, (600 - 250) / 500, 0.4])


def test_identity_bbox_nonbucket_random_crop_is_unknown():
    item = _item(dataset_config=SimpleNamespace(buckets=False, random_crop=True))
    assert map_face_bbox_to_training_frame(_bbox(0.4, 0.2, 0.6, 0.4), item) == UNKNOWN_FRAME


def test_identity_bbox_zero_means_no_face():
    assert map_face_bbox_to_training_frame(_bbox(0, 0, 0, 0), _item()) is None


# ---------------------------------------------------------------------------
# Config: perceptual-fork face_id layout
# ---------------------------------------------------------------------------

def _fork_process(**extra):
    process = {
        "type": "sd_trainer",
        "model": {"name_or_path": "x", "arch": "flux"},
        "train": {"optimizer": "adamw", "batch_size": 1},
        "datasets": [{"folder_path": "x", "buckets": True}],
        "face_id": {
            "identity_loss_weight": 0.1,
            "body_proportion_loss_weight": 0.05,
            "body_proportion_include_head": True,
            "body_shape_loss_weight": 0.02,
            "body_shape_loss_min_cos": 0.3,
            "normal_loss_weight": 0.01,
            "vae_anchor_loss_weight": 0.2,
            "vae_anchor_model_path": "/vae.safetensors",
        },
    }
    process.update(extra)
    return process


def test_supported_face_id_keys_match_face_id_config():
    cfg = FaceIDConfig()
    for key in SUPPORTED_FACE_ID_KEYS:
        assert hasattr(cfg, key), key
    attrs = {k for k in vars(cfg) if not k.startswith('_')}
    assert attrs <= SUPPORTED_FACE_ID_KEYS


def test_fork_face_id_keys_move_to_sections():
    raw = _fork_process()
    report = normalize_perceptual_config(raw)
    p = report.process
    assert p["face_id"] == {"identity_loss_weight": 0.1}
    assert p["body_proportion"] == {"loss_weight": 0.05, "include_head": True}
    assert p["body_shape"] == {"loss_weight": 0.02, "loss_min_cos": 0.3}
    assert p["normal_id"] == {"loss_weight": 0.01}
    assert p["vae_anchor"] == {"loss_weight": 0.2, "vae_model_path": "/vae.safetensors"}
    assert not report.conflicts
    # input untouched
    assert "body_shape_loss_weight" in raw["face_id"] and "body_shape" not in raw


def test_fork_key_merges_into_existing_section():
    report = normalize_perceptual_config(_fork_process(body_shape={"loss_max_t": 0.9}))
    assert report.process["body_shape"] == {"loss_max_t": 0.9, "loss_weight": 0.02, "loss_min_cos": 0.3}


def test_conflicting_fork_key_is_reported_and_left_in_place():
    report = normalize_perceptual_config(_fork_process(body_shape={"loss_weight": 0.5}))
    assert [(k, d) for k, d, _, _ in report.conflicts] == [("body_shape_loss_weight", "body_shape.loss_weight")]
    assert report.process["body_shape"]["loss_weight"] == 0.5
    assert report.process["face_id"]["body_shape_loss_weight"] == 0.02


def test_unsupported_fork_settings_are_reported():
    process = _fork_process(body_id={"enabled": True})
    process["face_id"]["identity_metrics"] = True
    process["face_id"]["face_suppression_weight"] = 0.5
    process["datasets"][0]["depth_as_control"] = True
    report = normalize_perceptual_config(process)
    assert set(report.ignored_face_id) == {"identity_metrics", "face_suppression_weight"}
    assert report.ignored_dataset == [(0, "depth_as_control")]
    assert report.ignored_sections == ["body_id"]


def test_depth_section_relying_on_fork_defaults_is_reported():
    assert normalize_perceptual_config(_fork_process(depth_consistency={})).depth_default_keys == [
        "loss_weight", "mask_source",
    ]
    explicit = {"loss_weight": 0.1, "mask_source": "subject"}
    assert normalize_perceptual_config(_fork_process(depth_consistency=explicit)).depth_default_keys == []


def _ids(process):
    result = collect_admission_diagnostics({"process": [process]})
    return result, {(d["rule_id"], d["severity"]) for d in result["diagnostics"]}


def test_admission_warns_on_translation():
    _, ids = _ids(_fork_process())
    assert (RULE_LEGACY_FACE_ID, "warning") in ids


def test_admission_rejects_conflict():
    result, ids = _ids(_fork_process(body_shape={"loss_weight": 0.5}))
    assert (RULE_LEGACY_FACE_ID_CONFLICT, "error") in ids
    assert not result["valid"]


def test_admission_warns_on_unsupported_and_depth_defaults():
    process = _fork_process(depth_consistency={})
    process["face_id"]["landmark_loss_weight"] = 0.1
    _, ids = _ids(process)
    assert (RULE_UNSUPPORTED_PERCEPTUAL_KEYS, "warning") in ids
    assert (RULE_DEPTH_FORK_DEFAULTS, "warning") in ids


def test_native_layout_is_silent():
    process = _fork_process(
        face_id={"identity_loss_weight": 0.1, "face_model": "buffalo_l", "identity_loss_use_average": False},
        body_shape={"loss_weight": 0.02},
        depth_consistency={"loss_weight": 0.0, "mask_source": "none"},
    )
    report = normalize_perceptual_config(process)
    assert not report.has_notes
    _, ids = _ids(process)
    compat_rules = {RULE_LEGACY_FACE_ID, RULE_LEGACY_FACE_ID_CONFLICT,
                    RULE_UNSUPPORTED_PERCEPTUAL_KEYS, RULE_DEPTH_FORK_DEFAULTS}
    assert not {rule for rule, _ in ids} & compat_rules


# ---------------------------------------------------------------------------
# Trainer wiring (fakes; no model weights)
# ---------------------------------------------------------------------------

def _fake_trainer(px_h=32, px_w=64):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    trainer = SDTrainer.__new__(SDTrainer)
    trainer.device_torch = torch.device("cpu")
    vae = nn.Linear(1, 1)
    trainer.sd = SimpleNamespace(
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
        is_flow_matching=True,
        vae=vae,
        vae_torch_dtype=torch.float32,
        # Decoder stand-in: a grad-carrying [-1, 1] image of the decode size.
        decode_latents=lambda x0, device=None, dtype=None: torch.tanh(
            x0.mean() + torch.zeros(x0.shape[0], 3, px_h, px_w)
        ),
    )
    return trainer


class _RecordingFaceEncoder:
    def __init__(self):
        self.bboxes = None
        self.n = None

    def __call__(self, pixels, bboxes=None, return_crops=False):
        self.bboxes = bboxes
        self.n = pixels.shape[0]
        emb = torch.nn.functional.normalize(pixels.flatten(1)[:, :512].repeat(1, 1) + 1.0, dim=-1)
        if emb.shape[1] < 512:
            emb = torch.nn.functional.pad(emb, (0, 512 - emb.shape[1]))
        return emb, None


def _identity_batch(file_items, bboxes):
    B = len(file_items)
    return SimpleNamespace(
        identity_embedding=torch.ones(B, 512),
        face_bboxes=bboxes,
        file_items=file_items,
        get_is_reg_list=lambda: [False] * B,
        identity_loss_weight_list=None,
        identity_loss_min_t_list=None,
        identity_loss_max_t_list=None,
        identity_loss_min_cos_list=None,
    )


def _run_identity(file_items, bboxes):
    trainer = _fake_trainer()
    trainer.face_id_config = FaceIDConfig(identity_loss_weight=1.0, identity_loss_min_cos=-2.0)
    enc = _RecordingFaceEncoder()
    trainer._id_loss_model = enc
    trainer._identity_mean_embed = None
    trainer._id_face_detector = None
    B = len(file_items)
    noise_pred = torch.zeros(B, 4, 4, 8, requires_grad=True)
    loss = trainer._compute_face_identity_anchor_loss(
        noise_pred, torch.zeros(B, 4, 4, 8), torch.full((B,), 500.0), _identity_batch(file_items, bboxes),
    )
    return enc, loss


def test_trainer_identity_uses_flipped_bbox():
    enc, _ = _run_identity([_item(flip_x=True)], [_bbox(0.1, 0.2, 0.3, 0.4)])
    # decode is 64x32; flipped x 0.7..0.9
    assert enc.bboxes[0] == pytest.approx([0.7 * 64, 0.2 * 32, 0.9 * 64, 0.4 * 32])


def test_trainer_identity_skips_sample_whose_face_was_cropped_out():
    cropped = _item(scale_to_width=512, scale_to_height=256, crop_x=256, crop_width=256, crop_height=256)
    enc, loss = _run_identity([cropped, _item()], [_bbox(0.05, 0.2, 0.2, 0.4), _bbox(0.1, 0.2, 0.3, 0.4)])
    assert enc.n == 1  # only the in-frame sample reaches ArcFace
    assert torch.is_tensor(loss)


def test_trainer_body_shape_passes_full_frame_bbox():
    from toolkit.config_modules import BodyShapeConfig

    trainer = _fake_trainer(px_h=48, px_w=32)
    trainer.body_shape_config = BodyShapeConfig(loss_weight=1.0, loss_min_t=0.0, loss_max_t=1.0, loss_min_cos=-2.0)
    seen = {}

    def _encoder(pixels, person_bboxes=None):
        seen["bboxes"] = person_bboxes
        return pixels.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 10) + 1.0

    trainer._body_shape_perceptor = _encoder
    batch = SimpleNamespace(
        body_shape_gt=torch.ones(2, 10),
        get_is_reg_list=lambda: [False, False],
        body_shape_loss_weight_list=None,
        body_shape_loss_min_t_list=None,
        body_shape_loss_max_t_list=None,
        body_shape_loss_min_cos_list=None,
    )
    trainer._compute_body_shape_anchor_loss(
        torch.zeros(2, 4, 6, 4, requires_grad=True), torch.zeros(2, 4, 6, 4), torch.full((2,), 600.0), batch,
    )
    assert seen["bboxes"] == [[0.0, 0.0, 32.0, 48.0]] * 2


# ---------------------------------------------------------------------------
# Config checker: depth preview_only + implicit identity mode
# ---------------------------------------------------------------------------

from toolkit.admission import RULE_DEPTH_PREVIEW_ONLY_WEIGHT, RULE_IDENTITY_MODE_IMPLICIT  # noqa: E402


def _native(**extra):
    process = _fork_process(face_id={"identity_loss_weight": 0.0})
    process.update(extra)
    return process


@pytest.mark.parametrize("depth,dataset_w,expected", [
    ({"loss_weight": 0.1, "mask_source": "none", "preview_only": True}, None, True),
    ({"loss_weight": 0.0, "mask_source": "none", "preview_only": True}, 0.2, True),
    ({"loss_weight": 0.0, "mask_source": "none", "preview_only": True}, None, False),
    ({"loss_weight": 0.1, "mask_source": "none", "preview_only": False}, None, False),
])
def test_depth_preview_only_with_weight_warns(depth, dataset_w, expected):
    process = _native(depth_consistency=depth)
    if dataset_w is not None:
        process["datasets"][0]["depth_loss_weight"] = dataset_w
    assert normalize_perceptual_config(process).depth_preview_only_blocks_weight is expected
    _, ids = _ids(process)
    assert ((RULE_DEPTH_PREVIEW_ONLY_WEIGHT, "warning") in ids) is expected


@pytest.mark.parametrize("face_id,dataset_w,expected", [
    ({"identity_loss_weight": 0.1}, None, True),
    ({"identity_loss_weight": 0.0}, 0.1, True),
    ({"identity_loss_weight": 0.1, "identity_loss_use_average": False}, None, False),
    ({"identity_loss_weight": 0.1, "identity_loss_use_average": True}, None, False),
    ({"identity_loss_weight": 0.0}, None, False),
])
def test_identity_mode_implicit_warns_only_when_identity_active(face_id, dataset_w, expected):
    process = _native(face_id=face_id)
    if dataset_w is not None:
        process["datasets"][0]["identity_loss_weight"] = dataset_w
    _, ids = _ids(process)
    assert ((RULE_IDENTITY_MODE_IMPLICIT, "warning") in ids) is expected


def test_identity_mode_keys_are_supported_not_ignored():
    process = _native(face_id={
        "identity_loss_weight": 0.1, "identity_loss_use_average": True, "identity_loss_average_blend": 0.2,
        "identity_loss_use_random": True, "identity_loss_num_refs": 3,
    })
    assert normalize_perceptual_config(process).ignored_face_id == []
    cfg = FaceIDConfig(**process["face_id"])
    assert cfg.identity_loss_use_average is True and cfg.identity_loss_num_refs == 3


def test_identity_mode_defaults_and_validation():
    cfg = FaceIDConfig()
    assert cfg.identity_loss_use_average is False
    assert cfg.identity_loss_average_blend == 0.0
    assert cfg.identity_loss_use_random is False and cfg.identity_loss_num_refs == 0
    with pytest.raises(ValueError):
        FaceIDConfig(identity_loss_average_blend=1.5)
    with pytest.raises(ValueError):
        FaceIDConfig(identity_loss_num_refs=-1)


# ---------------------------------------------------------------------------
# Identity reference modes vs the perceptual fork's algorithm
# ---------------------------------------------------------------------------

from toolkit.face_id_loss import (  # noqa: E402
    bias_corrected_cosine,
    build_identity_reference_tables,
    identity_dataset_key,
    identity_reference_cosine,
)

F = torch.nn.functional


def _unit(n, seed):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, 512, generator=g), dim=-1)


def _source_average_mode(gen, orig, dataset_embeds, mean):
    """Transcription of the fork: refs replaced by the dataset mean, clean-cos targets."""
    avg = torch.stack(dataset_embeds).mean(dim=0)
    avg = avg / (avg.norm() + 1e-8)
    avg_c = F.normalize(avg - mean, p=2, dim=-1)
    clean = torch.tensor([
        max(F.cosine_similarity(F.normalize(o - mean, dim=-1).unsqueeze(0), avg_c.unsqueeze(0)).item(), 0.1)
        for o in orig
    ])
    gen_c = F.normalize(gen - mean.unsqueeze(0), dim=-1)
    ref_c = F.normalize(avg.unsqueeze(0).expand_as(gen) - mean.unsqueeze(0), dim=-1)
    cos = F.cosine_similarity(gen_c, ref_c, dim=-1)
    return cos, clean, torch.clamp(1.0 - cos / clean, min=0.0)


def test_average_mode_matches_fork():
    dataset = list(_unit(6, 1))
    orig = torch.stack(dataset[:3])
    gen = _unit(3, 2)
    mean = _unit(1, 3)[0] * 0.3
    cfg = FaceIDConfig(identity_loss_use_average=True)
    avgs, pools = build_identity_reference_tables({"ds": dataset}, use_average=True)
    cos, clean = identity_reference_cosine(gen, orig, ["ds"] * 3, cfg, avgs, pools, mean)
    src_cos, src_clean, src_loss = _source_average_mode(gen, orig, dataset, mean)
    assert torch.allclose(cos, src_cos, atol=1e-5)
    assert torch.allclose(clean, src_clean, atol=1e-5)
    assert torch.allclose(torch.clamp(1.0 - cos / clean, min=0.0), src_loss, atol=1e-5)
    # the fork built its random pools after the replacement: only the average
    assert pools["ds"].shape[0] == 1


def test_per_image_mode_is_plain_bias_corrected_cosine():
    orig, gen, mean = _unit(3, 4), _unit(3, 5), _unit(1, 6)[0] * 0.3
    cos, clean = identity_reference_cosine(gen, orig, ["ds"] * 3, FaceIDConfig(), {}, {}, mean)
    assert clean is None
    assert torch.allclose(cos, bias_corrected_cosine(gen, orig, mean))


def test_blend_mode_matches_fork():
    dataset = list(_unit(5, 7))
    orig, gen = torch.stack(dataset[:2]), _unit(2, 8)
    cfg = FaceIDConfig(identity_loss_average_blend=0.25)
    avgs, pools = build_identity_reference_tables({"ds": dataset}, use_average=False)
    cos, clean = identity_reference_cosine(gen, orig, ["ds"] * 2, cfg, avgs, pools, None)
    blended = 0.75 * orig + 0.25 * avgs["ds"]
    blended = blended / (blended.norm(dim=-1, keepdim=True) + 1e-8)
    assert clean is None
    assert torch.allclose(cos, bias_corrected_cosine(gen, blended, None), atol=1e-6)


class _SeqRng:
    def __init__(self, picks):
        self.picks = list(picks)

    def randint(self, a, b):
        return self.picks.pop(0)


def test_random_and_multi_ref_modes():
    dataset = list(_unit(4, 9))
    gen = dataset[2].unsqueeze(0).clone()  # identical to pool entry 2
    orig = dataset[0].unsqueeze(0)
    avgs, pools = build_identity_reference_tables({"ds": dataset}, use_average=False)

    rnd = FaceIDConfig(identity_loss_use_random=True)
    cos, _ = identity_reference_cosine(gen, orig, ["ds"], rnd, avgs, pools, None, rng=_SeqRng([2]))
    assert cos.item() == pytest.approx(1.0, abs=1e-5)

    multi = FaceIDConfig(identity_loss_num_refs=3)
    cos, _ = identity_reference_cosine(gen, orig, ["ds"], multi, avgs, pools, None, rng=_SeqRng([1, 2]))
    assert cos.item() == pytest.approx(1.0, abs=1e-5)  # best of {ref, pool[1], pool[2]}
    assert cos.item() >= bias_corrected_cosine(gen, orig, None).item()


def test_reference_tables_skip_zero_embeddings():
    dataset = list(_unit(2, 10)) + [torch.zeros(512)]
    avgs, pools = build_identity_reference_tables(
        {"ds": dataset, "empty": [torch.zeros(512)]}, use_average=False,
    )
    assert set(avgs) == {"ds"} and pools["ds"].shape[0] == 2


def test_trainer_average_mode_uses_clean_targets():
    ds_cfg = SimpleNamespace(buckets=True, random_crop=False, folder_path="/data/ds")
    items = [_item(dataset_config=ds_cfg), _item(dataset_config=ds_cfg)]
    trainer = _fake_trainer()
    trainer.face_id_config = FaceIDConfig(
        identity_loss_weight=1.0, identity_loss_min_cos=-2.0, identity_loss_use_average=True,
    )
    trainer._id_loss_model = _RecordingFaceEncoder()
    trainer._identity_mean_embed = None
    trainer._id_face_detector = None
    dataset = list(_unit(4, 11))
    trainer._identity_avg_embeds, trainer._identity_embed_pools = build_identity_reference_tables(
        {identity_dataset_key(ds_cfg): dataset}, use_average=True,
    )
    batch = _identity_batch(items, [_bbox(0.1, 0.2, 0.3, 0.4)] * 2)
    batch.identity_embedding = torch.stack(dataset[:2])
    loss = trainer._compute_face_identity_anchor_loss(
        torch.zeros(2, 4, 4, 8, requires_grad=True), torch.zeros(2, 4, 4, 8),
        torch.full((2,), 500.0), batch,
    )
    assert torch.is_tensor(loss) and loss.requires_grad
    # Expected value from the fork's formula, using the fake encoder's output
    # for the fake decode (all-zero x0 -> tanh(0) -> 0.5 pixels).
    gen = _RecordingFaceEncoder()(torch.full((2, 3, 32, 64), 0.5))[0]
    _, _, src_loss = _source_average_mode(gen, torch.stack(dataset[:2]), dataset, torch.zeros(512))
    expected = (src_loss * 0.5).sum() / 2  # t_ratio 0.5, weight 1, both samples in the mask
    assert loss.item() == pytest.approx(expected.item(), rel=1e-4)
