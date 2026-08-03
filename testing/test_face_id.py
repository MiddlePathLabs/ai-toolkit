"""Focused tests for the Phase 3 face-identity (ArcFace) anchor.

Covers: FaceIDConfig defaults + validation, DatasetConfig per-dataset
overrides, bias-corrected cosine math, the lazy mixin read/cleanup/reject, and
preflight_face_id. The perceptor (onnx2torch + insightface) is faked -- no
ArcFace weights or detection runs here; those are covered by a separate
CUDA-required acceptance run.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, FaceIDConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.face_id_loss import bias_corrected_cosine, compute_identity_loss


# ---------------------------------------------------------------------------
# Config defaults + validation
# ---------------------------------------------------------------------------

def test_no_args_yields_safe_disabled_defaults():
    c = FaceIDConfig()
    assert c.identity_loss_weight == 0.0
    assert c.identity_loss_min_t == 0.0
    assert c.identity_loss_max_t == 1.0
    assert c.identity_loss_min_cos == 0.2
    assert c.face_model == 'buffalo_l'
    assert c.identity_loss_decoded_det_threshold == 0.5


def test_config_has_no_enabled_field():
    assert not hasattr(FaceIDConfig(), 'enabled')


def test_partial_object_merge():
    c = FaceIDConfig(identity_loss_weight=0.1, identity_loss_min_cos=0.3)
    assert c.identity_loss_weight == 0.1
    assert c.identity_loss_min_cos == 0.3
    assert c.face_model == 'buffalo_l'


def test_timestep_validation():
    with pytest.raises(ValueError, match="identity_loss_min_t"):
        FaceIDConfig(identity_loss_min_t=-0.01)
    with pytest.raises(ValueError, match="identity_loss_max_t"):
        FaceIDConfig(identity_loss_max_t=1.01)
    with pytest.raises(ValueError, match="identity_loss_min_t"):
        FaceIDConfig(identity_loss_min_t=0.7, identity_loss_max_t=0.3)


def test_dataset_overrides_default_none():
    d = DatasetConfig(dataset_path=".", resolution=64)
    assert d.identity_loss_weight is None
    assert d.identity_loss_min_t is None
    assert d.identity_loss_max_t is None
    assert d.identity_loss_min_cos is None


# ---------------------------------------------------------------------------
# Bias-corrected cosine math
# ---------------------------------------------------------------------------

def test_cosine_identical_is_one():
    emb = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    cos = bias_corrected_cosine(emb, emb, mean_emb=None)
    assert torch.allclose(cos, torch.ones(2), atol=1e-5)


def test_loss_identical_is_zero():
    emb = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    loss = compute_identity_loss(emb, emb, mean_emb=None)
    assert torch.allclose(loss, torch.zeros(1), atol=1e-5)


def test_bias_correction_zero_mean_is_identity():
    # subtracting a zero mean direction is a no-op (renormalize is idempotent
    # on already-normalized embeddings).
    emb = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    cos_none = bias_corrected_cosine(emb, emb.flip(0), mean_emb=None)
    cos_zero = bias_corrected_cosine(emb, emb.flip(0), mean_emb=torch.zeros(512))
    assert torch.allclose(cos_none, cos_zero, atol=1e-5)


def test_bias_correction_nonzero_mean_changes_cosine():
    # a non-zero mean direction changes the cosine for non-identical pairs
    emb = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    cos_none = bias_corrected_cosine(emb, emb.flip(0), mean_emb=None)
    cos_mean = bias_corrected_cosine(emb, emb.flip(0), mean_emb=torch.randn(512))
    assert not torch.allclose(cos_none, cos_mean, atol=1e-4)


def test_bias_correction_preserves_identity_match():
    # an embedding matched against itself stays cosine 1.0 after centering
    emb = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    mean = torch.randn(512)
    cos = bias_corrected_cosine(emb, emb, mean_emb=mean)
    assert torch.allclose(cos, torch.ones(1), atol=1e-5)


def test_gradient_flows_through_gen():
    leaf = torch.randn(1, 512, requires_grad=True)
    emb = torch.nn.functional.normalize(leaf, dim=-1)
    ref = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    loss = compute_identity_loss(emb, ref, mean_emb=None)
    loss.backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()


# ---------------------------------------------------------------------------
# Lazy mixin read/cleanup/reject (face_identity)
# ---------------------------------------------------------------------------

def _make_image(path):
    from PIL import Image
    Image.new("RGB", (32, 32), (100, 100, 100)).save(str(path))
    return str(path)


def _make_file_item(image_path, dataset_config):
    return FileItemDTO(path=image_path, dataset_config=dataset_config)


def _stamp_identity_cache(file_item, cache_path, emb=None, bbox=None, *, nan=False, no_version=False):
    from safetensors.torch import save_file
    from toolkit.face_id import CACHE_VERSION_KEY
    if emb is None:
        emb = torch.randn(512)
    if bbox is None:
        bbox = torch.tensor([0.1, 0.1, 0.3, 0.3])
    if nan:
        emb = torch.full((512,), float("nan"))
    data = {"identity_embedding": emb, "face_bbox": bbox}
    if not no_version:
        data[CACHE_VERSION_KEY] = torch.ones(1)
    save_file(data, str(cache_path))
    file_item._face_cache_path = str(cache_path)
    file_item._face_cache_key = "identity_embedding"
    file_item._face_bbox_key = "face_bbox"
    file_item.is_face_identity_cached = True


def test_lazy_read_delivers_embedding_and_bbox(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    emb = torch.randn(512)
    bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _stamp_identity_cache(fi, tmp_path / "id.safetensors", emb=emb, bbox=bbox)
    fi.load_and_process_image(__import__("torchvision").transforms.Compose([__import__("torchvision").transforms.ToTensor()]))
    assert torch.equal(fi.identity_embedding, emb)
    assert torch.equal(fi.face_bbox, bbox)


def test_cleanup_releases_but_keeps_metadata(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    _stamp_identity_cache(fi, tmp_path / "id.safetensors")
    fi.get_face_identity_gt()
    assert fi.identity_embedding is not None
    fi.cleanup_face_identity()
    assert fi.identity_embedding is None
    assert fi.face_bbox is None
    assert fi.is_face_identity_cached is True  # metadata intact -> re-read works


def test_non_finite_cache_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=32)
    fi = _make_file_item(image_path, cfg)
    _stamp_identity_cache(fi, tmp_path / "id.safetensors", nan=True)
    assert fi.get_face_identity_gt() is None


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def test_batch_aggregates_identity(tmp_path):
    from torchvision import transforms
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path), resolution=32,
        identity_loss_weight=0.05, identity_loss_min_t=0.1, identity_loss_max_t=0.9, identity_loss_min_cos=0.25,
    )
    fi = _make_file_item(image_path, cfg)
    _stamp_identity_cache(fi, tmp_path / "id.safetensors")
    fi.load_and_process_image(transforms.Compose([transforms.ToTensor()]))
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.identity_embedding.shape == (1, 512)
    assert batch.identity_loss_weight_list == [0.05]
    assert batch.identity_loss_min_cos_list == [0.25]
    assert batch.face_bboxes is not None and len(batch.face_bboxes) == 1


# ---------------------------------------------------------------------------
# preflight_face_id
# ---------------------------------------------------------------------------

def test_preflight_inert_when_absent():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_face_id
    assert preflight_face_id(None, [], None, False) is None


def test_preflight_dataset_only_activation():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_face_id
    dc = SimpleNamespace(identity_loss_weight=0.1,
                         identity_loss_min_t=None, identity_loss_max_t=None, identity_loss_min_cos=None)
    cfg = preflight_face_id(None, [dc], 'krea2', False)
    assert cfg is not None and cfg.identity_loss_weight == 0.0


def test_preflight_krea2_low_vram_rejected():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_face_id
    cfg = FaceIDConfig(identity_loss_weight=0.05)
    with pytest.raises(ValueError, match="low_vram"):
        preflight_face_id(cfg, [], 'krea2', True)


def test_preflight_low_vram_ok_non_krea():
    from extensions_built_in.sd_trainer.SDTrainer import preflight_face_id
    cfg = FaceIDConfig(identity_loss_weight=0.05)
    out = preflight_face_id(cfg, [], 'flux1', True)
    assert out is not None and out.identity_loss_weight == 0.05
