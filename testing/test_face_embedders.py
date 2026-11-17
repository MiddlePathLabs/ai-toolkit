"""AdaFace ViT embedder (scripts/face_embedders.py) — architecture and weight mapping.

The real weights (0.46 GB) are downloaded on first eval use, never by tests: these cases
build tiny AdaFaceViT variants so every check runs in milliseconds on CPU.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
_spec = importlib.util.spec_from_file_location(
    "face_embedders", os.path.join(_SCRIPTS, "face_embedders.py")
)
face_embedders = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(face_embedders)


def tiny_net():
    # patch 8 over 112 keeps the real token count; everything else shrinks for speed
    return face_embedders.AdaFaceViT(out_dim=16, dim=32, depth=2, heads=4)


def test_forward_shape():
    net = tiny_net()
    out = net(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 16)


def test_embed_aligned_unit_norm_and_mirror_invariance():
    net = tiny_net().eval()
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    e = face_embedders.embed_aligned(net, img)
    assert e.shape == (16,)
    assert e.dtype == np.float32
    assert pytest.approx(float(np.linalg.norm(e)), abs=1e-5) == 1.0
    # the embedding averages the image with its mirror, so a mirrored input embeds identically
    e_flip = face_embedders.embed_aligned(net, np.ascontiguousarray(img[:, ::-1]))
    assert np.allclose(e, e_flip, atol=1e-6)


def test_load_into_maps_model_net_prefix_and_tolerates_missing_batch_counts():
    net = tiny_net()
    sd = {f"model.net.{k}": v for k, v in net.state_dict().items()}
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    sd["unrelated.key"] = torch.zeros(1)
    loaded = face_embedders.load_into(tiny_net(), sd)
    for k, v in net.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_load_into_rejects_missing_weights():
    net = tiny_net()
    sd = {f"model.net.{k}": v for k, v in net.state_dict().items()
          if not k.endswith("num_batches_tracked")}
    dropped = sorted(sd)[0]
    del sd[dropped]
    with pytest.raises(RuntimeError, match="missing"):
        face_embedders.load_into(tiny_net(), sd)


def test_ensure_adaface_weights_roundtrip_and_sha_reject(tmp_path, monkeypatch):
    # fully offline: the "remote" is a local file:// URI whose digest becomes the pin
    good = tmp_path / "good.safetensors"
    save_file({"model.net.dummy": torch.zeros(1)}, str(good))
    digest = face_embedders._sha256(str(good))
    dest = str(tmp_path / "dest" / "weights.safetensors")
    monkeypatch.setitem(face_embedders.ADAFACE_WEIGHTS, "url", good.as_uri())
    monkeypatch.setitem(face_embedders.ADAFACE_WEIGHTS, "sha256", digest)
    assert face_embedders.ensure_adaface_weights(dest) == dest   # downloads, verifies, moves into place
    assert face_embedders.ensure_adaface_weights(dest) == dest   # second call: sha hit, no download
    monkeypatch.setitem(face_embedders.ADAFACE_WEIGHTS, "sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        face_embedders.ensure_adaface_weights(dest)              # bad digest: rejected, no .part left
    assert not os.path.exists(dest + ".part")


def test_ensure_adaface_weights_rejects_directory(tmp_path):
    with pytest.raises(RuntimeError, match="directory"):
        face_embedders.ensure_adaface_weights(str(tmp_path))
