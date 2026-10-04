"""Independent behavioral checks for H3 cache identity and provenance."""
from types import SimpleNamespace

import torch

from toolkit.dataloader_mixins import (
    LatentCachingMixin,
    _cache_model_identity,
    _dto_extras_from_state_dict,
    _latent_cache_model_identity,
    content_fingerprint,
)
from toolkit.subject_mask import _mask_cache_identity


def test_content_fingerprint_rejects_same_size_same_mtime_replacement(tmp_path):
    source = tmp_path / "clip.bin"
    source.write_bytes(b"abcd")
    first = content_fingerprint(str(source))
    stat = source.stat()
    source.write_bytes(b"wxyz")
    # Make the usual stat-only signature indistinguishable.
    import os

    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert content_fingerprint(str(source)) != first
def test_text_encoder_override_is_part_of_cache_identity(tmp_path):
    first_path = tmp_path / "te-a.bin"
    second_path = tmp_path / "te-b.bin"
    first_path.write_bytes(b"a")
    second_path.write_bytes(b"b")
    first = SimpleNamespace(
        model_config=SimpleNamespace(
            model_kwargs={"text_encoder_path": str(first_path)},
            te_name_or_path=None,
        ),
        arch="minimax_h3",
    )
    second = SimpleNamespace(
        model_config=SimpleNamespace(
            model_kwargs={"text_encoder_path": str(second_path)},
            te_name_or_path=None,
        ),
        arch="minimax_h3",
    )
    assert _cache_model_identity(first) != _cache_model_identity(second)


def test_subject_mask_identity_contains_full_transform_and_source(tmp_path):
    source = tmp_path / "image.bin"
    source.write_bytes(b"source-a")
    item = SimpleNamespace(
        path=str(source),
        source_content_fingerprint=content_fingerprint(str(source)),
        flip_x=False,
        flip_y=False,
        scale_to_width=128,
        scale_to_height=96,
        crop_x=0,
        crop_y=0,
        crop_width=96,
        crop_height=96,
    )
    config = SimpleNamespace(
        cache_resolution=256,
        body_close_radius=2,
        mask_dilate_radius=0,
        skin_bias=0.0,
        primary_only=True,
        sam_size="small",
        segformer_res=768,
    )
    first = _mask_cache_identity(item, config, 96, 96)
    item.crop_x = 8
    second = _mask_cache_identity(item, config, 96, 96)
    item.crop_x = 0
    item.flip_x = True
    third = _mask_cache_identity(item, config, 96, 96)
    assert len({first, second, third}) == 3

def test_caption_dropout_key_keeps_visual_control_identity(tmp_path):
    from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin

    control = tmp_path / "control.png"
    control.write_bytes(b"control")
    dataset_config = SimpleNamespace(
        do_i2v=False,
        fps=24,
        num_frames=1,
        auto_frame_count=False,
        trim_auto_frame_count_tail=False,
    )
    item = SimpleNamespace(
        caption="a person",
        text_embedding_space_version="h3",
        text_embedding_version=2,
        _cache_model_identity="encoder",
        encode_control_in_text_embeddings=True,
        control_path=str(control),
        control_video_paths=[],
        text_embedding_uses_target_size=False,
        dataset_config=dataset_config,
        is_video=False,
        source_content_fingerprint="source",
    )
    item.load_caption = lambda *args, **kwargs: None
    normal = TextEmbeddingFileItemDTOMixin.get_text_embedding_info_dict(item)
    dropout = TextEmbeddingFileItemDTOMixin.get_text_embedding_info_dict(
        item, caption_override="", text_only=True
    )
    assert normal["control_paths"] == dropout["control_paths"]

def test_cached_audio_presence_is_scalar_extra():
    extras = _dto_extras_from_state_dict(
        {
            "latent": torch.zeros(2),
            "audio_present": torch.tensor([1.0]),
        }
    )
    assert extras["audio_present"].shape == (1,)
    assert extras["audio_present"].item() == 1.0


def test_latent_identity_includes_resolved_vae_source(tmp_path):
    first_path = tmp_path / "vae-a.safetensors"
    second_path = tmp_path / "vae-b.safetensors"
    first_path.write_bytes(b"vae-a")
    second_path.write_bytes(b"vae-b")
    first = SimpleNamespace(
        model_config=SimpleNamespace(
            name_or_path="transformer",
            vae_path=str(first_path),
            model_kwargs={},
        ),
        arch="minimax_h3",
        latent_space_version="h3",
    )
    second = SimpleNamespace(
        model_config=SimpleNamespace(
            name_or_path="transformer",
            vae_path=str(second_path),
            model_kwargs={},
        ),
        arch="minimax_h3",
        latent_space_version="h3",
    )
    assert _latent_cache_model_identity(first) != _latent_cache_model_identity(second)


def test_disk_latent_cache_round_trip_writes_metadata_and_num_frames(tmp_path):
    from safetensors import safe_open
    from PIL import Image

    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    source = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (64, 96, 128)).save(source)
    config = DatasetConfig(
        dataset_path=str(tmp_path),
        resolution=32,
        cache_latents_to_disk=True,
    )
    item = FileItemDTO(path=str(source), dataset_config=config)
    item.tensor = torch.ones(3, 32, 32)
    item.num_frames = 1
    calls = {"n": 0}

    class _Model:
        torch_dtype = torch.float32
        device_torch = torch.device("cpu")
        cache_latents_as_uint8 = False

        @staticmethod
        def encode_images(images):
            calls["n"] += 1
            return images * 2

    dataset = SimpleNamespace(sd=_Model(), dataset_config=config)
    latent_path = item.get_latent_path(recalculate=True)
    LatentCachingMixin._cache_one_latent(
        dataset,
        item,
        latent_path,
        None,
        True,
        True,
        False,
    )
    assert calls["n"] == 1
    assert item._encoded_latent is None
    assert latent_path and latent_path.endswith(".safetensors")

    with safe_open(latent_path, framework="pt", device="cpu") as handle:
        assert "latent" in handle.keys()
        assert "num_frames" in handle.keys()
        metadata = handle.metadata()
    assert metadata["filename"] == "image.png"

    item.is_latent_cached = True
    loaded = item.get_latent()
    assert loaded is not None
    assert item.num_frames == 1
    assert torch.equal(loaded.tensor, torch.full((3, 32, 32), 2.0))


def test_first_frame_cache_uses_condition_encoder_hook(tmp_path):
    from PIL import Image

    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    source = tmp_path / "video.png"
    Image.new("RGB", (8, 8), (64, 96, 128)).save(source)
    config = DatasetConfig(
        dataset_path=str(tmp_path),
        resolution=8,
        num_frames=3,
        do_i2v=True,
    )
    item = FileItemDTO(path=str(source), dataset_config=config)
    item.is_video = True
    item.num_frames = 3
    item.tensor = torch.ones(3, 3, 8, 8)
    calls = {"target": 0, "condition": 0}

    class _Model:
        torch_dtype = torch.float32
        device_torch = torch.device("cpu")
        cache_latents_as_uint8 = False

        @staticmethod
        def encode_images(images):
            calls["target"] += 1
            return torch.ones(1, 2, 3, 2, 2)

        @staticmethod
        def encode_condition_images(images):
            calls["condition"] += 1
            return torch.full((1, 2, 1, 2, 2), 7.0)

    item.is_caching_to_memory = True
    dataset = SimpleNamespace(sd=_Model(), dataset_config=config)
    LatentCachingMixin._cache_one_latent(
        dataset,
        item,
        str(tmp_path / "unused.safetensors"),
        None,
        True,
        False,
        True,
    )
    assert calls == {"target": 1, "condition": 1}
    assert torch.equal(item._cached_first_frame_latent, torch.full((2, 1, 2, 2), 7.0))


def test_ref_video_fingerprint_is_memoized_until_explicit_rekey(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from extensions_built_in.diffusion_models.minimax_h3.src import ref_video_cache
    from extensions_built_in.diffusion_models.minimax_h3.src import text_encoder

    source = str(tmp_path / "reference.mp4")
    fingerprint_calls = {"n": 0}
    video_opens = {"n": 0}

    class _Capture:
        def __init__(self, _path):
            video_opens["n"] += 1

        def isOpened(self):
            return True

        def get(self, property_id):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 8,
                cv2.CAP_PROP_FRAME_HEIGHT: 8,
                cv2.CAP_PROP_FRAME_COUNT: 5,
                cv2.CAP_PROP_FPS: 24,
            }[property_id]

        def release(self):
            pass

    def _fingerprint(_path):
        fingerprint_calls["n"] += 1
        return f"sha256:{fingerprint_calls['n']}"

    monkeypatch.setattr(ref_video_cache.cv2, "VideoCapture", _Capture)
    monkeypatch.setattr(ref_video_cache, "_content_fingerprint", _fingerprint)
    monkeypatch.setattr(text_encoder, "video_has_audio", lambda _path: False)
    monkeypatch.setattr(
        ref_video_cache,
        "read_frames_at",
        lambda _cap, indices: [
            np.zeros((8, 8, 3), dtype=np.uint8) for _ in indices
        ],
    )
    monkeypatch.setattr(
        ref_video_cache,
        "_cache_path",
        lambda _path, _recipe: str(tmp_path / "reference.safetensors"),
    )

    config = SimpleNamespace(
        auto_frame_count=False,
        num_frames=5,
        trim_auto_frame_count_tail=False,
        fps=24,
    )

    class _Model:
        sample_rate = 32000

        @staticmethod
        def encode_condition_images(pixels):
            return torch.zeros(1, 2, pixels.shape[2], 1, 1)

        @staticmethod
        def get_frame_count_snapper():
            return None

    model = _Model()
    first = ref_video_cache.load_ref_video_latent(model, source, config, 8, 8)
    second = ref_video_cache.load_ref_video_latent(model, source, config, 8, 8)
    assert first["num_frames"] == second["num_frames"] == 5
    assert fingerprint_calls["n"] == 1
    # One preparation probe and one decode probe; the memory hit performs no
    # additional source open/probe.
    assert video_opens["n"] == 2

    ref_video_cache.rekey_ref_video_source(model, source)
    assert fingerprint_calls["n"] == 2
