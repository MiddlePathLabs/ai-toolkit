"""Independent behavioral checks for H3 cache identity and provenance."""
import json
import os
from types import SimpleNamespace

import pytest
import torch


def _toy_prompt_tensor(caption, control_rgb=None):
    raw = (caption or "").encode("utf-8")
    caption_features = torch.tensor(
        [
            float(len(raw)),
            float(sum(raw)),
            float(raw[0] if raw else 0),
            float(raw[-1] if raw else 0),
        ],
        dtype=torch.float32,
    )
    if control_rgb is None:
        control_rgb = torch.zeros(3, dtype=torch.float32)
    return torch.cat((caption_features, control_rgb.to(torch.float32))).reshape(1, 1, -1)


def _toy_prompt_embeds(caption, control_images=None):
    from toolkit.prompt_utils import PromptEmbeds

    if isinstance(control_images, (list, tuple)):
        control_images = control_images[0] if control_images else None
    if torch.is_tensor(control_images):
        if control_images.ndim == 3:
            control_images = control_images.unsqueeze(0)
        control_rgb = control_images.detach().to(torch.float32).mean(dim=(0, 2, 3)).cpu()
    else:
        control_rgb = None
    return PromptEmbeds(_toy_prompt_tensor(caption, control_rgb))


class _FrozenToyEncoderModel:
    load_rgba = False
    use_raw_control_images = False
    arch = "minimax_h3"
    model_config = SimpleNamespace(model_kwargs={}, te_name_or_path=None)
    device = torch.device("cpu")
    device_torch = torch.device("cpu")
    torch_dtype = torch.float32
    has_multiple_control_images = False
    encode_first_frame_in_text_embeddings = False
    is_audio_model = False
    is_multimodal_llm = False
    is_xl = False
    is_vega = False
    is_ssd = False
    vae = None
    unet = None
    te_padding_side = "right"
    sample_rate = 48000

    def __init__(self, encode_control=False):
        self.encode_control_in_text_embeddings = encode_control
        self.encoder = torch.nn.Linear(1, 1)
        self.encoder.train()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.device_state = None
    def get_bucket_divisibility(self):
        return 1

    def get_latent_space_version(self):
        return "toy-latent"

    def get_text_embedding_space_version(self):
        return "toy-text"


    def _state(self):
        return {
            "training": self.encoder.training,
            "requires_grad": tuple(
                parameter.requires_grad for parameter in self.encoder.parameters()
            ),
            "device": tuple(parameter.device for parameter in self.encoder.parameters()),
        }

    def save_device_state(self):
        self.device_state = self._state()

    def restore_device_state(self):
        if self.device_state is None:
            return
        state = self.device_state
        self.encoder.train(state["training"])
        for parameter, requires_grad, device in zip(
            self.encoder.parameters(), state["requires_grad"], state["device"]
        ):
            parameter.requires_grad_(requires_grad)
            parameter.data = parameter.data.to(device)
        self.device_state = None

    def set_device_state_preset(self, _name):
        self.save_device_state()
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def encode_prompt(self, caption, control_images=None, target_size=None):
        del target_size
        return _toy_prompt_embeds(
            caption,
            control_images if self.encode_control_in_text_embeddings else None,
        )

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


def test_latent_identity_memoizes_duplicate_checkpoint_paths(tmp_path, monkeypatch):
    import toolkit.dataloader_mixins as mixins

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            name_or_path=str(checkpoint),
            vae_path=str(checkpoint),
            model_kwargs={
                "video_vae_path": str(checkpoint),
                "audio_vae_path": str(checkpoint),
            },
        ),
        arch="minimax_h3",
    )
    calls = {"count": 0}
    original = mixins.content_fingerprint

    def fingerprint(path):
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(mixins, "content_fingerprint", fingerprint)
    first = _latent_cache_model_identity(model)
    second = _latent_cache_model_identity(model)
    assert first == second
    assert calls["count"] == 1


def test_file_item_skips_latent_checkpoint_identity_without_latent_cache(tmp_path, monkeypatch):
    import toolkit.dataloader_mixins as mixins
    from PIL import Image
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(source)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    config = DatasetConfig(dataset_path=str(tmp_path), resolution=8, buckets=False)
    model = SimpleNamespace(
        load_rgba=False,
        use_raw_control_images=False,
        model_config=SimpleNamespace(arch="sd1", model_kwargs={"name": str(checkpoint)}),
    )

    def unexpected_identity(_model):
        raise AssertionError("latent checkpoint identity was prepared without a cache")

    monkeypatch.setattr(mixins, "_latent_cache_model_identity", unexpected_identity)
    item = FileItemDTO(path=str(source), dataset_config=config, sd=model)
    assert item._latent_model_identity == "latent-cache-disabled"


@pytest.mark.parametrize("source_size", [(8, 4), (20, 14), (14, 20)])
def test_subject_mask_transform_matches_nonbucket_training_geometry(tmp_path, source_size):
    from PIL import Image
    from torchvision.transforms import functional as TF
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.subject_mask import _apply_dataloader_transform

    source = tmp_path / "image.png"
    image = Image.new("RGB", source_size)
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), (x * 11, (y * 17) % 256, 127))
    image.save(source)
    config = DatasetConfig(
        dataset_path=str(tmp_path),
        buckets=False,
        scale=0.5,
        resolution=4,
        random_crop=False,
    )
    item = FileItemDTO(
        path=str(source),
        dataset_config=config,
        sd=_FrozenToyEncoderModel(),
    )
    item.load_and_process_image(TF.to_tensor)
    transformed = _apply_dataloader_transform(image, item)

    assert tuple(item.tensor.shape[-2:]) == (transformed.height, transformed.width)
    assert item.tensor.shape[-2:] == torch.Size((4, 4))
    torch.testing.assert_close(item.tensor, TF.to_tensor(transformed), rtol=0, atol=0)

def test_prompt_embed_cache_keeps_previous_entry_on_publication_failure(tmp_path, monkeypatch):
    import toolkit.prompt_utils as prompt_utils

    cache_path = tmp_path / "prompt.safetensors"
    previous = torch.tensor([[1.0, 2.0, 3.0]])
    prompt_utils.PromptEmbeds(previous).save(str(cache_path))

    def fail_publication(_source, _destination):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(prompt_utils.os, "replace", fail_publication)
    with pytest.raises(OSError, match="synthetic publication failure"):
        prompt_utils.PromptEmbeds(torch.tensor([[4.0, 5.0, 6.0]])).save(str(cache_path))

    loaded = prompt_utils.PromptEmbeds.load(str(cache_path))
    torch.testing.assert_close(loaded.text_embeds, previous)
    assert set(tmp_path.iterdir()) == {cache_path}




def test_text_cache_consumer_preserves_caption_authority_and_dropout_control(
    tmp_path, monkeypatch
):
    import toolkit.dataloader_mixins as mixins
    from PIL import Image
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    source = tmp_path / "image.png"
    control = tmp_path / "control.png"
    Image.new("RGB", (8, 8), color="red").save(source)
    Image.new("RGB", (8, 8), color="blue").save(control)
    source.with_suffix(".txt").write_text("sidecar caption", encoding="utf-8")
    config = DatasetConfig(
        dataset_path=str(tmp_path),
        resolution=8,
        buckets=False,
        cache_text_embeddings=True,
        caption_dropout_rate=1.0,
    )

    model = _FrozenToyEncoderModel(encode_control=True)
    item = FileItemDTO(
        path=str(source),
        dataset_config=config,
        sd=model,
        encode_control_in_text_embeddings=True,
    )
    item.control_path = str(control)
    dataset = SimpleNamespace(
        sd=model,
        dataset_path=str(tmp_path),
        dataset_config=config,
        file_list=[item],
        caption_dict={str(source): {"caption": "JSON authority"}},
    )
    mixins.TextEmbeddingCachingMixin.cache_text_embeddings(dataset)
    model.restore_device_state()

    assert item.caption == "JSON authority"

    expected_control = torch.tensor([0.0, 0.0, 1.0])
    monkeypatch.setattr(mixins.random, "random", lambda: 1.0)
    item.load_prompt_embedding()
    assert torch.equal(
        item.prompt_embeds.text_embeds,
        _toy_prompt_tensor("JSON authority", expected_control),
    )
    assert not torch.equal(
        item.prompt_embeds.text_embeds,
        _toy_prompt_tensor("sidecar caption", expected_control),
    )

    item.prompt_embeds = None
    monkeypatch.setattr(mixins.random, "random", lambda: 0.0)
    item.load_prompt_embedding()
    assert torch.equal(
        item.prompt_embeds.text_embeds,
        _toy_prompt_tensor("", expected_control),
    )


def test_midrun_caption_rekey_restores_encoder_state_on_success_noop_and_failure(
    tmp_path, monkeypatch
):
    from PIL import Image
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_loader import AiToolkitDataset

    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(source)
    caption_source = tmp_path / "captions.json"
    caption_source.write_text(json.dumps({str(source): "before"}), encoding="utf-8")
    config = DatasetConfig(
        dataset_path=str(caption_source),
        resolution=8,
        buckets=False,
        cache_text_embeddings=True,
    )
    model = _FrozenToyEncoderModel()
    dataset = AiToolkitDataset(config, batch_size=1, sd=model)
    model.restore_device_state()
    expected_state = model._state()
    caption_source.write_text(json.dumps({str(source): "after"}), encoding="utf-8")

    result = dataset[0]
    assert result.path == str(source)
    assert result.caption == "after"
    torch.testing.assert_close(
        result.prompt_embeds.text_embeds, _toy_prompt_tensor("after"),
        rtol=0, atol=0,
    )
    assert model._state() == expected_state
    assert model.device_state is None
    dataset._regenerate_text_embeddings([result])
    assert model._state() == expected_state
    assert model.device_state is None

    def fail_cache(**_kwargs):
        model.set_device_state_preset("cache_text_encoder")
        raise RuntimeError("synthetic cache failure")

    monkeypatch.setattr(dataset, "cache_text_embeddings", fail_cache)
    with pytest.raises(RuntimeError, match="synthetic cache failure"):
        dataset._regenerate_text_embeddings([result])
    assert model._state() == expected_state
    assert model.device_state is None




def test_worker_dto_reloads_authoritative_caption_source(tmp_path):
    from PIL import Image
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    source = tmp_path / "image.png"
    caption_source = tmp_path / "captions.json"
    Image.new("RGB", (8, 8)).save(source)
    caption_source.write_text(
        json.dumps({str(source): "disk authority"}), encoding="utf-8"
    )
    config = DatasetConfig(dataset_path=str(tmp_path), resolution=8, buckets=False)
    item = FileItemDTO(
        path=str(source),
        dataset_config=config,
        caption_source_path=str(caption_source),
    )
    item.load_caption({str(source): "stale worker copy"}, force=True)
    assert item.caption == "disk authority"
    original_stat = caption_source.stat()
    caption_source.write_text(
        json.dumps({str(source): "same authority"}), encoding="utf-8"
    )
    os.utime(
        caption_source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    item.load_caption({str(source): "stale worker copy"}, force=True)
    assert item.caption == "same authority"
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("use_short_captions", [False, True])
def test_dataset_caption_authority_and_edits_with_cache_on_and_off(
    tmp_path, cache_enabled, use_short_captions
):
    from PIL import Image
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_loader import AiToolkitDataset

    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(source)
    source.with_suffix(".txt").write_text("wrong sidecar", encoding="utf-8")
    caption_source = tmp_path / "captions.json"
    original = json.dumps({
        str(source): {"caption": "first caption", "caption_short": "first short"}
    })
    caption_source.write_text(original, encoding="utf-8")
    config = DatasetConfig(
        dataset_path=str(caption_source),
        resolution=8,
        buckets=False,
        cache_text_embeddings=cache_enabled,
        use_short_captions=use_short_captions,
        default_caption="wrong default",
    )
    model = _FrozenToyEncoderModel()
    dataset = AiToolkitDataset(config, batch_size=1, sd=model)
    model.restore_device_state()
    expected_state = model._state()
    item = dataset[0]
    expected_caption = "first short" if use_short_captions else "first caption"
    assert item.path == str(source)
    assert item.caption == expected_caption
    assert item.tensor.shape == (3, 8, 8)
    if cache_enabled:
        torch.testing.assert_close(
            item.prompt_embeds.text_embeds, _toy_prompt_tensor(expected_caption),
            rtol=0, atol=0,
        )
    assert caption_source.read_text(encoding="utf-8") == original

    original_stat = caption_source.stat()
    edited = json.dumps({
        str(source): {"caption": "other caption", "caption_short": "other short"}
    })
    caption_source.write_text(edited, encoding="utf-8")
    os.utime(
        caption_source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    refreshed = dataset[0]
    expected_caption = "other short" if use_short_captions else "other caption"
    assert refreshed.path == str(source)
    assert refreshed.caption == expected_caption
    if cache_enabled:
        torch.testing.assert_close(
            refreshed.prompt_embeds.text_embeds, _toy_prompt_tensor(expected_caption),
            rtol=0, atol=0,
        )
    assert caption_source.read_text(encoding="utf-8") == edited
    assert model._state() == expected_state
    assert model.device_state is None



def test_pending_worker_embedding_defers_dto_and_parent_rebuilds(
    tmp_path, monkeypatch
):
    import copy
    import toolkit.dataloader_mixins as mixins
    from PIL import Image
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import (
        DataLoaderBatchDTO,
        FileItemDTO,
    )

    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(source)
    config = DatasetConfig(
        dataset_path=str(tmp_path), resolution=8, buckets=False,
        cache_text_embeddings=True,
    )

    model = _FrozenToyEncoderModel()
    item = FileItemDTO(
        path=str(source),
        dataset_config=config,
        sd=model,
        cache_owner_token="owner-b",
    )
    dataset = SimpleNamespace(
        sd=model,
        dataset_path=str(tmp_path),
        dataset_config=config,
        cache_owner_token="owner-b",
        file_list=[item],
        caption_dict={str(source): "before"},
        caption_source_path=None,
    )
    mixins.TextEmbeddingCachingMixin.cache_text_embeddings(dataset)
    model.restore_device_state()
    expected_state = model._state()

    worker_item = copy.deepcopy(item)
    assert worker_item.cache_owner_token == "owner-b"
    worker_item.load_caption({str(source): "after"}, force=True)
    worker_item.tensor = torch.zeros(3, 8, 8)
    worker_item.is_text_embedding_cached = True
    worker_item.load_prompt_embedding()
    assert worker_item.pending_text_embedding is True
    batch = DataLoaderBatchDTO(file_items=[worker_item])
    assert batch.pending_text_embedding_items == [worker_item]
    assert batch.prompt_embeds is None
    wrong_config = DatasetConfig(
        dataset_path=str(tmp_path), resolution=8, buckets=False,
        cache_text_embeddings=True, caption_dropout_rate=1.0,
    )
    wrong_item = FileItemDTO(
        path=str(source),
        dataset_config=wrong_config,
        sd=model,
        cache_owner_token="owner-a",
    )
    wrong_dataset = SimpleNamespace(
        sd=model,
        dataset_path=str(tmp_path),
        dataset_config=wrong_config,
        cache_owner_token="owner-a",
        file_list=[wrong_item],
        caption_dict={str(source): "wrong-owner-caption"},
    )
    wrong_dataset.repair_pending_text_embeddings = (
        AiToolkitDataset.repair_pending_text_embeddings.__get__(wrong_dataset)
    )
    wrong_dataset._refresh_caption_source_for_workers = (
        AiToolkitDataset._refresh_caption_source_for_workers.__get__(wrong_dataset)
    )
    wrong_dataset.cache_text_embeddings = (
        mixins.TextEmbeddingCachingMixin.cache_text_embeddings.__get__(wrong_dataset)
    )
    wrong_dataset._regenerate_text_embeddings = (
        AiToolkitDataset._regenerate_text_embeddings.__get__(wrong_dataset)
    )
    wrong_dataset.repair_pending_text_embeddings(batch)
    assert worker_item.pending_text_embedding is True
    assert model._state() == expected_state
    assert model.device_state is None


    dataset.caption_dict = {str(source): "after"}
    dataset._refresh_caption_source_for_workers = (
        AiToolkitDataset._refresh_caption_source_for_workers.__get__(dataset)
    )
    dataset.repair_pending_text_embeddings = (
        AiToolkitDataset.repair_pending_text_embeddings.__get__(dataset)
    )
    dataset.cache_text_embeddings = (
        mixins.TextEmbeddingCachingMixin.cache_text_embeddings.__get__(dataset)
    )
    dataset._regenerate_text_embeddings = (
        AiToolkitDataset._regenerate_text_embeddings.__get__(dataset)
    )
    def unexpected_collective():
        raise AssertionError("incremental cache repair entered a collective")

    monkeypatch.setattr(
        mixins.accelerator,
        "main_process_first",
        unexpected_collective,
    )
    dataset.repair_pending_text_embeddings(batch)
    batch.rebuild_prompt_embeddings()
    assert worker_item.pending_text_embedding is False
    assert torch.equal(
        batch.prompt_embeds.text_embeds,
        _toy_prompt_tensor("after"),
    )
    assert model._state() == expected_state
    assert model.device_state is None


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


@pytest.mark.parametrize("num_frames, expected_indices", [
    (3, (0, 2, 4)),
    (5, (0, 1, 2, 3, 4)),
])
def test_ref_video_cache_complete_identity_persists_across_call_orders(
    tmp_path, monkeypatch, num_frames, expected_indices
):
    import cv2
    import numpy as np
    from extensions_built_in.diffusion_models.minimax_h3.src import ref_video_cache
    from extensions_built_in.diffusion_models.minimax_h3.src import text_encoder

    source = str(tmp_path / "reference.mp4")
    encode_calls = {"count": 0}

    class _Capture:
        def __init__(self, _path):
            pass

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

    monkeypatch.setattr(ref_video_cache.cv2, "VideoCapture", _Capture)
    monkeypatch.setattr(ref_video_cache, "_content_fingerprint", lambda _path: "sha256:source")
    monkeypatch.setattr(text_encoder, "video_has_audio", lambda _path: False)
    monkeypatch.setattr(
        ref_video_cache,
        "read_frames_at",
        lambda _cap, indices: [
            np.full((8, 8, 3), (10 + index, 80 + index, 200 + index), dtype=np.uint8)
            for index in indices
        ],
    )


    config = SimpleNamespace(
        auto_frame_count=False,
        num_frames=num_frames,
        trim_auto_frame_count_tail=False,
        fps=24,
    )

    class _Model:
        sample_rate = 32000

        def __init__(self):
            self.encode_calls = 0

        def encode_condition_images(self, pixels):
            self.encode_calls += 1
            encode_calls["count"] += 1
            return torch.nn.functional.avg_pool3d(pixels, (1, 4, 4))

        @staticmethod
        def get_frame_count_snapper():
            return None

    first_model = _Model()
    square = ref_video_cache.load_ref_video_latent(first_model, source, config, 128, 128)
    square_memory = ref_video_cache.load_ref_video_latent(
        first_model, source, config, 128, 128
    )
    wide = ref_video_cache.load_ref_video_latent(first_model, source, config, 256, 256)
    expected_rgb = torch.tensor([
        [200 + index for index in expected_indices],
        [80 + index for index in expected_indices],
        [10 + index for index in expected_indices],
    ], dtype=torch.float32) / 255.0 * 2.0 - 1.0
    expected_rgb = expected_rgb.reshape(3, num_frames, 1, 1).to(torch.float16)
    torch.testing.assert_close(
        square["latent"], expected_rgb.expand(3, num_frames, 32, 32),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        wide["latent"], expected_rgb.expand(3, num_frames, 64, 64),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(square_memory["latent"], square["latent"], rtol=0, atol=0)
    for _ in range(10):
        for size, expected in [(256, wide), (128, square)]:
            hit = ref_video_cache.load_ref_video_latent(first_model, source, config, size, size)
            torch.testing.assert_close(hit["latent"], expected["latent"], rtol=0, atol=0)
    assert first_model.encode_calls == 2

    second_model = _Model()
    wide_disk = ref_video_cache.load_ref_video_latent(
        second_model, source, config, 256, 256
    )
    square_disk = ref_video_cache.load_ref_video_latent(
        second_model, source, config, 128, 128
    )
    torch.testing.assert_close(wide_disk["latent"], wide["latent"], rtol=0, atol=0)
    torch.testing.assert_close(square_disk["latent"], square["latent"], rtol=0, atol=0)
    assert second_model.encode_calls == 0
    assert encode_calls["count"] == 2
