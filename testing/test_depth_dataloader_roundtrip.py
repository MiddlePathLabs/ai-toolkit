"""End-to-end round-trip test for the depth-caching dataloader path.

This is the load-bearing test for the whole depth feature (guide sec.4.3 item 4):
without it, a cache file can exist on disk while every training batch still
receives ``depth_gt_list=None`` because nothing wires the file-item's lazy read
into the batch DTO.

It exercises the real ``FileItemDTO`` / ``DataLoaderBatchDTO`` against a fake
safetensors depth-cache whose key is produced by the real
``build_depth_cache_fingerprint`` helper (so the key is realistic), drives
``load_and_process_image`` through BOTH the ordinary image path and the
latent-cached early-return path, and verifies:

  * the cached depth tensor is delivered into ``batch.depth_gt_list`` with the
    right values and shape,
  * the per-dataset depth / loss-split scalars are collected into the batch,
  * ``cleanup()`` releases the resident tensor while leaving the cache path/key
    metadata intact,
  * a corrupt, mismatched, or non-finite cache file is rejected rather than
    silently feeding garbage through training.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.dataloader_mixins import DepthCachingFileItemDTOMixin
from toolkit.depth_consistency import build_depth_cache_fingerprint, depth_cache_key

ARCH = "krea2"
VAE_ID = "AutoencoderKLQwenImage::krea-vae"


def _perceptor_cfg():
    return SimpleNamespace(
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        input_size=518,
        pixel_blur_sigma=0.0,
    )


def _make_image(path):
    # 64x64 solid RGB image; buckets + scale=1.0 keep the transform deterministic
    Image.new("RGB", (64, 64), (128, 128, 128)).save(str(path))
    return str(path)


def _make_file_item(image_path, dataset_config):
    """Construct a real FileItemDTO over a temp image."""
    return FileItemDTO(path=image_path, dataset_config=dataset_config)


def _stamp_depth_cache(file_item, cache_path, *, tensor=None, key=None,
                       garbage=False, wrong_key=False, nan_data=False):
    """Write a fake depth-GT safetensors under the key the file_item expects.

    Returns the (path, key, tensor) triple that was written.
    """
    fingerprint = build_depth_cache_fingerprint(
        file_item, _perceptor_cfg(), arch=ARCH, vae_id=VAE_ID
    )
    expected_key = key if key is not None else depth_cache_key(fingerprint)

    if garbage:
        # not a valid safetensors file at all -> header unreadable
        with open(cache_path, "wb") as f:
            f.write(b"this is definitely not a safetensors file")
        written_tensor = None
    else:
        if nan_data:
            written_tensor = torch.full((32, 32), float("nan"))
        elif tensor is not None:
            written_tensor = tensor
        else:
            written_tensor = torch.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
        write_key = "some_other_key" if wrong_key else expected_key
        save_file({write_key: written_tensor.contiguous()}, str(cache_path))
        # sanity: the file we wrote is a real safetensors
        with safe_open(str(cache_path), framework="pt", device="cpu") as f:
            assert write_key in f.keys()

    file_item._depth_cache_path = str(cache_path)
    file_item._depth_cache_key = expected_key
    file_item.is_depth_cached = True
    return str(cache_path), expected_key, written_tensor


def _to_tensor_transform():
    return transforms.Compose([transforms.ToTensor()])


# ---------------------------------------------------------------------------
# Delivery: depth GT flows through both load paths into the batch DTO
# ---------------------------------------------------------------------------

def test_ordinary_path_delivers_depth_gt(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    cache_path = tmp_path / "depth.safetensors"
    expected = torch.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
    _stamp_depth_cache(fi, cache_path, tensor=expected)

    # ordinary (non-latent) image load path
    assert fi.depth_gt is None  # not yet materialized
    fi.load_and_process_image(_to_tensor_transform())
    assert torch.equal(fi.depth_gt, expected)

    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.depth_gt_list is not None
    assert len(batch.depth_gt_list) == 1
    assert torch.equal(batch.depth_gt_list[0], expected)
    # right shape too
    assert tuple(batch.depth_gt_list[0].shape) == (32, 32)


def test_latent_cached_path_delivers_depth_gt(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    cache_path = tmp_path / "depth.safetensors"
    expected = torch.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
    _stamp_depth_cache(fi, cache_path, tensor=expected)

    # put the item into the latent-cached path: pre-populate the latent so
    # get_latent() does not touch disk, and take the early return in
    # load_and_process_image. Depth GT must still be produced above that return.
    fi.is_latent_cached = True
    fi.is_caching_to_memory = True
    fi._encoded_latent = torch.zeros(1, 4, 8, 8)

    fi.load_and_process_image(_to_tensor_transform())
    assert torch.equal(fi.depth_gt, expected)

    batch = DataLoaderBatchDTO(file_items=[fi])
    assert batch.depth_gt_list is not None
    assert torch.equal(batch.depth_gt_list[0], expected)
    # latent path populated latents, not the raw image tensor
    assert batch.latents is not None
    assert batch.latents.shape[0] == 1


def test_depth_gt_values_are_exact(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    cache_path = tmp_path / "depth.safetensors"
    expected = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    _stamp_depth_cache(fi, cache_path, tensor=expected)

    fi.load_and_process_image(_to_tensor_transform())
    assert torch.equal(fi.depth_gt, expected)
    assert fi.depth_gt.dtype == torch.float32


# ---------------------------------------------------------------------------
# Per-dataset depth / loss-split scalars are collected into the batch
# ---------------------------------------------------------------------------

def test_batch_collects_depth_loss_fields(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(
        dataset_path=str(tmp_path),
        resolution=64,
        depth_loss_weight=0.25,
        depth_loss_min_t=0.1,
        depth_loss_max_t=0.8,
        loss_split="diffusion_depth",
    )
    fi = _make_file_item(image_path, cfg)
    cache_path = tmp_path / "depth.safetensors"
    _stamp_depth_cache(fi, cache_path)

    fi.load_and_process_image(_to_tensor_transform())
    batch = DataLoaderBatchDTO(file_items=[fi])

    assert batch.depth_loss_weight_list == [0.25]
    assert batch.depth_loss_min_t_list == [0.1]
    assert batch.depth_loss_max_t_list == [0.8]
    assert batch.loss_split_list == ["diffusion_depth"]


# ---------------------------------------------------------------------------
# Cleanup: resident tensor released, cache metadata retained
# ---------------------------------------------------------------------------

def test_cleanup_releases_tensor_keeps_metadata(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    cache_path = tmp_path / "depth.safetensors"
    _stamp_depth_cache(fi, cache_path)

    fi.load_and_process_image(_to_tensor_transform())
    assert fi.depth_gt is not None

    batch = DataLoaderBatchDTO(file_items=[fi])
    # batch.cleanup() must propagate to cleanup_depth() on each file item
    batch.cleanup()

    assert fi.depth_gt is None  # resident tensor released
    # cache metadata survives so the next epoch re-reads from the same file/key
    assert fi._depth_cache_path == str(cache_path)
    assert fi._depth_cache_key is not None
    assert fi.is_depth_cached is True


def test_cleanup_depth_is_idempotent_when_not_cached(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    # never marked depth-cached -> cleanup_depth must be a no-op (no AttributeError)
    fi.cleanup_depth()
    assert fi.depth_gt is None


# ---------------------------------------------------------------------------
# Non-cached item contributes no depth_gt_list
# ---------------------------------------------------------------------------

def test_non_cached_item_has_no_depth_in_batch(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    # is_depth_cached stays False (default)
    fi.load_and_process_image(_to_tensor_transform())
    batch = DataLoaderBatchDTO(file_items=[fi])
    assert getattr(batch, "depth_gt_list", None) is None
    assert fi.depth_gt is None


# ---------------------------------------------------------------------------
# Corrupt / mismatched cache must be rejected, never silently loaded
# ---------------------------------------------------------------------------

def test_corrupt_cache_file_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    _stamp_depth_cache(fi, tmp_path / "depth.safetensors", garbage=True)

    fi.load_and_process_image(_to_tensor_transform())
    assert fi.depth_gt is None  # corrupt header -> not loaded as garbage


def test_wrong_cache_key_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    _stamp_depth_cache(fi, tmp_path / "depth.safetensors", wrong_key=True)

    fi.load_and_process_image(_to_tensor_transform())
    assert fi.depth_gt is None  # expected key absent -> treated as a miss


def test_non_finite_cache_rejected(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    _stamp_depth_cache(fi, tmp_path / "depth.safetensors", nan_data=True)

    fi.load_and_process_image(_to_tensor_transform())
    assert fi.depth_gt is None  # NaN tensor must not feed training silently


# ---------------------------------------------------------------------------
# Mixin is wired into FileItemDTO at the right place
# ---------------------------------------------------------------------------

def test_depth_mixin_is_in_file_item_bases():
    bases = FileItemDTO.__mro__
    assert DepthCachingFileItemDTOMixin in bases
    # contract: immediately after LatentCachingFileItemDTOMixin in the MRO
    from toolkit.dataloader_mixins import LatentCachingFileItemDTOMixin
    assert bases.index(DepthCachingFileItemDTOMixin) == bases.index(LatentCachingFileItemDTOMixin) + 1


def test_file_item_has_depth_attributes_by_default(tmp_path):
    image_path = _make_image(tmp_path / "img.png")
    cfg = DatasetConfig(dataset_path=str(tmp_path), resolution=64)
    fi = _make_file_item(image_path, cfg)
    assert fi.depth_gt is None
    assert fi.is_depth_cached is False
    assert fi._depth_cache_path is None
    assert fi._depth_cache_key is None
