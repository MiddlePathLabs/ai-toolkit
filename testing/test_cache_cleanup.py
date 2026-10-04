from pathlib import Path

import pytest

from toolkit.cache_cleanup import apply_cache_cleanup, preview_cache_cleanup


def _cache(root: Path, directory: str, name: str = "image_hash.safetensors") -> Path:
    folder = root / directory
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    target.write_bytes(b"regenerable cache")
    return target


def test_preview_and_apply_preserve_media_checkpoints_and_unselected_caches(tmp_path):
    latent = _cache(tmp_path / "nested", "_latent_cache")
    text = _cache(tmp_path, "_t_e_cache")
    face = _cache(tmp_path, "_face_id_cache")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    media = tmp_path / "image.png"
    media.write_bytes(b"source")
    unrelated = _cache(tmp_path, "_latent_cache", "notes.txt")
    plan = preview_cache_cleanup(tmp_path)
    assert {item["path"] for item in plan["files"]} == {
        "nested/_latent_cache/image_hash.safetensors", "_t_e_cache/image_hash.safetensors",
    }
    assert latent.exists() and text.exists()
    result = apply_cache_cleanup(tmp_path, plan["token"])
    assert set(result["removed"]) == {item["path"] for item in plan["files"]}
    assert not latent.exists() and not text.exists()
    assert checkpoint.read_bytes() == b"checkpoint"
    assert media.read_bytes() == b"source"
    assert face.read_bytes() == b"regenerable cache"
    assert unrelated.read_bytes() == b"regenerable cache"


@pytest.mark.parametrize("change", ["replace", "add", "remove", "selection"])
def test_changed_preview_refuses_deletion_before_touching_other_files(tmp_path, change):
    first = _cache(tmp_path, "_latent_cache", "first.safetensors")
    second = _cache(tmp_path, "_latent_cache", "second.safetensors")
    plan = preview_cache_cleanup(tmp_path)
    kinds = ("latent", "text")
    if change == "replace":
        second.write_bytes(b"new cache bytes")
    elif change == "add":
        _cache(tmp_path, "_t_e_cache")
    elif change == "remove":
        second.unlink()
    else:
        kinds = ("latent",)
    with pytest.raises(ValueError, match="preview"):
        apply_cache_cleanup(tmp_path, plan["token"], kinds)
    assert first.read_bytes() == b"regenerable cache"
    if change != "remove":
        assert second.exists()


def test_cache_symlinks_and_linked_dataset_roots_are_not_cleanup_targets(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    external = _cache(tmp_path / "outside", "_latent_cache")
    try:
        (root / "linked_folder").symlink_to(external.parent.parent, target_is_directory=True)
        (root / "_latent_cache").symlink_to(external.parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")
    assert preview_cache_cleanup(root)["files"] == []
    with pytest.raises(ValueError, match="symbolic link"):
        preview_cache_cleanup(root / "linked_folder")
    assert external.read_bytes() == b"regenerable cache"


def test_replacing_cache_directory_with_link_invalidates_preview(tmp_path):
    root = tmp_path / "dataset"
    target = _cache(root, "_latent_cache")
    plan = preview_cache_cleanup(root)
    external = _cache(tmp_path / "outside", "_latent_cache")
    target.unlink()
    target.parent.rmdir()
    try:
        target.parent.symlink_to(external.parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="preview"):
        apply_cache_cleanup(root, plan["token"])
    assert external.read_bytes() == b"regenerable cache"
