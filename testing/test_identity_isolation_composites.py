"""Offline-only compositor contracts; all portraits here are synthetic color tiles."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from toolkit import identity_isolation_composites as composites


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_pin(folder, revision="synthetic-fixture-v1"):
    """A private test pin, deliberately unrelated to the production HF identity."""
    rows = json.loads((folder / "strangers.json").read_text(encoding="utf-8"))
    (folder / composites.REVISION_FILE).write_text(revision, encoding="utf-8")
    return composites._AssetPin(
        repo="synthetic-test-fixture",
        revision=revision,
        image_sha256={r["file"]: composites._sha256(folder / r["file"]) for r in rows},
        manifest_blob=composites._git_blob((folder / "strangers.json").read_bytes()),
        readme_blob=composites._git_blob((folder / "README.md").read_bytes()),
    )


@pytest.fixture
def recipe(tmp_path):
    source = tmp_path / "subject"
    assets = tmp_path / "strangers"
    source.mkdir()
    assets.mkdir()
    Image.new("RGB", (180, 100), "blue").save(source / "a.png")
    Image.new("RGB", (100, 180), "red").save(source / "z.png")
    (source / "a.txt").write_text("zxy, a woman smiling", encoding="utf-8")
    (source / "z.txt").write_text("zxy, a woman sitting", encoding="utf-8")
    rows = []
    for index, color in enumerate(("lime", "yellow", "magenta")):
        name = f"stranger_{index:02d}.png"
        Image.new("RGB", (120, 80), color).save(assets / name)
        rows.append({"file": name, "description": f"a synthetic stranger {index}"})
    _write_json(assets / "strangers.json", rows)
    (assets / "README.md").write_text(
        "Synthetic test tiles, not the pinned portrait dataset.", encoding="utf-8"
    )
    pin = _fixture_pin(assets)
    config = SimpleNamespace(
        composites=True,
        fused_captions=True,
        composite_count=16,
        seed=42,
        subjects=[
            {"folder_path": str(source), "trigger": "zxy", "class_name": "woman"}
        ],
        generated_dir=str(tmp_path / "generated"),
        stranger_asset_dir=str(assets),
    )
    return config, pin


def _prepare(recipe):
    config, pin = recipe
    return composites._prepare_composites(config, pin)


def _output_bytes(folder):
    return {
        p.name: p.read_bytes()
        for p in Path(folder).iterdir()
        if p.suffix in {".png", ".txt"}
    }


def test_exact_rng_dimensions_caption_order_and_placement(recipe):
    config, pin = recipe
    folder = Path(_prepare(recipe)[0]["folder_path"])
    rows = json.loads((Path(config.stranger_asset_dir) / "strangers.json").read_text())
    rng = random.Random(config.seed * 1000)
    order = list(range(len(rows)))
    rng.shuffle(order)
    for k in range(config.composite_count):
        stranger = rows[order[k % len(order)]]
        source_name = rng.choice(["a.png", "z.png"])
        left = rng.random() < 0.5
        me, them = ("left", "right") if left else ("right", "left")
        caption = (folder / f"fizgig_mp_{k:02d}.txt").read_text(encoding="utf-8")
        assert (
            caption
            == f"{stranger['description']} on the {them}, and zxywoman on the {me}, {composites.TAIL}"
        )
        with Image.open(folder / f"fizgig_mp_{k:02d}.png") as image:
            assert image.size == (1024, 512)
            assert image.mode == "RGB"
            with Image.open(
                Path(config.subjects[0]["folder_path"]) / source_name
            ) as source:
                assert image.getpixel((256 if left else 768, 256)) == source.getpixel(
                    (50, 50)
                )
            with Image.open(
                Path(config.stranger_asset_dir) / stranger["file"]
            ) as other:
                assert image.getpixel((768 if left else 256, 256)) == other.getpixel(
                    (50, 50)
                )
    manifest = json.loads((folder / composites.MANIFEST_FILE).read_text())
    assert manifest["inputs"]["assets"]["repo"] == "synthetic-test-fixture"
    assert manifest["inputs"]["assets"]["license"] == "synthetic-test-fixture"
    assert manifest["inputs"]["assets"]["revision"] == pin.revision


def test_center_square_crop_ignores_outer_borders(recipe):
    config, _ = recipe
    source = Path(config.subjects[0]["folder_path"])
    for name in ("a.png", "z.png"):
        with Image.new("RGB", (200, 100), "black") as image:
            image.paste("red", (50, 0, 150, 100))
            image.save(source / name)
    folder = Path(_prepare(recipe)[0]["folder_path"])
    for k in range(config.composite_count):
        caption = (folder / f"fizgig_mp_{k:02d}.txt").read_text()
        left = "zxywoman on the left" in caption
        with Image.open(folder / f"fizgig_mp_{k:02d}.png") as image:
            x = 0 if left else 512
            assert image.getpixel((x, 0)) == (255, 0, 0)
            assert image.getpixel((x + 511, 511)) == (255, 0, 0)


def test_source_images_and_caption_bytes_are_untouched(recipe):
    config, _ = recipe
    folder = Path(config.subjects[0]["folder_path"])
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    _prepare(recipe)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_no_fusion_uses_raw_trigger(recipe):
    config, _ = recipe
    config.fused_captions = False
    folder = Path(_prepare(recipe)[0]["folder_path"])
    caption = (folder / "fizgig_mp_00.txt").read_text()
    assert ", and zxy on the " in caption
    assert "zxywoman" not in caption


def test_subject_index_offsets_rng_and_returns_metadata(recipe):
    config, _ = recipe
    source = Path(config.subjects[0]["folder_path"])
    other = source.with_name("second")
    other.mkdir()
    Image.new("RGB", (100, 100), "cyan").save(other / "a.png")
    config.subjects.append(
        {"folder_path": str(other), "trigger": "abc", "class_name": "man"}
    )
    results = _prepare(recipe)
    assert [r["subject_index"] for r in results] == [0, 1]
    assert len({r["cache_identity"] for r in results}) == 2
    rng = random.Random(config.seed * 1000 + 1)
    order = list(range(3))
    rng.shuffle(order)
    rng.choice(["a.png"])
    left = rng.random() < 0.5
    caption = (Path(results[1]["folder_path"]) / "fizgig_mp_00.txt").read_text()
    assert caption.startswith(
        f"a synthetic stranger {order[0]} on the {'right' if left else 'left'}"
    )
    assert f"abcman on the {'left' if left else 'right'}" in caption


def test_cache_reuse_does_not_render_or_touch_files(recipe, monkeypatch):
    result = _prepare(recipe)
    folder = Path(result[0]["folder_path"])
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    monkeypatch.setattr(
        composites, "_render", lambda *args: pytest.fail("valid cache was regenerated")
    )
    assert _prepare(recipe) == result
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()
    } == before


def test_same_recipe_deterministic_across_output_roots(recipe):
    config, _ = recipe
    first = _prepare(recipe)[0]
    config.generated_dir += "-second"
    second = _prepare(recipe)[0]
    assert first["cache_identity"] == second["cache_identity"]
    assert _output_bytes(first["folder_path"]) == _output_bytes(second["folder_path"])


@pytest.mark.parametrize(
    "changed", ["image", "caption", "mapping", "count", "seed", "fusion"]
)
def test_changed_input_creates_distinct_cache_without_modifying_old_cache(
    recipe, changed
):
    config, _ = recipe
    first = _prepare(recipe)[0]
    before = _output_bytes(first["folder_path"])
    source = Path(config.subjects[0]["folder_path"])
    if changed == "image":
        # Content rather than timestamps identifies the input.
        Image.new("RGB", (180, 100), "orange").save(source / "a.png")
    elif changed == "caption":
        (source / "a.txt").write_bytes(b"different caption bytes, ignored by recipe")
    elif changed == "mapping":
        config.subjects[0]["trigger"] = "newword"
    elif changed == "count":
        config.composite_count += 1
    elif changed == "seed":
        config.seed += 1
    else:
        config.fused_captions = False
    second = _prepare(recipe)[0]
    assert second["cache_identity"] != first["cache_identity"]
    assert second["folder_path"] != first["folder_path"]
    assert _output_bytes(first["folder_path"]) == before


def test_all_subject_mappings_participate_in_each_cache(recipe):
    config, _ = recipe
    source = Path(config.subjects[0]["folder_path"])
    second = source.with_name("second")
    second.mkdir()
    Image.new("RGB", (30, 30), "white").save(second / "a.png")
    config.subjects.append(
        {"folder_path": str(second), "trigger": "abc", "class_name": "man"}
    )
    first = _prepare(recipe)
    config.subjects[1]["trigger"] = "changed"
    after = _prepare(recipe)
    assert all(a["cache_identity"] != b["cache_identity"] for a, b in zip(first, after))


def test_new_verified_fixture_revision_invalidates_cache(recipe):
    config, _ = recipe
    first = _prepare(recipe)[0]
    assets = Path(config.stranger_asset_dir)
    Image.new("RGB", (120, 80), "purple").save(assets / "stranger_00.png")
    pin = _fixture_pin(assets, revision="synthetic-fixture-v2")
    second = composites._prepare_composites(config, pin)[0]
    assert second["cache_identity"] != first["cache_identity"]


@pytest.mark.parametrize(
    "damage",
    [
        "missing-image",
        "wrong-pin",
        "tampered-image",
        "tampered-manifest",
        "missing-readme",
    ],
)
def test_missing_incomplete_or_tampered_assets_rejected(recipe, damage):
    config, _ = recipe
    assets = Path(config.stranger_asset_dir)
    if damage == "missing-image":
        (assets / "stranger_00.png").unlink()
    elif damage == "wrong-pin":
        (assets / composites.REVISION_FILE).write_text("wrong revision")
    elif damage == "tampered-image":
        Image.new("RGB", (120, 80), "purple").save(assets / "stranger_00.png")
    elif damage == "tampered-manifest":
        _write_json(assets / "strangers.json", [])
    else:
        (assets / "README.md").unlink()
    with pytest.raises(ValueError, match="download-assets"):
        _prepare(recipe)
    assert not Path(config.generated_dir).exists()


def test_public_preparation_never_accepts_synthetic_fixture_as_pinned_assets(recipe):
    config, _ = recipe
    with pytest.raises(ValueError, match="wrong revision marker"):
        composites.prepare_composites(config)


def test_missing_assets_actionable_without_network(recipe, monkeypatch):
    config, _ = recipe
    config.stranger_asset_dir += "-missing"
    import socket

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("network attempted")
    )
    with pytest.raises(ValueError, match="Preparation never downloads"):
        composites.prepare_composites(config)


def test_disabled_composites_require_no_assets_or_sources(recipe):
    config, _ = recipe
    config.composites = False
    config.stranger_asset_dir = "not-present"
    config.subjects = []
    assert composites.prepare_composites(config) == []


@pytest.mark.parametrize(
    "damage",
    [
        "missing-image",
        "missing-caption",
        "corrupt-image",
        "changed-caption",
        "missing-manifest",
        "extra-image",
    ],
)
def test_incomplete_or_corrupt_cache_is_rejected_not_regenerated(recipe, damage):
    folder = Path(_prepare(recipe)[0]["folder_path"])
    if damage == "missing-image":
        (folder / "fizgig_mp_00.png").unlink()
    elif damage == "missing-caption":
        (folder / "fizgig_mp_00.txt").unlink()
    elif damage == "corrupt-image":
        (folder / "fizgig_mp_00.png").write_bytes(b"corrupt")
    elif damage == "changed-caption":
        (folder / "fizgig_mp_00.txt").write_text("a different caption")
    elif damage == "extra-image":
        Image.new("RGB", (2, 2), "white").save(folder / "unexpected.png")
    else:
        (folder / composites.MANIFEST_FILE).unlink()
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(ValueError, match="Incomplete/corrupt"):
        _prepare(recipe)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


@pytest.mark.parametrize("encoding", ["same-bytes", "different-encoding"])
def test_known_source_stranger_content_overlap_rejected(recipe, encoding):
    config, _ = recipe
    source = Path(config.subjects[0]["folder_path"])
    stranger = Path(config.stranger_asset_dir) / "stranger_00.png"
    if encoding == "same-bytes":
        (source / "overlap.png").write_bytes(stranger.read_bytes())
    else:
        with Image.open(stranger) as image:
            image.save(source / "overlap.bmp")
    with pytest.raises(ValueError, match="content overlap"):
        _prepare(recipe)


@pytest.mark.parametrize("key", ["identity_ids", "provenance_ids"])
def test_declared_identity_or_provenance_overlap_rejected(recipe, key):
    config, _ = recipe
    source = Path(config.subjects[0]["folder_path"])
    assets = Path(config.stranger_asset_dir)
    _write_json(source / composites.PROVENANCE_FILE, {key: ["KNOWN-ID"]})
    _write_json(assets / composites.PROVENANCE_FILE, {key: ["known-id"]})
    with pytest.raises(ValueError, match="declared .* overlap"):
        _prepare(recipe)


@pytest.mark.parametrize(
    "location", ["source", "nested-source", "source-ancestor", "assets"]
)
def test_generated_directory_cannot_contain_or_overlap_inputs(recipe, location):
    config, _ = recipe
    source = Path(config.subjects[0]["folder_path"])
    config.generated_dir = str(
        {
            "source": source,
            "nested-source": source / "generated",
            "source-ancestor": source.parent,
            "assets": Path(config.stranger_asset_dir),
        }[location]
    )
    with pytest.raises(ValueError, match="must be separate"):
        _prepare(recipe)


def test_render_failure_never_publishes_partial_output(recipe, monkeypatch):
    config, _ = recipe

    def fail(folder, *args):
        (folder / "partial.txt").write_text("not a dataset")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(composites, "_render", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _prepare(recipe)
    assert list(Path(config.generated_dir).iterdir()) == []


def test_source_mutation_during_generation_never_publishes_cache(recipe, monkeypatch):
    config, _ = recipe
    render = composites._render

    def mutate(*args):
        render(*args)
        (Path(config.subjects[0]["folder_path"]) / "a.txt").write_text(
            "changed during render"
        )

    monkeypatch.setattr(composites, "_render", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        _prepare(recipe)
    assert list(Path(config.generated_dir).iterdir()) == []


def test_explicit_download_uses_only_pinned_revision(tmp_path, monkeypatch):
    # Downloader contract only: no fixture ever claims to be authenticated assets.
    calls = []
    import huggingface_hub

    def snapshot(*args, **kwargs):
        calls.append((args, kwargs))

    verified = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(
        composites, "_load_assets", lambda folder, pin: verified.append((folder, pin))
    )
    destination = tmp_path / "download"
    assert composites.download_assets(str(destination)) == str(destination.resolve())
    args, kwargs = calls[0]
    assert args == (composites.REPO,)
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == "8aeb77502234781cb5dca524994c80efcaa51024"
    assert verified[0][1] is composites._ASSET_PIN
    assert (
        destination / composites.REVISION_FILE
    ).read_text().strip() == composites.REVISION
    assert len(calls) == 1


def test_cli_download_is_explicit(tmp_path, monkeypatch, capsys):
    destination = str(tmp_path / "explicit")
    calls = []
    monkeypatch.setattr(
        composites, "download_assets", lambda path: calls.append(path) or path
    )
    composites.main(["download-assets", "--asset-dir", destination])
    assert calls == [destination]
    assert capsys.readouterr().out.strip() == destination


def test_toolkit_training_cache_directories_do_not_invalidate_recipe(recipe):
    first = _prepare(recipe)
    folder = Path(first[0]["folder_path"])
    for name in (
        "_latent_cache",
        "_t_e_cache",
        "_normal_cache",
        "_body_proportion_cache",
        "_body_shape_cache",
        "_face_id_cache",
        "_vae_anchor_cache",
    ):
        cache = folder / name
        cache.mkdir()
        (cache / "training-owned.bin").write_bytes(b"training cache")
    (folder / ".aitk_size.json").write_text("{}", encoding="utf-8")
    assert _prepare(recipe) == first


@pytest.mark.parametrize("name", ["_latent_cache", "_t_e_cache", "_clip_vision_cache"])
def test_cache_allowlist_does_not_hide_files_or_unknown_directories(recipe, name):
    folder = Path(_prepare(recipe)[0]["folder_path"])
    if name == "_clip_vision_cache":
        (folder / name).mkdir()
    else:
        (folder / name).write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="Incomplete/corrupt"):
        _prepare(recipe)


@pytest.mark.parametrize("class_name", ["cat", "dog", "horse", "object", "style"])
def test_non_person_subjects_cannot_use_stranger_people_composites(recipe, class_name):
    config, _ = recipe
    config.subjects[0]["class_name"] = class_name
    with pytest.raises(ValueError, match="require person-class subjects"):
        composites.prepare_composites(config)
