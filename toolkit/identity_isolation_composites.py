# Copyright 2026 Peter Neill
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Adapted from Fizgig families/multi_person.py (v8.4.0), by Peter Neill.

AI-Toolkit adaptation: offline-only preparation, verified pinned assets,
content-addressed atomic caches and explicit download command. The panel
layout and stranger-first caption recipe are preserved; the v8.4 center
square crop is replaced by face-aware panels (SCRFD/YuNet detection via the
identity-eval detector, face-centered square with margin) plus a usability
pool so a panel never publishes without a clear, unambiguous face.

Optional portraits: ShootTheSound/fizgig-multi-person-strangers, revision
8aeb77502234781cb5dca524994c80efcaa51024, CC-BY-4.0,
https://creativecommons.org/licenses/by/4.0/ . Attribution: ShootTheSound/Fizgig,
https://huggingface.co/datasets/ShootTheSound/fizgig-multi-person-strangers .
The pinned README declares 32 Qwen Image 2.1 generated portraits, seeds
1000-1031. Composites crop/resize these portraits and combine them with user
images. Their attribution and modification notice are retained in each cache.

Optional identity_isolation_provenance.json files in source/asset directories
may declare identity_ids and provenance_ids (lists of strings). Matching IDs,
file digests or decoded pixels reject known overlap. These checks cannot
identify a real person or certify that two different portraits depict different
people; users remain responsible for source identity and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PIL import __version__ as PILLOW_VERSION

import numpy as np

if TYPE_CHECKING:
    from toolkit.identity_isolation import IsolationConfig

REPO = "ShootTheSound/fizgig-multi-person-strangers"
REVISION = "8aeb77502234781cb5dca524994c80efcaa51024"
REVISION_FILE = "fizgig_revision.txt"
PANEL = 512
PREFIX = "fizgig_mp_"
TAIL = "two people side by side, two close-up portraits"
MANIFEST_FILE = "identity_isolation_manifest.json"
PROVENANCE_FILE = "identity_isolation_provenance.json"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_RECIPE_VERSION = "fizgig-v8.5.1-face-square-frontal-v1"
# Face-aware paneling: square side = FACE_MARGIN x face height (a face is
# ~45% of the panel at 2.2, matching the "close-up portraits" caption tail),
# a source face must be at least MIN_FACE_PX tall before upscale, and a
# second face taller than RIVAL_FACE_RATIO x the largest makes the photo
# ambiguous for identity pairing, so it is excluded from the pool.
FACE_MARGIN = 2.2
MIN_FACE_PX = 96
RIVAL_FACE_RATIO = 0.6
# Subject panels also require a frontal-ish face, estimated from the
# detector's 5 landmarks: |nose_x - eye_midpoint_x| / interocular distance
# stays small while both eyes are visible and grows toward/past 1 in
# profile, where the far-eye landmark also collapses toward the nose. A
# yaw proxy above MAX_YAW_PROXY (~50-55 degrees) or interocular below
# MIN_INTEROCULAR_RATIO x face width rejects the photo; mild three-quarter
# views (~35-40 degrees) still pass. Stranger panels are exempt: the pinned
# portraits are close-ups and pose variety there is harmless.
MAX_YAW_PROXY = 0.62
MIN_INTEROCULAR_RATIO = 0.12
# SHA256 LFS object identities from the pinned HF tree; no image download needed.
_IMAGE_DIGESTS = (
    "d12e2abebcf981c3ae8808c13831838cbbaf21fda163026e7ea18ff7b86c3d61",
    "f1b27f9485510889e23467ef93b40ae1e1b7a0817a25a2ae413a64c2653ea850",
    "811095978481c7dbc77a0fdafc6bfe432d1fc308522493a1039dae5d98286c8a",
    "14a32e2083bebb21938c5f77fb0282a08f50285884879bfbb8c6d761fd5b2bd2",
    "c44b51e6a45ea8d967126396f09b782553efd1fe390567d7a54d219b11caf6c3",
    "7b4b56888e64c608e75377e165666a7fc3c430717e801e233b977fa6721d3771",
    "4834de786609aac681677a4f76d83c3b6a63ce677dc1be67ddfe855af9b2b245",
    "2efbbb8a87f973f8679000534ba4aa35e2510ea86c6a8dbdcd683a588a525f26",
    "137b7538cd12e667cbaffa624bf773be52502e7652fbb64a29736a49812f26ee",
    "6cef2efc6ba921da6171bf62027471d59cdc01c80b07ad9abfd78542a28d920e",
    "6a6a514ba7a785dd69f25ac0eacdbb64ce50133a95d0ba04cef1b919c19d7d7c",
    "7dbb9e1e41364f9079168c2200003a9990b11df5b9bdab9bfe6ae3d753597b74",
    "99ab01a49f6a5cd41c35a75ac0af5507a7f88bb076ece91c4fe03bae48824621",
    "41240513c73af511e50386a5cd8dadd60377d9af6d2156944075010866ae3d7f",
    "41a75a6ffa6c237c9fc61e0142534d62f275cb5e358dcb8508b12d1d6a2107d0",
    "12859d91c4860df7d9d841a53c0135fd2d1927aa33a9756c546bf5cdc9b8bf2d",
    "48c2d786d1e29a0124dcfe87ee87a2a5a8ed75bc959c14fc775267cb6e86f772",
    "d80a3f3ae282952438e76ecc2a56e880a73ed125ab9d4366c9f86677cbb05eec",
    "4b1e56f9860d7daa42740d1d20e8b223ac71d025f4a7e8ea0eed47d777088070",
    "cdaeafb2f4dd990045e27883e06b56bac01de0f5e03b8bd11cc06a05ce2319dc",
    "dbee236fa8c68a17a4416fedb23fc1254e19f94a112707128622ebeb74a35ddc",
    "00261c81f8823b14181aa6af4e05068c54d93bf6de850b988951318f439d1b03",
    "fd38b6379f39e96b50ef66b65b3e6e6305755c5a994506456d6b44d2f9ba0e65",
    "88008ba87eaf8917f2e9191e6bbfec7ac1e977806ba84a93681f87a8a6d32ae0",
    "e2961e0da422429c913952b9ef48c5034d393f8adf73b7eb4a9222cab2d55d7b",
    "63fdb723bd040b22d10aa141ac46cc4f5eeb31b3120bc973bac817f4dbd9c729",
    "0145e0412156d725e746df5080fd86cffc9e0b964e1a34934cba5e69cdd95a0c",
    "7c6fcf70a0c895a77f20db19f94ebc1cc3a5f34689cdd9b555d91b1dbfb0ba7d",
    "6f583a016854490de8a1b9ad8255b405ad4d9f3dfc46518c397a29354e0a6e57",
    "10d021f0f2e6d21b8035671a3063b9f2ee9c109965117557f43a567ee4b8b1d8",
    "42635ef5ffebb16f648ad0049a82a8076edb8f03e47446aa0ea19ee6e7f591db",
    "97833efd3c3f920c898918ffb90cf4dfb56cf4b04421512d57617eeedc198e29",
)


@dataclass(frozen=True)
class _AssetPin:
    repo: str
    revision: str
    image_sha256: dict[str, str]
    manifest_blob: str
    readme_blob: str


_ASSET_PIN = _AssetPin(
    REPO,
    REVISION,
    {f"stranger_{i:02d}.png": digest for i, digest in enumerate(_IMAGE_DIGESTS)},
    "27f1cb1df991de29976628d294c09256b9231746",
    "b7944928e5561248e149944a0bb8f8542d82bb79",
)


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def _pixel_digest(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(_json_bytes(image.size) + image.tobytes()).hexdigest()


def _provenance(folder: Path) -> dict:
    path = folder / PROVENANCE_FILE
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid identity provenance object: {path}")
    for key in ("identity_ids", "provenance_ids"):
        ids = raw.get(key, [])
        if not isinstance(ids, list) or any(
            not isinstance(i, str) or not i.strip() for i in ids
        ):
            raise ValueError(f"{path}: {key} must be a list of nonempty strings")
    return raw


def _asset_error(folder: Path, reason: str) -> ValueError:
    return ValueError(
        f"Identity isolation stranger assets at {folder}: {reason}. Preparation never downloads. "
        "Supply a complete pinned local copy or explicitly run "
        f'python -m toolkit.identity_isolation_composites download-assets --asset-dir "{folder}" '
        "(use a fresh directory if this copy is damaged)."
    )


def _load_assets(folder: Path, pin: _AssetPin) -> tuple[list[dict], dict]:
    try:
        if (folder / REVISION_FILE).read_text(encoding="utf-8").strip() != pin.revision:
            raise ValueError("wrong revision marker")
        manifest = (folder / "strangers.json").read_bytes()
        readme = (folder / "README.md").read_bytes()
        if (
            _git_blob(manifest) != pin.manifest_blob
            or _git_blob(readme) != pin.readme_blob
        ):
            raise ValueError("manifest or license README differs from pinned revision")
        rows = json.loads(manifest)
        if not isinstance(rows, list) or len(rows) != len(pin.image_sha256):
            raise ValueError("incomplete stranger manifest")
        names = [r["file"] for r in rows]
        if len(set(names)) != len(names) or set(names) != set(pin.image_sha256):
            raise ValueError("incomplete or duplicate stranger files")
        files = {}
        for row in rows:
            name = row["file"]
            if (
                Path(name).name != name
                or not isinstance(row["description"], str)
                or not row["description"].strip()
            ):
                raise ValueError("invalid stranger filename or description")
            path = folder / name
            digest = _sha256(path)
            if digest != pin.image_sha256[name]:
                raise ValueError(f"tampered stranger image: {name}")
            files[name] = {"sha256": digest, "pixels": _pixel_digest(path)}
        evidence = {
            "repo": pin.repo,
            "revision": pin.revision,
            "files": files,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "readme_sha256": hashlib.sha256(readme).hexdigest(),
            "provenance": _provenance(folder),
            "attribution": "ShootTheSound/Fizgig",
            "license": "CC-BY-4.0" if pin.repo == REPO else "synthetic-test-fixture",
            "license_url": "https://creativecommons.org/licenses/by/4.0/"
            if pin.repo == REPO
            else None,
            "source_url": f"https://huggingface.co/datasets/{pin.repo}/tree/{pin.revision}"
            if pin.repo == REPO
            else None,
            "generation": "Qwen Image 2.1, seeds 1000-1031"
            if pin.repo == REPO
            else "synthetic-test-fixture",
            "modifications": "center-square RGB LANCZOS resize; side-by-side composite",
            "identity_limitations": "Digest/declared-ID checks reject known overlap; they do not verify real identity.",
        }
        return rows, evidence
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise _asset_error(folder, str(error)) from error


def _source_snapshot(folder: Path) -> dict:
    photos = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not photos:
        raise ValueError(f"No source photos for identity isolation in {folder}")
    return {
        "folder_path": str(folder),
        "images": {
            p.name: {"sha256": _sha256(p), "pixels": _pixel_digest(p)} for p in photos
        },
        "captions": {
            p.name: _sha256(p)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() == ".txt"
        },
        "provenance": _provenance(folder),
    }


def _reject_overlap(source: dict, assets: dict) -> None:
    for key in ("sha256", "pixels"):
        known = {record[key] for record in assets["files"].values()}
        if any(record[key] in known for record in source["images"].values()):
            raise ValueError(
                f"Source/stranger content overlap in {source['folder_path']}; use unrelated stranger portraits"
            )
    for key in ("identity_ids", "provenance_ids"):
        mine = {v.strip().casefold() for v in source["provenance"].get(key, [])}
        theirs = {v.strip().casefold() for v in assets["provenance"].get(key, [])}
        if mine & theirs:
            raise ValueError(
                f"Source/stranger declared {key} overlap in {source['folder_path']}"
            )


def _build_detector():
    """SCRFD (buffalo_l) via the identity-eval detector, YuNet fallback.

    Preparation stays offline: discovery never downloads weights, so a
    machine without buffalo_l/YuNet fails fast with placement instructions
    instead of silently regressing to the v8.4 center crop.
    """
    try:
        from toolkit.identity_eval.detector import FaceDetector
    except Exception as error:  # noqa: BLE001 -- optional eval extras
        raise ValueError(
            f"Identity isolation composites require the identity-eval face "
            f"detector stack: {error}"
        ) from error
    try:
        return FaceDetector(device="cpu")
    except Exception as error:  # noqa: BLE001
        raise ValueError(
            "Identity isolation composites require a face detector (buffalo_l "
            f"or YuNet, see docs/identity_eval.md): {error}"
        ) from error


def _usable_face(
    detector, image: Image.Image, require_frontal: bool = False
) -> tuple[list[int] | None, str | None]:
    """Largest-face usability check for one RGB image.

    Returns an int bbox (x1, y1, x2, y2) clamped to the image, or a stable
    rejection reason. ``detector.detect`` receives BGR (the identity-eval
    contract) and returns area-sorted faces already score-thresholded.
    ``require_frontal`` adds the landmark yaw gate for subject panels.
    """
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
    faces = detector.detect(bgr).faces
    if not faces:
        return None, "no_face"
    x1, y1, x2, y2 = (float(v) for v in faces[0].bbox)
    face_h = min(y2, image.height) - max(y1, 0.0)
    if face_h < MIN_FACE_PX:
        return None, "small_face"
    if len(faces) > 1:
        other = faces[1].bbox
        other_h = min(float(other[3]), image.height) - max(float(other[1]), 0.0)
        if other_h >= RIVAL_FACE_RATIO * face_h:
            return None, "rival_face"
    if require_frontal:
        kps = np.asarray(faces[0].kps, dtype=np.float32).reshape(-1, 2)
        interocular = abs(float(kps[1, 0]) - float(kps[0, 0]))
        face_w = max(x2 - x1, 1.0)
        if interocular < MIN_INTEROCULAR_RATIO * face_w:
            return None, "profile_face"
        eye_mid_x = (float(kps[0, 0]) + float(kps[1, 0])) / 2
        yaw = (float(kps[2, 0]) - eye_mid_x) / max(interocular, 1e-6)
        if abs(yaw) > MAX_YAW_PROXY:
            return None, "profile_face"
    box = [
        int(round(max(0.0, min(x1, image.width)))),
        int(round(max(0.0, min(y1, image.height)))),
        int(round(max(0.0, min(x2, image.width)))),
        int(round(max(0.0, min(y2, image.height)))),
    ]
    return box, None


def _face_square(width: int, height: int, bbox) -> tuple[int, int, int, int] | None:
    """Face-centered square crop box, clamped to the image bounds.

    The square keeps the detected face fully inside it as long as the face
    lies inside the image: centering on the face then clamping only shifts
    the window toward the image interior.
    """
    x1 = max(0.0, min(float(bbox[0]), width))
    y1 = max(0.0, min(float(bbox[1]), height))
    x2 = max(x1, min(float(bbox[2]), width))
    y2 = max(y1, min(float(bbox[3]), height))
    face_h = y2 - y1
    if face_h <= 0 or x2 - x1 <= 0:
        return None
    side = int(min(FACE_MARGIN * face_h, width, height))
    left = max(0, min(int(round((x1 + x2) / 2 - side / 2)), width - side))
    top = max(0, min(int(round((y1 + y2) / 2 - side / 2)), height - side))
    return left, top, left + side, top + side


def _panel(image: Image.Image, bbox) -> Image.Image:
    """512 close-up panel cropped around the detected face."""
    square = _face_square(image.width, image.height, bbox)
    if square is None:
        raise ValueError(f"Degenerate face box {bbox} for {image.size} image")
    return image.crop(square).resize((PANEL, PANEL), Image.Resampling.LANCZOS)


def _face_pool(
    detector, folder: Path, names: list[str], require_frontal: bool = False
) -> dict:
    """Usability pool for one image folder: face box per usable photo."""
    usable = []
    rejected: dict[str, int] = {}
    for name in names:
        with Image.open(folder / name) as image:
            box, reason = _usable_face(detector, image, require_frontal)
        if box is None:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            usable.append({"file": name, "face": box})
    return {"usable": usable, "rejected": rejected}


def _render(
    folder: Path,
    config,
    index: int,
    strangers: list[dict],
    source: dict,
    assets_dir: Path,
    subject_pool: list[dict],
) -> None:
    """Render composites from usable faces only.

    Both sides cycle a shuffled usability pool without replacement per pass,
    so subject photo diversity is maximal for the requested count; every
    published panel contains one clear face by construction.
    """
    subject = config.subjects[index]
    trigger = subject["trigger"]
    if config.fused_captions:
        trigger = "".join(trigger.split()) + subject["class_name"].split()[-1].lower()
    rng = random.Random(config.seed * 1000 + index)
    stranger_order = list(range(len(strangers)))
    subject_order = list(range(len(subject_pool)))
    rng.shuffle(stranger_order)
    rng.shuffle(subject_order)
    for k in range(config.composite_count):
        stranger = strangers[stranger_order[k % len(stranger_order)]]
        mine = subject_pool[subject_order[k % len(subject_order)]]
        left = rng.random() < 0.5
        with Image.open(Path(source["folder_path"]) / mine["file"]) as image:
            a = _panel(image.convert("RGB"), mine["face"])
        with Image.open(assets_dir / stranger["file"]) as image:
            b = _panel(image.convert("RGB"), stranger["face"])
        with Image.new("RGB", (2 * PANEL, PANEL)) as sheet:
            sheet.paste(a if left else b, (0, 0))
            sheet.paste(b if left else a, (PANEL, 0))
            sheet.save(folder / f"{PREFIX}{k:02d}.png")
        a.close()
        b.close()
        me, them = ("left", "right") if left else ("right", "left")
        caption = f"{stranger['description']} on the {them}, and {trigger} on the {me}, {TAIL}"
        (folder / f"{PREFIX}{k:02d}.txt").write_text(caption, encoding="utf-8")


def _output_records(folder: Path, count: int) -> dict:
    records = {}
    expected = {
        f"{PREFIX}{i:02d}{ext}" for i in range(count) for ext in (".png", ".txt")
    }
    # Toolkit owns these caches after training; they are not recipe outputs.
    training_caches = {
        "_latent_cache",
        "_t_e_cache",
        "_normal_cache",
        "_body_proportion_cache",
        "_body_shape_cache",
        "_face_id_cache",
        "_vae_anchor_cache",
    }
    actual = {
        p.name
        for p in folder.iterdir()
        if not (
            p.name in training_caches
            and p.is_dir()
            and not p.is_symlink()
            or p.name == ".aitk_size.json"
            and p.is_file()
            and not p.is_symlink()
        )
    }
    if actual - {MANIFEST_FILE} != expected:
        raise ValueError(
            f"Incomplete/corrupt identity isolation cache: {folder}; expected exactly {count} image/caption pairs"
        )
    for name in sorted(expected):
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid identity isolation cache file: {path}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                if image.size != (2 * PANEL, PANEL) or image.mode != "RGB":
                    raise ValueError(f"Invalid composite dimensions/mode: {path}")
                image.load()
        elif not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Empty composite caption: {path}")
        records[name] = _sha256(path)
    return records


def _reuse_cache(folder: Path, identity: str, inputs: dict, count: int) -> None:
    try:
        manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest["cache_identity"] != identity or manifest["inputs"] != inputs:
            raise ValueError("cache manifest identity mismatch")
        if manifest["outputs"] != _output_records(folder, count):
            raise ValueError("cache output digest mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Incomplete/corrupt identity isolation cache at {folder}: {error}. Remove this generated cache explicitly before rebuilding."
        ) from error


def _prepare_composites(config, pin: _AssetPin, detector=None) -> list[dict]:
    if not config.composites:
        return []
    if not 1 <= len(config.subjects) <= 4 or config.composite_count < 1:
        raise ValueError(
            "Composites require 1-4 subjects and a positive composite_count"
        )
    people = {
        "woman",
        "man",
        "girl",
        "boy",
        "lady",
        "guy",
        "person",
        "child",
        "kid",
        "baby",
        "teenager",
    }
    if any(
        s["class_name"].split()[-1].casefold() not in people for s in config.subjects
    ):
        raise ValueError(
            "Composites use stranger people portraits and require person-class subjects"
        )
    assets_dir = Path(config.stranger_asset_dir).resolve()
    root = Path(config.generated_dir).resolve()
    sources = [Path(s["folder_path"]).resolve() for s in config.subjects]
    for source in [*sources, assets_dir]:
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError(
                "Generated composite directory must be separate from source and stranger directories (not nested)"
            )
    rows, assets = _load_assets(assets_dir, pin)
    snapshots = [_source_snapshot(source) for source in sources]
    for snapshot in snapshots:
        _reject_overlap(snapshot, assets)
    if detector is None:
        detector = _build_detector()
    stranger_pool = _face_pool(detector, assets_dir, [row["file"] for row in rows])
    if not stranger_pool["usable"]:
        raise ValueError(
            "No pinned stranger portrait has a usable face; the asset set is "
            "damaged or the detector is misbehaving"
        )
    subject_pools = []
    for snapshot in snapshots:
        pool = _face_pool(
            detector,
            Path(snapshot["folder_path"]),
            sorted(snapshot["images"]),
            require_frontal=True,
        )
        if not pool["usable"]:
            raise ValueError(
                f"No usable face-forward photo for subject {snapshot['folder_path']} "
                f"(rejections: {pool['rejected']}). Composites need at least one "
                "photo with a detected face at least "
                f"{MIN_FACE_PX}px tall, no rival face of comparable size, and a "
                "frontal-ish head pose for subject panels."
            )
        subject_pools.append(pool)
    usable_strangers = []
    by_file = {entry["file"]: entry for entry in stranger_pool["usable"]}
    for row in rows:
        entry = by_file.get(row["file"])
        if entry:
            usable_strangers.append({**row, "face": entry["face"]})
    mapping = [
        {**s, "folder_path": str(path)} for s, path in zip(config.subjects, sources)
    ]
    common = {
        "recipe": _RECIPE_VERSION,
        "panel": PANEL,
        "pillow": PILLOW_VERSION,
        "detector": detector.det_sig,
        "face_recipe": {
            "margin": FACE_MARGIN,
            "min_face_px": MIN_FACE_PX,
            "rival_ratio": RIVAL_FACE_RATIO,
            "max_yaw_proxy": MAX_YAW_PROXY,
            "min_interocular_ratio": MIN_INTEROCULAR_RATIO,
        },
        "face_pools": {
            "strangers": stranger_pool,
            "subjects": subject_pools,
        },
        "mapping": mapping,
        "sources": snapshots,
        "assets": assets,
        "seed": config.seed,
        "count": config.composite_count,
        "fused_captions": config.fused_captions,
    }
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for index in range(len(sources)):
        inputs = {**common, "subject_index": index}
        identity = hashlib.sha256(_json_bytes(inputs)).hexdigest()
        folder = root / identity
        if folder.exists():
            _reuse_cache(folder, identity, inputs, config.composite_count)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".identity-isolation-", dir=root))
            try:
                _render(
                    temporary,
                    config,
                    index,
                    usable_strangers,
                    snapshots[index],
                    assets_dir,
                    subject_pools[index]["usable"],
                )
                # Never publish an output if its inputs changed during preparation.
                if (
                    snapshots != [_source_snapshot(p) for p in sources]
                    or assets != _load_assets(assets_dir, pin)[1]
                ):
                    raise ValueError(
                        "Identity isolation inputs changed during composite preparation"
                    )
                manifest = {
                    "cache_identity": identity,
                    "inputs": inputs,
                    "outputs": _output_records(temporary, config.composite_count),
                }
                (temporary / MANIFEST_FILE).write_bytes(_json_bytes(manifest))
                try:
                    os.rename(temporary, folder)
                except OSError:
                    if not folder.exists():
                        raise
                    _reuse_cache(folder, identity, inputs, config.composite_count)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        result.append(
            {
                "folder_path": str(folder),
                "subject_index": index,
                "cache_identity": identity,
            }
        )
    return result


def prepare_composites(config: IsolationConfig) -> list[dict]:
    """Prepare offline, content-verified, face-aware stranger composites.

    The detector is built lazily inside preparation, after asset and source
    validation, so configuration errors never depend on model availability.
    """
    return _prepare_composites(config, _ASSET_PIN)


def download_assets(asset_dir: str) -> str:
    """Explicitly download and verify the pinned optional CC-BY-4.0 portraits."""
    destination = Path(asset_dir).resolve()
    if destination.exists():
        _load_assets(destination, _ASSET_PIN)
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".identity-assets-", dir=destination.parent)
    )
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(temporary),
            allow_patterns=["stranger_*.png", "strangers.json", "README.md"],
        )
        (temporary / REVISION_FILE).write_text(REVISION + "\n", encoding="utf-8")
        _load_assets(temporary, _ASSET_PIN)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(destination)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download-assets",
        help="Explicitly fetch the pinned CC-BY-4.0 stranger portraits",
    )
    download.add_argument("--asset-dir", required=True)
    args = parser.parse_args(argv)
    print(download_assets(args.asset_dir))


if __name__ == "__main__":
    main()
