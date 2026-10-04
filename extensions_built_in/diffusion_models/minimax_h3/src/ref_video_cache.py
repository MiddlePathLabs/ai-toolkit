"""Reference-video latents for ref2va, without dataloader machinery.

 A control VIDEO gets the dataset's temporal treatment — num_frames /
 auto_frame_count, fps — and is area-matched to the target (own aspect kept, /32
 grid), then a single VAE encode whose condition-specific result is cached next
 to the video in ``_latent_cache/``. The disk and memory entries share one
 complete provenance recipe: source content, target geometry, frame/audio
 recipe, and model encoder identity.
 Everything is deterministic (even frame spread, no random start) so the cache
 is stable; the audio track is encoded alongside when possible (cached for
 later use, unused in conditioning for now).
"""

import base64
import hashlib
import json
import os

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .packing import reference_video_pixel_size


def ref_frame_indices(total, src_fps, num_frames, dataset_fps, trim_tail):
    """Source frame indices a reference video is sampled at (dataset-identical)."""
    if trim_tail:
        # dataset trim mode: real-time pacing from the start, tail trimmed —
        # keeps motion speed honest and the soundtrack in sync
        fps_ratio = src_fps / dataset_fps if src_fps > 0 else 1.0
        return [min(round(i * fps_ratio), total - 1) for i in range(num_frames)]
    # deterministic even frame spread across the clip
    return [
        min(round(i * (total - 1) / max(num_frames - 1, 1)), total - 1)
        for i in range(num_frames)
    ]


def ref_video_num_frames(model, path, dataset_config):
    """Dataset-identical frame count for a reference video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or dataset_config.fps
    cap.release()
    if dataset_config.auto_frame_count:
        num_frames = int(total / src_fps * dataset_config.fps)
        snapper = model.get_frame_count_snapper()
        if snapper is not None:
            num_frames = snapper(num_frames)
    else:
        num_frames = dataset_config.num_frames
    return num_frames, total, src_fps


def load_video_ref_for_te(model, path, dataset_config=None, max_frames=None):
    """Build the Qwen presentation from the SAME frames the latent rows use:
    2 fps over the frame-count-treated 24 fps clip (ComfyUI: frames[::12],
    timestamps i/2). Without a dataset config (sampling), the clip is
    treated as its own length capped at ``max_frames``, snapped to 17n+5."""
    from PIL import Image as _Image

    from .packing import align_num_frames_down
    from .text_encoder import VideoRef, video_has_audio

    if dataset_config is not None:
        num_frames, total, src_fps = ref_video_num_frames(model, path, dataset_config)
        ds_fps = dataset_config.fps
        trim = bool(
            dataset_config.auto_frame_count
            and dataset_config.trim_auto_frame_count_tail
        )
    else:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        ds_fps = 24
        n = int(total / src_fps * ds_fps)
        if max_frames:
            n = min(n, max_frames)
        num_frames = align_num_frames_down(max(n, 5))
        trim = True
    indices = ref_frame_indices(total, src_fps, num_frames, ds_fps, trim)
    # 2 fps over the treated 24fps clip: every 12th treated frame
    step = max(1, int(ds_fps // 2))
    picks = list(range(0, len(indices), step))
    cap = cv2.VideoCapture(path)
    raw_frames = read_frames_at(cap, [indices[j] for j in picks])
    cap.release()
    frames = [
        _Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in raw_frames
    ]
    times = [j / ds_fps for j in picks]
    return VideoRef(frames=frames, timestamps=times, has_audio=video_has_audio(path))


def static_image_video_ref(image, num_frames: int, fps: int = 24):
    """Present a still IMAGE as a silent static reference video: the same
    frame held for ``num_frames`` at ``fps``, sampled at 2 fps like
    :func:`load_video_ref_for_te` (frame picks every fps//2, timestamps
    j/fps). Used when image references are routed through the video-ref path
    (``image_refs_as_video``)."""
    from .text_encoder import VideoRef

    step = max(1, int(fps // 2))
    picks = list(range(0, int(num_frames), step))
    return VideoRef(
        frames=[image] * len(picks),
        timestamps=[j / fps for j in picks],
        has_audio=False,
    )


def read_frames_at(cap, indices):
    """Read the frames at sorted ``indices`` by decoding SEQUENTIALLY (seek-per-
    frame is both slow and unreliable on VBR/web clips). Container frame counts
    routinely overstate the decodable count by a few frames; when the stream
    ends early, missing tail frames repeat the last decoded frame instead of
    failing. Returns a list of BGR frames aligned with ``indices``."""
    wanted = list(indices)
    out = {}
    need = sorted(set(wanted))
    if not need:
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pos = 0
    max_idx = need[-1]
    ptr = 0
    last = None
    while ptr < len(need) and pos <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        while ptr < len(need) and need[ptr] == pos:
            out[pos] = frame
            ptr += 1
        last = frame
        pos += 1
    if not out and last is None:
        raise ValueError("Could not decode any frames")
    frames = []
    for idx in wanted:
        if idx in out:
            frames.append(out[idx])
        else:
            # ran past the decodable end: hold the last real frame
            frames.append(last)
    return frames


def _content_fingerprint(path: str) -> str:
    """Hash the source bytes, never just size/mtime."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _model_cache_identity(model) -> str:
    cached = getattr(model, "_aitk_ref_video_model_identity", None)
    if cached is not None:
        return cached
    config = getattr(model, "model_config", None)
    if isinstance(config, dict):
        kwargs = config.get("model_kwargs", {}) or {}
    else:
        kwargs = getattr(config, "model_kwargs", {}) or {}
    checkpoint_fields = {}
    for key, value in kwargs.items():
        entry = {"value": str(value)}
        try:
            if os.path.isfile(value):
                entry["fingerprint"] = _content_fingerprint(value)
        except (OSError, IOError, TypeError):
            pass
        checkpoint_fields[str(key)] = entry
    fields = {
        "arch": getattr(model, "arch", model.__class__.__qualname__),
        "latent_space_version": getattr(model, "latent_space_version", ""),
        "condition_encoder": "encode_condition_images",
        "vae_encode_fp32": bool(getattr(model, "vae_encode_fp32", False)),
        "model_kwargs": checkpoint_fields,
    }
    te_identity = getattr(model, "te_cache_identity", None)
    if callable(te_identity):
        try:
            fields["text_encoder_identity"] = str(te_identity())
        except Exception:
            fields["text_encoder_identity"] = repr(te_identity)
    identity = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    try:
        setattr(model, "_aitk_ref_video_model_identity", identity)
    except Exception:
        pass
    return identity


def _cache_path(path: str, hash_dict: dict) -> str:
    latent_dir = os.path.join(os.path.dirname(path), "_latent_cache")
    name = os.path.splitext(os.path.basename(path))[0]
    hash_input = json.dumps(hash_dict, sort_keys=True).encode("utf-8")
    hash_str = (
        base64.urlsafe_b64encode(hashlib.md5(hash_input).digest())
        .decode("ascii")
        .replace("=", "")
    )
    return os.path.join(latent_dir, f"{name}_{hash_str}.safetensors")


def prepare_ref_video_source(model, path: str, *, force: bool = False) -> dict:
    """Memoize immutable source facts used by every reference-cache lookup.

    A training forward must not reopen/probe/hash a reference video merely to
    discover that its full recipe is already in ``_ref_video_cache``. Source
    replacement is an explicit operation: callers that intentionally re-key a
    path pass ``force=True`` (via :func:`rekey_ref_video_source`).
    """
    source_cache = getattr(model, "_ref_video_source_cache", None)
    if source_cache is None:
        source_cache = {}
        model._ref_video_source_cache = source_cache
    if not force and path in source_cache:
        return source_cache[path]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open reference video {path}")
    source = {
        "source_content_fingerprint": _content_fingerprint(path),
        "source_geometry": [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        ],
    }
    cap.release()
    from .text_encoder import video_has_audio

    source["audio_present"] = bool(video_has_audio(path))
    source_cache[path] = source
    return source


def rekey_ref_video_source(model, path: str) -> dict:
    """Explicitly re-fingerprint one source and invalidate its memory entries."""
    source_cache = getattr(model, "_ref_video_source_cache", {})
    previous = source_cache.get(path)
    refreshed = prepare_ref_video_source(model, path, force=True)
    old_fingerprint = (
        previous.get("source_content_fingerprint") if previous is not None else None
    )
    if old_fingerprint is not None:
        memory_cache = getattr(model, "_ref_video_cache", None) or {}
        for key in list(memory_cache):
            try:
                if json.loads(key).get("source_content_fingerprint") == old_fingerprint:
                    del memory_cache[key]
            except (TypeError, ValueError):
                continue
    return refreshed


@torch.no_grad()
def load_ref_video_latent(
    model, path: str, dataset_config, target_height: int, target_width: int
) -> dict:
    """Load one reference latent using the exact disk and memory identity.

    The in-memory key is the canonical serialized recipe used to derive the
    disk filename. It therefore cannot return a path-only entry for a changed
    target geometry, frame recipe, model encoder, or source file.
    """
    source = prepare_ref_video_source(model, path)
    src_w, src_h, total, src_fps = source["source_geometry"]
    src_fps = src_fps or dataset_config.fps
    if dataset_config.auto_frame_count:
        num_frames = int(total / src_fps * dataset_config.fps)
        snapper = model.get_frame_count_snapper()
        if snapper is not None:
            num_frames = snapper(num_frames)
    else:
        num_frames = dataset_config.num_frames
    trim_tail = bool(
        dataset_config.auto_frame_count and dataset_config.trim_auto_frame_count_tail
    )
    out_h, out_w = reference_video_pixel_size(
        src_w, src_h, target_height, target_width
    )
    source_audio_present = bool(source["audio_present"])
    hash_dict = {
        "recipe_namespace": "h3_ref_video_condition_v4",
        "source_content_fingerprint": source["source_content_fingerprint"],
        "source_geometry": [src_w, src_h, total, float(src_fps)],
        "ref_sizing": "match_target_area",
        "target_geometry": [int(target_height), int(target_width)],
        "output_geometry": [int(out_h), int(out_w)],
        "num_frames": int(num_frames),
        "fps": dataset_config.fps,
        "trim_tail": trim_tail,
        "auto_frame_count": bool(dataset_config.auto_frame_count),
        "audio_present": source_audio_present,
        "audio_recipe": {
            "sample_rate": int(getattr(model, "sample_rate", 0) or 0),
            "trim_to_frames": trim_tail,
        },
        "model_identity": _model_cache_identity(model),
        "is_ref_video": True,
    }
    memory_key = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
    mem_cache = getattr(model, "_ref_video_cache", None)
    if mem_cache is None:
        mem_cache = {}
        model._ref_video_cache = mem_cache
    if memory_key in mem_cache:
        return mem_cache[memory_key]

    cache_file = _cache_path(path, hash_dict)
    if os.path.exists(cache_file):
        sd = load_file(cache_file, device="cpu")
        entry = {
            "latent": sd["latent"],
            "num_frames": int(sd["num_frames"].item()),
            "audio_rows": sd.get("audio_latent"),
            "audio_present": bool(
                sd.get("audio_present", torch.tensor(float(source_audio_present)))
                .reshape(-1)[0].item()
            ),
        }
        mem_cache[memory_key] = entry
        return entry

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open reference video {path}")
    indices = ref_frame_indices(
        total, src_fps, num_frames, dataset_config.fps, trim_tail
    )
    try:
        raw_frames = read_frames_at(cap, indices)
    except ValueError as e:
        raise ValueError(f"{e}: {path}") from e
    cap.release()
    frames = []
    for frame in raw_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    pixels = torch.from_numpy(np.stack(frames)).float() / 255.0 * 2.0 - 1.0
    pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W), [-1, 1]
    # H3's condition-specific API owns seed-42 posterior sampling and the
    # fp16-round-before-normalization rule. Do not substitute encode_images.
    latent = model.encode_condition_images(pixels)[0].to("cpu", torch.float16)

    state_dict = {
        "latent": latent,
        "num_frames": torch.tensor(num_frames, dtype=torch.int64),
        "audio_present": torch.tensor(
            [1.0 if source_audio_present else 0.0], dtype=torch.float32
        ),
    }
    audio_rows = None
    if source_audio_present:
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(path)
            if trim_tail:
                keep = int(round(num_frames / dataset_config.fps * sample_rate))
                waveform = waveform[:, :keep]
            audio_latent = model.encode_audio(
                [{"waveform": waveform, "sample_rate": sample_rate}]
            )[0]
            audio_rows = audio_latent.to("cpu", torch.float16)
            state_dict["audio_latent"] = audio_rows
        except Exception:
            # The physical source fact remains true even when optional audio
            # decoding/encoding is unavailable; absent rows are not fabricated.
            pass

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    save_file(state_dict, cache_file)
    entry = {
        "latent": latent,
        "num_frames": num_frames,
        "audio_rows": audio_rows,
        "audio_present": source_audio_present,
    }
    mem_cache[memory_key] = entry
    return entry
