"""Standalone voice recordings for MiniMax-H3.

H3 is not an `is_audio_model` architecture. Voice files (`.wav` / `.mp3` /
`.flac` / `.m4a`, plus the rest of the global audio extension list) train as
their own media kind: `FileItemDTO.is_audio_only`. The packed sequence still
needs video rows, so each item carries a zeros placeholder sized from the
audio duration's 17n+5 frame grid. That placeholder is not a video target —
audio-only loss omits video MSE entirely rather than multiplying it by zero.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

# Keep these in lockstep with minimax_h3.src.packing / audio_vae. This module
# is imported from the dataloader, so it must not import the H3 package.
FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENTS_PER_SECOND = 40
HOP_LENGTH = AUDIO_SAMPLE_RATE // AUDIO_LATENTS_PER_SECOND  # 800 = 25 ms
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
AUDIO_LATENT_CHANNELS = 24
SPATIAL_COMPRESSION = 16
# Pinned square so audio items never upscale through the image bucket selector
# (Fizgig AUDIO_SENTINEL_RESO). 128 is divisible by H3's 32-px bucket grid;
# the VAE spatial factor of 16 yields an 8x8 latent.
AUDIO_PLACEHOLDER_PIXELS = 128

# Clip grid minus the 5-frame / 0.208 s slot — too short to train a voice.
AUDIO_GRID_FRAMES = tuple(17 * n + 5 for n in range(1, 8))
DURATION_TOLERANCE_SAMPLES = HOP_LENGTH


class AudioRejected(ValueError):
    """A voice file that cannot become a training item."""


def uses_h3_standalone_audio(sd: Any, dataset_config: Any) -> bool:
    """H3 + `do_audio` enumerates standalone audio files. Not ACE-Step."""
    if dataset_config is None or not bool(getattr(dataset_config, "do_audio", False)):
        return False
    if sd is None or bool(getattr(sd, "is_audio_model", False)):
        return False
    arch = getattr(getattr(sd, "model_config", None), "arch", None)
    return str(arch or "").startswith("minimax_h3")


def validate_audio_only_caption_dropout(dataset_config: Any, *, has_audio_only: bool) -> None:
    """Empty-prompt dropout on a voice-only set trains empty-prompt → that voice."""
    if not has_audio_only:
        return
    rate = float(getattr(dataset_config, "caption_dropout_rate", 0.0) or 0.0)
    if rate > 0:
        raise ValueError(
            "caption_dropout_rate > 0 on an H3 audio-only dataset trains "
            "empty-prompt → that voice. Set caption_dropout_rate: 0."
        )


def is_audio_path(path: str, audio_extensions: Iterable[str]) -> bool:
    return os.path.splitext(path)[1].lower() in set(audio_extensions)


def valid_durations_text() -> str:
    secs = [f / FPS for f in AUDIO_GRID_FRAMES]
    return ", ".join(f"{s:.3f}" for s in secs[:-1]) + f" or {secs[-1]:.3f} seconds"


def video_latent_num_frames(num_frames: int) -> int:
    """17n+5 pixel frames -> 5n+2 latent frames."""
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form 17n+5, got {num_frames}")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def hop_exact_samples(frames: int) -> int:
    return audio_latent_num_frames(frames) * HOP_LENGTH


def grid_frames_for_samples(n_samples: int, name: str = "audio") -> int:
    """Pixel-frame count whose duration matches, or AudioRejected with valid lengths."""
    for frames in AUDIO_GRID_FRAMES:
        target = round(frames / FPS * AUDIO_SAMPLE_RATE)
        if abs(n_samples - target) <= DURATION_TOLERANCE_SAMPLES:
            return frames
    got = n_samples / AUDIO_SAMPLE_RATE
    ceiling = AUDIO_GRID_FRAMES[-1] / FPS
    extra = (
        " Cut it into segments that match the H3 audio grid."
        if got > ceiling + 0.025
        else " Cut it to a valid H3 audio-grid duration."
    )
    raise AudioRejected(
        f"{name}: {got:.3f} s of audio. A voice training item has to be exactly "
        f"{valid_durations_text()} long.{extra}"
    )


def load_h3_voice_waveform(path: str) -> torch.Tensor:
    """Decode any supported file to float32 (2, L) at 32 kHz.

    Rate and channel count are converted. Undecodable files and digital
    silence are refused — a voice item has nothing else to train.
    """
    import torchaudio
    from toolkit.dataloader_mixins import waveform_to_stereo

    name = os.path.basename(path)
    try:
        waveform, sample_rate = torchaudio.load(path)
    except Exception as e:
        raise AudioRejected(f"{name}: no decodable audio in this file ({e})") from e
    if waveform.numel() == 0:
        raise AudioRejected(f"{name}: no decodable audio in this file")
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform_to_stereo(waveform.float())
    if int(sample_rate) != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, int(sample_rate), AUDIO_SAMPLE_RATE
        )
    if not waveform.any():
        raise AudioRejected(
            f"{name}: decodes to digital silence — nothing to train a voice on"
        )
    return waveform


def admit_voice_file(path: str) -> int:
    """Decode, refuse silence / bad duration, return the matching pixel-frame count."""
    wav = load_h3_voice_waveform(path)
    return grid_frames_for_samples(wav.shape[-1], os.path.basename(path))


def trim_to_grid(wav: torch.Tensor, frames: int) -> torch.Tensor:
    """Exactly `audio_latent_num_frames(frames) * 800` samples, trim or zero-pad."""
    want = hop_exact_samples(frames)
    have = wav.shape[-1]
    if have >= want:
        return wav[..., :want]
    return F.pad(wav, (0, want - have))


def make_placeholder_latents(
    num_frames: int,
    bucket_height: int,
    bucket_width: int,
) -> torch.Tensor:
    """zeros(24, latent_frames, bucket_height/16, bucket_width/16). Not a video target."""
    t_lat = video_latent_num_frames(num_frames)
    h_lat = bucket_height // SPATIAL_COMPRESSION
    w_lat = bucket_width // SPATIAL_COMPRESSION
    if h_lat < 1 or w_lat < 1:
        raise ValueError(
            f"audio-only placeholder spatial size is 0 for bucket "
            f"{bucket_width}x{bucket_height}"
        )
    return torch.zeros(AUDIO_LATENT_CHANNELS, t_lat, h_lat, w_lat)


def is_audio_only_batch(batch: Any) -> bool:
    """True when every file item is audio-only. Mixed batches fail closed."""
    items = getattr(batch, "file_items", None) or ()
    if not items:
        return False
    flags = [bool(getattr(it, "is_audio_only", False)) for it in items]
    if any(flags) and not all(flags):
        raise ValueError(
            "mixed audio-only and visual items in one batch; enable buckets "
            "or use batch_size 1 so voice recordings do not collate with "
            "photos or clips"
        )
    return all(flags)


def compute_audio_only_objective(
    audio_pred: torch.Tensor,
    audio_target: torch.Tensor,
    audio_loss_multiplier: float,
) -> torch.Tensor:
    """Audio MSE only. Callers must not add a video term, including `0 * video`."""
    if audio_pred is None or audio_target is None:
        raise ValueError("audio-only item produced no audio objective")
    loss = F.mse_loss(audio_pred.float(), audio_target.float(), reduction="mean")
    return loss * float(audio_loss_multiplier)
