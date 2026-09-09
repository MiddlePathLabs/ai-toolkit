import math
import wave
from types import SimpleNamespace

import pytest
import torch


from toolkit.h3_audio_only import (
    AUDIO_GRID_FRAMES,
    AUDIO_LATENT_CHANNELS,
    AUDIO_PLACEHOLDER_PIXELS,
    AUDIO_SAMPLE_RATE,
    AudioRejected,
    SPATIAL_COMPRESSION,
    admit_voice_file,
    audio_latent_num_frames,
    compute_audio_only_objective,
    grid_frames_for_samples,
    hop_exact_samples,
    is_audio_only_batch,
    load_h3_voice_waveform,
    make_placeholder_latents,
    trim_to_grid,
    uses_h3_standalone_audio,
    validate_audio_only_caption_dropout,
    valid_durations_text,
    video_latent_num_frames,
)


def _sine(n_samples, sr=AUDIO_SAMPLE_RATE, freq=220.0):
    t = torch.arange(n_samples, dtype=torch.float32) / sr
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0).repeat(2, 1)


def _write_wav(path, waveform, sr=AUDIO_SAMPLE_RATE):
    pcm = (waveform.clamp(-1, 1) * 32767).short().cpu().numpy()
    interleaved = pcm.T.reshape(-1).tobytes()
    with wave.open(str(path), "w") as w:
        w.setnchannels(waveform.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(interleaved)


def test_grid_matches_plan_durations():
    expected = (22, 39, 56, 73, 90, 107, 124)
    assert AUDIO_GRID_FRAMES == expected
    secs = [f / 24 for f in expected]
    assert [round(s, 3) for s in secs] == [
        0.917, 1.625, 2.333, 3.042, 3.75, 4.458, 5.167,
    ]


def test_grid_matches_h3_packing():
    from extensions_built_in.diffusion_models.minimax_h3.src import packing

    for frames in AUDIO_GRID_FRAMES:
        assert packing.align_num_frames_down(frames) == frames
        assert packing.video_latent_num_frames(frames) == video_latent_num_frames(frames)
        assert packing.audio_latent_num_frames(frames) == audio_latent_num_frames(frames)


def test_grid_frames_for_samples_admits_each_duration():
    for frames in AUDIO_GRID_FRAMES:
        target = round(frames / 24 * AUDIO_SAMPLE_RATE)
        assert grid_frames_for_samples(target) == frames
        assert grid_frames_for_samples(target + 800) == frames
        assert grid_frames_for_samples(target - 800) == frames


def test_out_of_grid_duration_names_valid_lengths():
    with pytest.raises(AudioRejected, match=valid_durations_text()):
        grid_frames_for_samples(round(2.0 * AUDIO_SAMPLE_RATE), "poem.wav")
    with pytest.raises(AudioRejected, match="6.000 s"):
        grid_frames_for_samples(round(6.0 * AUDIO_SAMPLE_RATE), "long.wav")


def test_trim_to_grid_is_hop_exact():
    frames = 39
    want = hop_exact_samples(frames)
    assert want == audio_latent_num_frames(frames) * 800
    long = _sine(want + 100)
    assert trim_to_grid(long, frames).shape[-1] == want
    short = _sine(want - 50)
    padded = trim_to_grid(short, frames)
    assert padded.shape[-1] == want
    assert torch.equal(padded[..., : want - 50], short)
    assert not padded[..., want - 50 :].any()


def test_placeholder_shape_from_bucket():
    frames = 39
    ph = make_placeholder_latents(frames, AUDIO_PLACEHOLDER_PIXELS, AUDIO_PLACEHOLDER_PIXELS)
    t_lat = video_latent_num_frames(frames)
    spatial = AUDIO_PLACEHOLDER_PIXELS // SPATIAL_COMPRESSION
    assert ph.shape == (AUDIO_LATENT_CHANNELS, t_lat, spatial, spatial)
    assert not ph.any()


def test_uses_h3_standalone_audio_not_ace_step():
    h3 = SimpleNamespace(is_audio_model=False, model_config=SimpleNamespace(arch="minimax_h3"))
    ace = SimpleNamespace(is_audio_model=True, model_config=SimpleNamespace(arch="ace_step_15"))
    flux = SimpleNamespace(is_audio_model=False, model_config=SimpleNamespace(arch="flux"))
    ds_on = SimpleNamespace(do_audio=True)
    ds_off = SimpleNamespace(do_audio=False)
    assert uses_h3_standalone_audio(h3, ds_on) is True
    assert uses_h3_standalone_audio(h3, ds_off) is False
    assert uses_h3_standalone_audio(ace, ds_on) is False
    assert uses_h3_standalone_audio(flux, ds_on) is False
    assert uses_h3_standalone_audio(None, ds_on) is False



def test_audio_only_rejects_caption_dropout():
    validate_audio_only_caption_dropout(SimpleNamespace(caption_dropout_rate=0.0), has_audio_only=True)
    validate_audio_only_caption_dropout(SimpleNamespace(caption_dropout_rate=0.05), has_audio_only=False)
    with pytest.raises(ValueError, match="empty-prompt"):
        validate_audio_only_caption_dropout(
            SimpleNamespace(caption_dropout_rate=0.05), has_audio_only=True
        )


def test_is_audio_only_batch_mixed_fails_closed():
    voice = SimpleNamespace(is_audio_only=True)
    clip = SimpleNamespace(is_audio_only=False, is_video=True)
    assert is_audio_only_batch(SimpleNamespace(file_items=[voice, voice])) is True
    assert is_audio_only_batch(SimpleNamespace(file_items=[clip])) is False
    with pytest.raises(ValueError, match="mixed audio-only"):
        is_audio_only_batch(SimpleNamespace(file_items=[voice, clip]))


def test_audio_only_objective_has_no_video_contribution():
    video_pred = torch.ones(1, 24, 4, 8, 8, requires_grad=True)
    audio_pred = torch.randn(1, 20, 32, requires_grad=True)
    audio_target = torch.randn(1, 20, 32)
    loss = compute_audio_only_objective(audio_pred, audio_target, 1.0)
    loss.backward()
    assert audio_pred.grad is not None
    assert audio_pred.grad.abs().sum() > 0
    assert video_pred.grad is None
    with pytest.raises(ValueError, match="no audio objective"):
        compute_audio_only_objective(None, audio_target, 1.0)


def test_decode_stereo_32k_and_silence(tmp_path):
    frames = 22
    duration = frames / 24
    path = tmp_path / "voice.wav"
    _write_wav(path, _sine(round(duration * 48000), sr=48000), sr=48000)
    wav = load_h3_voice_waveform(str(path))
    assert wav.shape[0] == 2
    assert grid_frames_for_samples(wav.shape[-1]) == frames
    assert admit_voice_file(str(path)) == frames

    silent = tmp_path / "silence.wav"
    _write_wav(silent, torch.zeros(2, round(duration * AUDIO_SAMPLE_RATE)), sr=AUDIO_SAMPLE_RATE)
    with pytest.raises(AudioRejected, match="digital silence"):
        load_h3_voice_waveform(str(silent))



def test_undecodable_file_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(AudioRejected, match="no decodable audio"):
        load_h3_voice_waveform(str(path))
