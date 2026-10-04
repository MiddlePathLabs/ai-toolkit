import argparse
import os
import subprocess
import sys
import time
from fractions import Fraction

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class FakeSD:
    def __init__(self):
        self.use_raw_control_images = False

    def encode_control_in_text_embeddings(self, *args, **kwargs):
        return None

    def get_bucket_divisibility(self):
        return 32


def _tensor_to_uint8_video(frames_fchw: torch.Tensor) -> torch.Tensor:
    """Convert ``[F,C,H,W]`` float/uint8 frames to ``[F,H,W,C]`` uint8."""
    x = frames_fchw.detach()
    if x.dtype != torch.uint8:
        x = x.to(torch.float32)
        if torch.isfinite(x).all() and x.min().item() < 0.0:
            x = x * 0.5 + 0.5
        x = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        x = x.to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().cpu()


def _write_video(path: str, frames_hwc: torch.Tensor, fps: float) -> None:
    """Write H.264 using the installed PyAV writer (not removed torchvision IO)."""
    import av

    frames = frames_hwc.numpy()
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=Fraction(str(float(fps))))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _mux_with_ffmpeg(video_in: str, wav_in: str, mp4_out: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_in,
            "-i",
            wav_in,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            mp4_out,
        ],
        check=True,
    )


def run_fixture(
    dataset_folder: str,
    *,
    epochs: int = 1,
    num_frames: int = 121,
    output_path: str = "output/dataset_test",
) -> None:
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)
    dataset_config = DatasetConfig(
        dataset_path=dataset_folder,
        resolution=512,
        default_caption="default",
        buckets=True,
        bucket_tolerance=64,
        shrink_video_to_frames=True,
        num_frames=num_frames,
        do_i2v=True,
        fps=24,
        do_audio=True,
        debug=True,
        audio_preserve_pitch=False,
        audio_normalize=True,
    )
    dataloader: DataLoader = get_dataloader_from_datasets(
        [dataset_config], batch_size=1, sd=FakeSD()
    )

    idx = 0
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            batch: DataLoaderBatchDTO
            img_batch = batch.tensor
            if img_batch.ndim == 5:
                batch_size, frames, channels, height, width = img_batch.shape
            else:
                batch_size, channels, height, width = img_batch.shape

            audio_tensor = batch.audio_tensor
            audio_data = batch.audio_data
            fps = float(getattr(dataset_config, "fps", None) or 1.0)
            for b in range(batch_size):
                if img_batch.ndim == 5:
                    frames_fchw = img_batch[b]
                else:
                    frames_fchw = img_batch[b].unsqueeze(0)
                video_uint8 = _tensor_to_uint8_video(frames_fchw)
                out_mp4 = os.path.join(output_path, f"{idx:06d}_{b:02d}.mp4")

                item_audio = None
                item_sr = None
                if isinstance(audio_data, (list, tuple)) and len(audio_data) > b:
                    audio_item = audio_data[b]
                    if (
                        isinstance(audio_item, dict)
                        and audio_item.get("waveform") is not None
                        and audio_item.get("sample_rate") is not None
                    ):
                        item_audio = audio_item["waveform"]
                        item_sr = int(audio_item["sample_rate"])
                elif torch.is_tensor(audio_tensor):
                    if audio_tensor.ndim == 3 and audio_tensor.shape[0] > b:
                        item_audio = audio_tensor[b]
                    elif audio_tensor.ndim == 2 and b == 0:
                        item_audio = audio_tensor
                    if item_audio is not None and isinstance(audio_data, dict):
                        try:
                            item_sr = int(audio_data["sample_rate"])
                        except (KeyError, TypeError, ValueError):
                            item_sr = None

                tmp_video = out_mp4 + ".tmp_video.mp4"
                tmp_wav = out_mp4 + ".tmp_audio.wav"
                try:
                    _write_video(tmp_video, video_uint8, fps)
                    if item_audio is not None and item_sr is not None and item_audio.numel() > 0:
                        import torchaudio

                        wav = item_audio.detach()
                        if wav.ndim == 1:
                            wav = wav.unsqueeze(0)
                        torchaudio.save(tmp_wav, wav.cpu().to(torch.float32), item_sr)
                        _mux_with_ffmpeg(tmp_video, tmp_wav, out_mp4)
                    else:
                        os.replace(tmp_video, out_mp4)
                except Exception as exc:
                    if os.path.exists(tmp_video):
                        os.replace(tmp_video, out_mp4)
                    else:
                        _write_video(out_mp4, video_uint8, fps)
                    if getattr(dataset_config, "debug", False):
                        print(f"Warning: failed to mux audio into mp4 for {out_mp4}: {exc}")
                finally:
                    for temporary in (tmp_video, tmp_wav):
                        try:
                            if os.path.exists(temporary):
                                os.remove(temporary)
                        except OSError:
                            pass
                time.sleep(0.2)
            idx += 1
        if epoch < epochs - 1:
            trigger_dataloader_setup_epoch(dataloader)
    print("done")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--output_path", type=str, default="output/dataset_test")
    args = parser.parse_args(argv)
    if args.output_path is None:
        raise ValueError("output_path is required for this test script")
    run_fixture(
        args.dataset_folder,
        epochs=args.epochs,
        num_frames=args.num_frames,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
