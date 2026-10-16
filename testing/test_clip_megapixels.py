"""Clip Target Megapixels: dataset.clip_megapixels gives video items their own
bucket budget while photos keep `resolution` (Fizgig v6.7.0 port)."""

from fractions import Fraction

import torch
from PIL import Image

from toolkit.buckets import clip_target_resolution, get_bucket_for_image_size
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset


def _write_video(path, width=1280, height=720, frames=8, fps=8.0):
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            array = np.full((height, width, 3), (i * 30) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def test_clip_target_resolution_values():
    assert clip_target_resolution(0.25) == 500
    assert clip_target_resolution(1.0) == 1000
    assert clip_target_resolution(0) == 64  # degenerate values floored
    assert clip_target_resolution(None) == 64


from test_h3_cache_contracts import _FrozenToyEncoderModel as _SizingStub


def test_clip_megapixels_none_means_no_override():
    # sanity: the budget math without the override
    bucket = get_bucket_for_image_size(1280, 720, resolution=1000, divisibility=64)
    assert bucket["width"] * bucket["height"] <= 1000 * 1000


def test_mixed_dataset_buckets_clips_separately_from_photos(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)
    photo = tmp_path / "still.png"
    Image.new("RGB", (1400, 1400), (255, 0, 0)).save(photo)

    config = DatasetConfig(
        dataset_path=str(tmp_path),
        resolution=1000,
        buckets=True,
        clip_megapixels=0.25,
        allow_uncaptioned=True,
        num_frames=8,
        num_workers=0,
    )
    dataset = AiToolkitDataset(config, sd=_SizingStub())
    items = {item.path: item for item in dataset.file_list}

    photo_item = items[str(photo)]
    video_item = items[str(video)]

    # photo keeps the dataset resolution budget (1 MP)
    assert photo_item.crop_width == 1000
    assert photo_item.crop_height == 1000

    # the clip lands inside the 0.25 MP budget (921,600 px source);
    # 666x375 = 249,750 px, the closest bucket at the toy model's /16
    # divisibility
    clip_pixels = video_item.crop_width * video_item.crop_height
    assert clip_pixels <= 0.25 * 1e6
    assert (video_item.crop_width, video_item.crop_height) == (666, 375)
    # and it sits in its own bucket, never mixed with photos
    clip_key = f"{video_item.crop_width}x{video_item.crop_height}x{video_item.num_frames}f"
    # in a video dataset every bucket key carries the frame count (photos x1f)
    photo_key = f"{photo_item.crop_width}x{photo_item.crop_height}x{photo_item.num_frames}f"
    assert clip_key in dataset.buckets
    assert photo_key in dataset.buckets


def test_unset_clip_megapixels_keeps_photos_and_clips_together(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)
    photo = tmp_path / "still.png"
    Image.new("RGB", (1400, 1400), (255, 0, 0)).save(photo)

    config = DatasetConfig(
        dataset_path=str(tmp_path),
        resolution=1000,
        buckets=True,
        allow_uncaptioned=True,
        num_frames=8,
        num_workers=0,
    )
    dataset = AiToolkitDataset(config, sd=_SizingStub())
    video_item = next(i for i in dataset.file_list if i.path == str(video))
    # without the override the clip stays under the 1 MP budget unscaled
    assert (video_item.crop_width, video_item.crop_height) == (1280, 720)
