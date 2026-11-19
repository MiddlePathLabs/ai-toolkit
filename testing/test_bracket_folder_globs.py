"""Folder names containing [ ] are glob character classes: an unescaped
``glob.glob(os.path.join(folder, "*"))`` over a dataset folder named
"photos [v2]" matches nothing, so the same-folder control/clip pools and the
inference engine's output listing came back silently empty (Fizgig 7b51bdd2
swept the same bug class repo-wide). The folder half is now glob.escape'd at
every data-facing site."""

import os

from toolkit.data_transfer_object.data_loader import FileItemDTO
from extensions_built_in.inference_engine.engine import _list_outputs


class _DatasetConfig:
    control_from_same_folder = True
    num_controls_from_same_folder = 1
    clip_image_from_same_folder = True


def _dto_in(folder, name):
    item = object.__new__(FileItemDTO)
    item.dataset_config = _DatasetConfig()
    item.path = os.path.join(str(folder), name)
    item.clip_image_path = None
    return item


def test_control_pool_survives_bracket_folder_name(tmp_path):
    folder = tmp_path / "photos [v2]"
    folder.mkdir()
    for i in range(3):
        (folder / f"img{i}.png").write_bytes(b"x")

    item = _dto_in(folder, "img0.png")
    picks = set()
    for _ in range(30):
        picks.update(item.get_new_control_paths())
    # the pool holds the folder's other files, not nothing
    assert picks == {
        os.path.join(str(folder), f"img{i}.png") for i in (1, 2)
    }


def test_clip_image_pool_survives_bracket_folder_name(tmp_path):
    folder = tmp_path / "clips [test]"
    folder.mkdir()
    for i in range(2):
        (folder / f"c{i}.jpg").write_bytes(b"x")

    item = _dto_in(folder, "c0.jpg")
    picks = {item.get_new_clip_image_path() for _ in range(20)}
    assert picks == {os.path.join(str(folder), "c1.jpg")}


def test_inference_output_listing_survives_bracket_folder_name(tmp_path):
    folder = tmp_path / "out [v1]"
    folder.mkdir()
    (folder / "sample_00001.png").write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"x")

    listed = _list_outputs(str(folder))
    assert [os.path.basename(p) for p in listed] == ["sample_00001.png"]
