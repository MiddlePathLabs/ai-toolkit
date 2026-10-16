"""validate_dataset_captions: fail fast on uncaptioned media, and
decode_caption_file: tolerant caption sidecar decoding."""

from types import SimpleNamespace

import pytest

from toolkit.dataloader_mixins import (
    decode_caption_file,
    validate_dataset_captions,
)


def _cfg(**kw):
    base = dict(caption_ext='.txt', default_caption=None, allow_uncaptioned=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _item(path):
    return SimpleNamespace(path=path)


# ----------------------------------------------------------- decode_caption_file

def test_decode_utf8(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("a photo of the subject".encode("utf-8"))
    assert decode_caption_file(str(p)) == "a photo of the subject"


def test_decode_utf8_with_bom(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"\xef\xbb\xbfsubject")
    assert decode_caption_file(str(p)) == "subject"


def test_decode_utf16_bom(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("subject".encode("utf-16"))
    assert decode_caption_file(str(p)) == "subject"


def test_decode_cp1252(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"caf\xe9")  # é in cp1252 is invalid utf-8
    assert decode_caption_file(str(p)) == "café"


def test_decode_binary_never_crashes(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(bytes(range(256)))
    out = decode_caption_file(str(p))  # latin-1 decodes any byte string
    assert isinstance(out, str) and len(out) == 256


# ------------------------------------------------- validate_dataset_captions

def test_uncaptioned_file_raises_and_names_it(tmp_path):
    img = tmp_path / "photo_0001.jpg"
    img.write_bytes(b"x")
    (tmp_path / "ok.jpg").write_bytes(b"x")
    (tmp_path / "ok.txt").write_text("captioned")
    with pytest.raises(ValueError, match="photo_0001.jpg"):
        validate_dataset_captions(
            [_item(str(img)), _item(str(tmp_path / "ok.jpg"))],
            None, _cfg(), str(tmp_path),
        )


def test_json_caption_entry_counts_as_captioned():
    items = [_item("/ds/a.jpg")]
    validate_dataset_captions(items, {"/ds/a.jpg": "captioned"}, _cfg(), "/ds")


def test_json_dict_entry_with_caption_counts():
    validate_dataset_captions(
        [_item("/ds/a.jpg")], {"/ds/a.jpg": {"caption": "c"}}, _cfg(), "/ds"
    )


def test_json_entry_without_caption_does_not_count():
    with pytest.raises(ValueError):
        validate_dataset_captions(
            [_item("/ds/a.jpg")], {"/ds/a.jpg": {"something": "else"}}, _cfg(), "/ds"
        )


def test_default_caption_is_an_opt_out():
    validate_dataset_captions([_item("/ds/a.jpg")], None, _cfg(default_caption="x"), "/ds")


def test_allow_uncaptioned_is_an_opt_out():
    validate_dataset_captions([_item("/ds/a.jpg")], None, _cfg(allow_uncaptioned=True), "/ds")


def test_caption_ext_none_disables_the_guard():
    validate_dataset_captions([_item("/ds/a.jpg")], None, _cfg(caption_ext=None), "/ds")


def test_error_lists_count_and_more():
    paths = [f"/ds/img_{i:04d}.jpg" for i in range(9)]
    with pytest.raises(ValueError, match=r"9 file\(s\).*and 3 more"):
        validate_dataset_captions([_item(p) for p in paths], None, _cfg(), "/ds")
