"""D-OPSD other-photo recipe from musubi-tuner's `subject_ref` teacher."""
from types import SimpleNamespace

from toolkit.config_modules import ModelConfig
from toolkit.dataloader_mixins import CaptionProcessingDTOMixin
from toolkit.h3_dopsd import (
    SUBJECT_REF_TOKEN,
    parse_dopsd_settings,
    recipe_warnings,
    wrap_subject_reference_caption,
)


def _cfg(**kw):
    return ModelConfig(name_or_path="x", arch="minimax_h3_ref2va", model_kwargs=kw)


def test_musubi_recipe_other():
    s = parse_dopsd_settings(_cfg(dopsd=True, dopsd_ref_mode="other", dopsd_musubi_recipe=True))
    assert (s.teacher_sigma_max, s.teacher_sigma_min) == (1.0, 0.15)
    assert (s.loss_mag_weight, s.loss_dc_weight) == (0.5, 0.3)
    assert s.subject_declaration and not s.copy_declaration
    assert recipe_warnings(s) == []


def test_musubi_recipe_self():
    s = parse_dopsd_settings(_cfg(dopsd=True, dopsd_musubi_recipe=True))
    assert s.teacher_sigma_max == 0.75 and s.loss_dc_weight == 0.3
    assert s.copy_declaration and not s.subject_declaration
    assert s.loss_mag_weight == 1.0
    assert recipe_warnings(s) == []


def test_explicit_keys_win_over_the_recipe():
    s = parse_dopsd_settings(_cfg(
        dopsd=True, dopsd_ref_mode="other", dopsd_musubi_recipe=True, dopsd_loss_dc_weight=1.0))
    assert s.loss_dc_weight == 1.0 and s.loss_mag_weight == 0.5


def test_recipe_is_off_by_default():
    s = parse_dopsd_settings(_cfg(dopsd=True, dopsd_ref_mode="other"))
    assert s.teacher_sigma_min == 0.0 and s.loss_mag_weight == 1.0
    assert not s.subject_declaration


def test_warnings_for_bands_that_contradict_the_mode():
    other = parse_dopsd_settings(_cfg(dopsd=True, dopsd_ref_mode="other",
                                      dopsd_teacher_sigma_max=0.75))
    assert any("identity" in w for w in recipe_warnings(other))
    self_ref = parse_dopsd_settings(_cfg(dopsd=True))  # ungated self-ref: default sigma_max 1.0
    assert any("0.85" in w for w in recipe_warnings(self_ref))


def test_subject_wrap_text():
    text = wrap_subject_reference_caption(f"{SUBJECT_REF_TOKEN} smiling at a cafe.", still_image=True)
    assert text.startswith("subject_definitions:\n<Subject 1> is the subject whose appearance comes from <Picture 1>")
    assert "attribute_transfer" in text and "pose, framing, outfit and setting follow the description" in text
    assert "single still image" in text
    assert text.endswith("detailed_description:\n<Subject 1> smiling at a cafe.")


def test_other_photo_token_becomes_subject():
    item = SimpleNamespace(is_video=False, dopsd_other_ref=True, dopsd_subject_declaration=True)
    assert CaptionProcessingDTOMixin.get_dopsd_ref_token(item) == "<Subject 1>"
    item.dopsd_subject_declaration = False
    assert CaptionProcessingDTOMixin.get_dopsd_ref_token(item) == "<Picture 1>"
