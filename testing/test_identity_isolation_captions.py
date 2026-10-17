"""Synthetic caption/configuration coverage; no model, files, or network required."""

import builtins
import hashlib
import json
import os

import pytest

from toolkit.identity_isolation import IsolationConfig


def subject(trigger="zwxem", class_name="woman", folder_path="synthetic/subject"):
    return dict(folder_path=folder_path, trigger=trigger, class_name=class_name)


def config(**overrides):
    raw = dict(preset="full", subjects=[subject()])
    raw.update(overrides)
    return IsolationConfig(raw)


@pytest.mark.parametrize(
    "original, training, preservation",
    [
        ("zwxem, a woman smiling", "zwxemwoman smiling", "a woman smiling"),
        ("zwxem a woman smiling", "zwxemwoman smiling", "a woman smiling"),
        ("ZWXEM, A WOMAN smiling", "zwxemwoman smiling", "a woman smiling"),
        ("zwxem smiling", "zwxemwoman smiling", "a woman smiling"),
        ("(zwxem), zwxem!", "(zwxemwoman), zwxemwoman!", "(a woman), a woman!"),
        ("zwxemwoman smiling", "zwxemwoman smiling", "a woman smiling"),
        ("ZWXEMWOMAN smiling", "ZWXEMWOMAN smiling", "a woman smiling"),
        ("zwxem, a young woman", "zwxemwoman, a young woman", "a woman, a young woman"),
        ("zwxem a young woman", "zwxemwoman a young woman", "a woman a young woman"),
        ("zwxem, the woman", "zwxemwoman, the woman", "a woman, the woman"),
        ("zwxem,  a woman", "zwxemwoman,  a woman", "a woman,  a woman"),
        (
            "prezwxem zwxem_suffix zwxem2",
            "prezwxem zwxem_suffix zwxem2",
            "prezwxem zwxem_suffix zwxem2",
        ),
        ("ézwxem zwxemé", "ézwxem zwxemé", "ézwxem zwxemé"),
        (
            "  a stranger,  smiling  ",
            "  a stranger,  smiling  ",
            "  a stranger,  smiling  ",
        ),
        ("", "", ""),
    ],
)
def test_caption_matrix(original, training, preservation):
    isolation = config()
    assert isolation.training_caption(original) == training
    assert isolation.training_caption(training) == training
    assert isolation.preservation_caption(original) == preservation
    assert isolation.preservation_caption(training) == preservation
    assert isolation.preservation_caption(preservation) == preservation


def test_multiple_subjects_and_repeated_raw_fused_occurrences():
    isolation = config(subjects=[subject(), subject("qrx", "man", "synthetic/second")])
    caption = "qrx, a man beside zwxem a woman; qrxman greets zwxem and qrx."
    assert (
        isolation.training_caption(caption)
        == "qrxman beside zwxemwoman; qrxman greets zwxemwoman and qrxman."
    )
    assert (
        isolation.preservation_caption(caption)
        == "a man beside a woman; a man greets a woman and a man."
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_multiword_triggers_are_longest_first(reverse):
    subjects = [
        subject("nova", "woman"),
        subject("nova rose", "man", "synthetic/second"),
    ]
    isolation = config(subjects=subjects[::-1] if reverse else subjects)
    caption = "nova rose, a man and nova, a woman greet nova rose and nova."
    assert (
        isolation.training_caption(caption)
        == "novaroseman and novawoman greet novaroseman and novawoman."
    )
    assert (
        isolation.preservation_caption(caption)
        == "a man and a woman greet a man and a woman."
    )


def test_trigger_regex_metacharacters_are_literal():
    isolation = config(subjects=[subject("z.w+x")])
    assert isolation.training_caption("z.w+x, a woman; zXwwx") == "z.w+xwoman; zXwwx"
    assert (
        isolation.preservation_caption("z.w+xwoman; z.w+x a woman")
        == "a woman; a woman"
    )


@pytest.mark.parametrize("class_name", ["woman", "a woman", "an woman", " A WOMAN "])
def test_explicit_class_normalizes_article_and_case(class_name):
    normalized = config(subjects=[subject(class_name=class_name)]).subjects[0]
    assert normalized["class_name"] == "woman"
    assert normalized["phrase"] == "a woman"
    assert normalized["fused"] == "zwxemwoman"


def test_vowel_class_and_whitespace_in_trigger():
    isolation = config(subjects=[subject("z wx", "a owl")])
    assert isolation.subjects[0]["phrase"] == "an owl"
    assert isolation.training_caption("z wx, an owl") == "zwxowl"
    assert (
        isolation.preservation_caption("zwxowl and z wx an owl") == "an owl and an owl"
    )


@pytest.mark.parametrize("raw", [None, {}, {"preset": "off"}])
def test_off_is_inert(raw):
    isolation = IsolationConfig(raw)
    assert not isolation.active
    assert not isolation.fused_captions
    assert not isolation.composites
    assert not isolation.dop_enabled
    assert isolation.caption_preview("zwxem a woman") == {
        "original": "zwxem a woman",
        "training": "zwxem a woman",
        "preservation": "zwxem a woman",
    }


def test_off_with_stored_subjects_does_not_rewrite_captions():
    isolation = config(preset="off")
    assert isolation.training_caption("zwxem, a woman") == "zwxem, a woman"
    assert isolation.preservation_caption("zwxemwoman") == "zwxemwoman"


def test_full_preset_defaults():
    isolation = config()
    assert (
        isolation.active
        and isolation.fused_captions
        and isolation.composites
        and isolation.dop_enabled
    )
    assert isolation.composite_count == 16
    assert isolation.composite_lr_multiplier == 2.0
    assert isolation.dop_interval == 2
    assert isolation.dop_weight == 2.0
    assert isolation.seed == 42
    assert not isolation.balance_subjects


def test_dop_only_preset_and_no_subject_legacy_route():
    isolation = IsolationConfig({"preset": "dop_only"})
    assert isolation.active and isolation.dop_enabled
    assert not isolation.fused_captions and not isolation.composites
    assert isolation.dop_interval == 1 and isolation.dop_weight == 1.0
    assert isolation.preservation_caption("a stranger") == "a stranger"
    mapped = config(preset="dop_only")
    assert mapped.training_caption("zwxem, a woman") == "zwxem, a woman"
    assert mapped.preservation_caption("zwxem, a woman") == "a woman"


def test_explicit_component_overrides_and_fusion_only():
    isolation = config(
        preset="off", fused_captions=True, dop_interval=3, dop_weight=0.5
    )
    assert isolation.active and isolation.fused_captions
    assert not isolation.composites and not isolation.dop_enabled
    assert isolation.training_caption("zwxem") == "zwxemwoman"
    assert isolation.dop_interval == 3 and isolation.dop_weight == 0.5
    disabled = IsolationConfig(
        {
            "preset": "full",
            "fused_captions": False,
            "composites": False,
            "dop_enabled": False,
        }
    )
    assert not disabled.active


def test_caption_preview_reports_both_representations():
    assert config().caption_preview("zwxem, a woman") == {
        "original": "zwxem, a woman",
        "training": "zwxemwoman",
        "preservation": "a woman",
    }


@pytest.mark.parametrize("count", [0, 5])
@pytest.mark.parametrize("component", ["fused_captions", "composites"])
def test_caption_and_composite_subject_count_limits(count, component):
    subjects = [
        subject(f"trigger{i}", folder_path=f"synthetic/{i}") for i in range(count)
    ]
    with pytest.raises(ValueError):
        IsolationConfig({"subjects": subjects, component: True})


def test_four_subjects_are_supported():
    isolation = config(
        subjects=[
            subject(f"trigger{i}", folder_path=f"synthetic/{i}") for i in range(4)
        ]
    )
    assert (
        isolation.training_caption("trigger0 and trigger3")
        == "trigger0woman and trigger3woman"
    )


@pytest.mark.parametrize(
    "subjects, message",
    [
        (
            [subject(), subject("ZWXEM", "man", "synthetic/second")],
            "duplicate triggers",
        ),
        (
            [subject("z wx"), subject("zwx", "woman", "synthetic/second")],
            "fused collisions",
        ),
        (
            [subject(), subject("zwxemwoman", "man", "synthetic/second")],
            "cross-mapping alias",
        ),
        (
            [subject("person"), subject("other", "person", "synthetic/second")],
            "cross-mapping alias",
        ),
        ([subject("woman")], "cross-mapping alias"),
        (
            [subject("x"), subject("xwoman-y", "man", "synthetic/second")],
            "cascading fused aliases",
        ),
        (
            [subject(), subject("other", "man", "synthetic/subject/../subject")],
            "duplicate source folders",
        ),
    ],
)
def test_ambiguous_mappings_are_rejected(subjects, message):
    with pytest.raises(ValueError, match=message):
        config(subjects=subjects)


def test_duplicate_folders_follow_platform_case_rules():
    if os.path.normcase("SYNTHETIC") != os.path.normcase("synthetic"):
        pytest.skip("Case-sensitive platform paths are distinct")
    with pytest.raises(ValueError, match="duplicate source folders"):
        config(subjects=[subject(), subject("other", "man", "SYNTHETIC/SUBJECT")])


@pytest.mark.parametrize(
    "class_name",
    ["", "young woman", "a young woman", "woman/man", "woman_2", "2", "a", "an", None],
)
def test_invalid_explicit_classes(class_name):
    with pytest.raises(ValueError, match="class_name"):
        config(subjects=[subject(class_name=class_name)])


@pytest.mark.parametrize("field", ["trigger", "folder_path"])
@pytest.mark.parametrize("value", ["", "  ", None, 12])
def test_subject_strings_must_be_nonempty(field, value):
    mapping = subject()
    mapping[field] = value
    with pytest.raises(ValueError, match=field):
        config(subjects=[mapping])


@pytest.mark.parametrize("field", ["composite_count", "dop_interval"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None, float("inf")])
def test_positive_integer_settings(field, value):
    with pytest.raises(ValueError, match=field):
        config(**{field: value})


@pytest.mark.parametrize("value", [-1, 1.5, True, "42", None])
def test_seed_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="seed"):
        config(seed=value)


@pytest.mark.parametrize("field", ["composite_lr_multiplier", "dop_weight"])
@pytest.mark.parametrize(
    "value",
    [0, -1, float("nan"), float("inf"), -float("inf"), True, "2", None, 10**1000],
)
def test_weight_settings_are_finite_and_positive(field, value):
    with pytest.raises(ValueError, match=field):
        config(**{field: value})


@pytest.mark.parametrize(
    "field", ["fused_captions", "composites", "dop_enabled", "balance_subjects"]
)
def test_component_settings_require_booleans(field):
    with pytest.raises(ValueError, match=field):
        config(**{field: "false"})


@pytest.mark.parametrize(
    "raw", [[], "full", {"preset": "unknown"}, {"subjects": {}}, {"subjects": [None]}]
)
def test_invalid_top_level_configuration(raw):
    with pytest.raises(ValueError):
        IsolationConfig(raw)


def test_signature_is_canonical_resolved_json_and_round_trips():
    isolation = config()
    reordered = IsolationConfig(dict(reversed(list(isolation.as_dict().items()))))
    assert isolation.signature == reordered.signature
    canonical = json.dumps(
        isolation.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert isolation.signature == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert config(dop_weight=3).signature != isolation.signature
    assert config(subjects=[subject("different")]).signature != isolation.signature
    propagated = isolation.as_dict()
    propagated["subjects"][0]["trigger"] = "changed"
    assert isolation.raw["subjects"][0]["trigger"] == "zwxem"


def test_parsing_and_caption_transforms_do_not_access_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration must not access files")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(os, "makedirs", forbidden)
    monkeypatch.setattr(os.path, "exists", forbidden)
    assert not IsolationConfig().active
    isolation = config()
    assert isolation.training_caption("zwxem") == "zwxemwoman"


def test_unicode_ignorecase_equivalent_triggers_are_duplicates():
    with pytest.raises(ValueError, match="duplicate triggers"):
        config(subjects=[subject("iris"), subject("ırıs", "man", "synthetic/second")])


@pytest.mark.parametrize("field", ["stranger_asset_dir", "generated_dir"])
@pytest.mark.parametrize("value", ["", "  ", None, 12])
def test_asset_directories_require_nonempty_strings(field, value):
    with pytest.raises(ValueError, match=field):
        config(**{field: value})


def test_resolved_config_does_not_mutate_input():
    mapping = subject(class_name="A WOMAN")
    raw = {"preset": "full", "subjects": [mapping]}
    isolation = IsolationConfig(raw)
    assert mapping == subject(class_name="A WOMAN")
    assert set(raw) == {"preset", "subjects"}
    assert isolation.subjects[0]["class_name"] == "woman"


def test_off_accepts_unfinished_subject_drafts_without_applying_them():
    isolation = IsolationConfig(
        {
            "preset": "off",
            "subjects": [
                {"folder_path": "", "trigger": "", "class_name": ""},
            ],
        }
    )
    assert not isolation.active
    assert isolation.caption_preview("zwxem, a woman") == {
        "original": "zwxem, a woman",
        "training": "zwxem, a woman",
        "preservation": "zwxem, a woman",
    }


def test_dop_mappings_reject_more_than_four_subjects():
    with pytest.raises(ValueError):
        IsolationConfig(
            {
                "preset": "dop_only",
                "subjects": [
                    subject(f"trigger{i}", folder_path=f"synthetic/{i}")
                    for i in range(5)
                ],
            }
        )
