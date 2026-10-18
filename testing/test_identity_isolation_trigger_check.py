# Offline trigger-word checker regressions: a fake greedy-piece tokenizer
# drives verdicts, suggestions, and the bridge action. No network, no torch.

import pytest

from toolkit.identity_isolation_trigger_check import (
    check_triggers,
    load_trigger_tokenizer,
    suggest_triggers,
    trigger_verdict,
)


class FakeTokenizer:
    """Greedy longest-match piece segmentation over a fixed vocabulary;
    characters with no vocabulary entry become single-character pieces."""

    def __init__(self, vocabulary):
        self.pieces = list(vocabulary)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for run in text.split():
            position = 0
            while position < len(run):
                for length in range(len(run) - position, 0, -1):
                    candidate = run[position : position + length]
                    if candidate in self.pieces:
                        ids.append(self.pieces.index(candidate))
                        position += length
                        break
                else:
                    self.pieces.append(run[position])
                    ids.append(len(self.pieces) - 1)
                    position += 1
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def _krea_process(tmp_path, subjects, **extra):
    return {
        "training_folder": str(tmp_path / "output"),
        "model": {"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
        "datasets": [
            {"folder_path": str(tmp_path / name), "num_workers": 0, "resolution": 32}
            for name, *_ in subjects
        ],
        "identity_isolation": {
            "preset": "full",
            "subjects": [
                {
                    "folder_path": str(tmp_path / name),
                    "trigger": trigger,
                    "class_name": class_name,
                }
                for name, trigger, class_name in subjects
            ],
            **extra,
        },
    }


def test_clean_join_is_ok_at_start_and_mid():
    tokenizer = FakeTokenizer(["zwx", "em", "woman"])
    verdict = trigger_verdict(tokenizer, "zwxem", "woman")
    assert verdict["ok"] is True
    assert verdict["notes"] == []
    assert verdict["joined"] == "zwxemwoman"
    assert verdict["pieces"] == ["zwx", "em", "woman"]
    assert verdict["start_pieces"] == ["zwx", "em", "woman"]


def test_trigger_that_breaks_the_class_word_is_not_ok():
    tokenizer = FakeTokenizer(["f", "yx", "ow", "oman", "woman"])
    verdict = trigger_verdict(tokenizer, "fyxo", "woman")
    assert verdict["ok"] is False
    assert verdict["pieces"] == ["f", "yx", "ow", "oman"]
    assert any("breaks 'woman'" in note for note in verdict["notes"])
    assert any("vowel" in note for note in verdict["notes"])  # fyxo ends in a vowel


def test_consonant_break_gets_the_ending_tip():
    # 'umm' spans the krum/man boundary, so 'man' never reappears whole
    tokenizer = FakeTokenizer(["kr", "umm", "an", "woman"])
    verdict = trigger_verdict(tokenizer, "krum", "man")
    assert verdict["ok"] is False
    assert any("runs into 'man'" in note for note in verdict["notes"])


def test_single_piece_trigger_warns_but_joins_clean():
    tokenizer = FakeTokenizer(["dlor", "woman"])
    verdict = trigger_verdict(tokenizer, "dlor", "woman")
    assert verdict["ok"] is True
    assert any("single piece" in note for note in verdict["notes"])


def test_same_first_letter_as_another_subject_is_not_ok():
    tokenizer = FakeTokenizer(["zwx", "em", "zw", "ilk", "woman", "man"])
    verdict = trigger_verdict(tokenizer, "zwxem", "woman", others=["zwilk"])
    assert verdict["ok"] is False
    assert any("same letter as 'zwilk'" in note for note in verdict["notes"])


def test_shared_piece_with_another_subject_is_not_ok():
    tokenizer = FakeTokenizer(["zwx", "em", "xor", "woman", "man"])
    verdict = trigger_verdict(tokenizer, "zwxem", "woman", others=["emxor"])
    assert verdict["ok"] is False
    assert any("shares the piece em with 'emxor'" in note for note in verdict["notes"])


def test_multiword_trigger_is_rejected():
    tokenizer = FakeTokenizer(["woman"])
    verdict = trigger_verdict(tokenizer, "two words", "woman")
    assert verdict["ok"] is False
    assert any("one word" in note for note in verdict["notes"])


def test_suggestions_are_clean_disjoint_and_deterministic():
    tokenizer = FakeTokenizer(["zwx", "em", "woman"])
    first = suggest_triggers(tokenizer, "woman", ["zwxem"], "fyxo", seed=7)
    second = suggest_triggers(tokenizer, "woman", ["zwxem"], "fyxo", seed=7)
    assert first == second
    assert first
    assert len(set(first)) == len(first)
    assert "fyxo" not in first
    assert all(word[0] != "z" for word in first)  # zwxem's first letter
    for word in first:
        verdict = trigger_verdict(tokenizer, word, "woman", ["zwxem", *first])
        assert verdict["ok"] is True
        assert verdict["notes"] == []
        assert len(verdict["pieces"]) >= 2


def test_check_triggers_reports_each_subject(tmp_path):
    tokenizer = FakeTokenizer(["zwx", "em", "f", "yx", "ow", "oman", "woman", "man"])
    process = _krea_process(
        tmp_path,
        [("subject-one", "zwxem", "woman"), ("subject-two", "fyxo", "man")],
    )
    result = check_triggers(process, tokenizer=tokenizer)
    assert result["encoder"] == "provided tokenizer"
    assert result["ok"] is False
    assert [entry["subject_index"] for entry in result["subjects"]] == [0, 1]
    clean, broken = result["subjects"]
    assert clean["ok"] is True and clean["suggestions"] == []
    assert broken["ok"] is False and broken["suggestions"]
    assert broken["class_name"] == "man"


def test_check_triggers_parses_subjects_under_inactive_preset(tmp_path):
    tokenizer = FakeTokenizer(["zwx", "em", "woman"])
    process = _krea_process(tmp_path, [("subject-one", "zwxem", "woman")])
    process["identity_isolation"]["preset"] = "off"
    process["identity_isolation"]["fused_captions"] = False
    result = check_triggers(process, tokenizer=tokenizer)
    assert len(result["subjects"]) == 1
    assert result["ok"] is True


def test_check_triggers_requires_subjects(tmp_path):
    process = _krea_process(tmp_path, [("subject-one", "zwxem", "woman")])
    process["identity_isolation"]["subjects"] = []
    with pytest.raises(ValueError, match="subject"):
        check_triggers(process, tokenizer=FakeTokenizer(["woman"]))


def test_check_triggers_rejects_other_architectures(tmp_path):
    process = _krea_process(tmp_path, [("subject-one", "zwxem", "woman")])
    process["model"]["arch"] = "flux"
    with pytest.raises(ValueError, match="Krea 2"):
        check_triggers(process)


def test_load_trigger_tokenizer_is_offline_only(tmp_path):
    process = _krea_process(tmp_path, [("subject-one", "zwxem", "woman")])
    process["model"]["model_kwargs"] = {
        "text_encoder_path": str(tmp_path / "not-cached")
    }
    with pytest.raises(ValueError, match="not available offline"):
        load_trigger_tokenizer(process)


def test_bridge_action_trigger_check(tmp_path, monkeypatch):
    from toolkit import identity_isolation_trigger_check
    from ui_scripts.identity_isolation import inspect_request

    tokenizer = FakeTokenizer(["zwx", "em", "woman"])
    monkeypatch.setattr(
        identity_isolation_trigger_check,
        "load_trigger_tokenizer",
        lambda _process: (tokenizer, "fake tokenizer"),
    )
    process = _krea_process(tmp_path, [("subject-one", "zwxem", "woman")])
    result = inspect_request({"action": "trigger_check", "process": process})
    check = result["trigger_check"]
    assert check["ok"] is True
    assert check["subjects"][0]["joined"] == "zwxemwoman"
    with pytest.raises(ValueError, match="action must be"):
        inspect_request({"action": "nonsense", "process": process})
