import copy

from toolkit.admission import collect_process_admission_diagnostics


def _process():
    return {
        "model": {"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
        "network": {"type": "lora", "linear": 4, "linear_alpha": 4},
        "train": {"optimizer": "adamw", "batch_size": 1},
        "datasets": [{"folder_path": "nonexistent-synthetic-source", "num_workers": 0}],
    }


def test_off_admission_preserves_existing_legacy_dop_diagnostics_and_resolution():
    process = _process()
    process.update(trigger_word="legacy")
    process["train"].update(
        diff_output_preservation=True, diff_output_preservation_multiplier=7
    )
    ordinary = collect_process_admission_diagnostics(process)
    process["identity_isolation"] = {"preset": "off"}
    assert collect_process_admission_diagnostics(process) == ordinary


def test_active_static_admission_reports_dop_conflict_without_source_access(
    monkeypatch,
):
    import toolkit.identity_isolation_data as integration

    process = _process()
    process["train"]["diff_output_preservation"] = False
    process["identity_isolation"] = {
        "preset": "full",
        "subjects": [
            {
                "folder_path": "nonexistent-synthetic-source",
                "trigger": "zwxem",
                "class_name": "woman",
            }
        ],
    }
    original = copy.deepcopy(process)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("static admission accessed source files")

    monkeypatch.setattr(integration, "_images", unexpected)
    result = collect_process_admission_diagnostics(process)
    assert not result["valid"]
    assert any(
        item["rule_id"] == "admission.identity_isolation"
        for item in result["diagnostics"]
    )
    assert process == original


def test_active_static_admission_rejects_incompatible_category_accumulation():
    process = _process()
    process["train"]["gradient_accumulation"] = 2
    process["identity_isolation"] = {
        "preset": "full",
        "subjects": [
            {
                "folder_path": "nonexistent-synthetic-source",
                "trigger": "zwxem",
                "class_name": "woman",
            }
        ],
    }
    result = collect_process_admission_diagnostics(process)
    assert not result["valid"]
    assert any(
        item["rule_id"] == "admission.identity_isolation"
        for item in result["diagnostics"]
    )
