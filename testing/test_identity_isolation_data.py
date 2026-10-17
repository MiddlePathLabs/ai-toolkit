import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.identity_isolation_data import (
    configure_isolation,
    inspect_isolation,
    isolation_exposure_report,
)


def _source(tmp_path, name, count, caption):
    folder = tmp_path / name
    folder.mkdir()
    for index in range(count):
        Image.new("RGB", (32, 16), (index, 30, 90)).save(folder / f"{index}.png")
        (folder / f"{index}.txt").write_text(caption, encoding="utf-8")
    return folder


def _process(tmp_path, counts=(2, 5), **settings):
    subjects, datasets = [], []
    for index, count in enumerate(counts):
        trigger = f"person{index}"
        folder = _source(
            tmp_path, f"subject{index}", count, "person0 and person1 outdoors"
        )
        subjects.append(
            {"folder_path": str(folder), "trigger": trigger, "class_name": "person"}
        )
        datasets.append(
            {"folder_path": str(folder), "resolution": 32, "num_workers": 0}
        )
    return {
        "identity_isolation": {
            "preset": "off",
            "fused_captions": True,
            "subjects": subjects,
            **settings,
        },
        "datasets": datasets,
        "train": {"steps": 100, "batch_size": 1},
        "model": {"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
        "network": {"type": "lora", "linear": 4, "linear_alpha": 4},
    }


def _item(dataset, *, cached=False, dropout=0):
    dataset = {
        **dataset,
        "cache_text_embeddings": cached,
        "caption_dropout_rate": dropout,
    }
    config = DatasetConfig(**dataset)
    path = sorted(Path(config.folder_path).glob("*.png"))[0]
    item = FileItemDTO(path=str(path), dataset_config=config)
    item.load_caption()
    return item


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"preset": "off"},
        {
            "preset": "full",
            "fused_captions": False,
            "composites": False,
            "dop_enabled": False,
        },
    ],
)
def test_disabled_configuration_is_mutation_filesystem_and_rng_inert(
    tmp_path, monkeypatch, raw
):
    import toolkit.identity_isolation_data as integration

    process = {
        "identity_isolation": raw,
        "datasets": [{"folder_path": "missing"}],
        "train": {
            "diff_output_preservation": True,
            "diff_output_preservation_multiplier": 7,
        },
    }
    original = copy.deepcopy(process)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("disabled isolation accessed source data")

    monkeypatch.setattr(integration, "_images", unexpected)
    monkeypatch.setattr(integration, "_digest_file", unexpected)
    assert configure_isolation(process) is None
    assert process == original
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_every_mapped_dataset_transforms_all_subjects_and_preserves_cached_live_parity(
    tmp_path,
):
    process = _process(tmp_path, dop_enabled=True)
    process["datasets"][0]["resolution"] = [16, 32]
    process["datasets"].append({**process["datasets"][1], "resolution": 16})
    configure_isolation(process)
    for dataset in preprocess_dataset_raw_config(process["datasets"]):
        live, cached = _item(dataset), _item(dataset, cached=True)
        assert (
            live.caption == cached.caption == "person0person and person1person outdoors"
        )
        assert (
            live.caption_dop == cached.caption_dop == "a person and a person outdoors"
        )
        assert live.dataset_config.identity_isolation is not None
        assert live.get_caption() == live.caption


@pytest.mark.parametrize("category", ["normal", "composite"])
def test_caption_dropout_matches_cached_training_and_preservation(tmp_path, category):
    process = _process(tmp_path, dop_enabled=True)
    config = configure_isolation(process)
    dataset = process["datasets"][0]
    dataset["isolation_category"] = category
    if category == "composite":
        dataset["trigger_word"] = None
    live, cached = _item(dataset, dropout=1), _item(dataset, cached=True, dropout=1)
    assert live.caption == cached.get_dropout_caption()
    assert live.caption_dop == cached.get_dop_dropout_caption()
    assert live.caption_dop == config.preservation_caption(live.caption)


def test_already_fused_caption_does_not_receive_raw_trigger_again(tmp_path):
    process = _process(tmp_path, counts=(1,))
    configure_isolation(process)
    folder = process["datasets"][0]["folder_path"]
    from pathlib import Path

    (Path(folder) / "0.txt").write_text("person0person outdoors", encoding="utf-8")
    assert _item(process["datasets"][0]).caption == "person0person outdoors"


def test_fusion_only_preserves_independent_legacy_dop_with_mapping_aware_teacher(
    tmp_path,
):
    process = _process(tmp_path)
    process["train"]["diff_output_preservation"] = True
    process["train"]["diff_output_preservation_multiplier"] = 3
    config = configure_isolation(process)
    assert not config.dop_enabled
    assert process["train"]["diff_output_preservation_multiplier"] == 3
    item = _item(process["datasets"][0])
    assert item.caption_dop == "a person and a person outdoors"


@pytest.mark.parametrize(
    "train,error",
    [
        ({"diff_output_preservation": False}, "conflicts"),
        ({"diff_output_preservation_multiplier": 9}, "multiplier conflicts"),
        ({"diff_output_preservation_class": "woman"}, "subject classes"),
    ],
)
def test_dop_contradictions_are_actionable_and_do_not_mutate_input(
    tmp_path, train, error
):
    process = _process(tmp_path, dop_enabled=True)
    process["train"].update(train)
    original = copy.deepcopy(process)
    with pytest.raises(ValueError, match=error):
        configure_isolation(process)
    assert process == original


def test_no_subject_dop_only_uses_existing_legacy_trigger_and_class(tmp_path):
    folder = _source(tmp_path, "legacy", 1, "legacy outdoors")
    process = {
        "identity_isolation": {"preset": "dop_only"},
        "datasets": [
            {
                "folder_path": str(folder),
                "num_workers": 0,
                "trigger_word": "legacy",
                "diff_output_preservation_class": "person",
            }
        ],
        "train": {},
        "model": {"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
        "network": {"type": "lora", "linear": 4, "linear_alpha": 4},
    }
    config = configure_isolation(process)
    assert config.subjects == []
    assert _item(process["datasets"][0]).caption_dop == "person outdoors"


def test_text_cache_namespace_changes_only_when_isolation_is_active(tmp_path):
    process = _process(tmp_path)
    ordinary = _item(process["datasets"][0])
    off = _item({**process["datasets"][0], "identity_isolation": {"preset": "off"}})
    assert ordinary.get_text_embedding_info_dict() == off.get_text_embedding_info_dict()
    assert ordinary.get_text_embedding_path() == off.get_text_embedding_path()
    configure_isolation(process)
    active = _item(process["datasets"][0])
    first = active.get_text_embedding_info_dict()
    assert (
        first["identity_isolation"]
        == active.dataset_config.identity_isolation.signature
    )
    assert first["identity_isolation_caption_version"] == 1
    changed = copy.deepcopy(process["datasets"][0])
    changed["identity_isolation"]["seed"] += 1
    assert (
        _item(changed).get_text_embedding_info_dict()["identity_isolation"]
        != first["identity_isolation"]
    )


def test_balancing_multiplies_raw_repeats_and_reports_resolution_exposures(tmp_path):
    process = _process(tmp_path, counts=(2, 5), balance_subjects=True)
    process["datasets"][0].update(num_repeats=3, resolution=[16, 32])
    configure_isolation(process)
    first = process["datasets"][0]
    assert first["isolation_raw_repeats"] == 3
    assert (
        first["isolation_balance_multiplier"] == 2
    )  # Python round(2.5), matching Fizgig.
    assert first["num_repeats"] == 6
    report = isolation_exposure_report(process)
    assert report["effective_count"] == 29
    assert report["per_subject_probabilities"][0] == pytest.approx(24 / 29)
    assert report["expected_total_sample_exposures"] == 100


@pytest.mark.parametrize(
    "change,error",
    [
        ({"gradient_accumulation": 2}, "gradient_accumulation"),
        ({"gradient_accumulation_steps": 2}, "gradient_accumulation"),
        ({"pool_datasets": True}, "pool_datasets"),
        ({"batch_size": 2}, "batch_size"),
    ],
)
def test_composite_lr_rejects_mixed_optimizer_update_modes(tmp_path, change, error):
    process = _process(tmp_path, composites=True)
    process["train"].update(change)
    for dataset in process["datasets"]:
        dataset["buckets"] = False
    with pytest.raises(ValueError, match=error):
        configure_isolation(process, prepare=False)


def test_composite_auxiliary_conflicts_are_not_silently_removed(tmp_path):
    process = _process(tmp_path, composites=True)
    process["datasets"][0]["mask_path"] = "explicit-mask"
    with pytest.raises(ValueError, match="explicit auxiliaries"):
        configure_isolation(process, prepare=False)


def test_composites_clone_preprocessing_and_resolution_but_not_trigger_or_repeats(
    tmp_path,
):
    process = _process(tmp_path, composites=True, balance_subjects=True)
    process["datasets"][0].update(
        resolution=[16, 32], num_repeats=3, cache_latents_to_disk=True, random_crop=True
    )
    configure_isolation(process, prepare=False)
    source, composite = process["datasets"][0], process["datasets"][2]
    assert composite["resolution"] == source["resolution"]
    assert composite["cache_latents_to_disk"] is True
    assert composite["random_crop"] is True
    assert composite["num_repeats"] == 1
    assert composite["trigger_word"] is None
    assert composite["isolation_category"] == "composite"
    assert composite["isolation_subject_index"] == 0
    assert composite["isolation_original_count"] == 16


def test_inspect_never_prepares_assets_and_leaves_process_unmodified(
    tmp_path, monkeypatch
):
    import toolkit.identity_isolation_composites as composites

    process = _process(tmp_path, composites=True)
    original = copy.deepcopy(process)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("inspect generated composites")

    monkeypatch.setattr(composites, "prepare_composites", unexpected)
    preview = inspect_isolation(process)
    assert process == original
    assert preview["resolved"]["composites"] is True
    assert len(preview["captions"]) == 7
    assert preview["exposure"]["composite_count"] == 32
    assert preview["captions"][0]["subject_index"] == 0


def test_resume_recipe_changes_on_source_bytes_or_caption_content(tmp_path):
    process = _process(tmp_path)
    first, second, third = (copy.deepcopy(process) for _ in range(3))
    configure_isolation(first)
    from pathlib import Path

    source = Path(process["datasets"][0]["folder_path"])
    Image.new("RGB", (32, 16), "red").save(source / "0.png")
    configure_isolation(second)
    (source / "0.txt").write_text("changed caption", encoding="utf-8")
    configure_isolation(third)
    assert (
        len(
            {
                item["identity_isolation_recipe_digest"]
                for item in (first, second, third)
            }
        )
        == 3
    )


def test_actual_loader_slots_include_partial_bucket_batch_effects(tmp_path):
    process = _process(tmp_path, counts=(2, 1))
    configure_isolation(process)
    configs = [DatasetConfig(**dataset) for dataset in process["datasets"]]
    datasets = []
    for config, count in zip(configs, (2, 1)):
        datasets.append(
            SimpleNamespace(
                dataset_config=config,
                source_configs=[config],
                file_list=[
                    SimpleNamespace(dataset_config=config) for _ in range(count)
                ],
                batch_indices=[list(range(count))],
            )
        )
    from torch.utils.data import ConcatDataset

    class LoaderDataset:
        def __init__(self, value):
            self.__dict__.update(vars(value))

        def __len__(self):
            return len(self.batch_indices)

    loader = SimpleNamespace(
        dataset=ConcatDataset([LoaderDataset(value) for value in datasets])
    )
    report = isolation_exposure_report(process, loader)
    assert report["basis"] == "actual_loader_slots"
    assert report["expected_total_sample_exposures"] == 150
    assert report["datasets"][0]["loader_draw_probability"] == pytest.approx(0.5)
    assert report["datasets"][0]["expected_sample_exposures"] == pytest.approx(100)


@pytest.mark.parametrize("dropout", [0, 1])
def test_real_text_cache_and_caption_refresh_match_live_encoder_inputs(
    tmp_path, dropout
):
    from testing.test_h3_cache_contracts import _FrozenToyEncoderModel
    from toolkit.data_loader import AiToolkitDataset

    process = _process(tmp_path, dop_enabled=True)
    configure_isolation(process)
    raw = {
        **process["datasets"][0],
        "cache_text_embeddings": True,
        "caption_dropout_rate": dropout,
    }
    model = _FrozenToyEncoderModel()
    dataset = AiToolkitDataset(DatasetConfig(**raw), sd=model)
    from pathlib import Path

    (Path(raw["folder_path"]) / "0.txt").write_text(
        "person0 beside person1", encoding="utf-8"
    )
    index = next(
        index
        for index, item in enumerate(dataset.file_list)
        if Path(item.path).name == "0.png"
    )
    cached = dataset._get_single_item(index)
    live = _item(raw, cached=False, dropout=dropout)
    assert live.caption == (
        "person0person " if dropout else "person0person beside person1person"
    )
    torch.testing.assert_close(
        cached.prompt_embeds.text_embeds, model.encode_prompt(live.caption).text_embeds
    )
    torch.testing.assert_close(
        cached.dop_prompt_embeds.text_embeds,
        model.encode_prompt(live.caption_dop).text_embeds,
    )


def test_actual_loader_sampler_and_padding_exposure(tmp_path):
    from testing.test_h3_cache_contracts import _FrozenToyEncoderModel
    from toolkit.data_loader import StatefulRandomSampler, get_dataloader_from_datasets

    process = _process(tmp_path)
    configure_isolation(process)
    loader = get_dataloader_from_datasets(
        process["datasets"], batch_size=3, sd=_FrozenToyEncoderModel()
    )
    assert isinstance(loader.sampler, StatefulRandomSampler)
    report = isolation_exposure_report(process, loader)
    assert report["effective_count"] == 7
    assert report["original_count"] == 7
    assert sum(row["loader_sample_slots"] for row in report["datasets"]) == 9
    assert report["expected_total_sample_exposures"] == 300
    assert report["per_subject_probabilities"][0] == pytest.approx(1 / 3)


def test_prepare_binds_generated_content_even_if_returned_cache_identity_is_unchanged(
    tmp_path, monkeypatch
):
    import toolkit.identity_isolation_composites as composites

    process = _process(tmp_path, composites=True, composite_count=1)
    outputs = [
        _source(tmp_path, f"composite{index}", 1, f"person{index} next to a person")
        for index in range(2)
    ]
    monkeypatch.setattr(
        composites,
        "prepare_composites",
        lambda _config: [
            {
                "folder_path": str(folder),
                "subject_index": index,
                "cache_identity": "synthetic-fixed-recipe",
            }
            for index, folder in enumerate(outputs)
        ],
    )
    first, second = copy.deepcopy(process), copy.deepcopy(process)
    configure_isolation(first)
    Image.new("RGB", (32, 16), "blue").save(outputs[0] / "0.png")
    configure_isolation(second)
    assert (
        first["identity_isolation_recipe_digest"]
        != second["identity_isolation_recipe_digest"]
    )
    composite = first["datasets"][2]
    assert composite["isolation_category"] == "composite"
    assert composite["isolation_cache_identity"] == "synthetic-fixed-recipe"


def test_configure_rejects_derived_config_instead_of_reapplying_balancing(tmp_path):
    process = _process(tmp_path, balance_subjects=True)
    configure_isolation(process)
    original = copy.deepcopy(process)
    with pytest.raises(ValueError, match="once on an unprepared"):
        configure_isolation(process)
    assert process == original


def test_preview_rejects_nonperson_composite_class_before_asset_preparation(tmp_path):
    process = _process(tmp_path, composites=True)
    process["identity_isolation"]["subjects"][0]["class_name"] = "dog"
    with pytest.raises(ValueError, match="person classes only"):
        inspect_isolation(process)


@pytest.mark.parametrize("cached", [False, True])
def test_unmatched_dataset_receives_global_subject_mapping_without_owner_injection(
    tmp_path, cached
):
    process = _process(tmp_path, dop_enabled=True)
    folder = _source(tmp_path, "unrelated", 1, "person0 and person1 in a crowd")
    process["datasets"].append(
        {"folder_path": str(folder), "num_workers": 0, "resolution": 32}
    )
    configure_isolation(process)
    raw = process["datasets"][-1]
    item = _item(raw, cached=cached)
    assert raw.get("trigger_word") is None
    assert raw.get("isolation_subject_index") is None
    assert item.caption == "person0person and person1person in a crowd"
    assert item.caption_dop == "a person and a person in a crowd"
    assert item.get_dropout_caption() == item.get_dop_dropout_caption() == ""


@pytest.mark.parametrize("dataset_trigger", [None, "globalmark", "person0", ""])
def test_dop_validates_process_trigger_only_when_inherited(tmp_path, dataset_trigger):
    process = _process(tmp_path, dop_enabled=True)
    process["trigger_word"] = "globalmark"
    folder = _source(tmp_path, "unrelated", 1, "a scene without identity words")
    process["datasets"].append(
        {
            "folder_path": str(folder),
            "num_workers": 0,
            "resolution": 32,
            "trigger_word": dataset_trigger,
        }
    )
    if dataset_trigger in (None, "globalmark"):
        with pytest.raises(ValueError, match="unmapped dataset trigger_word"):
            configure_isolation(process)
    else:
        configure_isolation(process)
        item = _item(process["datasets"][-1])
        expected = (
            "a person a scene without identity words"
            if dataset_trigger
            else "a scene without identity words"
        )
        assert item.caption_dop == expected


def test_pooled_buckets_partition_explicit_categories_not_filenames(tmp_path):
    from testing.test_h3_cache_contracts import _FrozenToyEncoderModel
    from toolkit.data_loader import get_dataloader_from_datasets

    process = _process(tmp_path, counts=(2,))
    configure_isolation(process)
    folder = _source(tmp_path, "ordinary_folder", 1, "person0 beside a person")
    composite = {
        **process["datasets"][0],
        "folder_path": str(folder),
        "isolation_category": "composite",
        "trigger_word": None,
        "isolation_dataset_index": 1,
    }
    loader = get_dataloader_from_datasets(
        [*process["datasets"], composite],
        batch_size=2,
        sd=_FrozenToyEncoderModel(),
        pool_datasets=True,
    )
    pooled = loader.dataset.datasets[0]
    assert len(pooled.source_configs) == 2
    categories = []
    for batch in pooled.batch_indices:
        batch_categories = {
            pooled.file_list[index].dataset_config.isolation_category for index in batch
        }
        assert len(batch_categories) == 1
        categories.extend(batch_categories)
    assert set(categories) == {"normal", "composite"}


def test_exposure_counts_legacy_accumulation_as_delayed_updates_not_extra_draws(
    tmp_path,
):
    process = _process(tmp_path)
    process["train"].update(steps=5, batch_size=2, gradient_accumulation_steps=2)
    configure_isolation(process)
    report = isolation_exposure_report(process)
    assert report["expected_total_sample_exposures"] == 10
    assert report["expected_optimizer_updates"] == 2
    assert report["optimizer_updates_with_partial_flush"] == 3


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("PERSON0PERSON outdoors", "PERSON0PERSON outdoors"),
        ("PERSON0 outdoors", "person0person outdoors"),
    ],
)
def test_case_insensitive_active_caption_does_not_duplicate_owner_trigger(
    tmp_path, caption, expected
):
    process = _process(tmp_path, counts=(1,))
    configure_isolation(process)
    folder = Path(process["datasets"][0]["folder_path"])
    (folder / "0.txt").write_text(caption, encoding="utf-8")
    assert _item(process["datasets"][0]).caption == expected


def test_declared_training_compatibility_fails_before_source_or_asset_access(
    tmp_path, monkeypatch
):
    import toolkit.identity_isolation_data as integration

    process = _process(tmp_path, composites=True)
    process["model"]["arch"] = "flux"

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unsupported training mode accessed image sources")

    monkeypatch.setattr(integration, "_images", unexpected)
    with pytest.raises(ValueError, match="Krea 2 RAW"):
        configure_isolation(process)


def test_exposure_distinguishes_source_originals_from_generated_inputs(tmp_path):
    process = _process(tmp_path, counts=(1, 2), composites=True, composite_count=2)
    configure_isolation(process, prepare=False)
    report = isolation_exposure_report(process)
    assert report["source_original_count"] == 3
    assert report["generated_original_count"] == 4
    assert report["original_count"] == 7
    assert report["normal_count"] == 3
    assert report["composite_count"] == 4


def test_balance_only_ablation_changes_repeats_without_fusion_or_preservation(tmp_path):
    process = _process(
        tmp_path, counts=(1, 3), fused_captions=False, balance_subjects=True
    )
    config = configure_isolation(process)
    assert config.active
    assert [dataset["num_repeats"] for dataset in process["datasets"]] == [3, 1]
    item = _item(process["datasets"][0])
    assert item.caption == "person0 and person1 outdoors"
    assert item.caption_dop is None
    assert not process["train"].get("diff_output_preservation", False)
