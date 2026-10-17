"""Actual disk-cache/live conditioning parity with an offline synthetic encoder."""

import copy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from testing.test_h3_cache_contracts import _FrozenToyEncoderModel
from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.dataloader_mixins import TextEmbeddingCachingMixin
from toolkit.identity_isolation_data import configure_isolation


@pytest.mark.parametrize("dropout", [0.0, 1.0])
def test_disk_training_and_preservation_embeds_match_live_multi_identity_conditioning(
    tmp_path, dropout
):
    subjects, datasets = [], []
    caption = "zwxem, a woman and qrxt, a man beside zwxemwoman"
    for trigger, noun in [("zwxem", "woman"), ("qrxt", "man")]:
        folder = tmp_path / trigger
        folder.mkdir()
        Image.new("RGB", (32, 16), "red").save(folder / "one.png")
        (folder / "one.txt").write_text(caption, encoding="utf-8")
        subjects.append(
            {"folder_path": str(folder), "trigger": trigger, "class_name": noun}
        )
        datasets.append(
            {
                "folder_path": str(folder),
                "resolution": 32,
                "buckets": False,
                "num_workers": 0,
            }
        )
    process = {
        "model": {"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
        "network": {"type": "lora", "linear": 4, "linear_alpha": 4},
        "train": {"optimizer": "adamw"},
        "datasets": datasets,
        "identity_isolation": {
            "preset": "off",
            "fused_captions": True,
            "dop_enabled": True,
            "subjects": subjects,
        },
    }
    isolation = configure_isolation(process)
    model = _FrozenToyEncoderModel()
    model.arch = "krea2"
    cached_config = DatasetConfig(
        **{
            **process["datasets"][0],
            "cache_text_embeddings": True,
            "caption_dropout_rate": dropout,
        }
    )
    source = tmp_path / "zwxem" / "one.png"
    cached = FileItemDTO(path=str(source), dataset_config=cached_config, sd=model)
    cache_dataset = SimpleNamespace(
        sd=model,
        dataset_path=str(source.parent),
        dataset_config=cached_config,
        file_list=[cached],
        caption_dict=None,
    )
    TextEmbeddingCachingMixin.cache_text_embeddings(cache_dataset)
    model.restore_device_state()
    cached.load_prompt_embedding()

    live_raw = copy.deepcopy(process["datasets"][0])
    live_raw.update(cache_text_embeddings=False, caption_dropout_rate=dropout)
    live = FileItemDTO(
        path=str(source), dataset_config=DatasetConfig(**live_raw), sd=model
    )
    live.load_caption()
    trainer = object.__new__(SDTrainer)
    trainer.isolation_config = isolation
    trainer.train_config = SimpleNamespace(diff_output_preservation_class="")
    trainer.trigger_word = None
    preservation = trainer._preservation_prompts([live.caption], [live])[0]
    assert "zwxem" not in preservation and "qrxt" not in preservation
    expected_training = model.encode_prompt(live.caption).text_embeds
    expected_preservation = model.encode_prompt(preservation).text_embeds
    assert torch.equal(cached.prompt_embeds.text_embeds, expected_training)
    assert torch.equal(cached.dop_prompt_embeds.text_embeds, expected_preservation)
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == caption


def test_unrelated_dataset_with_disabled_inheritance_never_injects_global_trigger(
    tmp_path, monkeypatch
):
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    from toolkit import identity_isolation_composites

    folders = []
    for name in ("subject", "unrelated", "composite"):
        folder = tmp_path / name
        folder.mkdir()
        Image.new("RGB", (32, 16), "red").save(folder / "one.png")
        caption = (
            "ZWXEMWOMAN beside a person"
            if name == "subject"
            else "a scene without identity words"
        )
        (folder / "one.txt").write_text(caption, encoding="utf-8")
        folders.append(folder)
    monkeypatch.setattr(
        identity_isolation_composites,
        "prepare_composites",
        lambda _config: [
            {
                "folder_path": str(folders[2]),
                "subject_index": 0,
                "cache_identity": "synthetic-only",
            },
        ],
    )
    process = {
        "training_folder": str(tmp_path / "output"),
        "trigger_word": "globalmark",
        "model": {"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
        "network": {"type": "lora", "linear": 4, "linear_alpha": 4},
        "train": {"optimizer": "adamw", "resume_mode": "weights_only"},
        "datasets": [
            {
                # the unrelated dataset cannot inherit the unmapped process
                # trigger under isolation DOP; an empty string disables it
                "folder_path": str(folder),
                "num_workers": 0,
                "resolution": 32,
                "caption_dropout_rate": 0,
                **({} if index == 0 else {"trigger_word": ""}),
            }
            for index, folder in enumerate(folders[:2])
        ],
        "identity_isolation": {
            "preset": "full",
            "composite_count": 1,
            "subjects": [
                {
                    "folder_path": str(folders[0]),
                    "trigger": "zwxem",
                    "class_name": "woman",
                }
            ],
        },
    }
    job = SimpleNamespace(name="synthetic-constructor", meta={}, raw_config={})
    original = copy.deepcopy(process)
    trainer = BaseSDTrainProcess(0, job, process)
    assert [dataset.trigger_word for dataset in trainer.dataset_configs] == [
        "zwxem",
        "",
        None,
    ]
    assert [dataset.isolation_category for dataset in trainer.dataset_configs] == [
        "normal",
        "normal",
        "composite",
    ]
    from toolkit.identity_isolation_data import inspect_isolation

    previews = inspect_isolation(original)["captions"]
    for preview, dataset in zip(previews, trainer.dataset_configs[:2]):
        item = FileItemDTO(path=preview["source_path"], dataset_config=dataset)
        item.load_caption()
        assert preview["training"] == item.caption
        assert preview["preservation"] == item.caption_dop


def test_legacy_dop_only_caption_preview_uses_existing_global_replacement():
    from ui_scripts.identity_isolation import inspect_request

    result = inspect_request(
        {
            "action": "caption",
            "process": {
                "identity_isolation": {"preset": "dop_only"},
                "trigger_word": "legacy",
                "train": {"diff_output_preservation_class": "person"},
            },
            "caption": "legacy beside legacy",
        }
    )
    assert result["caption"]["training"] == "legacy beside legacy"
    assert result["caption"]["preservation"] == "person beside person"
