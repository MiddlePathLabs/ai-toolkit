"""Opt-in dataset preparation and exposure accounting for identity isolation."""

import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from toolkit.identity_isolation import IsolationConfig

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_AUXILIARIES = (
    "control_path",
    "control_path_1",
    "control_path_2",
    "control_path_3",
    "control_from_same_folder",
    "controls",
    "inpaint_path",
    "mask_path",
    "alpha_mask",
    "unconditional_path",
    "clip_image_path",
    "clip_image_from_same_folder",
    "control_video_path",
    "control_video_paths",
)


def _path(value):
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(value))))


def _digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(folder):
    if not os.path.isdir(folder):
        raise ValueError(f"identity_isolation subject folder does not exist: {folder}")
    result = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "_controls")
        result.extend(
            os.path.join(root, name)
            for name in sorted(files)
            if not name.startswith(".")
            and Path(name).suffix.lower() in _IMAGE_EXTENSIONS
        )
    if not result:
        raise ValueError(f"identity_isolation requires images in {folder}")
    return sorted(result)


def _caption(path, dataset):
    extension = dataset.get("caption_type") or dataset.get("caption_ext", ".txt")
    if extension and not extension.startswith("."):
        extension = "." + extension
    sidecar = os.path.splitext(path)[0] + (extension or ".txt")
    if os.path.isfile(sidecar):
        from toolkit.dataloader_mixins import decode_caption_file

        text = decode_caption_file(sidecar)
        if text.strip():
            return text, sidecar
    default = dataset.get("default_caption")
    if default and str(default).strip():
        return str(default), None
    raise ValueError(f"identity_isolation requires a nonempty caption for {path}")


def _source_records(dataset):
    folder = dataset.get("dataset_path") or dataset.get("folder_path")
    records = []
    if os.path.isfile(folder) and str(folder).lower().endswith(".json"):
        with open(folder, encoding="utf-8") as stream:
            mapping = json.load(stream)
        for image, value in sorted(mapping.items()):
            text = value.get("caption", "") if isinstance(value, dict) else value
            records.append(
                {
                    "path": _path(image),
                    "image_sha256": _digest_file(image),
                    "caption": text,
                    "caption_source_sha256": _digest_file(folder),
                }
            )
        return records
    for image in _images(folder):
        caption, sidecar = _caption(image, dataset)
        records.append(
            {
                "path": _path(image),
                "image_sha256": _digest_file(image),
                "caption": caption,
                "caption_sha256": _digest_file(sidecar) if sidecar else None,
            }
        )
    return records


def _validate_compatibility(config, datasets, train):
    people = {
        "woman",
        "man",
        "girl",
        "boy",
        "lady",
        "guy",
        "person",
        "child",
        "kid",
        "baby",
        "teenager",
    }
    if config.composites and any(
        subject["class_name"] not in people for subject in config.subjects
    ):
        raise ValueError(
            "identity_isolation stranger composites support person classes only; disable composites for animals, objects or styles"
        )
    for dataset in datasets:
        folder, path = dataset.get("folder_path"), dataset.get("dataset_path")
        if folder is not None and path is not None and _path(folder) != _path(path):
            raise ValueError(
                "identity_isolation dataset folder_path and dataset_path must identify the same folder"
            )
    if any(int(item.get("num_workers", 2)) != 0 for item in datasets):
        raise ValueError(
            "identity_isolation requires dataset num_workers: 0 for resumable stateful-sampler cursors"
        )
    if not config.composites or config.composite_lr_multiplier == 1:
        return
    if (
        int(train.get("gradient_accumulation", 1)) != 1
        or int(train.get("gradient_accumulation_steps", 1)) != 1
    ):
        raise ValueError(
            "identity_isolation composite_lr_multiplier != 1 requires gradient_accumulation: 1 and gradient_accumulation_steps: 1"
        )
    if train.get("pool_datasets", False):
        raise ValueError(
            "identity_isolation composite_lr_multiplier != 1 requires pool_datasets: false"
        )
    buckets = [bool(item.get("buckets", True)) for item in datasets]
    if len(set(buckets)) > 1:
        raise ValueError(
            "identity_isolation requires consistently bucketed or nonbucketed datasets"
        )
    if not all(buckets) and (
        int(train.get("batch_size", 1)) != 1
        or any(int(item.get("batch_size") or 1) != 1 for item in datasets)
    ):
        raise ValueError(
            "identity_isolation composite_lr_multiplier != 1 requires buckets or nonbucket batch_size: 1"
        )


def _resolve_dop(config, train):
    if config.dop_enabled:
        if (
            "diff_output_preservation" in train
            and not train["diff_output_preservation"]
        ):
            raise ValueError(
                "identity_isolation DOP conflicts with train.diff_output_preservation: false; disable isolation dop or enable legacy DOP explicitly"
            )
        if (
            "diff_output_preservation_multiplier" in train
            and float(train["diff_output_preservation_multiplier"]) != config.dop_weight
        ):
            raise ValueError(
                "train.diff_output_preservation_multiplier conflicts with identity_isolation dop_weight; set them to the same value"
            )
        train["diff_output_preservation"] = True
        train["diff_output_preservation_multiplier"] = config.dop_weight
    if train.get("diff_output_preservation") and config.subjects:
        legacy_class = train.get("diff_output_preservation_class")
        if legacy_class and any(
            subject["class_name"] != legacy_class for subject in config.subjects
        ):
            raise ValueError(
                "train.diff_output_preservation_class conflicts with identity_isolation subject classes; remove the legacy class and use subject class_name mappings"
            )


def _split_paths(datasets):
    result = []
    for dataset in datasets:
        for key in ("folder_path", "dataset_path"):
            values = dataset.get(key)
            if isinstance(values, (list, tuple)):
                for value in values:
                    clone = copy.deepcopy(dataset)
                    clone[key] = value
                    result.append(clone)
                break
        else:
            result.append(copy.deepcopy(dataset))
    return result


def resolve_isolation_settings(process_config):
    """Resolve static settings without source, asset, model, or optimizer access."""
    config = IsolationConfig(process_config.get("identity_isolation"))
    if not config.active:
        return None, None, None
    train = copy.deepcopy(process_config.get("train", {}))
    datasets = _split_paths(process_config.get("datasets", []))
    if not datasets:
        raise ValueError("identity_isolation requires at least one training dataset")
    _validate_compatibility(config, datasets, train)
    _resolve_dop(config, train)
    return config, train, datasets


def configure_isolation(process_config, *, prepare=True):
    config, train, datasets = resolve_isolation_settings(process_config)
    if config is None:
        return None
    if "identity_isolation_recipe_digest" in process_config:
        raise ValueError(
            "configure_isolation must be called once on an unprepared process configuration"
        )
    from toolkit.identity_isolation_training import validate_isolation_process

    validate_isolation_process(
        {**process_config, "train": train, "datasets": datasets}, config
    )
    mapped = defaultdict(list)
    source_records = {}
    source_variants = {}

    def read_source(dataset):
        folder = dataset.get("dataset_path") or dataset.get("folder_path")
        if folder is None:
            raise ValueError(
                "identity_isolation datasets require folder_path or dataset_path"
            )
        key = json.dumps(
            [
                _path(folder),
                dataset.get("caption_type") or dataset.get("caption_ext", ".txt"),
                dataset.get("default_caption"),
            ],
            ensure_ascii=False,
        )
        if key not in source_variants:
            source_variants[key] = _source_records(dataset)
        return source_variants[key]

    resolved = config.as_dict()
    for index, subject in enumerate(config.subjects):
        folder = _path(subject["folder_path"])
        for dataset in datasets:
            candidate = dataset.get("dataset_path") or dataset.get("folder_path")
            if candidate is None or _path(candidate) != folder:
                continue
            if dataset.get("is_reg") or dataset.get("prior_reg"):
                raise ValueError(
                    "identity_isolation subjects cannot be regularization datasets"
                )
            if dataset.get("trigger_word") not in (None, "", subject["trigger"]):
                raise ValueError(
                    f"Dataset trigger_word conflicts with subject trigger for {folder}"
                )
            legacy_class = dataset.get("diff_output_preservation_class")
            if (
                train.get("diff_output_preservation")
                and legacy_class
                and legacy_class != subject["class_name"]
            ):
                raise ValueError(
                    f"Dataset diff_output_preservation_class conflicts with identity_isolation class_name for {folder}"
                )
            if config.composites:
                conflicts = [key for key in _AUXILIARIES if dataset.get(key)]
                if conflicts:
                    raise ValueError(
                        f"identity_isolation composites cannot inherit explicit auxiliaries {conflicts} from {folder}; use an auxiliary-free subject dataset"
                    )
            records = read_source(dataset)
            source_records.setdefault(folder, records)
            dataset.update(
                identity_isolation=resolved,
                isolation_category="normal",
                isolation_subject_index=index,
                trigger_word=subject["trigger"],
                isolation_original_count=len(records),
                isolation_raw_repeats=dataset.get("num_repeats", 1),
            )
            if train.get("diff_output_preservation"):
                dataset["diff_output_preservation"] = True
            mapped[index].append(dataset)
        if not mapped[index]:
            raise ValueError(
                f"identity_isolation subject folder must match an existing dataset folder: {subject['folder_path']}"
            )
    if config.subjects:
        counts = [
            len(source_records[_path(subject["folder_path"])])
            for subject in config.subjects
        ]
        largest = max(counts)
        for index, blocks in mapped.items():
            balance = (
                max(1, round(largest / counts[index])) if config.balance_subjects else 1
            )
            for dataset in blocks:
                raw_repeats = int(dataset.get("num_repeats", 1))
                if raw_repeats < 1:
                    raise ValueError("identity_isolation requires num_repeats >= 1")
                dataset["num_repeats"] = raw_repeats * balance
                dataset["isolation_balance_multiplier"] = balance
    for dataset in datasets:
        dataset["identity_isolation"] = resolved
        dataset.setdefault("isolation_category", "normal")
        if train.get("diff_output_preservation"):
            trigger = dataset.get("trigger_word")
            if trigger is None:
                trigger = process_config.get("trigger_word")
            if (
                config.subjects
                and trigger
                and trigger not in {subject["trigger"] for subject in config.subjects}
            ):
                raise ValueError(
                    "identity_isolation DOP cannot combine an unmapped dataset trigger_word with subject mappings; remove that trigger or add its subject mapping"
                )
            dataset["diff_output_preservation"] = True
        folder = dataset.get("dataset_path") or dataset.get("folder_path")
        records = read_source(dataset)
        source_records.setdefault(_path(folder), records)
        if dataset.get("isolation_original_count") is None:
            dataset["isolation_original_count"] = len(records)
    generated = []
    if config.composites:
        if prepare:
            from toolkit.identity_isolation_composites import prepare_composites

            generated = prepare_composites(config)
        else:
            generated = [
                {
                    "subject_index": index,
                    "folder_path": None,
                    "cache_identity": "preview-not-prepared",
                }
                for index in range(len(config.subjects))
            ]
        if len(generated) != len(mapped) or {
            item["subject_index"] for item in generated
        } != set(mapped):
            raise ValueError(
                "identity_isolation composite preparation did not return every subject"
            )
        for item in generated:
            index = item["subject_index"]
            if prepare:
                generated_images = _images(item["folder_path"])
                if len(generated_images) != config.composite_count:
                    raise ValueError(
                        f"identity_isolation expected {config.composite_count} composites for subject {index}, found {len(generated_images)}"
                    )
                generated_records = _source_records(
                    {"folder_path": item["folder_path"]}
                )
            else:
                generated_records = []
            for source in mapped[index]:
                clone = copy.deepcopy(source)
                clone.update(
                    folder_path=item["folder_path"],
                    dataset_path=None,
                    trigger_word=None,
                    caption_ext=".txt",
                    default_caption=None,
                    num_repeats=1,
                    flip_x=False,
                    isolation_category="composite",
                    isolation_raw_repeats=1,
                    isolation_balance_multiplier=1,
                    isolation_original_count=config.composite_count,
                    isolation_cache_identity=item["cache_identity"],
                )
                clone.pop("caption_type", None)
                for key in _AUXILIARIES:
                    clone.pop(key, None)
                datasets.append(clone)
            item["content"] = generated_records
    for index, dataset in enumerate(datasets):
        dataset["isolation_dataset_index"] = index
    recipe = {
        "schema": 1,
        "isolation": resolved,
        "sources": source_variants,
        "generated": generated,
        "datasets": datasets,
        "preview": not prepare,
    }
    digest = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
    process_config["datasets"] = datasets
    process_config["train"] = train
    process_config["identity_isolation_recipe_digest"] = digest
    return config


def _source_key(config):
    return (
        getattr(config, "isolation_dataset_index", None),
        config.folder_path,
        config.dataset_path,
        str(config.resolution),
        getattr(config, "isolation_category", "normal"),
    )


def isolation_exposure_report(process_config, loader=None):
    """Estimate sample exposures from effective items or actual loader batch slots."""
    config = IsolationConfig(process_config.get("identity_isolation"))
    if not config.active:
        return {}
    rows = []
    if loader is not None:
        from toolkit.data_loader import get_dataloader_datasets

        for dataset in get_dataloader_datasets(loader):
            batches = getattr(dataset, "batch_indices", None)
            if dataset.dataset_config.buckets and batches is not None:
                slots = defaultdict(int)
                batch_slots = defaultdict(float)
                for batch in batches:
                    for index in batch:
                        source_id = _source_key(dataset.file_list[index].dataset_config)
                        slots[source_id] += 1
                        batch_slots[source_id] += 1 / len(batch)
                for source in dataset.source_configs:
                    key = _source_key(source)
                    count = sum(
                        _source_key(item.dataset_config) == key
                        for item in dataset.file_list
                    )
                    rows.append(
                        _report_row(vars(source), count, slots[key], batch_slots[key])
                    )
            else:
                for source in dataset.source_configs:
                    count = sum(
                        _source_key(item.dataset_config) == _source_key(source)
                        for item in dataset.file_list
                    )
                    rows.append(_report_row(vars(source), count, count, None))
    else:
        from toolkit.config_modules import preprocess_dataset_raw_config

        for dataset in preprocess_dataset_raw_config(
            process_config.get("datasets", [])
        ):
            original = dataset.get("isolation_original_count")
            if original is None:
                folder = dataset.get("dataset_path") or dataset.get("folder_path")
                original = (
                    len(_images(folder)) if folder and os.path.isdir(folder) else 0
                )
            count = original * int(dataset.get("num_repeats", 1))
            count *= (2 if dataset.get("flip_x") else 1) * (
                2 if dataset.get("flip_y") else 1
            )
            rows.append(_report_row(dataset, count, count, None))
    total = sum(row["effective_count"] for row in rows)
    train = process_config.get("train", {})
    accumulation = int(train.get("gradient_accumulation", 1))
    update_accumulation = max(1, int(train.get("gradient_accumulation_steps", 1)))
    input_steps = int(train.get("steps", 0))
    sample_slots = sum(row["loader_sample_slots"] for row in rows)
    batch_slots = sum(row["loader_batch_count"] or 0 for row in rows)
    actual_bucket_batches = loader is not None and batch_slots > 0
    samples_per_draw = (
        sample_slots / batch_slots
        if actual_bucket_batches
        else int(train.get("batch_size", 1))
    )
    draws = input_steps * samples_per_draw * accumulation
    subjects = defaultdict(float)
    categories = defaultdict(int)
    category_slots = defaultdict(float)
    for row in rows:
        row["probability"] = (
            row["loader_sample_slots"] / sample_slots if sample_slots else 0
        )
        row["loader_draw_probability"] = (
            (row["loader_batch_count"] or 0) / batch_slots
            if actual_bucket_batches
            else row["probability"]
        )
        row["expected_sample_exposures"] = draws * row["probability"]
        categories[row["category"]] += row["effective_count"]
        category_slots[row["category"]] += row["loader_sample_slots"]
        if row["subject_index"] is not None:
            subjects[row["subject_index"]] += row["probability"]
    normal, composite = categories["normal"], categories["composite"]
    originals = {}
    for row in rows:
        folder = _path(row["folder_path"]) if row["folder_path"] else None
        key = (row["category"], row["subject_index"], folder)
        originals[key] = max(originals.get(key, 0), row["original_count"])
    return {
        "basis": "actual_loader_slots" if loader is not None else "configured_estimate",
        "datasets": rows,
        "original_count": sum(originals.values()),
        "source_original_count": sum(
            count
            for (category, _, _), count in originals.items()
            if category == "normal"
        ),
        "generated_original_count": sum(
            count
            for (category, _, _), count in originals.items()
            if category == "composite"
        ),
        "expanded_original_count": sum(row["original_count"] for row in rows),
        "effective_count": total,
        "normal_count": normal,
        "composite_count": composite,
        "normal_to_composite_ratio": category_slots["normal"]
        / category_slots["composite"]
        if category_slots["composite"]
        else None,
        "prepared_normal_to_composite_ratio": normal / composite if composite else None,
        "scheduled_normal_sample_slots": category_slots["normal"],
        "scheduled_composite_sample_slots": category_slots["composite"],
        "per_subject_probabilities": dict(subjects),
        "expected_total_sample_exposures": draws,
        "configured_input_steps": input_steps,
        "expected_optimizer_updates": input_steps // update_accumulation,
        "optimizer_updates_with_partial_flush": (input_steps + update_accumulation - 1)
        // update_accumulation,
        "repeat_semantics": "effective num_repeats = raw num_repeats * round(max_subject_images / subject_images) when balancing; composites repeat once",
        "caveat": "Exposure estimates are not equality guarantees: bucket padding/partial batches, per-dataset batch sizes, shuffling, epoch boundaries, category retirement and accumulation alter realized sample draws. Legacy gradient_accumulation_steps delays updates, not input-step draws; final partial-window flush determines the update-count bounds.",
    }


def _report_row(dataset, effective, slots, batch_count):
    original = dataset.get("isolation_original_count")
    if original is None:
        original = effective // max(1, int(dataset.get("num_repeats", 1)))
    return {
        "folder_path": dataset.get("folder_path") or dataset.get("dataset_path"),
        "category": dataset.get("isolation_category", "normal"),
        "subject_index": dataset.get("isolation_subject_index"),
        "resolution": dataset.get("resolution", 512),
        "original_count": original,
        "raw_repeats": dataset.get(
            "isolation_raw_repeats", dataset.get("num_repeats", 1)
        ),
        "effective_repeats": dataset.get("num_repeats", 1),
        "balance_multiplier": dataset.get("isolation_balance_multiplier", 1),
        "effective_count": effective,
        "loader_sample_slots": slots,
        "loader_batch_count": batch_count,
    }


def inspect_isolation(process_config):
    prepared = copy.deepcopy(process_config)
    config = configure_isolation(prepared, prepare=False)
    if config is None:
        return {"resolved": None, "captions": [], "exposure": {}}
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    captions = []
    seen = set()
    for dataset in prepared["datasets"]:
        index = dataset.get("isolation_subject_index")
        if dataset.get("isolation_category") == "composite":
            continue
        for record in _source_records(dataset):
            path, text = record["path"], record["caption"]
            if (index, path) in seen:
                continue
            seen.add((index, path))
            preview_dataset = copy.deepcopy(dataset)
            if preview_dataset.get("trigger_word") is None:
                preview_dataset["trigger_word"] = process_config.get("trigger_word")
            preview_dataset["cache_text_embeddings"] = True
            preview_dataset["caption_dropout_rate"] = 0
            if prepared.get("train", {}).get("diff_output_preservation"):
                preview_dataset["diff_output_preservation_class"] = prepared[
                    "train"
                ].get("diff_output_preservation_class", "")
            item = FileItemDTO(
                path=path, dataset_config=DatasetConfig(**preview_dataset)
            )
            item.load_caption(caption_dict={path: text})
            preview = {
                "original": text,
                "training": item.caption,
                "preservation": item.caption_dop
                if item.caption_dop is not None
                else config.preservation_caption(item.caption),
            }
            preview.update(subject_index=index, source_path=path)
            captions.append(preview)
    return {
        "resolved": config.as_dict(),
        "captions": captions,
        "exposure": isolation_exposure_report(prepared),
        "recipe_digest": prepared["identity_isolation_recipe_digest"],
    }
