"""Sample filenames preserve configured columns across EMA and raw passes."""

import ast
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_methods(path, class_name, method_names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    source_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    for method in methods:
        method.returns = None
        for argument in (
            method.args.posonlyargs + method.args.args + method.args.kwonlyargs
        ):
            argument.annotation = None
    extracted = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(extracted), str(ROOT / path), "exec"), namespace)  # noqa: S102 - trusted repository methods
    return {name: namespace[name] for name in method_names}


def _sample_process(tmp_path, *, ema=True, first=False, postprocess=None, fail=False):
    clock = iter(1234.5 + index for index in range(12))
    namespace = {
        "os": os,
        "json": json,
        "re": re,
        "tempfile": tempfile,
        "random": SimpleNamespace(randint=lambda *_: 24680),
        "time": SimpleNamespace(time=lambda: next(clock)),
        "ImgExt": "png",
    }
    config_methods = _load_methods(
        "toolkit/config_modules.py", "GenerateImageConfig",
        {"__init__", "_process_prompt_string", "set_gen_time", "_get_path_no_ext",
         "get_image_path", "get_prompt_path"}, namespace,
    )
    image_config = type("GenerateImageConfig", (), config_methods)
    namespace.update({
        "GenerateImageConfig": image_config,
        "flush": lambda: None,
        "print_acc": lambda *_: None,
        "ClipVisionAdapter": type("ClipVisionAdapter", (), {}),
        "CustomAdapter": type("CustomAdapter", (), {}),
    })
    sample = _load_methods(
        "jobs/process/BaseSDTrainProcess.py", "BaseSDTrainProcess", {"sample"}, namespace,
    )["sample"]
    prompts = ["a sample" for _ in range(12)]
    prompts[1] = "raw override --seed 77 --steps 9"
    prompts[2] = "normal override --d 88 --s 11"
    samples = [
        SimpleNamespace(
            raw_weights=i % 2 == 1, seed=-1 if i == 0 else None,
            width=64, height=64, neg="", guidance_scale=1.0, sample_steps=4,
            network_multiplier=1.0, num_frames=1, fps=15, duration=None,
            ctrl_img=None, ctrl_idx=0, ctrl_img_1=None, ctrl_img_2=None,
            ctrl_img_3=None,
        )
        for i in range(len(prompts))
    ]
    config = SimpleNamespace(
        prompts=prompts, samples=samples, seed=100, walk_seed=True, ext="webp" if first else "png",
        guidance_rescale=0.0, adapter_conditioning_scale=1.0,
        refiner_start_at=0.5, extra_values=[], do_cfg_norm=False, sampler="test",
    )
    generated = []
    modes = []

    def generate_images(configs, sampler):
        assert sampler == "test"
        plan_files = list((tmp_path / "samples" / ".sample-plans").glob("*.json"))
        assert len(plan_files) == 1
        planned_names = {item["filename"] for item in json.loads(plan_files[0].read_text())["samples"]}
        assert not list((tmp_path / "samples" / ".sample-plans").glob("*.tmp"))
        paths = []
        for count, item in enumerate(configs):
            item.set_gen_time(999999 + count)
            image_path = item.get_image_path(count, len(configs))
            assert Path(image_path).name in planned_names
            paths.append((item, image_path))
            if not fail:
                Path(image_path).write_bytes(b"generated sample")
        if fail:
            raise RuntimeError("generation failed")
        generated.append(paths)

    process = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=True), save_root=str(tmp_path),
        sample_config=SimpleNamespace(ext="png") if first else config,
        first_sample_config=config, adapter_config=None, embedding=None,
        adapter=None, trigger_word=None, logger=None,
        ema=SimpleNamespace(eval=lambda: modes.append("ema"), train=lambda: modes.append("raw")) if ema else None,
        sd=SimpleNamespace(generate_images=generate_images),
        post_process_generate_image_config_list=postprocess or (lambda configs: configs),
    )
    sample(process, step=42, is_first=first)
    return generated, modes


@pytest.mark.parametrize("first", [False, True])
def test_mixed_raw_samples_keep_global_columns_and_resolved_metadata(tmp_path, first):
    generated, modes = _sample_process(tmp_path, first=first)
    plan_path, = (tmp_path / "samples" / ".sample-plans").glob("*.json")
    timestamp = json.loads(plan_path.read_text())["timestamp"]
    assert modes == ["ema", "raw", "raw"]
    assert len(generated) == 2
    ext = "webp" if first else "png"
    for rows, indices, prefix in (
        (generated[0], range(0, 12, 2), ""),
        (generated[1], range(1, 12, 2), "RAW_"),
    ):
        assert len(rows) == len(indices)
        for (config, path), index in zip(rows, indices):
            seed = {0: 24680, 1: 77, 2: 88}.get(index, 100 + index)
            steps = {1: 9, 2: 11}.get(index, 4)
            filename = f"{prefix}{timestamp}__000000042_{index:02d}_seed-{seed}_steps-{steps}.{ext}"
            assert Path(path).name == filename
            assert config.seed == seed
            assert config.num_inference_steps == steps
            assert "[count]" not in config.output_path
            assert config.output_path == path
            assert Path(config.get_prompt_path(0, 1)).stem == Path(path).stem


def test_without_ema_raw_flags_keep_single_full_index_pass(tmp_path):
    generated, modes = _sample_process(tmp_path, ema=False)
    plan_path, = (tmp_path / "samples" / ".sample-plans").glob("*.json")
    timestamp = json.loads(plan_path.read_text())["timestamp"]
    assert modes == []
    assert len(generated) == 1
    assert len(generated[0]) == 12
    for index, (_, path) in enumerate(generated[0]):
        assert Path(path).name.startswith(f"{timestamp}__000000042_{index:02d}_seed-")


def test_plan_records_resolved_postprocessed_identity_and_both_weight_passes(tmp_path):
    def postprocess(configs):
        for config in configs:
            config.seed += 1
            config.num_inference_steps += 2
            config.output_ext = "mp4"
        return configs

    generated, _ = _sample_process(tmp_path, postprocess=postprocess)
    plan_path, = (tmp_path / "samples" / ".sample-plans").glob("*.json")
    plan = json.loads(plan_path.read_text())
    assert plan["timestamp"] == 1234500
    assert plan["trainingStep"] == 42
    expected = []
    for rows in generated:
        for config, sample_path in rows:
            assert Path(sample_path).is_file()
            assert config.gen_time != plan["timestamp"]
            assert Path(config.get_prompt_path()).stem == Path(sample_path).stem
            expected.append({
                "filename": Path(sample_path).name,
                "index": config.sample_index,
                "seed": config.seed,
                "steps": config.num_inference_steps,
            })
    assert plan["samples"] == expected
    assert {item["index"] for item in plan["samples"]} == set(range(12))
    assert all(item["filename"].endswith(".mp4") for item in plan["samples"])


def test_failed_generation_leaves_complete_plan_without_media(tmp_path):
    with pytest.raises(RuntimeError, match="generation failed"):
        _sample_process(tmp_path, fail=True)
    plan_path, = (tmp_path / "samples" / ".sample-plans").glob("*.json")
    plan = json.loads(plan_path.read_text())
    assert len(plan["samples"]) == 12
    assert not any((tmp_path / "samples" / item["filename"]).exists() for item in plan["samples"])
    assert not list(plan_path.parent.glob("*.tmp"))
