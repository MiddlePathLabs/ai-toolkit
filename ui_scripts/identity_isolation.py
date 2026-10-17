"""Local Identity Isolation inspection/preparation bridge; never starts training."""

import argparse
import base64
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def inspect_request(payload):
    from toolkit.identity_isolation import IsolationConfig
    from toolkit.identity_isolation_data import configure_isolation, inspect_isolation

    process = payload.get("process")
    if not isinstance(process, dict):
        raise ValueError("process must contain one training process configuration")
    action = payload.get("action", "inspect")
    if action not in {"inspect", "prepare", "caption", "trigger_check"}:
        raise ValueError("action must be inspect, prepare, caption, or trigger_check")
    if action == "trigger_check":
        from toolkit.identity_isolation_trigger_check import check_triggers

        return check_triggers(process)
    config = IsolationConfig(process.get("identity_isolation"))
    if action == "caption":
        text = payload.get("caption", "")
        if not isinstance(text, str):
            raise ValueError("caption must be text")
        preview = config.caption_preview(text)
        train = process.get("train", {})
        if not config.subjects and (
            config.dop_enabled or train.get("diff_output_preservation")
        ):
            trigger = process.get("trigger_word")
            if trigger is None:
                triggers = {
                    dataset["trigger_word"]
                    for dataset in process.get("datasets", [])
                    if dataset.get("trigger_word")
                }
                if len(triggers) > 1:
                    raise ValueError(
                        "Legacy DOP has multiple dataset triggers; use Inspect datasets for per-dataset preservation previews"
                    )
                trigger = next(iter(triggers), None)
            if trigger:
                preview["preservation"] = preview["training"].replace(
                    trigger, train.get("diff_output_preservation_class", "")
                )
        return {"caption": preview, "resolved": config.as_dict()}
    if action == "inspect":
        return inspect_isolation(process)

    prepared = copy.deepcopy(process)
    configure_isolation(prepared)
    result = inspect_isolation(process)
    result["composites"] = []
    from PIL import Image

    seen_composites = set()
    for dataset in prepared.get("datasets", []):
        if dataset.get("isolation_category") != "composite":
            continue
        folder = Path(dataset["folder_path"])
        for path in sorted(folder.glob("*.png"))[:4]:
            if path in seen_composites:
                continue
            seen_composites.add(path)
            with Image.open(path) as image:
                image.thumbnail((512, 256))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
            caption = path.with_suffix(".txt").read_text(encoding="utf-8")
            result["composites"].append(
                {
                    "path": str(path),
                    "subject_index": dataset.get("isolation_subject_index"),
                    "caption": caption,
                    "thumbnail": "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="JSON request file; otherwise read stdin")
    args = parser.parse_args()
    try:
        payload = json.loads(
            Path(args.request).read_text(encoding="utf-8")
            if args.request
            else sys.stdin.read()
        )
        with contextlib.redirect_stdout(sys.stderr):
            result = inspect_request(payload)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=True))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
