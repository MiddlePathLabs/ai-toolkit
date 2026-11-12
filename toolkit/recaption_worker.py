"""Spare-GPU captioner process for toolkit/recaption.py. Started by the training job with
CUDA_VISIBLE_DEVICES set to one spare GPU. Protocol on stdin/stdout, one JSON object per line:

  in  (first line) {"captioner": "<extension uid>", "caption": {...caption config...}}
  out              {"ready": true}  or  {"ready": false, "error": "..."}
  in               {"id": "...", "path": "<image>", "prompt": "<instruction>"}
  out              {"id": "...", "caption": "..."}  or  {"id": "...", "error": "..."}

Protocol lines on stdout carry a fixed prefix; everything the model prints goes to stderr
(the job's recaption_worker.log). Closing stdin ends the process.
"""
import json
import os
import sys

PREFIX = "@@RECAPTION@@ "


def main() -> int:
    # Keep the real stdout for protocol messages only; route every other print to stderr.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(msg):
        proto.write(PREFIX + json.dumps(msg) + "\n")
        proto.flush()

    first = sys.stdin.readline()
    if not first:
        return 0
    try:
        setup = json.loads(first)
        import torch
        from toolkit.recaption import build_captioner

        captioner = build_captioner(setup["captioner"], setup.get("caption") or {})
        with torch.no_grad():
            captioner.load_model()
            captioner.load_loras()
    except Exception as e:  # report and exit; the parent falls back
        import traceback

        traceback.print_exc()
        send({"ready": False, "error": f"{e.__class__.__name__}: {e}"})
        return 1
    send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        try:
            captioner.caption_config.caption_prompt = req["prompt"]
            with torch.no_grad():
                caption = captioner.get_caption_for_file(req["path"])
            send({"id": req.get("id"), "caption": caption or ""})
        except Exception as e:
            import traceback

            traceback.print_exc()
            send({"id": req.get("id"), "error": f"{e.__class__.__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
