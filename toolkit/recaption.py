"""Auto-recaption for images the per-image adaptive-LR watcher confirms stuck.

Flow (all decisions on the main process, training never waits):
  1. At a window boundary, the watcher reports images that just became confirmed-stuck.
  2. Each one is queued (attempt 1). A background thread sends it to a captioner:
       - the OpenAI-compatible endpoint, if configured and `prefer: api` (the default), or
       - a local captioner in a separate process on a spare GPU (one this job isn't using,
         with enough free VRAM), started on demand and stopped again when idle, or
       - whichever of those two exists when the preferred one doesn't.
  3. When a caption comes back, the main thread writes it where the image's caption comes
     from (sidecar file or the dataset's caption JSON), keeping your original once in a
     backup, and resets that image in the watcher so it is judged only on the new caption.
     The dataloader re-reads captions on every fetch, so the next time the image is drawn
     it trains on the new text.
  4. If the image is confirmed stuck again, it gets another attempt (with the failed
     captions listed so the captioner doesn't repeat them), up to `max_attempts`. After
     that it is given up on: it keeps the best-scoring caption (or your original, with
     `on_give_up: original`) and is never queued again. The watcher's throttle still
     applies to it.

Captions are written with the literal `[trigger]` token, which the dataloader replaces with
the dataset's trigger word. State lives in <save_root>/recaption_state.json, which also lets
a resumed run continue where it stopped and `python -m toolkit.recaption revert <that file>`
put every original caption back.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATE_FILENAME = "recaption_state.json"
PROPOSAL_SUFFIX = ".recaption_proposal.txt"
ORIG_SUFFIX = ".orig"
_PROTOCOL_PREFIX = "@@RECAPTION@@ "
_MAX_REQUEST_FAILURES = 3

DEFAULT_CAPTION_PROMPT = (
    "Caption this image as if you were going to try to generate it with an image generator. "
    "Be thorough and describe everything in the image. Be decisive by stating things as they "
    "are. Do not say things like \"It appears that\" or \"possibly\". No preamble. Just get to "
    "the point."
)


# ---------------------------------------------------------------------------------------
# prompt + caption text
# ---------------------------------------------------------------------------------------

def build_prompt(base_prompt: Optional[str], trigger: Optional[str], failed_captions: List[str]) -> str:
    parts = [(base_prompt or DEFAULT_CAPTION_PROMPT).strip()]
    if trigger:
        parts.append(
            "Refer to the main subject only as [trigger] (write that token literally, square "
            "brackets included) instead of naming or identifying who or what it is."
        )
    failed = [c.strip() for c in failed_captions if c and c.strip()]
    if len(failed) == 1:
        parts.append(
            "This image's current training caption is:\n"
            f"\"{failed[0]}\"\n"
            "It does not match the image well enough. Write a new caption from what you actually "
            "see, keeping anything in it that is correct."
        )
    elif failed:
        listed = "\n".join(f"- \"{c}\"" for c in failed)
        parts.append(
            "These earlier captions for this image did not work for training:\n"
            f"{listed}\n"
            "Write a new caption that is more literal and specific about what is visible. Do not "
            "reuse their wording."
        )
    parts.append("Output only the caption.")
    return "\n\n".join(parts)


_WRAPPING_QUOTES = re.compile(r'^\s*["“”\'](.*)["“”\']\s*$', re.DOTALL)
_CAPTION_LABEL = re.compile(r"^\s*(new\s+)?caption\s*:\s*", re.IGNORECASE)


def clean_new_caption(text: str, trigger: Optional[str]) -> str:
    caption = (text or "").strip()
    caption = _CAPTION_LABEL.sub("", caption)
    m = _WRAPPING_QUOTES.match(caption)
    if m:
        caption = m.group(1).strip()
    caption = " ".join(caption.split())
    if not caption:
        raise ValueError("captioner returned an empty caption")
    if trigger:
        has_token = "[trigger]" in caption or "[name]" in caption
        if not has_token and trigger.casefold() not in caption.casefold():
            caption = f"[trigger], {caption}"
    return caption


# ---------------------------------------------------------------------------------------
# caption storage (sidecar or dataset JSON)
# ---------------------------------------------------------------------------------------

def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: a dataloader worker may have the file open for a moment
            if attempt == 9:
                os.remove(tmp)
                raise
            time.sleep(0.05 * (attempt + 1))


def caption_target(file_item) -> dict:
    """Where this item's caption comes from, mirroring FileItemDTO.load_caption()."""
    source = getattr(file_item, "caption_source_path", None)
    if source and os.path.isfile(source):
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if file_item.path in data:
                use_short = bool(getattr(file_item.dataset_config, "use_short_captions", False))
                entry = data[file_item.path]
                field = "caption"
                if use_short and isinstance(entry, dict) and "caption_short" in entry:
                    field = "caption_short"
                return {"kind": "json", "path": source, "key": file_item.path, "field": field}
        except (OSError, ValueError):
            pass
    ext = getattr(file_item.dataset_config, "caption_ext", ".txt") or ".txt"
    return {"kind": "sidecar", "path": os.path.splitext(file_item.path)[0] + ext}


def read_caption(target: dict) -> Optional[str]:
    if target["kind"] == "sidecar":
        if not os.path.exists(target["path"]):
            return None
        with open(target["path"], "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    with open(target["path"], "r", encoding="utf-8") as f:
        entry = json.load(f).get(target["key"])
    if isinstance(entry, dict):
        return entry.get(target.get("field", "caption"))
    return entry


def write_caption(target: dict, caption: Optional[str]) -> None:
    """Write (or, with caption=None on a sidecar, remove) a caption. The first write to a
    sidecar keeps the original next to it as <caption file>.orig; JSON sources get one
    <json>.orig copy of the whole file."""
    path = target["path"]
    if target["kind"] == "sidecar":
        backup = path + ORIG_SUFFIX
        if caption is None:
            if os.path.exists(path):
                os.remove(path)
            return
        if not os.path.exists(backup) and os.path.exists(path):
            with open(path, "rb") as src, open(backup, "wb") as dst:
                dst.write(src.read())
        _atomic_write_text(path, caption)
        return
    backup = path + ORIG_SUFFIX
    if not os.path.exists(backup):
        with open(path, "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(target["key"])
    field = target.get("field", "caption")
    if isinstance(entry, dict):
        entry[field] = caption
    else:
        data[target["key"]] = caption
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def revert_all(state_path: str, log_fn: Callable[[str], None] = print) -> int:
    """Restore every original caption recorded in a recaption_state.json."""
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    restored = 0
    for key, entry in (state.get("items") or {}).items():
        if not entry.get("touched") or "target" not in entry:
            continue
        write_caption(entry["target"], entry.get("original"))
        restored += 1
        log_fn(f"[recaption] restored original caption for {os.path.basename(key)}")
    return restored


# ---------------------------------------------------------------------------------------
# captioner construction (used in-process for the API and inside the GPU worker)
# ---------------------------------------------------------------------------------------

class _StubJob:
    def __init__(self):
        self.name = "recaption"
        self.meta = OrderedDict()


def build_captioner(uid: str, caption_kwargs: dict):
    from extensions_built_in.captioner import AI_TOOLKIT_EXTENSIONS

    ext = next((e for e in AI_TOOLKIT_EXTENSIONS if e.uid == uid), None)
    if ext is None:
        known = ", ".join(e.uid for e in AI_TOOLKIT_EXTENSIONS)
        raise ValueError(f"unknown captioner {uid!r} (known: {known})")
    cls = ext.get_process()
    caption = dict(caption_kwargs)
    caption.setdefault("extensions", ["jpg", "jpeg", "png", "bmp", "webp"])
    caption.setdefault("path_to_caption", ".")
    config = OrderedDict(
        # a path that never exists, so the captioner never treats itself as a UI job
        # (that would write status into the training job's row)
        sqlite_db_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), f".no_db_{uuid.uuid4().hex}.db"),
        caption=caption,
    )
    # BaseProcess prints the whole config, api_key included; keep it out of the logs
    with contextlib.redirect_stdout(io.StringIO()):
        captioner = cls(0, _StubJob(), config)
    return captioner


# ---------------------------------------------------------------------------------------
# spare GPU detection
# ---------------------------------------------------------------------------------------

def _norm_uuid(value: str) -> str:
    value = str(value).strip().lower()
    return value[4:] if value.startswith("gpu-") else value


def query_gpus() -> List[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            util = float(parts[3])
        except ValueError:
            util = 0.0
        try:
            rows.append({"index": int(parts[0]), "uuid": parts[1], "free_mb": float(parts[2]), "util": util})
        except ValueError:
            continue
    return rows


def reserved_gpu_uuids(gpus: List[dict], world_size: int, visible_env: Optional[str] = None,
                       current_uuid: Optional[str] = None) -> Optional[set]:
    """Normalized UUIDs of the GPUs this job trains on, or None when that can't be told
    (then no GPU counts as spare)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_env is None else visible_env
    by_index = {g["index"]: _norm_uuid(g["uuid"]) for g in gpus}
    if visible is not None and visible.strip() != "":
        reserved = set()
        for entry in visible.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if entry.isdigit():
                if int(entry) not in by_index:
                    return None
                reserved.add(by_index[int(entry)])
            elif entry.lower().startswith("gpu-"):
                reserved.add(_norm_uuid(entry))
            else:
                return None  # MIG or something else we can't map
        return reserved
    if world_size > 1:
        return None  # every rank sees every GPU; we can't tell which ones are free
    if current_uuid is None:
        try:
            import torch
            current_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        except Exception:
            return None
    return {_norm_uuid(current_uuid)}


def find_spare_gpu(min_free_gb: float, world_size: int, max_util: float = 30.0) -> Optional[dict]:
    gpus = query_gpus()
    if len(gpus) < 2:
        return None
    reserved = reserved_gpu_uuids(gpus, world_size)
    if reserved is None:
        return None
    candidates = [
        g for g in gpus
        if _norm_uuid(g["uuid"]) not in reserved
        and g["free_mb"] >= min_free_gb * 1024
        and g["util"] <= max_util
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda g: g["free_mb"])


# ---------------------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------------------

class ApiBackend:
    name = "api"

    def __init__(self, api_cfg: dict):
        kwargs = dict(api_cfg)
        kwargs.setdefault("model_name_or_path", kwargs.get("model", "") or "default")
        kwargs.setdefault("max_new_tokens", 512)
        self.captioner = build_captioner("OpenAICaptioner", kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            self.captioner.load_model()
        self.describe = f"endpoint {self.captioner.chat_url}"

    def caption(self, path: str, prompt: str) -> str:
        self.captioner.caption_config.caption_prompt = prompt
        return self.captioner.get_caption_for_file(path)

    def close(self):
        session = getattr(self.captioner, "session", None)
        if session is not None:
            session.close()


class GpuWorkerBackend:
    """A local captioner in its own process, pinned to one spare GPU."""

    name = "gpu"

    def __init__(self, gpu: dict, gpu_cfg: dict, log_path: str, ready_timeout: float = 1800.0,
                 request_timeout: float = 900.0):
        cfg = dict(gpu_cfg)
        uid = cfg.pop("captioner", "Qwen3VLCaptioner")
        cfg.setdefault("max_new_tokens", 512)
        cfg["device"] = "cuda"
        self.request_timeout = request_timeout
        self.describe = f"{uid} on GPU {gpu['index']}"
        env = dict(os.environ)
        env.pop("AITK_JOB_ID", None)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        env["PYTHONUNBUFFERED"] = "1"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._log = open(log_path, "a", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "toolkit.recaption_worker"],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._responses: "queue.Queue[dict]" = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self._send({"captioner": uid, "caption": cfg})
        msg = self._wait(ready_timeout)
        if not msg.get("ready"):
            self.close()
            raise RuntimeError(f"captioner worker failed to start: {msg.get('error', msg)} (see {log_path})")

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                if line.startswith(_PROTOCOL_PREFIX):
                    try:
                        self._responses.put(json.loads(line[len(_PROTOCOL_PREFIX):]))
                    except ValueError:
                        pass
                else:
                    self._log.write(line)
        finally:
            self._responses.put({"eof": True})

    def _send(self, msg: dict):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _wait(self, timeout: float) -> dict:
        try:
            msg = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("captioner worker timed out")
        if msg.get("eof"):
            raise RuntimeError(f"captioner worker exited (code {self.proc.poll()})")
        return msg

    def alive(self) -> bool:
        return self.proc.poll() is None

    def caption(self, path: str, prompt: str) -> str:
        req_id = uuid.uuid4().hex
        self._send({"id": req_id, "path": path, "prompt": prompt})
        while True:
            msg = self._wait(self.request_timeout)
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg["caption"]

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass
        finally:
            try:
                self._log.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------------------

class RecaptionManager:
    """Owned by the main process of a training job. Only touches the watcher and dataset
    items from the training thread (on_window / poll); the background thread only talks to
    the captioner."""

    def __init__(self, cfg, *, save_root: str, total_steps: int, world_size: int = 1,
                 proposals_only_reason: Optional[str] = None, log_fn: Callable[[str], None] = print,
                 backend_factory: Optional[Callable[[], Optional[object]]] = None):
        self.cfg = cfg
        self.save_root = save_root
        self.total_steps = max(1, int(total_steps))
        self.world_size = world_size
        self.proposals_only = bool(cfg.proposals_only or proposals_only_reason)
        self._log = log_fn
        self.state_path = os.path.join(save_root, STATE_FILENAME)
        self.items: Dict[str, dict] = {}
        self.reset_seq = 0
        self._resets: List[list] = []
        self._requests: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._results: "queue.Queue[dict]" = queue.Queue()
        self._backend_factory = backend_factory or self._default_backend
        self._backend = None
        self._gpu_retry_after = 0.0
        self._no_backend_logged = False
        self._cutoff_logged = False
        self._closed = False
        self._load_state()
        if proposals_only_reason and not cfg.proposals_only:
            self._log(f"[recaption] proposals only: {proposals_only_reason}")
        self._log(f"[recaption] on: {self._backend_summary()}, up to {cfg.max_attempts} attempt(s) "
                  f"per image, none after step {self.cutoff_step}"
                  + (" (writing proposals, not applying them)" if self.proposals_only else ""))
        self._thread = threading.Thread(target=self._worker_loop, name="recaption", daemon=True)
        self._thread.start()

    @property
    def cutoff_step(self) -> int:
        return int(self.total_steps * self.cfg.stop_after)

    def _backend_summary(self) -> str:
        options = []
        if self.cfg.has_api:
            options.append("endpoint")
        if self.cfg.has_gpu_model:
            options.append(f"spare GPU ({self.cfg.gpu.get('captioner')})")
        if not options:
            return "no captioner configured (stuck images are only logged)"
        if len(options) == 2:
            first = "endpoint" if self.cfg.prefer == "api" else "spare GPU"
            return f"captioner: {first} first, then the other"
        return f"captioner: {options[0]}"

    # ---- persistence ------------------------------------------------------------

    def _load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.load_state_dict(state)
            self._log(f"[recaption] resumed state for {len(self.items)} image(s)")
        except (OSError, ValueError) as e:
            self._log(f"[recaption] could not read {self.state_path} ({e}); starting fresh")

    def state_dict(self) -> dict:
        return {"version": 1, "items": self.items, "reset_seq": self.reset_seq, "resets": self._resets}

    def load_state_dict(self, state: dict):
        self.items = dict(state.get("items") or {})
        self.reset_seq = int(state.get("reset_seq", 0))
        self._resets = list(state.get("resets") or [])
        for entry in self.items.values():
            if entry.get("status") == "pending":
                entry["status"] = "error"  # its request died with the previous process

    def save(self):
        try:
            os.makedirs(self.save_root, exist_ok=True)
            _atomic_write_text(self.state_path, json.dumps(self.state_dict(), indent=2, ensure_ascii=False))
        except OSError as e:
            self._log(f"[recaption] could not save state ({e})")

    # ---- training-thread API -------------------------------------------------------

    def on_window(self, step: int, watcher, file_items_by_path: Dict[str, list]) -> None:
        """Call after watcher.epoch_boundary(). Queues newly stuck images and retries
        requests that failed."""
        changed = False
        candidates = set(watcher.newly_stuck)
        for key, entry in self.items.items():
            if (entry.get("status") == "error" and key in watcher._confirmed_stuck
                    and entry.get("request_failures", 0) < _MAX_REQUEST_FAILURES):
                candidates.add(key)
        for key in sorted(candidates):
            items = file_items_by_path.get(key)
            if not items:
                continue
            changed |= self._consider(step, key, items[0], watcher, file_items_by_path)
        if changed:
            self.save()

    def _consider(self, step, key, file_item, watcher, file_items_by_path) -> bool:
        entry = self.items.get(key)
        if entry is not None and entry.get("status") in ("pending", "gave_up", "proposed", "skipped"):
            return False
        reason = self._skip_reason(file_item)
        if reason:
            self.items[key] = {"status": "skipped", "reason": reason}
            self._log(f"[recaption] skipping {os.path.basename(key)}: {reason}")
            return True
        if entry is None:
            target = caption_target(file_item)
            try:
                original = read_caption(target)
            except (OSError, ValueError) as e:
                self._log(f"[recaption] can't read caption for {os.path.basename(key)} ({e})")
                return False
            entry = {"status": "new", "attempts": 0, "target": target, "original": original,
                     "touched": False, "history": [{"caption": original, "source": "original", "score": None}]}
            self.items[key] = entry
        # score the caption that just failed (lower residual = better fit)
        if entry["history"] and entry["history"][-1].get("score") is None:
            entry["history"][-1]["score"] = watcher.last_residual.get(key)

        if entry["attempts"] >= self.cfg.max_attempts:
            self._give_up(key, entry, watcher, file_items_by_path)
            return True
        if step > self.cutoff_step:
            if not self._cutoff_logged:
                self._log(f"[recaption] past step {self.cutoff_step}: no new recaptions this late in the run")
                self._cutoff_logged = True
            return entry["status"] == "new"

        failed = [h["caption"] for h in entry["history"] if h.get("caption")]
        trigger = getattr(file_item.dataset_config, "trigger_word", None)
        prompt = build_prompt(self.cfg.caption_prompt, trigger, failed)
        entry["status"] = "pending"
        self._requests.put({"key": key, "path": file_item.path, "prompt": prompt, "trigger": trigger,
                            "attempt": entry["attempts"] + 1})
        self._log(f"[recaption] {os.path.basename(key)}: queued attempt {entry['attempts'] + 1}/{self.cfg.max_attempts}")
        return True

    @staticmethod
    def _skip_reason(file_item) -> Optional[str]:
        ext = os.path.splitext(file_item.path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return f"only images are recaptioned ({ext} file)"
        dc = file_item.dataset_config
        if getattr(dc, "identity_isolation", None) is not None and getattr(dc, "isolation_category", None) == "composite":
            return "composite subject triggers can't be written as a single [trigger]"
        if getattr(dc, "random_triggers", None):
            return "dataset uses random triggers"
        return None

    def _give_up(self, key, entry, watcher, file_items_by_path):
        entry["status"] = "gave_up"
        if self.cfg.on_give_up == "original":
            keep = entry["history"][0]
        else:
            scored = [h for h in entry["history"] if h.get("score") is not None]
            keep = min(scored, key=lambda h: h["score"]) if scored else entry["history"][-1]
        current = entry["history"][-1]
        if keep is not current and not self.proposals_only:
            try:
                write_caption(entry["target"], keep["caption"])
                self._reset(key, watcher, file_items_by_path)
            except (OSError, ValueError) as e:
                self._log(f"[recaption] couldn't restore caption for {os.path.basename(key)} ({e})")
        entry["kept"] = keep["source"]
        self._log(f"[recaption] {os.path.basename(key)}: still stuck after {entry['attempts']} attempt(s), "
                  f"giving up and keeping the {keep['source']} caption")

    def _reset(self, key, watcher, file_items_by_path):
        watcher.reset_item(key)
        for item in file_items_by_path.get(key, []):
            item.loss_multiplier = 1.0
        self.reset_seq += 1
        self._resets.append([self.reset_seq, key])

    def has_results(self) -> bool:
        return not self._results.empty()

    def poll(self, watcher, file_items_by_path: Dict[str, list]) -> None:
        """Apply finished captions. Cheap when nothing is ready; call every step."""
        changed = False
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            changed = True
            self._apply(result, watcher, file_items_by_path)
        if changed:
            self.save()

    def _apply(self, result, watcher, file_items_by_path):
        key = result["key"]
        entry = self.items.get(key)
        if entry is None:
            return
        name = os.path.basename(key)
        if "error" in result:
            entry["status"] = "error"
            entry["request_failures"] = entry.get("request_failures", 0) + 1
            if result.get("no_backend"):
                entry["status"] = "skipped"
                entry["reason"] = "no captioner available"
            else:
                self._log(f"[recaption] {name}: captioning failed ({result['error']})")
            return
        try:
            caption = clean_new_caption(result["caption"], result.get("trigger"))
        except ValueError as e:
            entry["status"] = "error"
            entry["request_failures"] = entry.get("request_failures", 0) + 1
            self._log(f"[recaption] {name}: {e}")
            return
        old = entry["history"][-1].get("caption") or ""
        if self.proposals_only:
            proposal_path = os.path.splitext(key)[0] + PROPOSAL_SUFFIX
            try:
                _atomic_write_text(proposal_path, caption)
            except OSError as e:
                self._log(f"[recaption] {name}: couldn't write proposal ({e})")
                entry["status"] = "error"
                return
            entry["status"] = "proposed"
            entry["proposal"] = caption
            self._log(f"[recaption] {name}: proposal written to {os.path.basename(proposal_path)}")
            self._log(f"[recaption]   old: {_short(old)}")
            self._log(f"[recaption]   new: {_short(caption)}")
            return
        try:
            write_caption(entry["target"], caption)
        except (OSError, ValueError) as e:
            entry["status"] = "error"
            entry["request_failures"] = entry.get("request_failures", 0) + 1
            self._log(f"[recaption] {name}: couldn't write caption ({e})")
            return
        entry["attempts"] += 1
        entry["touched"] = True
        entry["status"] = "applied"
        entry["history"].append({"caption": caption, "source": f"attempt {entry['attempts']}", "score": None,
                                 "backend": result.get("backend")})
        self._reset(key, watcher, file_items_by_path)
        self._log(f"[recaption] {name}: new caption (attempt {entry['attempts']}/{self.cfg.max_attempts}, "
                  f"{result.get('backend', '?')}); judging it from scratch")
        self._log(f"[recaption]   old: {_short(old)}")
        self._log(f"[recaption]   new: {_short(caption)}")

    def summary(self, watcher=None) -> None:
        if not self.items:
            return
        counts: Dict[str, int] = {}
        for key, entry in self.items.items():
            status = entry.get("status", "?")
            if status == "applied":
                stuck = watcher is not None and key in watcher._confirmed_stuck
                status = "applied, still stuck" if stuck else "applied, no longer stuck"
            counts[status] = counts.get(status, 0) + 1
        text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        self._log(f"[recaption] summary: {text}. Details and originals: {self.state_path}")
        if any(e.get("touched") for e in self.items.values()):
            self._log(f"[recaption] to restore every original caption: python -m toolkit.recaption revert \"{self.state_path}\"")

    def close(self, wait: float = 5.0):
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join(timeout=wait)
        self._close_backend()

    # ---- background thread --------------------------------------------------------

    def _default_backend(self):
        cfg = self.cfg
        order = ["api", "gpu"] if cfg.prefer == "api" else ["gpu", "api"]
        for kind in order:
            if kind == "api" and cfg.has_api:
                try:
                    return ApiBackend(cfg.api)
                except Exception as e:
                    self._log(f"[recaption] endpoint unusable ({e})")
            if kind == "gpu" and cfg.has_gpu_model and time.time() >= self._gpu_retry_after:
                gpu = find_spare_gpu(cfg.min_free_vram_gb, self.world_size)
                if gpu is None:
                    continue
                self._log(f"[recaption] starting {cfg.gpu.get('captioner')} on spare GPU {gpu['index']} "
                          f"({gpu['free_mb'] / 1024:.0f} GB free)")
                try:
                    return GpuWorkerBackend(gpu, cfg.gpu, os.path.join(self.save_root, "recaption_worker.log"))
                except Exception as e:
                    self._gpu_retry_after = time.time() + 600
                    self._log(f"[recaption] spare-GPU captioner failed ({e}); not retrying it for 10 minutes")
        return None

    def _close_backend(self):
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None

    def _get_backend(self):
        backend = self._backend
        if backend is not None and getattr(backend, "name", "") == "gpu" and not backend.alive():
            self._close_backend()
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def _worker_loop(self):
        last_used = time.time()
        while True:
            try:
                req = self._requests.get(timeout=5.0)
            except queue.Empty:
                backend = self._backend
                if (backend is not None and getattr(backend, "name", "") == "gpu"
                        and time.time() - last_used > self.cfg.idle_shutdown_seconds):
                    self._log("[recaption] spare-GPU captioner idle, shutting it down")
                    self._close_backend()
                continue
            if req is None:
                return
            result = {"key": req["key"], "trigger": req.get("trigger")}
            try:
                backend = self._get_backend()
                if backend is None:
                    result.update(error="no captioner available", no_backend=True)
                    if not self._no_backend_logged:
                        self._no_backend_logged = True
                        self._log("[recaption] no captioner available (no endpoint set, no spare GPU); "
                                  "stuck images are only logged")
                else:
                    result["caption"] = backend.caption(req["path"], req["prompt"])
                    result["backend"] = backend.describe
            except Exception as e:
                result["error"] = str(e) or e.__class__.__name__
                if self._backend is not None and getattr(self._backend, "name", "") == "gpu":
                    self._close_backend()
            last_used = time.time()
            self._results.put(result)


def _short(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def sync_peer_resets(state_path: str, watcher, last_seq: int) -> int:
    """Non-main ranks: reset watcher entries for images rank 0 recaptioned since `last_seq`.
    Returns the newest sequence number seen."""
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            resets = json.load(f).get("resets") or []
    except (OSError, ValueError):
        return last_seq
    for seq, key in resets:
        if seq > last_seq:
            watcher.reset_item(key)
            last_seq = seq
    return last_seq


def group_file_items(datasets: Iterable) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for ds in datasets:
        for item in getattr(ds, "file_list", []):
            out.setdefault(item.path, []).append(item)
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "revert":
        n = revert_all(sys.argv[2])
        print(f"restored {n} caption(s)")
    else:
        print("usage: python -m toolkit.recaption revert <save_root>/recaption_state.json")
        sys.exit(2)
