"""Auto-recaption of stuck images (toolkit/recaption.py) and the watcher hooks it uses."""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image

from toolkit.config_modules import RecaptionConfig, TrainConfig
from toolkit.loss_watch import PerImageAdaptiveLR
from toolkit import recaption as rc


# ---- helpers -----------------------------------------------------------------------------

def _image(tmp_path, name="a.png", caption="a photo of ohwx"):
    path = tmp_path / name
    Image.new("RGB", (32, 32), (200, 10, 10)).save(path)
    if caption is not None:
        (tmp_path / (os.path.splitext(name)[0] + ".txt")).write_text(caption, encoding="utf-8")
    return str(path)


def _item(path, trigger="ohwx", source=None):
    dc = SimpleNamespace(caption_ext=".txt", trigger_word=trigger, use_short_captions=False,
                         identity_isolation=None, random_triggers=[])
    return SimpleNamespace(path=path, dataset_config=dc, caption_source_path=source, loss_multiplier=0.5)


class _FakeBackend:
    name = "fake"
    describe = "fake"

    def __init__(self, captions):
        self.captions = list(captions)
        self.prompts = []

    def caption(self, path, prompt):
        self.prompts.append(prompt)
        return self.captions.pop(0)

    def close(self):
        pass


def _manager(tmp_path, backend, **cfg):
    cfg.setdefault("enabled", True)
    return rc.RecaptionManager(RecaptionConfig(**cfg), save_root=str(tmp_path / "out"), total_steps=1000,
                               log_fn=lambda *_: None, backend_factory=lambda: backend,
                               proposals_only_reason=cfg.pop("_reason", None))


def _wait_results(mgr, n=1, timeout=10.0):
    end = time.time() + timeout
    while mgr._results.qsize() < n:
        if time.time() > end:
            raise AssertionError("no captioner result")
        time.sleep(0.01)


def _mark_stuck(watcher, key, residual):
    watcher._confirmed_stuck.add(key)
    watcher.newly_stuck = {key}
    watcher.last_residual[key] = residual
    watcher._mult[key] = 0.5


# ---- config --------------------------------------------------------------------------------

def test_config_defaults_and_validation():
    cfg = RecaptionConfig()
    assert not cfg.enabled and cfg.prefer == "api" and cfg.max_attempts == 2
    assert cfg.stop_after == 0.75 and cfg.on_give_up == "best" and not cfg.has_api
    assert RecaptionConfig(api={"api_base_url": "http://x"}).has_api
    with pytest.raises(ValueError):
        RecaptionConfig(prefer="cloud")
    with pytest.raises(ValueError):
        RecaptionConfig(stop_after=0)
    tc = TrainConfig(per_image_adaptive_lr=True, per_image_adaptive_lr_recaption={"enabled": True, "prefer": "gpu"})
    assert tc.per_image_adaptive_lr_recaption.enabled and tc.per_image_adaptive_lr_recaption.prefer == "gpu"
    assert TrainConfig().per_image_adaptive_lr_recaption.enabled is False


# ---- watcher hooks ---------------------------------------------------------------------------

def test_watcher_reports_newly_stuck_and_reset_forgets_item():
    w = PerImageAdaptiveLR(warmup_epochs=0, log_fn=lambda *_: None)
    for ep in range(1, 7):
        for t in (0.1, 0.5, 0.9):
            w.observe(epoch=ep, item_key="bad", timestep=t, loss=1.0)
            for i in range(1, 7):  # healthy images, distinct losses that keep dropping
                w.observe(epoch=ep, item_key=f"g{i}", timestep=t, loss=0.05 * i / ep)
        w.epoch_boundary(ep)
        if "bad" in w._confirmed_stuck:
            break
    assert w.newly_stuck == {"bad"}
    assert w.last_residual["bad"] > w.last_residual["g1"]
    w.reset_item("bad")
    assert "bad" not in w._confirmed_stuck and w.multiplier("bad") == 1.0
    assert all(r[0] != "bad" for r in w._records)
    w.epoch_boundary(ep + 1)
    assert "bad" not in w.newly_stuck  # no history left to judge it on


# ---- prompt / caption text -------------------------------------------------------------------

def test_prompt_mentions_trigger_and_failed_captions():
    p1 = rc.build_prompt(None, "ohwx", ["old one"])
    assert "[trigger]" in p1 and "old one" in p1 and "current training caption" in p1
    p2 = rc.build_prompt("Describe.", None, ["a", "b"])
    assert p2.startswith("Describe.") and "[trigger]" not in p2 and "Do not reuse" in p2


def test_clean_caption_keeps_or_adds_trigger():
    assert rc.clean_new_caption('"[trigger] on a beach"', "ohwx") == "[trigger] on a beach"
    assert rc.clean_new_caption("Caption: ohwx on a beach", "ohwx") == "ohwx on a beach"
    assert rc.clean_new_caption("a man on a beach", "ohwx") == "[trigger], a man on a beach"
    assert rc.clean_new_caption("a man", None) == "a man"
    with pytest.raises(ValueError):
        rc.clean_new_caption("  ", "ohwx")


# ---- storage ---------------------------------------------------------------------------------

def test_sidecar_write_backs_up_once_and_reverts(tmp_path):
    path = _image(tmp_path, caption="original")
    target = rc.caption_target(_item(path))
    assert target["kind"] == "sidecar"
    rc.write_caption(target, "new 1")
    rc.write_caption(target, "new 2")
    assert rc.read_caption(target) == "new 2"
    assert (tmp_path / "a.txt.orig").read_text(encoding="utf-8") == "original"


def test_json_source_is_updated_in_place(tmp_path):
    path = _image(tmp_path, caption=None)
    src = tmp_path / "captions.json"
    src.write_text(json.dumps({path: {"caption": "orig", "extra": 1}}), encoding="utf-8")
    target = rc.caption_target(_item(path, source=str(src)))
    assert target["kind"] == "json"
    rc.write_caption(target, "better")
    data = json.loads(src.read_text(encoding="utf-8"))
    assert data[path] == {"caption": "better", "extra": 1}
    assert os.path.exists(str(src) + ".orig")


# ---- GPU selection -----------------------------------------------------------------------------

def test_reserved_gpus_from_visible_devices():
    gpus = [{"index": 0, "uuid": "GPU-aaa"}, {"index": 1, "uuid": "GPU-bbb"}]
    assert rc.reserved_gpu_uuids(gpus, 1, visible_env="0") == {"aaa"}
    assert rc.reserved_gpu_uuids(gpus, 1, visible_env="GPU-BBB") == {"bbb"}
    assert rc.reserved_gpu_uuids(gpus, 1, visible_env="7") is None
    assert rc.reserved_gpu_uuids(gpus, 2, visible_env="") is None
    assert rc.reserved_gpu_uuids(gpus, 1, visible_env="", current_uuid="aaa") == {"aaa"}


def test_find_spare_gpu_skips_own_busy_and_small(monkeypatch):
    gpus = [
        {"index": 0, "uuid": "GPU-a", "free_mb": 80000, "util": 0},   # ours
        {"index": 1, "uuid": "GPU-b", "free_mb": 8000, "util": 0},    # too little VRAM
        {"index": 2, "uuid": "GPU-c", "free_mb": 40000, "util": 95},  # busy
        {"index": 3, "uuid": "GPU-d", "free_mb": 30000, "util": 2},
    ]
    monkeypatch.setattr(rc, "query_gpus", lambda: gpus)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert rc.find_spare_gpu(20, 1)["index"] == 3
    monkeypatch.setattr(rc, "query_gpus", lambda: gpus[:1])
    assert rc.find_spare_gpu(20, 1) is None


def test_backend_preference(monkeypatch, tmp_path):
    made = []
    monkeypatch.setattr(rc, "ApiBackend", lambda cfg: made.append("api") or SimpleNamespace(name="api"))
    monkeypatch.setattr(rc, "GpuWorkerBackend", lambda *a, **k: made.append("gpu") or SimpleNamespace(name="gpu"))
    monkeypatch.setattr(rc, "find_spare_gpu", lambda *a, **k: {"index": 1, "uuid": "GPU-x", "free_mb": 50000})
    both = {"api": {"api_base_url": "http://x"}, "gpu": {"model_name_or_path": "m"}}

    def pick(**cfg):
        made.clear()
        mgr = rc.RecaptionManager(RecaptionConfig(enabled=True, **cfg), save_root=str(tmp_path / "o"),
                                  total_steps=10, log_fn=lambda *_: None)
        try:
            return mgr._default_backend().name, list(made)
        finally:
            mgr.close()

    assert pick(**both) == ("api", ["api"])                      # endpoint wins even with a spare GPU
    assert pick(prefer="gpu", **both) == ("gpu", ["gpu"])
    assert pick(gpu={"model_name_or_path": "m"}) == ("gpu", ["gpu"])  # no endpoint -> spare GPU
    monkeypatch.setattr(rc, "find_spare_gpu", lambda *a, **k: None)
    assert pick(prefer="gpu", **both) == ("api", ["api"])        # no spare GPU -> endpoint


# ---- full flow ---------------------------------------------------------------------------------

def test_flow_recaption_retry_then_give_up_keeping_best(tmp_path):
    path = _image(tmp_path, caption="wrong caption")
    item = _item(path)
    items = {path: [item]}
    backend = _FakeBackend(["[trigger] in a red room", "[trigger], red square"])
    mgr = _manager(tmp_path, backend)
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(w, path, 0.9)
        mgr.on_window(100, w, items)
        assert mgr.items[path]["status"] == "pending"
        _wait_results(mgr)
        mgr.poll(w, items)
        entry = mgr.items[path]
        assert entry["status"] == "applied" and entry["attempts"] == 1
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "[trigger] in a red room"
        assert path not in w._confirmed_stuck and item.loss_multiplier == 1.0
        assert "wrong caption" in backend.prompts[0] and "[trigger]" in backend.prompts[0]

        _mark_stuck(w, path, 0.3)  # attempt 1 fit better than the original
        mgr.on_window(200, w, items)
        _wait_results(mgr)
        mgr.poll(w, items)
        assert mgr.items[path]["attempts"] == 2
        assert "Do not reuse" in backend.prompts[1] and "in a red room" in backend.prompts[1]

        _mark_stuck(w, path, 0.6)
        mgr.on_window(300, w, items)
        entry = mgr.items[path]
        assert entry["status"] == "gave_up" and entry["kept"] == "attempt 1"
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "[trigger] in a red room"
        _mark_stuck(w, path, 0.6)
        mgr.on_window(400, w, items)
        assert mgr._requests.empty() and len(backend.prompts) == 2  # never again

        state_file = tmp_path / "out" / rc.STATE_FILENAME
        assert json.loads(state_file.read_text(encoding="utf-8"))["items"][path]["status"] == "gave_up"
        assert rc.revert_all(str(state_file), log_fn=lambda *_: None) == 1
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "wrong caption"
    finally:
        mgr.close()


def test_give_up_original_restores_users_caption(tmp_path):
    path = _image(tmp_path, caption="mine")
    items = {path: [_item(path)]}
    mgr = _manager(tmp_path, _FakeBackend(["[trigger] x"]), max_attempts=1, on_give_up="original")
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(w, path, 0.9)
        mgr.on_window(10, w, items)
        _wait_results(mgr)
        mgr.poll(w, items)
        _mark_stuck(w, path, 0.1)
        mgr.on_window(20, w, items)
        assert mgr.items[path]["status"] == "gave_up"
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "mine"
    finally:
        mgr.close()


def test_proposals_only_never_touches_captions_or_watcher(tmp_path):
    path = _image(tmp_path, caption="mine")
    items = {path: [_item(path)]}
    mgr = _manager(tmp_path, _FakeBackend(["[trigger] proposal"]), _reason="text embeddings are cached")
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(w, path, 0.9)
        mgr.on_window(10, w, items)
        _wait_results(mgr)
        mgr.poll(w, items)
        assert mgr.items[path]["status"] == "proposed"
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "mine"
        assert (tmp_path / "a.recaption_proposal.txt").read_text(encoding="utf-8") == "[trigger] proposal"
        assert path in w._confirmed_stuck
    finally:
        mgr.close()


def test_no_new_recaptions_after_cutoff(tmp_path):
    path = _image(tmp_path)
    mgr = _manager(tmp_path, _FakeBackend([]))
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(w, path, 0.9)
        mgr.on_window(900, w, {path: [_item(path)]})  # cutoff is step 750 of 1000
        assert mgr._requests.empty() and mgr.items[path]["status"] == "new"
    finally:
        mgr.close()


def test_failed_request_is_retried_and_resume_requeues_pending(tmp_path):
    path = _image(tmp_path)
    items = {path: [_item(path)]}

    class Flaky(_FakeBackend):
        def caption(self, p, prompt):
            if not self.prompts:
                self.prompts.append(prompt)
                raise RuntimeError("HTTP 503")
            return super().caption(p, prompt)

    mgr = _manager(tmp_path, Flaky(["[trigger] ok"]))
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(w, path, 0.9)
        mgr.on_window(10, w, items)
        _wait_results(mgr)
        mgr.poll(w, items)
        assert mgr.items[path]["status"] == "error"
        w.newly_stuck = set()  # still confirmed stuck, not "new" any more
        mgr.on_window(20, w, items)
        _wait_results(mgr)
        mgr.poll(w, items)
        assert mgr.items[path]["status"] == "applied"
    finally:
        mgr.close()

    state = {"items": {path: {"status": "pending", "attempts": 0, "history": []}}}
    m2 = rc.RecaptionManager.__new__(rc.RecaptionManager)
    m2.load_state_dict(state)
    assert m2.items[path]["status"] == "error"


def test_skips_non_images_and_composite(tmp_path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\0")
    mgr = _manager(tmp_path, _FakeBackend([]))
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(w, str(vid), 0.9)
        mgr.on_window(10, w, {str(vid): [_item(str(vid))]})
        assert mgr.items[str(vid)]["status"] == "skipped"
        img = _image(tmp_path, "b.png")
        it = _item(img)
        it.dataset_config.identity_isolation = object()
        it.dataset_config.isolation_category = "composite"
        _mark_stuck(w, img, 0.9)
        mgr.on_window(10, w, {img: [it]})
        assert "composite" in mgr.items[img]["reason"]
    finally:
        mgr.close()


def test_peer_sync_resets_recaptioned_items(tmp_path):
    state = tmp_path / rc.STATE_FILENAME
    state.write_text(json.dumps({"resets": [[1, "x"], [2, "y"]]}), encoding="utf-8")
    w = PerImageAdaptiveLR(log_fn=lambda *_: None)
    w._confirmed_stuck |= {"x", "y"}
    assert rc.sync_peer_resets(str(state), w, 1) == 2
    assert w._confirmed_stuck == {"x"}


# ---- real captioners against a mock endpoint -----------------------------------------------------

class _Server:
    def __init__(self):
        self.prompts = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.prompts.append(body["messages"][-1]["content"][0]["text"])
                out = {"choices": [{"message": {"content": "[trigger] standing in a red room"},
                                    "finish_reason": "stop"}]}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.httpd.shutdown()


def test_api_backend_sends_prompt(server, tmp_path):
    path = _image(tmp_path)
    backend = rc.ApiBackend({"api_base_url": server.url, "model_name_or_path": "m"})
    try:
        assert backend.caption(path, "PROMPT-1") == "[trigger] standing in a red room"
        assert backend.caption(path, "PROMPT-2")
        assert server.prompts == ["PROMPT-1", "PROMPT-2"]
    finally:
        backend.close()


def test_gpu_worker_process_protocol(server, tmp_path):
    """Runs the real worker process; OpenAICaptioner stands in for a local model."""
    path = _image(tmp_path)
    backend = rc.GpuWorkerBackend(
        {"index": 1, "uuid": "GPU-00000000-0000-0000-0000-000000000000"},
        {"captioner": "OpenAICaptioner", "api_base_url": server.url, "model_name_or_path": "m"},
        str(tmp_path / "worker.log"), ready_timeout=120, request_timeout=60,
    )
    try:
        assert backend.alive()
        assert backend.caption(path, "WORKER-PROMPT") == "[trigger] standing in a red room"
        assert server.prompts == ["WORKER-PROMPT"]
        with pytest.raises(RuntimeError):
            backend.caption(str(tmp_path / "missing.png"), "x")
        assert backend.alive()  # one bad file doesn't kill the worker
    finally:
        backend.close()
    assert not backend.alive()
    assert backend._log.closed


@pytest.mark.parametrize("failure", ["send", "wait", "not_ready", "kill"])
def test_failed_handshake_reaps_process_and_closes_log(monkeypatch, tmp_path, failure):
    popen = rc.subprocess.Popen
    children = []

    def start_reader(*args, **kwargs):
        child = popen(
            [rc.sys.executable, "-c",
             "import time; time.sleep(60)" if failure == "kill" else "import sys; sys.stdin.read()"],
            **kwargs,
        )
        if failure == "kill":
            wait = child.wait
            child.wait = lambda timeout=None: wait(timeout=0.01 if timeout is not None else None)
        children.append(child)
        return child

    def fail(*args):
        raise RuntimeError("handshake failed")

    monkeypatch.setattr(rc.subprocess, "Popen", start_reader)
    if failure == "send":
        monkeypatch.setattr(rc.GpuWorkerBackend, "_send", fail)
    else:
        monkeypatch.setattr(
            rc.GpuWorkerBackend, "_wait",
            fail if failure in ("wait", "kill") else lambda *args: {"ready": False, "error": "load failed"},
        )
    with pytest.raises(RuntimeError):
        rc.GpuWorkerBackend({"index": 1, "uuid": "GPU-test"}, {}, str(tmp_path / "worker.log"))
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed
    # Windows refuses this rename if the parent's log descriptor is still open.
    (tmp_path / "worker.log").rename(tmp_path / "closed.log")


def test_close_during_backend_creation_closes_late_backend(tmp_path):
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Backend(_FakeBackend):
        def close(self):
            closed.set()

    backend = Backend(["must not be used"])

    def factory():
        started.set()
        assert release.wait(10)
        return backend

    mgr = rc.RecaptionManager(
        RecaptionConfig(enabled=True), save_root=str(tmp_path), total_steps=100,
        backend_factory=factory, log_fn=lambda *_: None,
    )
    mgr._requests.put({"key": "a", "path": "a", "prompt": "p"})
    try:
        assert started.wait(10)
        mgr.close(wait=0)
    finally:
        release.set()
        mgr._thread.join(10)
    assert closed.is_set()
    assert not mgr._thread.is_alive()
    assert backend.prompts == []


@pytest.mark.parametrize("request_error", [True, False])
def test_gpu_request_failure_only_restarts_broken_worker(tmp_path, request_error):
    class Backend(_FakeBackend):
        name = "gpu"
        closed = False
        calls = 0

        def alive(self):
            return not self.closed

        def close(self):
            self.closed = True

        def caption(self, path, prompt):
            self.calls += 1
            if self.calls == 1:
                error = rc.CaptionRequestError if request_error else RuntimeError
                raise error("bad image" if request_error else "worker exited")
            return "[trigger] better"

    backend = Backend([])
    mgr = _manager(tmp_path, backend)
    try:
        mgr._requests.put({"key": "a", "path": "a", "prompt": "p"})
        _wait_results(mgr)
        assert mgr._results.get_nowait()["error"]
        assert backend.closed is not request_error
        if request_error:
            mgr._requests.put({"key": "b", "path": "b", "prompt": "p"})
            _wait_results(mgr)
            assert mgr._results.get_nowait()["caption"] == "[trigger] better"
            assert backend.calls == 2
    finally:
        mgr.close()


def test_proposal_write_failures_stop_retrying(monkeypatch, tmp_path):
    path = _image(tmp_path)
    items = {path: [_item(path)]}
    backend = _FakeBackend(["[trigger] proposal"] * 3)
    mgr = _manager(tmp_path, backend, proposals_only=True)
    watcher = PerImageAdaptiveLR(log_fn=lambda *_: None)
    atomic_write = rc._atomic_write_text

    def deny_proposal(target, text):
        if target.endswith(rc.PROPOSAL_SUFFIX):
            raise PermissionError("read only")
        atomic_write(target, text)

    monkeypatch.setattr(rc, "_atomic_write_text", deny_proposal)
    try:
        _mark_stuck(watcher, path, 1.0)
        for step in (10, 20, 30):
            mgr.on_window(step, watcher, items)
            _wait_results(mgr)
            mgr.poll(watcher, items)
            watcher.newly_stuck = set()
        mgr.on_window(40, watcher, items)
        assert mgr.items[path]["request_failures"] == 3
        assert mgr.items[path]["attempts"] == 0
        assert mgr._requests.empty()
        assert len(backend.prompts) == 3
    finally:
        mgr.close()


def test_temporarily_unavailable_backend_is_retried(tmp_path):
    path = _image(tmp_path)
    items = {path: [_item(path)]}
    backend = _FakeBackend(["[trigger] recovered"])
    available = False
    mgr = rc.RecaptionManager(
        RecaptionConfig(enabled=True, gpu={"model_name_or_path": "m"}),
        save_root=str(tmp_path / "out"), total_steps=100,
        backend_factory=lambda: backend if available else None, log_fn=lambda *_: None,
    )
    watcher = PerImageAdaptiveLR(log_fn=lambda *_: None)
    try:
        _mark_stuck(watcher, path, 1.0)
        for step in (1, 2, 3, 4):
            mgr.on_window(step, watcher, items)
            _wait_results(mgr)
            mgr.poll(watcher, items)
            watcher.newly_stuck = set()
        assert mgr.items[path]["status"] == "deferred"
        assert mgr.items[path].get("request_failures", 0) == 0
        available = True
        mgr.on_window(5, watcher, items)
        _wait_results(mgr)
        mgr.poll(watcher, items)
        assert mgr.items[path]["status"] == "applied"
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "[trigger] recovered"
    finally:
        mgr.close()


def test_revert_continues_after_failed_target(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("new", encoding="utf-8")
    state = tmp_path / rc.STATE_FILENAME
    state.write_text(json.dumps({"items": {
        "bad.png": {"touched": True, "target": {"kind": "sidecar", "path": str(tmp_path / "missing" / "bad.txt")},
                    "original": "old bad"},
        "good.png": {"touched": True, "target": {"kind": "sidecar", "path": str(good)}, "original": "old good"},
    }}), encoding="utf-8")
    messages = []
    assert rc.revert_all(str(state), messages.append) == 1
    assert good.read_text(encoding="utf-8") == "old good"
    assert any("couldn't restore bad.png" in message for message in messages)


@pytest.mark.parametrize("entry", [{"caption_short": "short"}, None, 42])
def test_invalid_json_caption_entry_uses_sidecar(tmp_path, entry):
    path = _image(tmp_path, caption="sidecar")
    source = tmp_path / "captions.json"
    source.write_text(json.dumps({path: entry}), encoding="utf-8")
    target = rc.caption_target(_item(path, source=str(source)))
    assert target["kind"] == "sidecar"
    rc.write_caption(target, "replacement")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "replacement"
    assert json.loads(source.read_text(encoding="utf-8"))[path] == entry


@pytest.mark.parametrize("kwargs", [
    {"max_attempts": 0}, {"max_attempts": -1}, {"min_free_vram_gb": -0.1},
    {"idle_shutdown_seconds": -1},
])
def test_invalid_resource_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RecaptionConfig(**kwargs)
