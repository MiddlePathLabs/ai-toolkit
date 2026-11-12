"""OpenAI-compatible API captioner against a local mock chat-completions server."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import requests
from PIL import Image

from extensions_built_in.captioner.OpenAICaptioner import (
    OpenAICaptioner,
    clean_caption,
    normalize_chat_url,
)


class _Server:
    def __init__(self):
        self.requests = []
        self.fail_first = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"busy")
                    return
                content = body["messages"][-1]["content"]
                n_images = sum(1 for p in content if p["type"] == "image_url")
                text = f"<think>hmm</think>caption with {n_images} image(s)"
                out = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.close()


def _make(tmp_path, base_url, **caption):
    cfg = {
        "model_name_or_path": "test-model",
        "extensions": ["png", "mp4"],
        "path_to_caption": str(tmp_path),
        "api_base_url": base_url,
        "max_res": 64,
        "max_new_tokens": 257,
        "concurrency": 2,
        "max_retries": 2,
        **caption,
    }
    job = SimpleNamespace(name="t", meta=OrderedDict())
    return OpenAICaptioner(0, job, OrderedDict(caption=cfg, sqlite_db_path=str(tmp_path / "none.db")))


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://h:8080", "http://h:8080/v1/chat/completions"),
        ("http://h:8080/v1/", "http://h:8080/v1/chat/completions"),
        ("h:1234/v1", "http://h:1234/v1/chat/completions"),
        ("https://x/api/v1/chat/completions", "https://x/api/v1/chat/completions"),
        ("https://x/chat/completions", "https://x/chat/completions"),
        (
            "https://x?api-version=2024-10-21#response",
            "https://x/v1/chat/completions?api-version=2024-10-21#response",
        ),
        (
            "https://x/?key=a%2Fb&mode=chat#fragment",
            "https://x/v1/chat/completions?key=a%2Fb&mode=chat#fragment",
        ),
        (
            "https://x/chat/completions?key=a%2Fb#response",
            "https://x/chat/completions?key=a%2Fb#response",
        ),
        (
            "https://example.openai.azure.com/openai/deployments/vision/chat/completions"
            "?api-version=2024-10-21#response",
            "https://example.openai.azure.com/openai/deployments/vision/chat/completions"
            "?api-version=2024-10-21#response",
        ),
        (
            "https://x/proxy/v1/?redirect=/chat/completions#chat/completions",
            "https://x/proxy/v1/chat/completions?redirect=/chat/completions#chat/completions",
        ),
    ],
)
def test_normalize_chat_url(base_url, expected):
    assert normalize_chat_url(base_url) == expected


def test_normalize_chat_url_requires_endpoint():
    with pytest.raises(ValueError):
        normalize_chat_url("")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>a\nb</think>\n A cat.", "A cat."),
        ("<THINK>a\nb</THINK>\n A cat.", "A cat."),
        ("reasoning without opener</think>A dog.", "A dog."),
        ("reasoning without opener</THINK>A dog.", "A dog."),
        ("<think>unfinished reasoning", ""),
        ("<THINK>unfinished reasoning", ""),
        ("A cat.<think>unfinished reasoning", "A cat."),
        ("A cat.<THINK>unfinished reasoning", "A cat."),
        ("<think>first</think>A cat.<think>unfinished", "A cat."),
        ("  plain  ", "plain"),
        ("I think this is a cat.", "I think this is a cat."),
    ],
)
def test_clean_caption(text, expected):
    assert clean_caption(text) == expected


def test_end_to_end_images_and_video(tmp_path, server, monkeypatch):
    import cv2

    for i in range(3):
        Image.new("RGB", (200, 100), (i * 40, 0, 0)).save(tmp_path / f"img{i}.png")
    # existing caption is skipped without recaption
    Image.new("RGB", (10, 10)).save(tmp_path / "done.png")
    (tmp_path / "done.txt").write_text("keep me", encoding="utf-8")
    writer = cv2.VideoWriter(str(tmp_path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV mp4v encoder is unavailable; cannot create the test video")
    for f in range(20):
        writer.write(np.full((48, 64, 3), f * 10, np.uint8))
    writer.release()

    server.fail_first = 1  # first request gets a 503 and is retried
    monkeypatch.setattr("time.sleep", lambda s: None)
    cap = _make(tmp_path, server.url, api_key="sk-test", video_frames=4, system_prompt="be brief",
                extra_body={"top_k": 20})
    cap.run()

    for i in range(3):
        assert (tmp_path / f"img{i}.txt").read_text(encoding="utf-8") == "caption with 1 image(s)"
    assert (tmp_path / "clip.txt").read_text(encoding="utf-8") == "caption with 4 image(s)"
    assert (tmp_path / "done.txt").read_text(encoding="utf-8") == "keep me"

    assert len(server.requests) == 5  # 4 files + 1 retry
    req = server.requests[-1]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "test-model" and body["top_k"] == 20 and body["max_tokens"] == 257
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    url = next(p for p in body["messages"][1]["content"] if p["type"] == "image_url")["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_all_failures_raise(tmp_path, server, monkeypatch):
    Image.new("RGB", (32, 32)).save(tmp_path / "a.png")
    server.fail_first = 100
    monkeypatch.setattr("time.sleep", lambda s: None)
    cap = _make(tmp_path, server.url, max_retries=1)
    with pytest.raises(RuntimeError, match="every request"):
        cap.run()
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ChunkedEncodingError("incomplete response body"),
        requests.exceptions.ContentDecodingError("invalid compressed response"),
        requests.exceptions.JSONDecodeError("invalid JSON", "{", 1),
    ],
    ids=["chunked-body", "content-decoding", "json-decoding"],
)
def test_response_read_errors_are_retried(tmp_path, monkeypatch, error):
    cap = _make(tmp_path, "http://endpoint.invalid", max_retries=1)
    expected = {"choices": [{"message": {"content": "A cat."}}]}
    response = Mock(status_code=200)
    response.json.return_value = expected
    cap.session = Mock()
    if isinstance(error, requests.exceptions.JSONDecodeError):
        response.json.side_effect = [error, expected]
        cap.session.post.return_value = response
    else:
        cap.session.post.side_effect = [error, response]
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    assert cap._post({"model": "test-model"}) == expected
    assert cap.session.post.call_count == 2
    sleep.assert_called_once()


@pytest.mark.parametrize("error_class", [requests.ConnectionError, requests.Timeout])
def test_dead_endpoint_aborts_after_first_file(tmp_path, monkeypatch, error_class):
    for i in range(5):
        Image.new("RGB", (32, 32)).save(tmp_path / f"img{i}.png")
    cap = _make(tmp_path, "http://endpoint.invalid", concurrency=1, max_retries=2)
    session = Mock(headers={})
    session.post.side_effect = error_class("endpoint unavailable")
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        cap.run()

    assert session.post.call_count == 3
    assert not list(tmp_path.glob("*.txt"))


@pytest.mark.parametrize("stop_error", [None, RuntimeError, KeyboardInterrupt])
def test_stop_preserves_already_completed_caption(tmp_path, monkeypatch, stop_error):
    cap = _make(tmp_path, "http://endpoint.invalid", concurrency=1)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cap.file_paths = [str(first), str(second)]
    cap.is_ui_captioner = True
    caption = Mock(return_value="Completed caption.")
    monkeypatch.setattr(cap, "get_caption_for_file", caption)
    monkeypatch.setattr(cap, "update_step", lambda: None)

    def completed_before_stop(futures, **kwargs):
        # Hold the completion out of the returned done set so only stop's drain
        # can preserve it, not the ordinary successful-result path.
        for future in futures:
            future.result(timeout=5)
        return set(), set(futures)

    def stop():
        cap.is_stopping = True
        if stop_error is not None:
            raise stop_error("test stop")

    monkeypatch.setattr("concurrent.futures.wait", completed_before_stop)
    monkeypatch.setattr(cap, "maybe_stop", stop)
    if stop_error is None:
        cap.run_caption_loop()
    else:
        with pytest.raises(stop_error, match="test stop"):
            cap.run_caption_loop()

    assert first.with_suffix(".txt").read_text(encoding="utf-8") == "Completed caption."
    assert not second.with_suffix(".txt").exists()
    assert cap.step_num == 1
    caption.assert_called_once_with(str(first))


def test_stopped_worker_does_not_hold_subprocess_open():
    script = """
import threading
from types import SimpleNamespace
from extensions_built_in.captioner.OpenAICaptioner import OpenAICaptioner

started = threading.Event()
blocked = threading.Event()
cap = object.__new__(OpenAICaptioner)
cap.caption_config = SimpleNamespace(concurrency=1)
cap.file_paths = ["blocked.png"]
cap.step_num = 0
cap.is_stopping = False
cap.is_ui_captioner = True
cap.update_step = lambda: None

def caption(path):
    started.set()
    blocked.wait(120)
    return "too late"

def stop():
    if not started.wait(5):
        raise RuntimeError("worker did not start")
    cap.is_stopping = True
    raise KeyboardInterrupt("test stop")

cap.get_caption_for_file = caption
cap.maybe_stop = stop
try:
    cap.run_caption_loop()
except KeyboardInterrupt:
    print("STOPPED", flush=True)
else:
    raise AssertionError("stop was not propagated")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STOPPED" in result.stdout
