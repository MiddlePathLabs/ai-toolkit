"""OpenAI-compatible API captioner against a local mock chat-completions server."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
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
        "concurrency": 2,
        "max_retries": 2,
        **caption,
    }
    job = SimpleNamespace(name="t", meta=OrderedDict())
    return OpenAICaptioner(0, job, OrderedDict(caption=cfg, sqlite_db_path=str(tmp_path / "none.db")))


def test_normalize_chat_url():
    assert normalize_chat_url("http://h:8080") == "http://h:8080/v1/chat/completions"
    assert normalize_chat_url("http://h:8080/v1/") == "http://h:8080/v1/chat/completions"
    assert normalize_chat_url("h:1234/v1") == "http://h:1234/v1/chat/completions"
    assert normalize_chat_url("https://x/api/v1/chat/completions") == "https://x/api/v1/chat/completions"
    with pytest.raises(ValueError):
        normalize_chat_url("")


def test_clean_caption():
    assert clean_caption("<think>a\nb</think>\n A cat.") == "A cat."
    assert clean_caption("reasoning without opener</think>A dog.") == "A dog."
    assert clean_caption("  plain  ") == "plain"


def test_end_to_end_images_and_video(tmp_path, server, monkeypatch):
    import cv2

    for i in range(3):
        Image.new("RGB", (200, 100), (i * 40, 0, 0)).save(tmp_path / f"img{i}.png")
    # existing caption is skipped without recaption
    Image.new("RGB", (10, 10)).save(tmp_path / "done.png")
    (tmp_path / "done.txt").write_text("keep me", encoding="utf-8")
    writer = cv2.VideoWriter(str(tmp_path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
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
    assert body["model"] == "test-model" and body["top_k"] == 20 and body["max_tokens"] == 128
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
