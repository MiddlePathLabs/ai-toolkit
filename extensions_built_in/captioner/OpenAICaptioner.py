"""Captioner that sends each file to an OpenAI-compatible chat completions endpoint
(llama.cpp server, vLLM, LM Studio, Ollama, SGLang, TabbyAPI, a remote box, or the
OpenAI API itself) instead of loading a model in this process.

Images go up as base64 data URLs, resized to max_res. Videos are sampled into
video_frames evenly spaced stills and sent as one multi-image message, which every
vision server understands. Audio is not supported (no common OpenAI-compatible
input format for it). Requests run concurrently up to `concurrency`.

Config (under `caption:`):
  api_base_url     e.g. http://localhost:8080/v1 (".../chat/completions" also accepted)
  api_key          optional; falls back to the AITK_CAPTION_API_KEY env var
  model_name_or_path  the model name the server expects in the request
  caption_prompt / system_prompt / max_res / max_new_tokens
  temperature      default 0.2
  concurrency      parallel requests, default 2
  video_frames     stills sampled per video, default 8
  request_timeout  seconds per request, default 300
  max_retries      retries on connection errors, 408/429/5xx, default 3
  extra_body       dict merged into the request body (top_k, chat_template_kwargs, ...)
"""

import base64
import concurrent.futures
import io
import os
import re
import time
import traceback
from collections import OrderedDict
from typing import List, Optional

from PIL import Image

from .BaseCaptioner import BaseCaptioner, CaptionConfig

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "webm", "mkv", "avi"}
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def normalize_chat_url(base_url: str) -> str:
    """http://host:8080 -> http://host:8080/v1/chat/completions,
    http://host:8080/v1 -> .../v1/chat/completions, a full URL is kept."""
    from urllib.parse import urlparse

    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("api_base_url is required for the OpenAI-compatible captioner")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    if url.endswith("/chat/completions"):
        return url
    if urlparse(url).path in ("", "/"):
        url += "/v1"
    return url + "/chat/completions"


def clean_caption(text: str) -> str:
    """Drop reasoning blocks and a dangling unclosed <think> preamble."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


class OpenAICaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_base_url = kwargs.get("api_base_url", None) or ""
        self.api_key = kwargs.get("api_key", None) or os.environ.get("AITK_CAPTION_API_KEY", "") or ""
        self.system_prompt = kwargs.get("system_prompt", None) or None
        temperature = kwargs.get("temperature", 0.2)
        self.temperature = None if temperature is None else float(temperature)
        self.concurrency = max(1, int(kwargs.get("concurrency", 2)))
        self.video_frames = max(1, int(kwargs.get("video_frames", 8)))
        self.request_timeout = float(kwargs.get("request_timeout", 300))
        self.max_retries = max(0, int(kwargs.get("max_retries", 3)))
        self.extra_body = kwargs.get("extra_body", None) or {}
        if not isinstance(self.extra_body, dict):
            raise ValueError("extra_body must be a mapping")


class OpenAICaptioner(BaseCaptioner):
    caption_config_class = OpenAICaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(OpenAICaptioner, self).__init__(process_id, job, config, **kwargs)
        self.chat_url = normalize_chat_url(self.caption_config.api_base_url)
        self.session = None

    # ---- lifecycle -------------------------------------------------------

    def load_model(self):
        import requests

        cc = self.caption_config
        self.session = requests.Session()
        if cc.api_key:
            self.session.headers["Authorization"] = f"Bearer {cc.api_key}"
        self.session.headers["Content-Type"] = "application/json"
        print(f"[AITK] Captioning through {self.chat_url} (model '{cc.model_name_or_path}', {cc.concurrency} parallel)")
        if cc.quantize or cc.low_vram or cc.compile or cc.loras:
            print("[AITK] quantize / low_vram / compile / loras are set on the server, not here; ignoring them.")

    def load_loras(self):
        return

    def maybe_compile_models(self):
        return

    # ---- request building ------------------------------------------------

    @staticmethod
    def _ext(file_path: str) -> str:
        return os.path.splitext(file_path)[1].lower().lstrip(".")

    def _image_to_data_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _resize(self, image: Image.Image) -> Image.Image:
        max_res = self.caption_config.max_res
        if not max_res:
            return image
        max_pixels = max_res * max_res
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.BICUBIC)
        return image

    def _video_frames(self, file_path: str) -> List[Image.Image]:
        import cv2

        cap = cv2.VideoCapture(file_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            n = self.caption_config.video_frames
            frames = []
            if total > 0:
                # centers of n equal segments, so the first and last frames are not wasted on fades
                indices = sorted({min(total - 1, int((i + 0.5) * total / n)) for i in range(n)})
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if ok:
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                # no frame count (some webm/mkv): read sequentially, keep the first n
                while len(frames) < n:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"could not read any frames from {file_path}")
        return [self._resize(f) for f in frames]

    def build_messages(self, file_path: str) -> list:
        cc = self.caption_config
        ext = self._ext(file_path)
        prompt = self.resolve_prompt(cc.caption_prompt, file_path) or ""
        content = []
        if ext in IMAGE_EXTENSIONS:
            images = [self._resize(Image.open(file_path).convert("RGB"))]
        elif ext in VIDEO_EXTENSIONS:
            images = self._video_frames(file_path)
            prompt = (
                f"The following {len(images)} images are frames sampled in order from one video. "
                f"Treat them as the video.\n\n{prompt}"
            )
        else:
            raise ValueError(f"unsupported file type for the OpenAI-compatible captioner: .{ext}")
        content.append({"type": "text", "text": prompt})
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}})
        messages = []
        if cc.system_prompt:
            messages.append({"role": "system", "content": cc.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def build_payload(self, file_path: str) -> dict:
        cc = self.caption_config
        payload = {
            "model": cc.model_name_or_path,
            "messages": self.build_messages(file_path),
            "max_tokens": int(cc.max_new_tokens),
            "stream": False,
        }
        if cc.temperature is not None:
            payload["temperature"] = cc.temperature
        payload.update(cc.extra_body)
        return payload

    # ---- request / response ---------------------------------------------

    @staticmethod
    def parse_response(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"response has no choices: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        caption = clean_caption(content or "")
        if not caption:
            reason = choices[0].get("finish_reason")
            raise ValueError(
                f"empty caption (finish_reason={reason}); a reasoning model may have used up max_new_tokens"
            )
        if choices[0].get("finish_reason") == "length":
            print("[AITK] Warning: caption hit max_new_tokens and may be cut off.")
        return caption

    def _post(self, payload: dict) -> dict:
        import requests

        cc = self.caption_config
        last_error = None
        for attempt in range(cc.max_retries + 1):
            if self.is_stopping:
                raise RuntimeError("stopping")
            try:
                resp = self.session.post(self.chat_url, json=payload, timeout=cc.request_timeout)
                if resp.status_code in RETRY_STATUS:
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                elif resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                else:
                    return resp.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
            if attempt < cc.max_retries:
                delay = min(30.0, 2.0 * (2**attempt))
                print(f"[AITK] Request failed ({last_error}); retry {attempt + 1}/{cc.max_retries} in {delay:.0f}s")
                time.sleep(delay)
        raise RuntimeError(f"request failed after {cc.max_retries + 1} attempts: {last_error}")

    def get_caption_for_file(self, file_path: str) -> Optional[str]:
        return self.parse_response(self._post(self.build_payload(file_path)))

    def run_caption_loop(self):
        import tqdm as tqdm_mod

        cc = self.caption_config
        pbar = tqdm_mod.tqdm(total=len(self.file_paths), desc="Captioning files", unit="file")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=cc.concurrency)
        failures = 0
        try:
            pending = {}
            file_iter = iter(self.file_paths)

            def submit_next():
                path = next(file_iter, None)
                if path is not None:
                    pending[executor.submit(self.get_caption_for_file, path)] = path

            for _ in range(cc.concurrency):
                submit_next()
            while pending:
                done, _ = concurrent.futures.wait(
                    list(pending), timeout=2.0, return_when=concurrent.futures.FIRST_COMPLETED
                )
                if self.is_ui_captioner:
                    try:
                        self.maybe_stop()
                    finally:
                        if self.is_stopping:
                            for fut in pending:
                                fut.cancel()
                if self.is_stopping:
                    break
                for fut in done:
                    path = pending.pop(fut)
                    try:
                        caption = fut.result()
                        if caption is not None:
                            self.save_caption_for_file(path, caption)
                    except Exception as e:
                        failures += 1
                        print(f"Error captioning {path}: {e}")
                        if not isinstance(e, (RuntimeError, ValueError)):
                            traceback.print_exc()
                    self.step_num += 1
                    self.update_step()
                    pbar.update(1)
                    submit_next()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()
        if failures:
            print(f"[AITK] {failures} file(s) failed to caption.")
        if self.file_paths and failures == len(self.file_paths):
            raise RuntimeError(f"every request to {self.chat_url} failed; check the endpoint and model name")
