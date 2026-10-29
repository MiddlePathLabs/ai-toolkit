"""Hugging Face downloads survive an expired saved login: one anonymous retry, then a short error."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

import toolkit.hf_token_retry as hf_token_retry

ROOT = Path(__file__).resolve().parents[1]

SAVED = "hf_oauth_saved"  # an explicit token, so no test depends on this machine's login


def _expired():
    """Stand-in for the hub's 401 RepositoryNotFoundError on an expired OAuth login."""
    err = RuntimeError(
        "401 Client Error. Repository Not Found for url: "
        "https://huggingface.co/some/public-repo/resolve/x/model.safetensors."
        ' OAuth token has expired: "exp" claim timestamp check failed')
    err.response = types.SimpleNamespace(status_code=401)
    return err


def _refused():
    """The anonymous retry is refused too (a gated or private repo)."""
    err = RuntimeError("401 Client Error for url: https://huggingface.co/some/gated-repo")
    err.response = types.SimpleNamespace(status_code=401)
    return err


class FakeDownload:
    """hf_hub_download stand-in: refuses any call that may carry the saved login."""

    def __init__(self, anon_error=None):
        self.calls = []
        self.anon_error = anon_error

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs.get("token", "<unset>"))
        if kwargs.get("token") is not False:
            raise _expired()
        if self.anon_error is not None:
            raise self.anon_error
        return "model.safetensors"


@pytest.fixture(autouse=True)
def quiet_note(monkeypatch):
    monkeypatch.setattr(hf_token_retry, "_noted", False)


def test_retry_once_without_the_login():
    hub = FakeDownload()
    got = hf_token_retry._with_retry(
        hub, (), {"repo_id": "some/public-repo", "filename": "x", "token": SAVED})
    assert got == "model.safetensors"
    assert hub.calls == [SAVED, False]


def test_the_note_prints_once(capsys):
    hf_token_retry._with_retry(FakeDownload(), (), {"token": SAVED})
    hf_token_retry._with_retry(FakeDownload(), (), {"token": SAVED})
    out = capsys.readouterr().out
    assert out.count("[hf] Saved Hugging Face login was expired") == 1


def test_a_refused_anonymous_retry_raises_login_expired():
    hub = FakeDownload(anon_error=_refused())
    with pytest.raises(hf_token_retry.LoginExpired, match="hf auth login"):
        hf_token_retry._with_retry(hub, (), {"token": SAVED})
    assert hub.calls == [SAVED, False]


def test_an_unrelated_error_is_not_retried():
    calls = []

    def not_found(*args, **kwargs):
        calls.append(kwargs.get("token"))
        raise FileNotFoundError("x is not a local folder and is not a valid model identifer")

    with pytest.raises(FileNotFoundError):
        hf_token_retry._with_retry(not_found, (), {"token": SAVED})
    assert calls == [SAVED]  # one call: nothing was retried


def test_no_retry_when_the_call_already_sent_no_token():
    calls = []

    def anon_401(*args, **kwargs):
        calls.append(kwargs.get("token"))
        raise _refused()

    with pytest.raises(RuntimeError):
        hf_token_retry._with_retry(anon_401, (), {"token": False})
    assert calls == [False]  # one call: no anonymous retry of an anonymous request


def test_positional_arguments_survive_the_retry():
    hub = FakeDownload()

    def picky(repo_id, filename, token=None, **kwargs):
        assert repo_id == "some/public-repo" and filename == "x"
        return FakeDownload.__call__(hub, repo_id=repo_id, filename=filename, token=token)

    got = hf_token_retry._with_retry(
        picky, ("some/public-repo", "x"), {"token": SAVED})
    assert got == "model.safetensors"
    assert hub.calls == [SAVED, False]


def test_install_patches_the_hub_and_rebinds_importers(monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    original_download = FakeDownload()
    original_snapshot = FakeDownload()
    hub.hf_hub_download = original_download
    hub.snapshot_download = original_snapshot
    hub.get_token = lambda: SAVED
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    # a module that bound the name at import time, as diffusers does
    diffusers_module = types.ModuleType("diffusers.pipelines.pipeline_utils")
    diffusers_module.hf_hub_download = original_download
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.pipeline_utils", diffusers_module)

    hf_token_retry.install_hf_token_retry()

    wrapped_download = hub.hf_hub_download
    assert getattr(wrapped_download, "_aitk_token_retry", False)
    assert diffusers_module.hf_hub_download is wrapped_download
    assert getattr(hub.snapshot_download, "_aitk_token_retry", False)

    # the wired wrapper retries end to end
    assert wrapped_download(repo_id="some/public-repo", filename="x", token=SAVED) == "model.safetensors"
    assert original_download.calls == [SAVED, False]

    # idempotent: a second install leaves the wrapper in place
    hf_token_retry.install_hf_token_retry()
    assert hub.hf_hub_download is wrapped_download


def test_module_imports_without_huggingface_hub():
    script = """
import builtins

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith(("huggingface_hub", "yaml")):
        raise AssertionError(f"unexpected training dependency: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import toolkit.hf_token_retry
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
