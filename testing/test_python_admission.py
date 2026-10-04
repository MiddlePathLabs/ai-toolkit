"""Admission import and runtime progress regressions."""
from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import types
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]


def test_admission_import_requires_only_the_standard_library():
    script = """
import builtins

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith(("huggingface_hub", "yaml")):
        raise AssertionError(f"unexpected training dependency: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import toolkit.admission
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_hook_keeps_hub_progress_visible(monkeypatch):
    hf_tqdm = types.ModuleType("huggingface_hub.utils.tqdm")
    hf_tqdm.is_tqdm_disabled = lambda log_level: None
    utils = types.ModuleType("huggingface_hub.utils")
    hub = types.ModuleType("huggingface_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils.tqdm", hf_tqdm)

    progress = importlib.import_module("toolkit")
    progress.force_hf_hub_progress_bars()

    assert hf_tqdm.is_tqdm_disabled(logging.INFO) is False
    assert hf_tqdm.is_tqdm_disabled(logging.NOTSET) is None


