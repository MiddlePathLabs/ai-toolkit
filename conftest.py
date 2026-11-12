"""Keep every temp file a test run makes inside the repo's gitignored temp/ folder.

  temp/pytest/  pytest's tmp_path / tmpdir (pytest-of-<user>/pytest-N, last 3 runs kept)
  temp/tmp/     tempfile.* calls from tests and toolkit code, and subprocesses
                (TEMP/TMP/TMPDIR); one folder per run, older than a day is removed

Runs at import, before any test module is collected, so module-level tempfile use is
covered too. An explicitly set PYTEST_DEBUG_TEMPROOT is left alone.
"""
import os
import shutil
import tempfile
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TEMP = os.path.join(_ROOT, "temp")
_MAX_AGE_SECONDS = 24 * 3600


def _prune_old_runs(parent: str) -> None:
    now = time.time()
    try:
        entries = os.listdir(parent)
    except OSError:
        return
    for name in entries:
        path = os.path.join(parent, name)
        try:
            if os.path.isdir(path) and now - os.path.getmtime(path) > _MAX_AGE_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def _route_temp() -> None:
    pytest_root = os.path.join(_TEMP, "pytest")
    os.makedirs(pytest_root, exist_ok=True)
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", pytest_root)

    tmp_parent = os.path.join(_TEMP, "tmp")
    os.makedirs(tmp_parent, exist_ok=True)
    _prune_old_runs(tmp_parent)
    # per run (and per xdist worker), so concurrent runs never clean up each other's files
    run_dir = os.path.join(tmp_parent, f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)
    for var in ("TEMP", "TMP", "TMPDIR"):
        os.environ[var] = run_dir
    tempfile.tempdir = run_dir


_route_temp()
