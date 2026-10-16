"""_force_eager_stance: previews run torch.compile'd modules uncompiled when a
compile mode is active (Fizgig v7.2.2), degrading to a no-op without the API."""

import pytest
import torch

from jobs.process.BaseSDTrainProcess import _force_eager_stance


def _plain_probe(x):
    return torch.tensor(torch.compiler.is_compiling())


# backend="eager" still routes through Dynamo (is_compiling() is True inside)
# but skips inductor, so the test needs no C compiler
_compiled_probe = torch.compile(_plain_probe, backend="eager")


def _probe_runs_compiled():
    return bool(_compiled_probe(torch.randn(4)).item())


def test_stance_disabled_keeps_compiled_graphs():
    assert _probe_runs_compiled() is True  # warm the compile cache
    with _force_eager_stance(False):
        assert _probe_runs_compiled() is True


# without set_stance the helper is a no-op by design, so eager execution
# cannot be observed on those torch versions
@pytest.mark.skipif(
    not hasattr(torch.compiler, "set_stance"),
    reason="torch.compiler.set_stance unavailable",
)
def test_stance_enabled_runs_eager():
    assert _probe_runs_compiled() is True  # warm the compile cache
    with _force_eager_stance(True):
        assert _probe_runs_compiled() is False
    # and training-speed compiled graphs come back outside the stance
    assert _probe_runs_compiled() is True


def test_stance_errors_degrade_to_plain_body(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("no stance on this torch")

    monkeypatch.setattr(torch.compiler, "set_stance", _boom, raising=False)
    with _force_eager_stance(True):
        assert _probe_runs_compiled() is True  # compiled as usual, no raise
