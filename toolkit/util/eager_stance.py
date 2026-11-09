"""The force-eager torch.compiler stance as one shared context manager.

Two callers need the same thing: previews (jobs/process/BaseSDTrainProcess)
must not pay a fresh Dynamo/inductor build on the first sample, and the
training-maps trigger probe (toolkit/training_maps.py) needs its Python hooks,
which a compiled block never calls. Both wrap the forward in
torch.compiler.set_stance("force_eager"), degrading to a plain run where the
stance is unavailable (older torch, or set_stance itself raising).
"""
import contextlib

import torch


@contextlib.contextmanager
def force_eager_stance(enabled=True):
    """Run the body with torch's ``force_eager`` compiler stance when enabled:
    every torch.compile'd module executes its original eager code, so callers
    skip the first-call Dynamo/inductor build and Python hooks fire inside
    compiled blocks' forwards. Code outside the stance keeps its compiled
    graphs. Yields whether the stance is actually active, so a caller that
    REQUIRES eager execution can say so once instead of silently producing
    nothing."""
    if not enabled:
        yield False
        return
    ctx = None
    try:
        ctx = torch.compiler.set_stance("force_eager")
        ctx.__enter__()
    except Exception:
        ctx = None
    try:
        yield ctx is not None
    finally:
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
