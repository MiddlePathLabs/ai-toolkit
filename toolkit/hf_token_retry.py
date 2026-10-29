"""Public Hugging Face downloads that survive an expired saved login.

huggingface_hub attaches the saved login (``hf auth login``, ``HF_TOKEN``) to every request,
even for a public repo. Once that OAuth login expires (about a month) the Hub answers 401 and
a repo anyone could download fails with "Repository Not Found ... token has expired".

``install_hf_token_retry()`` wraps ``huggingface_hub.hf_hub_download`` and
``huggingface_hub.snapshot_download``: when a request that carried the login is refused with a
401 (or a token-shaped complaint), it tries once more with ``token=False`` and says so in one
line. Gated and private repos still need a working login: when the anonymous try is refused
too, :class:`LoginExpired` says so plainly.

The patch also rebinds modules that imported these names at import time (diffusers and
transformers bind ``hf_hub_download`` at module level, so patching the hub package alone would
not cover them). This module imports only the standard library so ``toolkit.admission`` keeps
working without training dependencies installed.
"""
from __future__ import annotations

import functools
import sys

NOTE = ("[hf] Saved Hugging Face login was expired; downloading without it. "
        "Run `hf auth login` again when you need private models.")
EXPIRED = ("Hugging Face login refused and this file needs one. Run `hf auth login` "
           "(or `hf auth logout` to download without a login), then retry.")

_noted = False


class LoginExpired(RuntimeError):
    """The saved Hugging Face login was refused and the file cannot be fetched without one."""

    def __init__(self, msg=EXPIRED):
        super().__init__(msg)


def _chain(e):
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def _status(e):
    for x in _chain(e):
        code = getattr(getattr(x, "response", None), "status_code", None)
        if code:
            return code
    return None


def _token_text(e):
    """True when the error text itself says the token was the problem."""
    text = " ".join(str(x) for x in _chain(e)).lower()
    return any(k in text for k in (
        "token has expired", "token is expired", "invalid credentials",
        "invalid token", "invalid user token", '"exp" claim',
    ))


def _sent_token(kwargs):
    token = kwargs.get("token")
    if token is False:
        return False
    if isinstance(token, str) and token:
        return True
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return True  # can't tell: one extra anonymous try is harmless


def is_login_error(e, kwargs=None):
    """A 401 (or a token-shaped complaint) on a request that carried a login."""
    if isinstance(e, LoginExpired):
        return True
    if kwargs is not None and not _sent_token(kwargs):
        return False
    return _status(e) == 401 or _token_text(e)


def _with_retry(fn, args, kwargs):
    """Call fn(*args, **kwargs); on a refused login, retry once with token=False."""
    global _noted
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if not is_login_error(e, kwargs):
            raise
        first = e
    try:
        got = fn(*args, **{**kwargs, "token": False})
    except Exception as second:
        if _status(second) in (401, 403) or "gated" in type(second).__name__.lower():
            raise LoginExpired() from first
        raise
    if not _noted:
        _noted = True
        print(NOTE)
    return got


def _wrap(original):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        return _with_retry(original, args, kwargs)
    wrapper._aitk_token_retry = True
    return wrapper


def install_hf_token_retry():
    """Patch the hub download functions with the expired-login retry (idempotent)."""
    import huggingface_hub
    replacements = []
    for name in ("hf_hub_download", "snapshot_download"):
        original = getattr(huggingface_hub, name, None)
        if original is None or getattr(original, "_aitk_token_retry", False):
            continue
        wrapper = _wrap(original)
        setattr(huggingface_hub, name, wrapper)
        replacements.append((name, original, wrapper))
    if not replacements:
        return
    # diffusers and transformers bind these names at import time; rebind what has
    # already been imported so the retry covers them too. Modules imported later pick
    # the wrapper up from the patched hub package attribute on their own.
    for module in list(sys.modules.values()):
        for name, original, wrapper in replacements:
            if getattr(module, name, None) is original:
                try:
                    setattr(module, name, wrapper)
                except Exception:
                    pass
