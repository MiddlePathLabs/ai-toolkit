# Copyright 2026 Peter Neill
# SPDX-License-Identifier: Apache-2.0
# Adapted from Fizgig v8.4.2 src/fizgig/families/trigger_check.py: piece
# extraction, clean-join verdicts, cross-subject piece/first-letter clashes,
# and the made-up-word suggester. Toolkit adaptations resolve the tokenizer
# from the process configuration (Krea 2 only) and never touch the network.
"""Check isolation trigger words against the training text encoder's tokenizer.

Fusion joins ``zwxem, a woman`` into ``zwxemwoman``; whether the class word
survives that join as a whole piece is a property of the model's tokenizer,
not of the caption rules. A trigger such as ``fyxo`` makes the encoder read
``fyxowoman`` as ``f | yx | ow | oman`` — the class word is destroyed and
nothing in the caption rules notices. This module answers, per subject:

- does the joined word split cleanly at the trigger/class boundary, both at a
  caption start and mid-caption,
- is the trigger a single piece the encoder already knows,
- does it share a first letter or a piece with another subject's trigger,
- which made-up alternatives would join cleanly.

Tokenizer loading is offline-only: the locally cached training tokenizer is
used when present and an explicit error is returned otherwise. No model
weights are loaded; ``transformers`` is imported lazily.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

# krea2's default text encoder (QWEN3_VL_PATH in
# extensions_built_in/diffusion_models/krea2/krea2.py); the tokenizer rides
# with the encoder repository, not the diffusion checkpoint.
DEFAULT_TEXT_ENCODER_PATH = "Qwen/Qwen3-VL-4B-Instruct"

_TOKENIZER_CACHE: dict[str, Any] = {}


def _pieces(tokenizer: Any, text: str) -> list[str]:
    """The word's pieces as plain text (no space markers), as its caption
    would tokenize it."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode([i]).replace("</w>", "") for i in ids]


def _splits_clean(pieces: list[str], trigger: str, class_name: str) -> bool:
    """The pieces fall exactly at the trigger / class border: some leading
    pieces spell the trigger, the rest spell the class word."""
    text = ""
    for i, piece in enumerate(pieces):
        text += piece.strip()
        if text.lower() == trigger.lower():
            return (
                "".join(q.strip() for q in pieces[i + 1 :]).lower()
                == class_name.lower()
            )
        if len(text) > len(trigger):
            return False
    return False


def trigger_verdict(
    tokenizer: Any,
    trigger: str,
    class_name: str,
    others: Iterable[str] = (),
) -> dict[str, Any]:
    """{ok, joined, pieces, notes}: ok when the joined word splits cleanly at
    both positions and shares no first letter or piece with the other
    subjects' words."""
    trigger, class_name = trigger.strip().strip(","), class_name.strip().lower()
    others = [o for o in (x.strip() for x in others or ()) if o]
    notes: list[str] = []
    ok = True
    if not trigger or " " in trigger:
        return {
            "ok": False,
            "joined": trigger + class_name if class_name else trigger,
            "pieces": [],
            "start_pieces": [],
            "notes": ["the trigger must be one word with no spaces"],
        }
    joined = trigger + class_name if class_name else trigger
    start = [p.strip() for p in _pieces(tokenizer, joined)]
    mid = [p.strip() for p in _pieces(tokenizer, " " + joined)]
    alone = [p.strip() for p in _pieces(tokenizer, " " + trigger)]
    if class_name and not (
        _splits_clean(start, trigger, class_name)
        and _splits_clean(mid, trigger, class_name)
    ):
        ok = False
        tip = (
            "Words ending in a vowel often do this; one ending in a consonant usually joins cleanly."
            if trigger[-1:].lower() in "aeiouy"
            else f"Its ending runs into '{class_name}'. Try a different last letter, or one of the words below."
        )
        notes.append(
            f"breaks '{class_name}': the model reads {joined} as {' | '.join(mid)}, "
            f"so '{class_name}' isn't a whole word any more. {tip}"
        )
    if len(alone) == 1 and len(trigger) > 2:
        notes.append(
            f"'{trigger}' is a single piece the text encoder already knows - it may carry a meaning of its own. "
            "A made-up word in two or three pieces is safer."
        )
    for other in others:
        if other.lower() == trigger.lower():
            continue
        if other[:1].lower() == trigger[:1].lower():
            ok = False
            notes.append(
                f"starts with the same letter as '{other}' - the two subjects can blend. Pick a different first letter."
            )
        shared = {p.lower() for p in alone if len(p) > 1} & {
            p.strip().lower()
            for p in _pieces(tokenizer, " " + other)
            if len(p.strip()) > 1
        }
        if shared:
            ok = False
            notes.append(
                f"shares the piece{'s' if len(shared) > 1 else ''} "
                f"{', '.join(sorted(shared))} with '{other}'."
            )
    return {
        "ok": ok,
        "joined": joined,
        "pieces": mid,
        "start_pieces": start,
        "notes": notes,
    }


# made-up words shaped like the guide's good ones (dlor, gwelt, knurv, zwxem):
# an onset, a vowel, a consonant end
_ONSET = (
    "dl",
    "zw",
    "vl",
    "vr",
    "kv",
    "sv",
    "gw",
    "dv",
    "zv",
    "kn",
    "thr",
    "skr",
    "bl",
    "gr",
    "tr",
    "j",
    "v",
    "z",
    "k",
)
_VOWEL = ("a", "e", "i", "o", "u", "y")
_CODA = (
    "rv",
    "lk",
    "vk",
    "kv",
    "xm",
    "rx",
    "lv",
    "zk",
    "x",
    "sk",
    "rk",
    "nk",
    "lt",
    "rd",
)
# letter pairs that never sit together in an English word: every suggestion
# holds one, so none can be a real word
_NOT_ENGLISH = {
    "zw",
    "dl",
    "vl",
    "vr",
    "kv",
    "vk",
    "sv",
    "zv",
    "dv",
    "zk",
    "kz",
    "xm",
    "rx",
    "wx",
    "xk",
    "vz",
    "jv",
    "xv",
    "vx",
    "qz",
    "zx",
    "fv",
    "vf",
    "pz",
    "jz",
    "zj",
    "bx",
    "xb",
    "mx",
    "dx",
    "vd",
    "tx",
    "kx",
    "xz",
    "gx",
    "xg",
    "lx",
    "xl",
    "nx",
    "vm",
    "vn",
    "vp",
    "vb",
    "zr",
    "lz",
}


def _not_english(word: str) -> bool:
    word = word.lower()
    return any(word[i : i + 2] in _NOT_ENGLISH for i in range(len(word) - 1))


def suggest_triggers(
    tokenizer: Any,
    class_name: str,
    others: Iterable[str],
    avoid: str,
    count: int = 5,
    seed: int = 0,
) -> list[str]:
    """Made-up words that join cleanly with the class word and stay clear of
    the other subjects' words."""
    others = list(others)
    rnd = random.Random(seed)
    used = {o[:1].lower() for o in others if o}
    out: list[str] = []
    tries = 0
    while len(out) < count and tries < 4000:
        tries += 1
        word = rnd.choice(_ONSET) + rnd.choice(_VOWEL) + rnd.choice(_CODA)
        if (
            not 4 <= len(word) <= 6
            or word[0] in used
            or word in out
            or word == avoid
            or not _not_english(word)
        ):
            continue
        verdict = trigger_verdict(tokenizer, word, class_name, [*others, *out])
        if verdict["ok"] and len(verdict["pieces"]) >= 2 and not verdict["notes"]:
            out.append(word)
            used.add(word[0])
    return out


def _resolve_tokenizer_source(process_config: dict) -> str:
    model = process_config.get("model")
    if not isinstance(model, dict):
        raise ValueError("trigger check requires a model configuration")
    arch = str(model.get("arch") or "").lower()
    if arch != "krea2":
        raise ValueError(
            "trigger check covers the Krea 2 text encoder only; "
            "other architectures are outside the declared Isolation boundary"
        )
    model_kwargs = model.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}
    return str(model_kwargs.get("text_encoder_path") or DEFAULT_TEXT_ENCODER_PATH)


def load_trigger_tokenizer(process_config: dict) -> Any:
    """The locally cached Krea 2 tokenizer, or an explicit error. Never
    fetches: this is an inspection helper, not a downloader."""
    source = _resolve_tokenizer_source(process_config)
    if source not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        try:
            _TOKENIZER_CACHE[source] = AutoTokenizer.from_pretrained(
                source, local_files_only=True
            )
        except Exception as error:  # not cached, or unreadable local dir
            raise ValueError(
                f"the {source} tokenizer is not available offline; run one "
                "training or embedding-cache pass with this model first so it "
                f"is cached locally, then retry. ({error})"
            ) from error
    return _TOKENIZER_CACHE[source], source


def check_triggers(process_config: dict, tokenizer: Any = None) -> dict[str, Any]:
    """Per-subject trigger verdicts for one process configuration.

    Subjects are parsed even when the preset is inactive, so words can be
    checked before anything is enabled. ``tokenizer`` overrides the resolved
    offline tokenizer (tests and offline inspection).
    """
    from toolkit.identity_isolation import IsolationConfig

    if not isinstance(process_config, dict):
        raise ValueError("process must contain one training process configuration")
    raw = process_config.get("identity_isolation")
    if not isinstance(raw, dict):
        raise ValueError("process must contain an identity_isolation mapping")
    probe = dict(raw)
    # parse and validate subjects under an inactive preset too
    probe["fused_captions"] = True
    config = IsolationConfig(probe)
    if not config.subjects:
        raise ValueError(
            "identity_isolation.subjects needs at least one subject with a trigger and class before checking"
        )
    if tokenizer is None:
        tokenizer, source = load_trigger_tokenizer(process_config)
    else:
        source = "provided tokenizer"
    results = []
    for index, subject in enumerate(config.subjects):
        others = [
            other["trigger"]
            for other_index, other in enumerate(config.subjects)
            if other_index != index
        ]
        verdict = trigger_verdict(
            tokenizer, subject["trigger"], subject["class_name"], others
        )
        verdict["suggestions"] = (
            []
            if verdict["ok"] and not verdict["notes"]
            else suggest_triggers(
                tokenizer,
                subject["class_name"],
                [*others, subject["trigger"]],
                subject["trigger"],
                seed=sum(map(ord, subject["trigger"])),
            )
        )
        verdict["subject_index"] = index
        verdict["trigger"] = subject["trigger"]
        verdict["class_name"] = subject["class_name"]
        results.append(verdict)
    return {
        "encoder": source,
        "ok": all(result["ok"] for result in results),
        "subjects": results,
    }
