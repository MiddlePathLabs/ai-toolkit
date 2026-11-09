"""Trigger-word attention for the training maps: how much of each image token's
attention goes to the caption's trigger word, averaged over heads and blocks
(port of Fizgig v9.3.0's training/attn_probe.py, adapted to the toolkit's
vendored models; see toolkit/training_maps.py).

A vendored model that supports the probe calls two hooks, both no-ops unless a
probe is active (a training-maps capture, run force-eager -- a compiled block
never calls back into Python):

  * layout(img_span, txt_start, total, valid, key_mask) - in the DiT forward,
    once the joint sequence is built: the image tokens' query rows as a
    [start, end) span, where the text tokens start (span start for the
    toolkit's text-first layout), the padded joint length attention sees, how
    many keys are real (a padded tail past `valid` is masked out and ignored),
    and the (B, L) bool key-padding mask. Krea 2 (krea2/src/mmdit.py):
    layout((txtlen, txtlen + imglen), 0, combined.shape[1], txtlen + imglen, padmask).
  * see(q, k) - in each Attention.forward, with q and k as attention uses them
    ((B, H, L, D) after QK-norm and RoPE; k may carry fewer heads, GQA). Calls
    whose sequence length isn't the joint one (the text-only refiner uses the
    same Attention class) are skipped by the probe.

The probe computes softmax(q k^T / sqrt(d)) itself, over every key (the
trigger's share of each row needs the whole row's normaliser), a few heads at a
time, and keeps only the trigger's columns, summed. trigger_positions maps the
trigger word to its positions in the text tokens the model sees, using the
encoder's own template arithmetic (krea2: the instruction prefix is tokenized
and its token count sliced off the features -- see krea2/src/text_encoder.py).
"""
import re
from contextlib import contextmanager

import torch

_ACTIVE = None
HEADS_PER_CHUNK = 4


class TriggerProbe:
    def __init__(self, positions):
        self.pos = sorted({int(p) for p in positions})
        self.layout = None
        self.key_mask = None
        self.acc = None
        self.n = 0

    def see(self, q, k):
        if self.layout is None or not self.pos:
            return
        (a, b), t0, total, valid = self.layout
        if q.shape[-2] != total or k.shape[-2] != total:
            return                      # not the joint sequence (e.g. text refiner)
        cols = [t0 + p for p in self.pos if t0 + p < valid]
        if not cols:
            return
        qi = q[0, :, a:b].float()                                   # [H, Ni, D]
        kk = k[0, :, :valid].float()                                # [Hk, valid, D]
        if kk.shape[0] != qi.shape[0]:                              # GQA: each key head serves a run of query heads
            if qi.shape[0] % kk.shape[0] != 0:
                return
            kk = kk.repeat_interleave(qi.shape[0] // kk.shape[0], dim=0)
        km = None
        if self.key_mask is not None and self.key_mask.dtype == torch.bool:
            km = self.key_mask.reshape(self.key_mask.shape[0], -1)[0, :valid]
        scale = qi.shape[-1] ** -0.5
        out = torch.zeros(qi.shape[1], device=qi.device)
        for h in range(0, qi.shape[0], HEADS_PER_CHUNK):
            s = qi[h:h + HEADS_PER_CHUNK] @ kk[h:h + HEADS_PER_CHUNK].transpose(-1, -2) * scale  # [h, Ni, valid]
            if km is not None:
                s.masked_fill_(~km, float("-inf"))
            lse = torch.logsumexp(s, dim=-1, keepdim=True)
            out += (s[..., cols] - lse).exp().sum(-1).sum(0)
            del s, lse
        out /= qi.shape[0]
        self.acc = out if self.acc is None else self.acc + out
        self.n += 1

    def result(self):
        """The trigger's mean share of each image token's attention, over the
        blocks seen, or None."""
        return None if self.acc is None or not self.n else self.acc / self.n


def layout(img_span, txt_start, total, valid=None, key_mask=None):
    p = _ACTIVE
    if p is not None:
        p.layout = ((int(img_span[0]), int(img_span[1])), int(txt_start), int(total),
                    int(total if valid is None else valid))
        p.key_mask = key_mask


def see(q, k):
    p = _ACTIVE
    if p is not None:
        p.see(q, k)


@contextmanager
def capturing(positions):
    """A probe for the trigger's text positions, active for the forward inside.
    Run it eagerly: a compiled block doesn't call back into Python."""
    global _ACTIVE
    probe = TriggerProbe(positions)
    _ACTIVE = probe
    try:
        yield probe
    finally:
        _ACTIVE = None


def trigger_positions(tokenizer, prefix, caption, trigger, max_length=512, prefix_idx=None):
    """The trigger's token positions in the text the model sees, for an encoder
    that builds `prefix + caption` and slices the prefix's tokens off its
    features (krea2's PROMPT_TEMPLATE_ENCODE_PREFIX / PROMPT_TEMPLATE_ENCODE_START_IDX).

    `prefix_idx`, when given, is the encoder's own slice index; it must equal
    len(tokenize(prefix)) or the mapping cannot be trusted ([] is returned).
    Every whole-word occurrence of each trigger (a comma list is several)
    counts. [] when the trigger isn't in the caption or the mapping is unsafe
    (prefix mismatch, or a caption long enough to hit the encoder's truncation,
    which would silently move every position)."""
    if not caption or not trigger:
        return []
    words = [w.strip() for w in str(trigger).split(",") if w.strip()]
    if not words:
        return []
    try:
        enc = tokenizer(prefix + caption, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        drop = len(tokenizer(prefix)["input_ids"])
    except Exception as e:                   # offset mapping needs a fast tokenizer
        raise ValueError(
            f"tokenizer cannot provide offset mappings ({type(e).__name__}: {e})") from e
    if prefix_idx is not None and drop != int(prefix_idx):
        return []                            # the encoder's slice is not the prefix's tokens
    # the encoder truncates at max_length + prefix_idx and only drops tokens
    # past that, so a caption whose tokens exactly fill the window is kept whole
    if len(enc["input_ids"]) > int(prefix_idx or drop) + int(max_length):
        return []                            # caption long enough to be truncated
    spans = []
    for word in words:
        for m in re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", caption, flags=re.IGNORECASE):
            spans.append((len(prefix) + m.start(), len(prefix) + m.end()))
    if not spans:
        return []
    out = []
    for i, (s, e) in enumerate(offsets):
        if i >= drop and e > s and any(e > a and s < b for a, b in spans):
            out.append(i - drop)
    return out
