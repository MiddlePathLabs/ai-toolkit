"""Tests for toolkit/attn_probe.py: the trigger-attention probe math and the
trigger-to-token position mapping."""
import pytest
import torch

from toolkit import attn_probe
from toolkit.attn_probe import TriggerProbe, capturing, layout, see, trigger_positions

PREFIX = "system words go here "
PREFIX_TOKENS = 4                                  # the naive tokenizer below makes one token per word


class FakeTok:
    """Whitespace tokenizer with offset mappings: one token per word."""

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        ids, offs = [], []
        i = 0
        for word in text.split(' '):
            if not word:
                i += 1
                continue
            start = text.index(word, i)
            end = start + len(word)
            ids.append(sum(ord(c) for c in word))
            offs.append((start, end))
            i = end + 1
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


def _manual_share(q, k, img_rows, cols, valid, drop=()):
    """Softmax(q k^T / sqrt(d)) summed over the trigger columns, all heads.
    Keys in `drop` are masked out (padding) before the softmax."""
    qi = q[0, :, img_rows[0]:img_rows[1]]         # [H, Ni, D]
    kk = k[0, :, :valid]                          # [H, valid, D]
    s = qi @ kk.transpose(-1, -2) * qi.shape[-1] ** -0.5
    if drop:
        s[..., list(drop)] = float("-inf")
    share = s.softmax(dim=-1)[..., cols].sum(-1)  # [H, Ni]
    return share.mean(0)


def test_probe_share_matches_softmax():
    torch.manual_seed(0)
    total, valid, txt_start, img_span = 40, 36, 0, (8, 32)
    H, D = 4, 8
    q = torch.randn(1, H, total, D)
    k = torch.randn(1, H, total, D)
    cols = [1, 3]                                  # trigger text positions
    with capturing(cols) as probe:
        layout(img_span, txt_start, total, valid=valid,
               key_mask=torch.ones(1, total, dtype=torch.bool))
        see(q, k)
        r = probe.result()
    manual = _manual_share(q, k, img_span, [txt_start + c for c in cols if txt_start + c < valid], valid)
    # a plain list is one group, reported under "" (the union key)
    assert r is not None and set(r) == {""} and r[""].shape == (img_span[1] - img_span[0],)
    assert torch.allclose(r[""], manual, atol=1e-5)


def test_probe_groups_each_word_and_their_union():
    torch.manual_seed(5)
    total, valid, img_span = 40, 36, (8, 32)
    q = torch.randn(1, 4, total, 8)
    k = torch.randn(1, 4, total, 8)
    groups = {"alpha": [1], "beta": [3, 5]}
    with capturing(groups) as probe:
        layout(img_span, 0, total, valid=valid,
               key_mask=torch.ones(1, total, dtype=torch.bool))
        see(q, k)
        r = probe.result()
    assert r is not None and set(r) == {"", "alpha", "beta"}
    for word, cols in groups.items():
        manual = _manual_share(q, k, img_span, cols, valid)
        assert torch.allclose(r[word], manual, atol=1e-5)
    # the union is the two words' columns together; disjoint columns of the
    # same softmax row, so it is exactly the sum of the separate shares
    manual_union = _manual_share(q, k, img_span, [1, 3, 5], valid)
    assert torch.allclose(r[""], manual_union, atol=1e-5)
    assert torch.allclose(r[""], r["alpha"] + r["beta"], atol=1e-5)


def test_probe_gqa_matches_expanded():
    torch.manual_seed(1)
    total, valid, img_span = 24, 24, (6, 18)
    H, Hk, D = 6, 3, 8                             # each key head serves 2 query heads
    q = torch.randn(1, H, total, D)
    k = torch.randn(1, Hk, total, D)
    cols = [2, 4]
    with capturing(cols) as probe:
        layout(img_span, 0, total, key_mask=torch.ones(1, total, dtype=torch.bool))
        see(q, k)
        r = probe.result()[""]
    k_exp = k.repeat_interleave(H // Hk, dim=1)
    manual = _manual_share(q, k_exp, img_span, cols, valid)
    assert r is not None and torch.allclose(r, manual, atol=1e-5)


def test_probe_skips_non_joint_lengths():
    torch.manual_seed(2)
    with capturing([1]) as probe:
        layout((2, 6), 0, 10, key_mask=torch.ones(1, 10, dtype=torch.bool))
        see(torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4))   # text refiner length
    assert probe.result() is None


def test_probe_masks_padded_tail():
    torch.manual_seed(3)
    total, valid, img_span = 32, 28, (6, 24)
    q = torch.randn(1, 2, total, 8)
    k = torch.randn(1, 2, total, 8)
    mask = torch.ones(1, total, dtype=torch.bool)
    mask[0, valid:] = False
    mask[0, 2] = False       # a masked text slot INSIDE the window, so the
                             # masked_fill_ branch actually sees a False
    with capturing([1]) as probe:
        layout(img_span, 0, total, valid=valid, key_mask=mask)
        see(q, k)
        r = probe.result()[""]
    manual = _manual_share(q, k, img_span, [1], valid, drop=(2,))
    assert torch.allclose(r, manual, atol=1e-5)


def test_probe_accumulates_over_blocks():
    torch.manual_seed(4)
    total, img_span = 12, (4, 10)
    qs = [torch.randn(1, 2, total, 8) for _ in range(3)]
    ks = [torch.randn(1, 2, total, 8) for _ in range(3)]
    with capturing([1]) as probe:
        layout(img_span, 0, total, key_mask=torch.ones(1, total, dtype=torch.bool))
        for q, k in zip(qs, ks):
            see(q, k)
        r = probe.result()[""]
    manuals = [_manual_share(q, k, img_span, [1], total) for q, k in zip(qs, ks)]
    mean = torch.stack(manuals).mean(0)
    assert torch.allclose(r, mean, atol=1e-5)


def test_capturing_resets_active_on_exception():
    try:
        with capturing([1]):
            assert attn_probe._ACTIVE is not None
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert attn_probe._ACTIVE is None


def test_trigger_positions_basic():
    caption = "a woman standing in a kitchen"
    tok = FakeTok()
    pos = trigger_positions(tok, PREFIX, caption, "woman",
                            prefix_idx=PREFIX_TOKENS)
    # PREFIX tokenizes to 4 tokens; 'woman' is full-sequence token 5, and the
    # model sees post-slice positions, so 5 - 4 = 1
    assert PREFIX_TOKENS == 4
    assert pos == [1]


def test_trigger_positions_word_boundary_and_list():
    tok = FakeTok()
    caption = "zwxemx woman, zwxem and ZWXEM too"
    # 'zwxemx' is not a whole-word match; the two whole-word zwxem occurrences
    # are caption words 2 and 4 -> post-slice positions 2 and 4
    assert trigger_positions(tok, PREFIX, caption, "zwxem", prefix_idx=PREFIX_TOKENS) == [2, 4]
    assert trigger_positions(tok, PREFIX, caption, "zwxem, woman", prefix_idx=PREFIX_TOKENS) == [1, 2, 4]


def test_trigger_positions_absent_or_unsafe():
    tok = FakeTok()
    assert trigger_positions(tok, PREFIX, "a plain caption", "woman", prefix_idx=4) == []
    assert trigger_positions(tok, PREFIX, "", "woman", prefix_idx=4) == []
    assert trigger_positions(tok, PREFIX, "a woman", "woman", prefix_idx=None) == [1]
    # the encoder's slice index disagrees with the tokenizer's prefix count
    assert trigger_positions(tok, PREFIX, "a woman", "woman", prefix_idx=99) == []
    # a caption long enough to hit the encoder's truncation window
    long_caption = " ".join(["word"] * 600)
    assert trigger_positions(tok, PREFIX, long_caption, "word", prefix_idx=4, max_length=512) == []


def test_trigger_positions_exactly_filling_window_kept():
    # the encoder truncates at max_length + prefix_idx and only drops tokens
    # PAST that, so a caption whose tokens exactly fill the window is kept
    # whole and the mapping stays valid
    tok = FakeTok()
    caption = " ".join(["word"] * 512)        # 4 prefix + 512 caption = the full window
    pos = trigger_positions(tok, PREFIX, caption, "word", prefix_idx=4, max_length=512)
    assert len(pos) == 512
    assert trigger_positions(tok, PREFIX, caption + " word", "word",
                             prefix_idx=4, max_length=512) == []      # one past: truncated


def test_trigger_positions_needs_offset_capable_tokenizer():
    class NoOffsets(FakeTok):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}       # never provides offset mappings
    with pytest.raises(ValueError):
        trigger_positions(NoOffsets(), PREFIX, "a woman", "woman", prefix_idx=3)
