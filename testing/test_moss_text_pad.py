"""MOSS-Image text-span padding (Fizgig #221 port).

``text_len_pad_multiple`` rounds a ragged batch's text span up so the
cuDNN-first masked attention plans only a few shapes. The pad columns must be
invisible: key-masked out of every attention and never read by the image-only
output, so the model's output is unchanged. Uniform batches keep the unmasked
fast path (no pad, no masks).
"""
import torch

from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    MossImageParams,
    MossImageTransformer,
)

_NUM_IMAGE = 8 * 8  # latent H=W=8 -> 64 image tokens


def _tiny_model():
    torch.manual_seed(0)
    params = MossImageParams(
        dim=64,
        ffn_dim=128,
        num_heads=2,
        num_layers=2,
        text_dim=32,
        freq_dim=8,
        latent_channels=4,
    )
    model = MossImageTransformer(params)
    model.eval()
    return model


def _inputs(seed, lengths):
    g = torch.Generator().manual_seed(seed)
    batch = len(lengths)
    hidden = torch.randn(batch, 4, 8, 8, generator=g)
    text = torch.randn(batch, max(lengths), 32, generator=g)
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    timestep = torch.rand(batch, generator=g) * 1000.0
    return hidden, timestep, text, mask


def _text_span(model, hidden, timestep, text, mask):
    """Sequence length reaching the first main block, minus the image tokens."""
    spans = []
    handle = model.layers[0].register_forward_hook(
        lambda module, args, output: spans.append(args[0].shape[1])
    )
    try:
        out = model(hidden, timestep, text, encoder_attention_mask=mask)
    finally:
        handle.remove()
    return spans[0] - _NUM_IMAGE, out


def test_ragged_batch_pads_span_and_output_is_unchanged():
    model = _tiny_model()
    hidden, timestep, text, mask = _inputs(1, lengths=(7, 13))

    model.text_len_pad_multiple = 0
    span_off, out_off = _text_span(model, hidden, timestep, text, mask)
    assert span_off == 13

    model.text_len_pad_multiple = 16
    span_on, out_on = _text_span(model, hidden, timestep, text, mask)
    assert span_on == 16, "ragged text span was not rounded up to the multiple"
    assert out_on.shape == out_off.shape
    assert torch.allclose(out_on, out_off, atol=1e-5), \
        "padding columns changed the model output"


def test_already_aligned_ragged_span_is_left_alone():
    model = _tiny_model()
    hidden, timestep, text, mask = _inputs(2, lengths=(7, 16))
    model.text_len_pad_multiple = 16
    span, _ = _text_span(model, hidden, timestep, text, mask)
    assert span == 16, "an already-aligned span should not gain pad columns"


def test_uniform_batch_keeps_unmasked_path():
    # A uniform batch has no padding mask; rounding would forfeit the unmasked
    # flash path for nothing, so the multiple must not engage.
    model = _tiny_model()
    hidden, timestep, text, mask = _inputs(3, lengths=(9, 9))
    model.text_len_pad_multiple = 16
    span_on, out_on = _text_span(model, hidden, timestep, text, mask)
    assert span_on == 9, "uniform batch was padded despite no ragged captions"
    model.text_len_pad_multiple = 0
    span_off, out_off = _text_span(model, hidden, timestep, text, mask)
    assert span_off == 9
    assert torch.allclose(out_on, out_off, atol=1e-6)


def test_gradient_flows_through_padded_ragged_batch():
    model = _tiny_model()
    model.text_len_pad_multiple = 16
    hidden, timestep, text, mask = _inputs(4, lengths=(5, 11))
    hidden = hidden.requires_grad_(True)
    out = model(hidden, timestep, text, encoder_attention_mask=mask)
    out.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
