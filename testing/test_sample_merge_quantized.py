"""Training samples keep the LoRA live on quantized bases instead of merging."""
from types import SimpleNamespace

import pytest
import torch

from toolkit.network_mixins import ToolkitNetworkMixin, network_base_is_quantized
from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer


def _convrot8_linear(n=256, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(n, n, bias=False)
    torch.nn.init.normal_(lin.weight, std=0.02)
    q = get_ostris_quantizer("convrot8")
    if q is None or not convert_linear_to_ostris(lin, q):
        pytest.skip("convrot8 backend unavailable")
    return lin, q


def _network(*base_modules):
    net = SimpleNamespace(
        get_all_modules=lambda: [SimpleNamespace(org_module=[m]) for m in base_modules]
    )
    net.base_is_quantized = lambda: ToolkitNetworkMixin.base_is_quantized(net)
    return net


def test_merge_into_int8_erases_small_deltas():
    # the reason for the rule: nearest re-quantization of an on-grid weight
    # plus a sub-half-step delta lands back on the same codes
    lin, q = _convrot8_linear()
    base = q.dequantize(lin).clone()
    d = torch.randn_like(base)
    d = d / d.pow(2).mean().sqrt() * base.pow(2).mean().sqrt() * 0.002
    q.requantize_(lin, base + d)
    kept = ((q.dequantize(lin) - base) * d).sum() / (d * d).sum()
    assert kept.item() < 0.1


def test_detects_quantized_and_plain_bases():
    lin, _ = _convrot8_linear()
    assert network_base_is_quantized(_network(torch.nn.Linear(4, 4), lin))
    assert not network_base_is_quantized(_network(torch.nn.Linear(4, 4)))


def test_networks_without_the_check_can_still_merge():
    assert not network_base_is_quantized(SimpleNamespace())
    assert not network_base_is_quantized(None)
