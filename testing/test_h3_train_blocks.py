import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from toolkit.config_modules import NetworkConfig
from toolkit.h3_train_blocks import prepare_train_blocks, verify_train_blocks
from toolkit.lora_special import LoRASpecialNetwork


class _Block(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.qkv = nn.Linear(width, width, bias=False)
        self.fc1 = nn.Linear(width, width, bias=False)
        self.adaln_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return x + self.fc1(torch.relu(self.qkv(x))) + self.adaln_proj(x)


class _Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


class ToyH3Transformer(nn.Module):
    """H3's layout: token_refiner.blocks.<i> nested next to the trunk blocks.<i>."""

    def __init__(self, n=25):
        super().__init__()
        self.token_refiner = _Refiner()
        self.blocks = nn.ModuleList([_Block() for _ in range(n)])
        self.proj_out = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)


class _H3:
    arch = "minimax_h3"
    use_old_lokr_format = False

    def __init__(self, transformer):
        self.model = transformer

    def get_transformer_block_names(self):
        return ["blocks"]


def _build_network(transformer, network_kwargs):
    for p in transformer.parameters():
        p.requires_grad_(False)
    config = NetworkConfig(type="lora", linear=4, linear_alpha=4)
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        lora_dim=4,
        multiplier=1.0,
        alpha=4,
        train_unet=True,
        train_text_encoder=False,
        network_config=config,
        network_type="lora",
        transformer_only=True,
        is_transformer=True,
        base_model=_H3(transformer),
        target_lin_modules=["ToyH3Transformer"],
        **network_kwargs,
    )
    network.force_to("cpu", dtype=torch.float32)
    network._update_torch_multiplier()
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.prepare_grad_etc(None, transformer)
    network.is_active = True  # the trainer's `with network:` context
    return network


def _config(spec):
    return NetworkConfig(type="lora", linear=4, linear_alpha=4, train_blocks=spec)


def test_network_config_default_off():
    assert NetworkConfig().train_blocks is None
    assert NetworkConfig(train_blocks="  ").train_blocks is None
    assert NetworkConfig(train_blocks="20-24").train_blocks == "20-24"


def test_prepare_off_passes_kwargs_through():
    kwargs = {"ignore_if_contains": ["x"]}
    out, indices = prepare_train_blocks(NetworkConfig(), _H3(ToyH3Transformer()), kwargs)
    assert out is kwargs and indices is None


def test_real_network_wraps_exactly_the_window():
    transformer = ToyH3Transformer(25)
    kwargs, indices = prepare_train_blocks(_config("2, 20-24"), _H3(transformer), {})
    assert indices == [2, 20, 21, 22, 23, 24]
    network = _build_network(transformer, kwargs)
    id_to_path = {id(m): n for n, m in transformer.named_modules()}
    paths = sorted(id_to_path[id(lora.orig_module_ref())] for lora in network.unet_loras)
    expected = sorted(
        f"blocks.{i}.{leaf}" for i in indices for leaf in ("qkv", "fc1", "adaln_proj")
    )
    # blocks.2. never matches blocks.20.; the refiner's own blocks.0/1 stay out
    assert paths == expected
    verify_train_blocks(_H3(transformer), network, indices, arm_runtime_check=False)


def test_ignore_if_contains_still_applies():
    transformer = ToyH3Transformer(25)
    kwargs, indices = prepare_train_blocks(
        _config("20-24"), _H3(transformer), {"ignore_if_contains": ["adaln_proj"]}
    )
    assert "token_refiner" in kwargs["ignore_if_contains"]
    network = _build_network(transformer, kwargs)
    assert len(network.unet_loras) == 5 * 2
    verify_train_blocks(_H3(transformer), network, indices, arm_runtime_check=False)


def test_prepare_fails_closed():
    transformer = ToyH3Transformer(25)
    with pytest.raises(ValueError, match="H3-only"):
        prepare_train_blocks(_config("20-24"), SimpleNamespace(arch="flux", model=transformer), {})
    with pytest.raises(ValueError, match="only_if_contains"):
        prepare_train_blocks(_config("20-24"), _H3(transformer), {"only_if_contains": ["blocks.1."]})
    with pytest.raises(ValueError, match="do not exist"):
        prepare_train_blocks(_config("20-49"), _H3(transformer), {})
    with pytest.raises(ValueError, match="text encoder"):
        prepare_train_blocks(_config("20-24"), _H3(transformer), {}, train_text_encoder=True)
    with pytest.raises(ValueError, match="LoRA/LoKr"):
        prepare_train_blocks(
            NetworkConfig(type="locon", train_blocks="20-24"), _H3(transformer), {}
        )


class _Adapter(nn.Module):
    def __init__(self, org):
        super().__init__()
        self.orig_module_ref = weakref.ref(org)
        self.down = nn.Linear(8, 2, bias=False)


def test_verify_rejects_adapters_outside_window_and_empty_blocks():
    transformer = ToyH3Transformer(25)
    outside = SimpleNamespace(
        unet_loras=[_Adapter(transformer.blocks[20].qkv), _Adapter(transformer.token_refiner.blocks[0].qkv)],
        text_encoder_loras=[],
    )
    with pytest.raises(ValueError, match="outside the window"):
        verify_train_blocks(_H3(transformer), outside, [20], arm_runtime_check=False)
    partial = SimpleNamespace(unet_loras=[_Adapter(transformer.blocks[20].qkv)], text_encoder_loras=[])
    with pytest.raises(ValueError, match="got no LoRA"):
        verify_train_blocks(_H3(transformer), partial, [20, 21], arm_runtime_check=False)


def test_runtime_check_passes_when_backward_stops():
    transformer = ToyH3Transformer(25)
    kwargs, indices = prepare_train_blocks(_config("20-24"), _H3(transformer), {})
    network = _build_network(transformer, kwargs)
    verify_train_blocks(_H3(transformer), network, indices)
    transformer(torch.randn(2, 8)).sum().backward()
    # only the window's LoRA params received gradients
    assert all(p.grad is not None for p in network.parameters())


def test_runtime_check_catches_upstream_trainable():
    transformer = ToyH3Transformer(25)
    kwargs, indices = prepare_train_blocks(_config("20-24"), _H3(transformer), {})
    network = _build_network(transformer, kwargs)
    verify_train_blocks(_H3(transformer), network, indices)
    transformer.blocks[3].fc1.weight.requires_grad_(True)  # something trainable upstream
    with pytest.raises(RuntimeError, match="block 19 requires grad"):
        transformer(torch.randn(2, 8))
