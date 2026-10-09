"""network.train_blocks on Qwen-Image 2.1: the 'identity' set (blocks 10-14)
gets the only LoRA modules, and the backward stops at block 10."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from toolkit.config_modules import NetworkConfig
from toolkit.h3_train_blocks import prepare_train_blocks, verify_train_blocks
from toolkit.lora_special import LoRASpecialNetwork

W = 8


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(W, W, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(W, W, bias=False)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()

    def forward(self, x):
        return x + self.attn.to_out[0](torch.relu(self.attn.to_q(x)))


class ToyQwen21Transformer(nn.Module):
    def __init__(self, n=32):
        super().__init__()
        self.img_in = nn.Linear(W, W, bias=False)
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n)])
        self.proj_out = nn.Linear(W, W, bias=False)

    def forward(self, x):
        x = self.img_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.proj_out(x)


class _Qwen:
    arch = "qwen_image_2"
    use_old_lokr_format = False

    def __init__(self, transformer):
        self.model = transformer

    def get_transformer_block_names(self):
        return ["transformer_blocks"]


def _network(transformer, network_kwargs):
    for p in transformer.parameters():
        p.requires_grad_(False)
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        lora_dim=4,
        multiplier=1.0,
        alpha=4,
        train_unet=True,
        train_text_encoder=False,
        network_config=NetworkConfig(type="lora", linear=4, linear_alpha=4),
        network_type="lora",
        transformer_only=True,
        is_transformer=True,
        base_model=_Qwen(transformer),
        target_lin_modules=["ToyQwen21Transformer"],
        **network_kwargs,
    )
    network.force_to("cpu", dtype=torch.float32)
    network._update_torch_multiplier()
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.prepare_grad_etc(None, transformer)
    network.is_active = True
    return network


def _config(spec):
    return NetworkConfig(type="lora", linear=4, linear_alpha=4, train_blocks=spec)


@pytest.mark.parametrize("spec", ["identity", " Identity ", "10-14"])
def test_identity_set_is_blocks_10_to_14(spec):
    transformer = ToyQwen21Transformer()
    kwargs, indices = prepare_train_blocks(_config(spec), _Qwen(transformer), {})
    assert indices == [10, 11, 12, 13, 14]
    assert kwargs["only_if_contains"] == [f"transformer_blocks.{i}." for i in indices]
    # H3's refiner exclusion does not leak into the Qwen layout
    assert "token_refiner" not in kwargs.get("ignore_if_contains", [])


def test_identity_lora_lives_only_in_the_window_and_backward_stops_at_10():
    transformer = ToyQwen21Transformer()
    kwargs, indices = prepare_train_blocks(_config("identity"), _Qwen(transformer), {})
    network = _network(transformer, kwargs)
    verify_train_blocks(_Qwen(transformer), network, indices)  # arms the hook

    names = [m.lora_name for m in network.unet_loras]
    assert len(names) == 5 * 2  # to_q + to_out.0 on each of the 5 blocks
    for module in network.unet_loras:
        torch.nn.init.normal_(module.lora_up.weight)

    out = transformer(torch.randn(3, W))  # the hook checks block 9's output
    out.sum().backward()
    for module in network.unet_loras:
        assert module.lora_down.weight.grad is not None


def test_named_set_is_unknown_on_h3_and_other_archs_are_rejected():
    transformer = ToyQwen21Transformer()
    with pytest.raises(ValueError, match="minimax_h3"):
        prepare_train_blocks(_config("identity"), SimpleNamespace(arch="flux", model=transformer), {})


def test_window_leak_fails_closed():
    transformer = ToyQwen21Transformer()
    _, indices = prepare_train_blocks(_config("identity"), _Qwen(transformer), {})
    leaky = _network(transformer, {"only_if_contains": ["transformer_blocks.3.", "transformer_blocks.10."]})
    with pytest.raises(ValueError, match="outside the window"):
        verify_train_blocks(_Qwen(transformer), leaky, indices, arm_runtime_check=False)
