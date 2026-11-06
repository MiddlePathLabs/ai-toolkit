"""MOSS-Image preview checkpoint (train on Base, sample on SFT): config
validation, and the frozen LoRA clone that carries the training network's
current weights onto the preview DiT — on a toy transformer with the real
module names."""
import pytest
import torch
from torch import nn

from extensions_built_in.diffusion_models.moss_image.moss_image import (
    bind_clone_modules,
    build_frozen_lora_clone,
    load_live_lora_weights,
)
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRAModule, LoRASpecialNetwork

W = 8
TARGETS = ["MossImageTransformer"]


class _Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(W, W, bias=False)
        self.k = nn.Linear(W, W, bias=False)
        self.v = nn.Linear(W, W, bias=False)
        self.o = nn.Linear(W, W, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Proj()

    def forward(self, x):
        return x + self.proj.o(self.proj.q(x))


class MossImageTransformer(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.x_in = nn.Linear(W, W, bias=False)
        self.layers = nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, x):
        x = self.x_in(x)
        for block in self.layers:
            x = block(x)
        return x


def _training_network(dit, rank=4, targets=None, **extra):
    config = NetworkConfig(type="lora", linear=rank, linear_alpha=rank)
    # the trainer builds the network from these and records them on it
    ctor_kwargs = dict(
        lora_dim=rank,
        multiplier=1.0,
        alpha=float(rank),
        train_unet=True,
        train_text_encoder=False,
        network_config=config,
        network_type="lora",
        is_transformer=True,
        target_lin_modules=TARGETS if targets is None else targets,
        ignore_if_contains=["text_embedder"],
        **extra,
    )
    net = LoRASpecialNetwork(text_encoder=None, unet=dit, **ctor_kwargs)
    net.aitk_ctor_kwargs = ctor_kwargs
    net.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
    net._update_torch_multiplier()  # the trainer primes this after apply_to
    return net


class TestPreviewCheckpointConfig:
    def test_defaults(self):
        config = ModelConfig(name_or_path="x", arch="moss_image")
        assert config.preview_checkpoint == ""
        assert config.preview_checkpoint_path is None
        assert config.preview_checkpoint_qtype == "convrot8"

    @pytest.mark.parametrize("mode", ["sft", "both", ""])
    def test_moss_modes_accepted(self, mode):
        config = ModelConfig(
            name_or_path="x", arch="moss_image", preview_checkpoint=mode
        )
        assert config.preview_checkpoint == mode

    def test_other_arch_rejected(self):
        with pytest.raises(ValueError, match="only supported for model.arch"):
            ModelConfig(name_or_path="x", arch="krea2", preview_checkpoint="sft")

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="must be 'sft'"):
            ModelConfig(
                name_or_path="x", arch="moss_image", preview_checkpoint="yes"
            )

    def test_bad_qtype_rejected(self):
        with pytest.raises(ValueError, match="qtype string"):
            ModelConfig(
                name_or_path="x",
                arch="moss_image",
                preview_checkpoint="both",
                preview_checkpoint_qtype=8,
            )


class TestFrozenLoraClone:
    def test_clone_matches_training_network_exactly(self):
        torch.manual_seed(0)
        training_dit = MossImageTransformer()
        preview_dit = MossImageTransformer()
        # same base weights: only the attached LoRA may differ
        preview_dit.load_state_dict(training_dit.state_dict())

        net = _training_network(training_dit)
        with torch.no_grad():  # give the LoRA a non-trivial delta
            for m in net.modules():
                if isinstance(m, LoRAModule):
                    m.lora_up.weight.normal_(0, 0.5)

        clone = build_frozen_lora_clone(net, preview_dit)
        assert clone.is_active is False
        assert len(clone.unet_loras) == len(net.unet_loras)

        # load of the live weights, exactly as the preview context does
        load_live_lora_weights(clone, net)

        x = torch.randn(2, W)
        net.is_active = True
        clone.is_active = True
        try:
            out_train = training_dit(x)
            out_preview = preview_dit(x)
            assert torch.equal(out_train, out_preview)
        finally:
            net.is_active = False
            clone.is_active = False

        # inactive: both nets pass through to identical base weights
        base = training_dit(x)
        assert torch.equal(preview_dit(x), base)
        assert not torch.equal(out_train, base)  # the active pass really differed

    def test_clone_is_frozen(self):
        training_dit = MossImageTransformer()
        net = _training_network(training_dit)
        clone = build_frozen_lora_clone(net, MossImageTransformer())
        assert not any(p.requires_grad for p in clone.parameters())

    def test_non_lora_network_rejected(self):
        training_dit = MossImageTransformer()
        net = _training_network(training_dit)
        net.network_type = "lokr"
        with pytest.raises(RuntimeError, match="supports LoRA networks"):
            build_frozen_lora_clone(net, MossImageTransformer())

    def test_untracked_network_rejected(self):
        training_dit = MossImageTransformer()
        net = _training_network(training_dit)
        del net.aitk_ctor_kwargs
        with pytest.raises(RuntimeError, match="aitk_ctor_kwargs"):
            build_frozen_lora_clone(net, MossImageTransformer())

    def test_clone_keeps_training_filters(self):
        # only_if_contains reaches the clone through the recorded kwargs, so
        # the clone carries exactly the trained modules, not every linear
        training_dit = MossImageTransformer()
        net = _training_network(training_dit, only_if_contains=["proj.q"])
        clone = build_frozen_lora_clone(net, MossImageTransformer())
        names = sorted(m.lora_name for m in clone.unet_loras)
        assert names == sorted(m.lora_name for m in net.unet_loras)
        assert names and all(n.endswith("proj$$q") for n in names)

    def test_holder_target_list_not_mutated(self):
        targets = list(TARGETS)
        training_dit = MossImageTransformer()
        net = _training_network(training_dit, targets=targets, conv_lora_dim=2)
        before = list(targets)
        build_frozen_lora_clone(net, MossImageTransformer())
        assert targets == before

    def test_bound_clone_follows_training_multiplier(self):
        torch.manual_seed(0)
        training_dit = MossImageTransformer()
        preview_dit = MossImageTransformer()
        preview_dit.load_state_dict(training_dit.state_dict())
        net = _training_network(training_dit)
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, LoRAModule):
                    m.lora_up.weight.normal_(0, 0.5)
        clone = build_frozen_lora_clone(net, preview_dit)
        load_live_lora_weights(clone, net)
        bind_clone_modules(clone, net)
        x = torch.randn(2, W)
        net.is_active = True
        try:
            for strength in (0.5, 0.0):
                net.multiplier = strength
                assert torch.equal(training_dit(x), preview_dit(x))
        finally:
            net.is_active = False
            bind_clone_modules(clone, clone)
        # unbound again: the clone's own (inactive) state governs
        net.is_active = True
        net.multiplier = 1.0
        try:
            assert not torch.equal(training_dit(x), preview_dit(x))
        finally:
            net.is_active = False
