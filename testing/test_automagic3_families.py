import torch
import torch.nn as nn

from toolkit.optimizers.automagic3 import Automagic3, tag_param_families


class TwoFamiliesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)


def _make_opt(net, **kwargs):
    params = [p for p in net.parameters() if p.requires_grad]
    kwargs.setdefault("fused", False)
    return Automagic3(params, lr=1e-6, **kwargs)


def _run(opt, grad_streams, steps):
    # grad_streams: list of (param, [grad sign per step]) -- fp32 grads
    for _ in range(steps):
        for p, signs in grad_streams:
            g = 1.0 if signs[_] > 0 else -1.0
            p.grad = torch.full_like(p, g)
        opt.step()


def test_default_sign_window_is_16():
    net = TwoFamiliesNet()
    opt = _make_opt(net)
    p = net.attn.weight
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.state[p]["sign_history"].shape[0] == 16


def test_polarity_history_override_still_works():
    net = TwoFamiliesNet()
    opt = _make_opt(net, polarity_history=8)
    p = net.attn.weight
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.state[p]["sign_history"].shape[0] == 8


def test_family_pools_engage_on_tagged_modules():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net)
    fams = set(opt._param_family_index.values())
    assert len(fams) == 2  # attn + mlp
    # the two params of one module share a family (coupled unit stays whole)
    assert (
        opt._param_family_index[net.attn.weight]
        == opt._param_family_index[net.attn.bias]
    )


def test_family_lrs_diverge_by_regime():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net)
    n = 25
    # attn holds a steady direction (up votes), mlp alternates (overshoot)
    streams = [
        (net.attn.weight, [1] * n),
        (net.attn.bias, [1] * n),
        (net.mlp.weight, [1, -1] * n),
        (net.mlp.bias, [1, -1] * n),
    ]
    _run(opt, streams, n)
    lr_attn = float(opt.state[net.attn.weight]["lr"])
    lr_mlp = float(opt.state[net.mlp.weight]["lr"])
    assert lr_attn > 1e-3  # rose by orders of magnitude
    assert lr_mlp < 1e-7  # backed off hard
    # A/B pair of one module moved together
    assert float(opt.state[net.attn.bias]["lr"]) == lr_attn


def test_single_bucket_falls_back_to_group_pool():
    net = TwoFamiliesNet()
    # pretend both modules belong to one family: untagged -> no split at all
    for p in net.parameters():
        p._optim_family = "attn"
    opt = _make_opt(net)
    assert len(opt._param_family_index) == 0


def test_family_groups_off_is_legacy():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net, family_groups=False)
    assert len(opt._param_family_index) == 0
    # identical vote streams -> identical group lrs (legacy behaviour)
    n = 25
    streams = [(p, [1] * n) for p in net.parameters()]
    _run(opt, streams, n)
    lrs = {float(opt.state[p]["lr"]) for p in net.parameters()}
    assert len(lrs) == 1


def test_untagged_params_stay_on_group_pool():
    net = TwoFamiliesNet()
    tag_param_families(net)
    extra = nn.Parameter(torch.zeros(4))
    params = list(net.parameters()) + [extra]
    opt = Automagic3(params, lr=1e-6, fused=False)
    assert extra not in opt._param_family_index
    n = 25
    streams = [
        (net.attn.weight, [1] * n),
        (net.attn.bias, [1] * n),
        (net.mlp.weight, [1, -1] * n),
        (net.mlp.bias, [1, -1] * n),
        (extra, [1] * n),
    ]
    _run(opt, streams, n)
    # the group pool holds only the untagged param's votes; its lr moved on
    # its own, independent of the family pools
    lr_extra = float(opt.state[extra]["lr"])
    assert 1e-6 < lr_extra <= 1e3


def test_resume_keeps_family_lrs_not_median_unified():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net)
    n = 25
    streams = [
        (net.attn.weight, [1] * n),
        (net.attn.bias, [1] * n),
        (net.mlp.weight, [1, -1] * n),
        (net.mlp.bias, [1, -1] * n),
    ]
    _run(opt, streams, n)
    saved = {k: {kk: vv for kk, vv in v.items()} for k, v in opt.state.items()}
    state_dict = opt.state_dict()

    opt2 = _make_opt(net)
    opt2.load_state_dict(state_dict)
    lr_attn = float(opt2.state[net.attn.weight]["lr"])
    lr_mlp = float(opt2.state[net.mlp.weight]["lr"])
    assert lr_attn > 1e-3 and lr_mlp < 1e-7  # family split survived the load


def test_family_of_rules():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net)
    f = opt._family_of
    assert f("lora_unet_blocks_17_self_attn_qkv") == "attn"
    assert f("lora_unet_blocks_3_mlp_fc1") == "mlp"
    assert f("txtfusion_0") == "txtfusion"
    assert f("lora_unet_text_fusion_blocks_1") == "txtfusion"
    assert f("lora_unet_blocks_5_thing_proj") == "other"


def test_custom_family_rules():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net, family_rules=(("a", ("attn",)),))
    fams = {opt._family_of(tag) for tag in ("attn", "mlp")}
    assert fams == {"a", "other"}


def test_legacy_equal_lr_state_dict_loads():
    net = TwoFamiliesNet()
    tag_param_families(net)
    opt = _make_opt(net)
    n = 2
    # one step of identical constant grads on every family: the saved lrs are
    # all equal, like an older single-pool checkpoint
    streams = [(p, [1] * n) for p in net.parameters()]
    _run(opt, streams, n)
    sd = opt.state_dict()
    lrs_before = {float(v["lr"]) for v in sd["state"].values()}
    assert len(lrs_before) == 1

    opt2 = _make_opt(net)
    opt2.load_state_dict(sd)  # must not raise
    # equal saved lrs are restored as saved (no median re-unify needed)
    assert {float(v["lr"]) for v in opt2.state_dict()["state"].values()} == lrs_before
    # and the controller keeps working: divergent streams split the families
    m = 25
    streams = [
        (net.attn.weight, [1] * m),
        (net.attn.bias, [1] * m),
        (net.mlp.weight, [1, -1] * m),
        (net.mlp.bias, [1, -1] * m),
    ]
    _run(opt2, streams, m)
    assert float(opt2.state[net.attn.weight]["lr"]) > float(
        opt2.state[net.mlp.weight]["lr"]
    )
