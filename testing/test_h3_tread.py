from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)
from toolkit.config_modules import TrainConfig
from toolkit.h3_tread import (
    MIN_POST_REJOIN_BLOCKS,
    bind_tread,
    gather_tread_state,
    plan_tread_keep_idx,
    scatter_tread_hidden,
    uses_tread,
    validate_tread_span,
)
from toolkit.optimizer import get_optimizer


TREAD_START = 1
TREAD_END = 5
N_LAYERS = 8  # remaining after rejoin = 3


class _H3:
    arch = "minimax_h3"

    def __init__(self, transformer):
        self.model = transformer


def _tiny_params(**kwargs):
    values = dict(
        hidden_size=32,
        num_layers=N_LAYERS,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=64,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=2,
    )
    values.update(kwargs)
    return MiniMaxH3TransformerParams(**values)


def _tiny_dit(**kwargs) -> MiniMaxH3Transformer:
    dit = MiniMaxH3Transformer(_tiny_params(**kwargs))
    dit.eval()
    return dit


def _pack(
    *,
    t_lat: int,
    n_cond: int = 2,
    pad: bool = False,
    batch: int = 1,
    grid_hw=(2, 2),
):
    """[text | cond video | audio | target video] (+ optional pad)."""
    n_text = 2
    n_audio = 2
    gh, gw = grid_hw
    n_tgt = t_lat * gh * gw
    n_video = n_cond + n_tgt
    seq = n_text + n_cond + n_audio + n_tgt
    if pad:
        seq += 1
    text_indices = torch.arange(n_text)
    cond = torch.arange(n_text, n_text + n_cond)
    audio_indices = torch.arange(n_text + n_cond, n_text + n_cond + n_audio)
    target = torch.arange(n_text + n_cond + n_audio, n_text + n_cond + n_audio + n_tgt)
    video_indices = torch.cat([cond, target]) if n_cond else target.clone()
    tags = torch.full((seq,), -1 if pad else 1, dtype=torch.long)
    tags[text_indices] = 1
    tags[video_indices] = 0
    tags[audio_indices] = 2
    token_tags = tags.unsqueeze(0).expand(batch, -1).contiguous()
    row_t = torch.full((batch, seq), 0.5)
    row_t[:, audio_indices] = 0.4
    pos = torch.zeros(batch, seq, 3)
    pos[:, :, 0] = torch.arange(seq).float()
    video_patch = 4 * 1 * 2 * 2
    return {
        "hidden_states": torch.randn(batch, n_video, video_patch),
        "audio_hidden_states": torch.randn(batch, n_audio, 8),
        "encoder_hidden_states": torch.randn(batch, n_text, 16),
        "row_timesteps": row_t,
        "token_tags": token_tags,
        "position_ids": pos,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "vsa_video_grid": (t_lat, gh, gw),
    }, {
        "seq_len": seq,
        "n_cond": n_cond,
        "n_tgt": n_tgt,
        "n_video": n_video,
        "n_audio": n_audio,
        "cond": cond,
        "target": target,
        "text_indices": text_indices,
        "audio_indices": audio_indices,
    }


def _clone_pack(pack):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in pack.items()}


def _enable_grads(pack):
    for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        pack[key] = pack[key].detach().requires_grad_(False)
    return pack


def test_train_config_tread_defaults_off():
    cfg = TrainConfig()
    assert cfg.tread_ratio == 0.0
    assert cfg.tread_start == 2
    assert cfg.tread_end == 47
    assert uses_tread(cfg) is False
    cfg = TrainConfig(tread_ratio=0.5, tread_start=2, tread_end=47)
    assert uses_tread(cfg) is True
    assert cfg.tread_ratio == pytest.approx(0.5)


def test_train_config_rejects_bad_ratio_and_span():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        TrainConfig(tread_ratio=1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        TrainConfig(tread_ratio=-0.1)
    with pytest.raises(ValueError, match="finite number"):
        TrainConfig(tread_ratio=float("nan"))
    with pytest.raises(ValueError, match="tread_start must be < tread_end"):
        TrainConfig(tread_ratio=0.5, tread_start=10, tread_end=10)
    with pytest.raises(ValueError, match="tread_start must be >= 0"):
        TrainConfig(tread_ratio=0.5, tread_start=-1, tread_end=10)
    with pytest.raises(ValueError, match="tread_start must be an integer"):
        TrainConfig(tread_ratio=0.5, tread_start=2.7, tread_end=47)
    with pytest.raises(ValueError, match="tread_end must be an integer"):
        TrainConfig(tread_ratio=0.5, tread_start=2, tread_end=None)
    cfg = TrainConfig(tread_ratio=0.5, tread_start=2.0, tread_end=47)
    assert cfg.tread_start == 2



def test_validate_span_allows_fizgig_prior_rejects_short_tail():
    validate_tread_span(2, 47, 50)
    with pytest.raises(ValueError, match="instability"):
        validate_tread_span(2, 48, 50)
    with pytest.raises(ValueError, match="not inside"):
        validate_tread_span(2, 51, 50)
    with pytest.raises(ValueError, match="not inside"):
        validate_tread_span(5, 5, 50)


def test_bind_off_clears_and_returns_none():
    dit = _tiny_dit()
    dit._tread = (0.5, TREAD_START, TREAD_END)
    dit._tread_generator = torch.Generator()
    assert bind_tread(TrainConfig(), _H3(dit)) is None
    assert dit._tread is None
    assert dit._tread_generator is None


def test_bind_installs_and_rejects_non_h3_and_vsa():
    dit = _tiny_dit()
    cfg = TrainConfig(tread_ratio=0.5, tread_start=TREAD_START, tread_end=TREAD_END)
    assert bind_tread(cfg, _H3(dit)) == (0.5, TREAD_START, TREAD_END)
    assert dit._tread == (0.5, TREAD_START, TREAD_END)
    assert isinstance(dit._tread_generator, torch.Generator)

    class Other:
        arch = "flux"

        def __init__(self):
            self.model = dit

    with pytest.raises(ValueError, match="H3-only"):
        bind_tread(cfg, Other())

    class VSA:
        arch = "minimax_h3_vsa"

        def __init__(self):
            self.model = dit

    with pytest.raises(ValueError, match="VSA"):
        bind_tread(cfg, VSA())

    vsa_dit = _tiny_dit(gate_compress=True)
    with pytest.raises(ValueError, match="gate_compress"):
        bind_tread(cfg, _H3(vsa_dit))

    sparse = _tiny_dit()
    sparse.vsa_sparsity = 0.9
    with pytest.raises(ValueError, match="vsa_sparsity"):
        bind_tread(cfg, _H3(sparse))


def test_bind_rejects_too_few_post_rejoin_blocks():
    dit = _tiny_dit()
    cfg = TrainConfig(
        tread_ratio=0.5,
        tread_start=1,
        tread_end=N_LAYERS - MIN_POST_REJOIN_BLOCKS + 1,
    )
    with pytest.raises(ValueError, match="instability"):
        bind_tread(cfg, _H3(dit))


def test_same_seed_selects_same_positions():
    pack, meta = _pack(t_lat=2)
    kwargs = dict(
        seq_len=meta["seq_len"],
        batch_size=1,
        video_indices=pack["video_indices"],
        target_video_grid=pack["vsa_video_grid"],
        ratio=0.5,
        start=TREAD_START,
        end=TREAD_END,
        n_blocks=N_LAYERS,
        device=torch.device("cpu"),
    )
    torch.manual_seed(0)
    a = plan_tread_keep_idx(**kwargs)
    torch.manual_seed(0)
    b = plan_tread_keep_idx(**kwargs)
    torch.manual_seed(1)
    c = plan_tread_keep_idx(**kwargs)
    assert a is not None and b is not None and c is not None
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_keep_idx_from_identical_generators_matches():
    pack, meta = _pack(t_lat=2)
    kwargs = dict(
        seq_len=meta["seq_len"],
        batch_size=1,
        video_indices=pack["video_indices"],
        target_video_grid=pack["vsa_video_grid"],
        ratio=0.5,
        start=TREAD_START,
        end=TREAD_END,
        n_blocks=N_LAYERS,
        device=torch.device("cpu"),
    )
    a = plan_tread_keep_idx(**kwargs, generator=torch.Generator().manual_seed(7))
    b = plan_tread_keep_idx(**kwargs, generator=torch.Generator().manual_seed(7))
    assert a is not None and b is not None
    assert torch.equal(a, b)


def test_tread_forward_does_not_consume_global_rng():
    dit = _tiny_dit()
    bind_tread(
        TrainConfig(tread_ratio=0.5, tread_start=TREAD_START, tread_end=TREAD_END),
        _H3(dit),
    )
    pack, _ = _pack(t_lat=2)

    torch.manual_seed(123)
    with torch.enable_grad():
        dit(**_clone_pack(pack))
    after_on = torch.randint(0, 1000, (1,))

    dit._tread = None
    torch.manual_seed(123)
    with torch.enable_grad():
        dit(**_clone_pack(pack))
    after_off = torch.randint(0, 1000, (1,))
    assert torch.equal(after_on, after_off)


def test_keep_idx_keeps_text_condition_audio_drops_half_target():
    pack, meta = _pack(t_lat=2, n_cond=2)
    torch.manual_seed(0)
    keep = plan_tread_keep_idx(
        seq_len=meta["seq_len"],
        batch_size=1,
        video_indices=pack["video_indices"],
        target_video_grid=pack["vsa_video_grid"],
        ratio=0.5,
        start=TREAD_START,
        end=TREAD_END,
        n_blocks=N_LAYERS,
        device=torch.device("cpu"),
    )
    assert keep is not None
    kept = set(keep.tolist())
    assert set(meta["text_indices"].tolist()).issubset(kept)
    assert set(meta["audio_indices"].tolist()).issubset(kept)
    assert set(meta["cond"].tolist()).issubset(kept)
    target = set(meta["target"].tolist())
    kept_tgt = target & kept
    n_keep = max(1, int(round(meta["n_tgt"] * 0.5)))
    assert len(kept_tgt) == n_keep
    assert keep.tolist() == sorted(keep.tolist())
    predicted = meta["seq_len"] - (meta["n_tgt"] - n_keep)
    assert keep.numel() == predicted


def test_stills_and_batch_gt1_do_not_route():
    still, meta = _pack(t_lat=1)
    assert plan_tread_keep_idx(
        seq_len=meta["seq_len"],
        batch_size=1,
        video_indices=still["video_indices"],
        target_video_grid=still["vsa_video_grid"],
        ratio=0.5,
        start=TREAD_START,
        end=TREAD_END,
        n_blocks=N_LAYERS,
        device=torch.device("cpu"),
    ) is None
    clip, meta = _pack(t_lat=2, batch=2)
    assert plan_tread_keep_idx(
        seq_len=meta["seq_len"],
        batch_size=2,
        video_indices=clip["video_indices"],
        target_video_grid=clip["vsa_video_grid"],
        ratio=0.5,
        start=TREAD_START,
        end=TREAD_END,
        n_blocks=N_LAYERS,
        device=torch.device("cpu"),
    ) is None


def test_scatter_restores_dropped_rows_identity():
    x_full = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
    keep = torch.tensor([0, 1, 3, 5])
    reduced = x_full.index_select(1, keep) + 1
    out = scatter_tread_hidden(reduced, x_full, keep)
    assert torch.equal(out[:, keep], reduced)
    drop = torch.tensor([2, 4])
    assert torch.equal(out[:, drop], x_full[:, drop])


def test_gather_subsets_mask_key_columns():
    x = torch.randn(1, 4, 8)
    cos = torch.randn(1, 4, 12)
    sin = torch.randn(1, 4, 12)
    adaln = torch.arange(4).view(1, 4)
    mask = torch.tensor([[[[True, True, False, True]]]])
    keep = torch.tensor([0, 2, 3])
    x_r, rotary_r, adaln_r, mask_r = gather_tread_state(
        x, (cos, sin), adaln, mask, keep
    )
    assert x_r.shape[1] == 3
    assert rotary_r[0].shape[1] == 3
    assert adaln_r.shape[1] == 3
    assert mask_r.shape[-1] == 3
    assert torch.equal(mask_r, mask.index_select(-1, keep))


def test_off_and_still_paths_are_bit_identical():
    torch.manual_seed(0)
    dit = _tiny_dit()
    still, _ = _pack(t_lat=1)
    clip, _ = _pack(t_lat=2)

    dit._tread = None
    with torch.no_grad():
        v0, a0 = dit(**still)
    dit._tread = (0.5, TREAD_START, TREAD_END)
    with torch.enable_grad():
        v1, a1 = dit(**still)
    assert torch.equal(v0, v1)
    assert torch.equal(a0, a1)

    dit._tread = None
    torch.manual_seed(3)
    with torch.enable_grad():
        cv0, ca0 = dit(**_clone_pack(clip))
    dit._tread = (0.0, TREAD_START, TREAD_END)
    torch.manual_seed(3)
    with torch.enable_grad():
        cv1, ca1 = dit(**_clone_pack(clip))
    # ratio 0 is treated as no-route by plan_tread_keep_idx
    assert torch.equal(cv0, cv1)
    assert torch.equal(ca0, ca1)


def test_inference_never_routes():
    torch.manual_seed(0)
    dit = _tiny_dit()
    pack, _ = _pack(t_lat=2)
    dit._tread = (0.5, TREAD_START, TREAD_END)
    with torch.no_grad():
        v_on, a_on = dit(**_clone_pack(pack))
    dit._tread = None
    with torch.no_grad():
        v_off, a_off = dit(**_clone_pack(pack))
    assert torch.equal(v_on, v_off)
    assert torch.equal(a_on, a_off)


def test_routed_block_length_and_output_covers_all_rows():
    torch.manual_seed(0)
    dit = _tiny_dit()
    dit._tread = (0.5, TREAD_START, TREAD_END)
    pack, meta = _pack(t_lat=2)
    n_keep = max(1, int(round(meta["n_tgt"] * 0.5)))
    predicted = meta["seq_len"] - (meta["n_tgt"] - n_keep)
    seen = []

    def _hook(_mod, args):
        seen.append(int(args[0].shape[1]))

    handle = dit.blocks[TREAD_START].register_forward_pre_hook(_hook)
    handle_end = dit.blocks[TREAD_END].register_forward_pre_hook(_hook)
    try:
        with torch.enable_grad():
            video_out, audio_out = dit(**pack)
    finally:
        handle.remove()
        handle_end.remove()
    assert seen[0] == predicted
    assert seen[1] == meta["seq_len"]
    assert video_out.shape[1] == meta["n_video"]
    assert audio_out.shape[1] == meta["n_audio"]


def test_gradient_checkpointing_finite_outputs_and_grads():
    torch.manual_seed(0)
    dit = _tiny_dit()
    dit.train()
    dit.enable_gradient_checkpointing(True)
    dit._tread = (0.5, TREAD_START, TREAD_END)
    pack, _ = _pack(t_lat=2)
    _enable_grads(pack)
    video_out, audio_out = dit(**pack)
    loss = video_out.float().pow(2).mean() + audio_out.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(video_out).all()
    assert torch.isfinite(audio_out).all()
    grads = [p.grad for p in dit.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_gradient_checkpointing_matches_eager_outputs_and_grads():
    torch.manual_seed(0)
    dit_eager = _tiny_dit()
    dit_ckpt = _tiny_dit()
    dit_ckpt.load_state_dict(dit_eager.state_dict())
    dit_ckpt.enable_gradient_checkpointing(True)
    dit_eager._tread = (0.5, TREAD_START, TREAD_END)
    dit_ckpt._tread = (0.5, TREAD_START, TREAD_END)
    dit_eager._tread_generator = torch.Generator().manual_seed(11)
    dit_ckpt._tread_generator = torch.Generator().manual_seed(11)
    pack, _ = _pack(t_lat=2)
    pack_e = _enable_grads(_clone_pack(pack))
    pack_c = _enable_grads(_clone_pack(pack))

    with torch.enable_grad():
        ve, ae = dit_eager(**pack_e)
        loss_e = ve.float().pow(2).mean() + ae.float().pow(2).mean()
        loss_e.backward()
        vc, ac = dit_ckpt(**pack_c)
        loss_c = vc.float().pow(2).mean() + ac.float().pow(2).mean()
        loss_c.backward()

    assert torch.equal(ve, vc)
    assert torch.equal(ae, ac)
    eager_grads = {n: p.grad for n, p in dit_eager.named_parameters() if p.grad is not None}
    ckpt_grads = {n: p.grad for n, p in dit_ckpt.named_parameters() if p.grad is not None}
    assert eager_grads.keys() == ckpt_grads.keys()
    for name in eager_grads:
        assert torch.equal(eager_grads[name], ckpt_grads[name]), name


def _trainable_one(dit: MiniMaxH3Transformer) -> nn.Parameter:
    for p in dit.parameters():
        p.requires_grad_(False)
    param = dit.blocks[0].attn.out_proj.weight
    param.requires_grad_(True)
    return param


def _smoke_step(optimizer_type: str):
    torch.manual_seed(0)
    dit = _tiny_dit()
    dit.train()
    dit._tread = (0.5, TREAD_START, TREAD_END)
    param = _trainable_one(dit)
    opt = get_optimizer([param], optimizer_type, learning_rate=1e-3)
    pack, _ = _pack(t_lat=2)
    _enable_grads(pack)
    before = param.detach().clone()
    opt.zero_grad(set_to_none=True)
    video_out, audio_out = dit(**pack)
    loss = video_out.float().pow(2).mean() + audio_out.float().pow(2).mean()
    loss.backward()
    if optimizer_type != "automagic2":
        opt.step()
    assert torch.isfinite(loss)
    assert not torch.equal(param.detach(), before)
    return loss.item()


def test_step_time_optimizer_survives_tread():
    _smoke_step("adam")


def test_fused_backward_optimizer_survives_tread():
    _smoke_step("automagic2")
