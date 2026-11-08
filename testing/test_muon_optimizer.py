import copy

import pytest
import torch

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.muon import BatchedMuon
from toolkit.optimizers.muon import Muon as VendoredMuon
from toolkit.optimizer_runtime import (
    OptimizerRuntimeAdapter,
    classify_optimizer,
)


def _finite_changed(before, model):
    changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    return changed, finite


def test_factory_uses_batched_muon_for_all_2d_params():
    model = torch.nn.Linear(4, 4, bias=False)
    opt = get_optimizer(model.parameters(), 'muon', 1e-4, {})
    # BatchedMuon when every parameter is 2D: torch.optim.Muon's update with the
    # per-matrix Newton-Schulz batched over same-shaped matrices (Fizgig #221,
    # v9.2.0 — the per-matrix loop was ~0.33 s of each MOSS batch-3 step; batched
    # the optimizer step is 12x faster, updates identical).
    assert type(opt) is BatchedMuon
    group = opt.param_groups[0]
    assert group['lr'] == pytest.approx(1e-4)
    # Factory defaults: RMS-matched to AdamW so AdamW learning rates carry over.
    assert group['adjust_lr_fn'] == 'match_rms_adamw'
    assert group['weight_decay'] == pytest.approx(0.01)
    assert group['momentum'] == pytest.approx(0.95)
    assert group['nesterov'] is True
    assert group['ns_steps'] == 5


def test_factory_falls_back_to_vendored_muon_for_non_2d_params():
    # A biased Linear mixes 2D weights with a 1D bias; a Conv2d adds a 4D factor.
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Conv2d(3, 4, 3),
    )
    opt = get_optimizer(model.parameters(), 'muon', 1e-4, {})
    assert type(opt) is VendoredMuon
    assert opt.param_groups[0]['adjust_lr_fn'] == 'match_rms_adamw'


def test_factory_passes_param_groups_through_muon():
    # krea2/H3 style: param-group dicts with per-group learning rates.
    w1 = torch.nn.Parameter(torch.randn(4, 4))
    w2 = torch.nn.Parameter(torch.randn(8, 8))
    groups = [
        {'params': [w1], 'lr': 1e-4},
        {'params': [w2], 'lr': 2e-4},
    ]
    opt = get_optimizer(groups, 'muon', 1e-4, {})
    lrs = [g['lr'] for g in opt.param_groups]
    assert lrs == pytest.approx([1e-4, 2e-4])


def test_factory_preserves_generator_backed_param_groups():
    # Regression: the 2D dimension check used to consume a group's generator,
    # handing the optimizer an empty group — a silent no-update run (AdamW
    # branches never pre-iterate, so only muon was affected).
    def build(nonlinearity):
        torch.manual_seed(0)
        if nonlinearity:
            model = torch.nn.Linear(4, 4)          # 2D weight + 1D bias -> vendored
        else:
            model = torch.nn.Linear(4, 4, bias=False)  # all 2D -> BatchedMuon
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        groups = [{'params': (p for p in model.parameters())}]
        opt = get_optimizer(groups, 'muon', 1e-2, {'weight_decay': 0.0})

        assert len(opt.param_groups[0]['params']) == len(before), \
            "generator-backed group lost its parameters"

        x = torch.randn(8, 4)
        target = torch.randn(8, 4)
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), target).backward()
        opt.step()
        changed, finite = _finite_changed(before, model)
        assert changed, "generator-backed Muon run performed no update"
        assert finite

    build(nonlinearity=False)  # BatchedMuon path
    build(nonlinearity=True)   # vendored fallback path (1D bias present)


def test_factory_accepts_bare_tensor_param_group():
    # torch allows a bare tensor as a group's "params"; the muon pre-check
    # must not iterate its rows.
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = get_optimizer([{'params': p}], 'muon', 1e-4, {})
    assert len(opt.param_groups[0]['params']) == 1
    assert opt.param_groups[0]['params'][0] is p


def test_factory_keeps_torch_error_for_group_without_params_key():
    # A malformed group dict must still reach torch's own validation with the
    # same KeyError every sibling branch produces, rather than dying in (or
    # being silently swallowed by) the muon pre-check.
    with pytest.raises(KeyError, match="params"):
        get_optimizer([{'lr': 1e-4}], 'muon', 1e-4, {})


def test_vendored_muon_step_changes_params_and_stays_finite():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = VendoredMuon(model.parameters(), lr=1e-2)

    x = torch.randn(8, 4)
    target = torch.randn(8, 4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), target).backward()
        opt.step()

    changed, finite = _finite_changed(before, model)
    assert changed, "Vendored Muon step did not change any parameter"
    assert finite, "Vendored Muon step produced non-finite parameters"
    assert all('momentum_buffer' in opt.state[p] for p in model.parameters())


def test_vendored_muon_handles_1d_and_4d_params():
    torch.manual_seed(1)
    model = torch.nn.Conv2d(3, 4, 3)  # 4D weight + 1D bias
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = VendoredMuon(model.parameters(), lr=1e-3)

    x = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 4, 6, 6)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), target).backward()
        opt.step()

    changed, finite = _finite_changed(before, model)
    assert changed, "Vendored Muon left conv/bias params unchanged"
    assert finite, "Vendored Muon step produced non-finite conv/bias params"


def test_muon_rms_matched_adjust_ratio_is_adamw_scaled():
    from toolkit.optimizers.muon import _adjust_lr
    # match_rms_adamw: adjusted lr = lr * 0.2 * sqrt(max(A, B))
    assert _adjust_lr(1e-4, 'match_rms_adamw', (32, 4)) == pytest.approx(1e-4 * 0.2 * 32 ** 0.5)
    assert _adjust_lr(1e-4, 'match_rms_adamw', (4, 32)) == pytest.approx(1e-4 * 0.2 * 32 ** 0.5)


def test_muon_rejects_invalid_lr_and_adjust_fn():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="Learning rate should be >= 0"):
        VendoredMuon([p], lr=-1.0)
    with pytest.raises(ValueError, match="Adjust learning rate function"):
        VendoredMuon([p], lr=1e-4, adjust_lr_fn='not-a-mode')


def _muon_cases():
    cases = []
    if hasattr(torch.optim, "Muon"):
        model = torch.nn.Linear(4, 4, bias=False)
        cases.append(torch.optim.Muon(model.parameters(), lr=1e-4))
    cases.append(BatchedMuon(torch.nn.Linear(4, 4, bias=False).parameters(), lr=1e-4))
    conv = torch.nn.Conv2d(3, 4, 3)
    cases.append(VendoredMuon(conv.parameters(), lr=1e-4))
    return cases


@pytest.mark.parametrize("opt", _muon_cases(), ids=lambda o: type(o).__module__)
def test_muon_classified_as_group_lr_step_time(opt):
    caps = classify_optimizer(opt)
    assert caps.step_scale_strategy == 'group_lr'
    assert caps.update_phase == 'step'
    assert caps.supports_step_scale is True
    assert caps.supports_active_param_mask is True
    assert caps.mask_strategy == 'grad_none'


def test_muon_group_lr_window_halves_update_and_restores_lr():
    # With weight decay off, Muon's update is exactly proportional to group lr,
    # so a 0.5 step scale must halve the parameter delta and restore the lr after.
    def run(scale):
        torch.manual_seed(7)
        model = torch.nn.Linear(8, 8, bias=False)
        opt = get_optimizer(model.parameters(), 'muon', 1e-3, {'weight_decay': 0.0})
        adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type='muon')
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), y).backward()
        before = model.weight.detach().clone()
        adapter.begin_window(opt, scale)
        assert opt.param_groups[0]['lr'] == pytest.approx(1e-3 * scale)
        opt.step()
        adapter.end_window(opt)
        assert opt.param_groups[0]['lr'] == pytest.approx(1e-3)
        return model.weight.detach() - before

    full = run(1.0)
    half = run(0.5)
    assert torch.allclose(half * 2.0, full, atol=1e-7), \
        "Muon group-lr window did not scale the update faithfully"


def test_muon_grad_none_mask_skips_inactive_params():
    torch.manual_seed(3)
    w1 = torch.nn.Parameter(torch.randn(4, 4))
    w2 = torch.nn.Parameter(torch.randn(4, 4))
    before1 = w1.detach().clone()
    before2 = w2.detach().clone()
    opt = get_optimizer([w1, w2], 'muon', 1e-2, {'weight_decay': 0.0})
    adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type='muon')

    w1.grad = torch.randn(4, 4)
    w2.grad = torch.randn(4, 4)
    adapter.begin_window(opt, 1.0, active_params=[w2])
    opt.step()
    adapter.end_window(opt)

    assert torch.equal(w1.detach(), before1), "inactive param moved under grad_none mask"
    assert not torch.equal(w2.detach(), before2), "active param did not move"


def test_admission_knows_muon_step_scale_support():
    from toolkit.admission import _known_step_scale_support
    assert _known_step_scale_support({'optimizer': 'muon'}) is True


# --- BatchedMuon parity (Fizgig #221, v9.2.0): the batched step must stay
# torch.optim.Muon's maths.  A LoRA's hundreds of small matrices are why the
# batching exists; these shapes mix repeats (same-shape grouping), tall and
# wide matrices, and a square.
_BATCHED_SHAPES = [(16, 64), (64, 16), (16, 64), (48, 16), (8, 8), (16, 96), (96, 16)]


def _batched_params(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.02) for s in _BATCHED_SHAPES]


def _batched_grads(step):
    g = torch.Generator().manual_seed(1000 + step)
    return [torch.randn(s, generator=g) for s in _BATCHED_SHAPES]


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="needs torch.optim.Muon")
@pytest.mark.parametrize(
    "kw",
    [
        dict(nesterov=True, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"),
        dict(nesterov=False, weight_decay=0.0, adjust_lr_fn="original"),
        dict(nesterov=True, weight_decay=0.1, adjust_lr_fn=None),
    ],
    ids=["nesterov-rms", "plain-original", "nesterov-original"],
)
def test_batched_muon_matches_torch_muon(kw):
    pa, pb = _batched_params(1), _batched_params(1)
    oa = torch.optim.Muon(pa, lr=1e-3, momentum=0.95, ns_steps=5, **kw)
    ob = BatchedMuon(pb, lr=1e-3, momentum=0.95, ns_steps=5, **kw)
    for step in range(6):
        for p, q, g in zip(pa, pb, _batched_grads(step)):
            p.grad, q.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb):
        assert (a - b).abs().max() <= 1e-3 * a.abs().max(), \
            (tuple(a.shape), (a - b).abs().max().item())


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="needs torch.optim.Muon")
def test_batched_muon_resumes_a_torch_muon_state():
    # A run paused on torch.optim.Muon (or the pre-batching factory path) goes on
    # under BatchedMuon exactly as it would have on torch's.
    pa, pb = _batched_params(2), _batched_params(2)
    kw = dict(lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
              weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    oa = torch.optim.Muon(pa, **kw)
    for step in range(3):
        for p, g in zip(pa, _batched_grads(step)):
            p.grad = g.clone()
        oa.step()
    with torch.no_grad():
        for p, q in zip(pa, pb):
            q.copy_(p)
    ob = BatchedMuon(pb, **kw)
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))  # as a resume loads it: nothing shared
    for step in range(3, 6):
        for p, q, g in zip(pa, pb, _batched_grads(step)):
            p.grad, q.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb):
        assert (a - b).abs().max() <= 1e-3 * a.abs().max()


def _torch_muon_preserves_bf16_non_nesterov_buffer():
    """Probe whether the installed torch.optim.Muon mutates the momentum buffer.

    torch < 2.15: with nesterov=False and bf16 params, ``update`` aliases the
    buffer and Newton-Schulz's no-op ``.bfloat16()`` cast + in-place ``div_``
    normalizes it every step (norm -> 1.0). torch 2.15 copies explicitly, so
    the buffer keeps its lerp value. BatchedMuon targets the fixed semantics
    (its stack always copies), so the bf16 nesterov=False parity case only
    applies where torch's reference does too.
    """
    p = torch.nn.Parameter(torch.eye(4, dtype=torch.bfloat16) * 3.0)
    opt = torch.optim.Muon([p], lr=1e-3, momentum=0.95, nesterov=False, ns_steps=1)
    p.grad = torch.eye(4, dtype=torch.bfloat16) * 3.0
    opt.step()
    buf_norm = opt.state[p]["momentum_buffer"].float().norm().item()
    preserved = buf_norm < 0.6  # mutated -> ~1.0; preserved -> ~0.3 (0.05 * lerp)
    if not preserved:
        del opt.state[p]["momentum_buffer"]  # keep the probe out of shared state
    return preserved


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="needs torch.optim.Muon")
@pytest.mark.parametrize("nesterov", [True, False], ids=["nesterov", "plain"])
def test_batched_muon_matches_torch_muon_bf16(nesterov):
    if not nesterov and not _torch_muon_preserves_bf16_non_nesterov_buffer():
        pytest.skip(
            "installed torch.optim.Muon normalizes the bf16 non-nesterov momentum "
            "buffer in place (pre-2.15 behavior); BatchedMuon targets the fixed "
            "2.15 semantics, so this reference is not comparable in that mode"
        )
    pa = [torch.nn.Parameter(t.to(torch.bfloat16)) for t in _batched_params(5)]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    kw = dict(lr=1e-3, momentum=0.95, nesterov=nesterov, ns_steps=5,
              weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    oa = torch.optim.Muon(pa, **kw)
    ob = BatchedMuon(pb, **kw)
    for step in range(6):
        for p, q, g in zip(pa, pb, _batched_grads(step)):
            p.grad, q.grad = g.clone().to(torch.bfloat16), g.clone().to(torch.bfloat16)
        oa.step()
        ob.step()
    # bf16 params carry ~0.4% relative precision; 6 steps of Newton-Schulz round
    # to a few ulps, so compare at bf16 scale rather than the fp32 tolerance.
    for a, b in zip(pa, pb):
        assert (a.float() - b.float()).abs().max() <= 2e-2 * a.float().abs().max(), \
            (tuple(a.shape), (a.float() - b.float()).abs().max().item())


def test_batched_muon_never_normalizes_its_momentum_buffer_in_place():
    # torch < 2.15's Muon resets the bf16 non-nesterov buffer to unit norm every
    # step; BatchedMuon (like fixed torch) must leave it at its lerp value.
    p = torch.nn.Parameter(torch.eye(4, dtype=torch.bfloat16) * 3.0)
    opt = BatchedMuon([p], lr=1e-3, momentum=0.95, nesterov=False, ns_steps=1)
    p.grad = torch.eye(4, dtype=torch.bfloat16) * 3.0
    opt.step()
    buf_norm = opt.state[p]["momentum_buffer"].float().norm().item()
    assert buf_norm == pytest.approx(0.3, rel=0.2), \
        f"momentum buffer was normalized in place (norm {buf_norm:.3f}, expected the ~0.3 lerp value)"


def test_batched_muon_leaves_params_without_a_grad_alone():
    ps = _batched_params(3)
    opt = BatchedMuon(ps, lr=1e-3, weight_decay=0.01)
    before = [p.detach().clone() for p in ps]
    for i, (p, g) in enumerate(zip(ps, _batched_grads(0))):
        p.grad = g if i % 2 == 0 else None
    opt.step()
    for i, (p, b) in enumerate(zip(ps, before)):
        assert torch.equal(p, b) == (i % 2 == 1)


def test_batched_muon_rejects_non_2d_params():
    p2d = torch.nn.Parameter(torch.randn(4, 4))
    p1d = torch.nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError, match="2D parameters only"):
        BatchedMuon([p2d, p1d], lr=1e-4)
    # ...and the factory routes such lists to the vendored fallback instead.
    opt = get_optimizer([p2d, p1d], 'muon', 1e-4, {})
    assert type(opt) is VendoredMuon


def test_identity_isolation_accepts_batched_muon():
    from toolkit.identity_isolation_training import validate_isolation_optimizer
    from toolkit.optimizer_runtime import OptimizerRuntimeAdapter

    opt = BatchedMuon(_batched_params(4), lr=1e-4)
    adapter = OptimizerRuntimeAdapter.inspect(opt, optimizer_type='muon')
    validate_isolation_optimizer(object(), opt, adapter)  # must not raise
