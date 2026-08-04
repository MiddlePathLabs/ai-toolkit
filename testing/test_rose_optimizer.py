import pytest
import torch

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.rose import Rose


def _finite_changed(before, model):
    changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    return changed, finite


def test_rose_fp32_step_changes_params_and_keeps_state_empty():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = Rose(model.parameters(), lr=1e-2)

    x = torch.randn(8, 4)
    target = torch.randn(8, 4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()

    changed, finite = _finite_changed(before, model)
    assert changed, "Rose FP32 step did not change any parameter"
    assert finite, "Rose FP32 step produced non-finite parameters"
    assert opt.state == {}, "Rose must not accumulate per-parameter state"


def test_factory_forwards_rose_params_into_param_group():
    model = torch.nn.Linear(4, 4)
    forwarded = {
        'weight_decay': 0.05,
        'wd_schedule': True,
        'centralize': False,
        'stabilize': False,
        'bf16_sr': False,
        'compute_dtype': 'fp32',
    }
    opt = get_optimizer(model.parameters(), 'rose', 1e-3, forwarded)

    assert isinstance(opt, Rose)
    group = opt.param_groups[0]
    assert group['lr'] == pytest.approx(1e-3)
    assert group['weight_decay'] == pytest.approx(0.05)
    assert group['wd_schedule'] is True
    assert group['centralize'] is False
    assert group['stabilize'] is False
    assert group['bf16_sr'] is False
    # 'fp32' string is resolved to the torch dtype, proving forwarding + conversion.
    assert group['compute_dtype'] == torch.float32


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="requires CUDA with BF16 support",
)
def test_rose_bf16_cuda_step_with_stochastic_rounding():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4).to('cuda', dtype=torch.bfloat16)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Defaults leave bf16_sr=True and compute_dtype=fp64, which is the
    # stochastic-rounding path exercised here.
    opt = Rose(model.parameters(), lr=1e-1)

    x = torch.randn(8, 4, device='cuda', dtype=torch.bfloat16)
    target = torch.randn(8, 4, device='cuda', dtype=torch.bfloat16)
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    opt.step()

    changed, finite = _finite_changed(before, model)
    assert changed, "Rose BF16 stochastic-rounding step did not change any parameter"
    assert finite, "Rose BF16 step produced non-finite parameters"
    assert opt.state == {}, "Rose must not accumulate per-parameter state"


def test_rose_rejects_invalid_compute_dtype_and_negative_lr():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="Invalid learning rate"):
        Rose([p], lr=-1.0)
    with pytest.raises(ValueError, match="Invalid compute_dtype string"):
        Rose([p], lr=1e-3, compute_dtype='not-a-dtype')
