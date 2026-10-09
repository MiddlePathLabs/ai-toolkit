"""A preview that runs out of GPU memory: training continues, the raw
training weights come back from under the EMA ones, and later previews are
skipped instead of hitting the same wall every sample_every steps."""
import random
from types import SimpleNamespace

import pytest
import torch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess, _is_out_of_memory
from toolkit.config_modules import SampleConfig
from toolkit.ema import ExponentialMovingAverage
from toolkit.stable_diffusion_model import PreviewStateRestorationError


class _FakeSD:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0
        self.weights_seen = None

    def generate_images(self, configs, sampler=None):
        self.calls += 1
        self.weights_seen = self.param.detach().clone()
        if self.error is not None:
            raise self.error


def _host(tmp_path, sd, param):
    host = object.__new__(BaseSDTrainProcess)
    host.accelerator = SimpleNamespace(is_main_process=True)
    host.save_root = str(tmp_path)
    host.device_torch = torch.device("cpu")
    host.sample_config = SampleConfig(prompts=["a photo of sks"], seed=7, sample_steps=8)
    host.first_sample_config = host.sample_config
    host.adapter_config = None
    host.adapter = None
    host.embedding = None
    host.trigger_word = None
    host.logger = SimpleNamespace(log_image=lambda *a, **k: None)
    sd.param = param
    host.sd = sd
    host.ema = ExponentialMovingAverage([param], decay=0.5, use_num_updates=False)
    return host


def _trained_param():
    param = torch.nn.Parameter(torch.zeros(4))
    return param


def _diverge(host, param):
    # EMA trails the raw weights, so eval() visibly swaps different values in
    with torch.no_grad():
        param.add_(1.0)
    host.ema.update()
    with torch.no_grad():
        param.add_(1.0)


@pytest.mark.parametrize(
    "error",
    [
        torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA error: out of memory"),
    ],
)
def test_oom_preview_restores_raw_weights_and_turns_previews_off(tmp_path, error):
    param = _trained_param()
    sd = _FakeSD(error)
    host = _host(tmp_path, sd, param)
    _diverge(host, param)
    raw = param.detach().clone()

    host.sample(100)  # must not raise

    assert sd.calls == 1
    assert not torch.equal(sd.weights_seen, raw)  # the preview ran on EMA weights
    assert torch.equal(param.detach(), raw)  # and the raw ones are back
    assert host._previews_disabled_by_oom

    host.sample(200)
    assert sd.calls == 1  # skipped, no second attempt


def test_non_oom_error_still_raises_but_restores_raw_weights(tmp_path):
    param = _trained_param()
    host = _host(tmp_path, _FakeSD(ValueError("bad prompt")), param)
    _diverge(host, param)
    raw = param.detach().clone()

    with pytest.raises(ValueError):
        host.sample(100)
    assert torch.equal(param.detach(), raw)
    assert not getattr(host, "_previews_disabled_by_oom", False)


def test_successful_preview_is_unchanged(tmp_path):
    param = _trained_param()
    sd = _FakeSD()
    host = _host(tmp_path, sd, param)
    _diverge(host, param)
    raw = param.detach().clone()

    host.sample(100)
    host.sample(200)
    assert sd.calls == 2
    assert torch.equal(param.detach(), raw)


def test_is_out_of_memory():
    assert _is_out_of_memory(torch.cuda.OutOfMemoryError("x"))
    assert _is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not _is_out_of_memory(RuntimeError("shape mismatch"))
    assert not _is_out_of_memory(ValueError("out of memory"))


@pytest.mark.parametrize("error", [None, torch.cuda.OutOfMemoryError("CUDA out of memory")])
def test_random_seed_preview_preserves_python_rng_on_generation_oom(tmp_path, error):
    param = _trained_param()
    host = _host(tmp_path, _FakeSD(error), param)
    host.sample_config = SampleConfig(prompts=["first", "second"], seed=-1)
    host.first_sample_config = host.sample_config
    generate = host.sd.generate_images

    def generate_with_python_rng(configs, sampler=None):
        random.seed(configs[0].seed)
        random.random()
        generate(configs, sampler=sampler)

    host.sd.generate_images = generate_with_python_rng
    state = random.getstate()
    try:
        host.sample(0, is_first=True)
        assert random.getstate() == state
    finally:
        random.setstate(state)


@pytest.mark.parametrize("raw_weights", [False, True])
def test_preview_restoration_failure_aborts_even_raw_weight_sampling(tmp_path, raw_weights):
    param = _trained_param()
    error = PreviewStateRestorationError("Preview training-state restoration failed")
    host = _host(tmp_path, _FakeSD(error), param)
    host.sample_config.samples[0].raw_weights = raw_weights
    _diverge(host, param)
    raw = param.detach().clone()

    with pytest.raises(PreviewStateRestorationError) as raised:
        host.sample(0, is_first=True)
    assert raised.value is error
    assert torch.equal(param.detach(), raw)
    assert not getattr(host, "_previews_disabled_by_oom", False)
