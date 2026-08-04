"""Instrumentation for Phase 2 depth-anchor hooks during real Krea 2 training.

Captures per-optimizer-step depth signals: the raw depth-consistency loss and
its SSI/gradient components, the resolved objective parity (diffusion vs depth
step), whether the depth block engaged, the LoRA-parameter gradient norm
(captured post-clip / pre-step), and the tagged LoRA tensor before and after
the optimizer step so the strict depth-only parameter-movement proof has a
definitive delta.

All captured tensors use ``.detach().cpu().clone()`` -- no retained autograd
graphs, no live GPU tensors held across steps.

Install on a process instance via ``recorder.install(process)`` BEFORE
``job.run()``, exactly like the Phase 1 NoisingRecorder.
"""
from dataclasses import dataclass, field

import torch


def _clone(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


@dataclass
class DepthStepRecord:
    """Per-optimizer-step capture for the depth-anchor path."""

    optimizer_step: int = 0
    is_accumulation_step: bool = False
    microbatch_index: int = 0
    # Losses (None when the producing path did not run this step).
    total_loss: float | None = None
    depth_loss: float | None = None
    depth_ssi: float | None = None
    depth_grad_component: float | None = None
    depth_applied: float | None = None
    depth_processed_indices: list = field(default_factory=list)
    # Objective parity (True = diffusion step, False = depth step). None when
    # depth is inactive so parity is meaningless.
    step_is_diffusion: bool | None = None
    depth_block_engaged: bool = False
    depth_gt_present: bool = False
    # Gradient + parameter movement on the tagged LoRA tensor.
    lora_grad_norm: float | None = None
    lora_grad_norm_all: float | None = None
    tagged_before_step: dict = field(default_factory=dict)
    tagged_after_step: dict = field(default_factory=dict)
    # CUDA memory snapshots at the step boundary (GB), None without CUDA.
    cuda_allocated_gb: float | None = None
    cuda_reserved_gb: float | None = None

    @property
    def tagged_delta_norm(self) -> float | None:
        """L2 norm of (after - before) for the tagged param, or None."""
        name = next(iter(self.tagged_after_step), None)
        if name is None or name not in self.tagged_before_step:
            return None
        diff = self.tagged_after_step[name].float() - self.tagged_before_step[name].float()
        return float(diff.norm().item())


class DepthRecorder:
    """Captures depth-anchor activity during a training run.

    Wraps the same process-method boundaries as the Phase 1 NoisingRecorder:
    ``hook_before_train_loop`` (defer until the optimizer is prepared),
    ``hook_train_loop`` (step boundary + loss dict + ``_last_depth_*`` attrs),
    ``_inject_gradient_noise`` (post-clip, pre-step -- the LoRA grad is live),
    and ``optimizer.step`` (the parameter update itself).
    """

    def __init__(self, tagged_param_name: str | None = None):
        self.records: list[DepthStepRecord] = []
        self.current_record: DepthStepRecord | None = None
        self.microbatch_count: int = 0
        self.optimizer_step_count: int = 0
        self.tagged_param = None
        self.tagged_param_name = tagged_param_name
        self.all_tagged_names: list[str] = []
        self._installed = False

    def _collect_params(self, process):
        tagged = []
        for group in process.optimizer.param_groups:
            for p in group["params"]:
                name = getattr(p, "_param_name", f"param_id_{id(p)}")
                if getattr(p, "_is_lora", False):
                    tagged.append((name, p))
                    self.all_tagged_names.append(name)
        if self.tagged_param is None and tagged:
            self.tagged_param_name = tagged[0][0]
            self.tagged_param = tagged[0][1]
        return tagged

    def install(self, process):
        if self._installed:
            return
        self._installed = True

        original_hook = process.hook_before_train_loop
        recorder = self

        def instrumented_hook():
            original_hook()
            recorder._post_prepare(process)

        process.hook_before_train_loop = instrumented_hook

    def _post_prepare(self, process):
        self._collect_params(process)
        recorder = self

        orig_grad_noise = process._inject_gradient_noise
        orig_optimizer_step = process.optimizer.step
        orig_train_single = process.train_single_accumulation
        orig_hook_train_loop = process.hook_train_loop

        def wrapped_hook_train_loop(batch):
            is_accum = bool(getattr(process, "is_grad_accumulation_step", False))
            recorder.current_record = DepthStepRecord(
                optimizer_step=int(getattr(process, "step_num", 0)),
                is_accumulation_step=is_accum,
            )
            result = orig_hook_train_loop(batch)

            rec = recorder.current_record
            if rec is not None:
                if isinstance(result, dict):
                    rec.total_loss = result.get("loss")
                # Depth metrics persist on the process as _last_depth_* (they
                # are NOT in hook_train_loop's loss-dict emit list, so they
                # survive until the next depth computation overwrites them).
                rec.depth_loss = getattr(process, "_last_depth_consistency_loss", None)
                rec.depth_ssi = getattr(process, "_last_depth_consistency_ssi", None)
                rec.depth_grad_component = getattr(
                    process, "_last_depth_consistency_grad", None
                )
                rec.depth_processed_indices = list(
                    getattr(process, "_last_depth_processed_indices", []) or []
                )
                rec.depth_block_engaged = bool(
                    getattr(process, "_last_depth_processed_indices", None)
                )
                # Parity is meaningful only when depth is active for this process.
                if process.depth_consistency_config is not None and process._depth_loss_active():
                    rec.step_is_diffusion = (process.step_num % 2 == 0)
                if torch.cuda.is_available():
                    rec.cuda_allocated_gb = torch.cuda.memory_allocated() / 1e9
                    rec.cuda_reserved_gb = torch.cuda.memory_reserved() / 1e9
                if not is_accum:
                    recorder.records.append(rec)
                    recorder.optimizer_step_count += 1
            return result

        process.hook_train_loop = wrapped_hook_train_loop

        def wrapped_grad_noise():
            # Runs post-clip, pre-step on non-accumulation boundaries: the
            # tagged LoRA parameter's gradient is live here (zeroed only inside
            # hook_train_loop after optimizer.step).
            rec = recorder.current_record
            if rec is not None and not rec.is_accumulation_step:
                if recorder.tagged_param is not None:
                    rec.tagged_before_step = {
                        recorder.tagged_param_name: _clone(recorder.tagged_param.data)
                    }
                    g = recorder.tagged_param.grad
                    if g is not None:
                        rec.lora_grad_norm = float(g.detach().norm().item())
                # Aggregate gradient norm across ALL trainable LoRA params --
                # the definitive "did depth reach the LoRA" signal. A single
                # tagged param can be near-zero while others carry the signal.
                sq_sum = 0.0
                found = False
                for group in process.optimizer.param_groups:
                    for p in group["params"]:
                        if not getattr(p, "_is_lora", False):
                            continue
                        if p.grad is None:
                            continue
                        found = True
                        sq_sum += float(p.grad.detach().pow(2).sum().item())
                if found:
                    rec.lora_grad_norm_all = sq_sum ** 0.5
            orig_grad_noise()

        process._inject_gradient_noise = wrapped_grad_noise

        def wrapped_optimizer_step(*args, **kwargs):
            ret = orig_optimizer_step(*args, **kwargs)
            rec = recorder.current_record
            if rec is not None and not rec.is_accumulation_step:
                if recorder.tagged_param is not None:
                    rec.tagged_after_step = {
                        recorder.tagged_param_name: _clone(recorder.tagged_param.data)
                    }
            return ret

        process.optimizer.step = wrapped_optimizer_step

        def wrapped_train_single(batch):
            recorder.microbatch_count += 1
            if recorder.current_record is not None:
                recorder.current_record.microbatch_index += 1
            # Whether this batch carried cached GT depth (the DTO must deliver
            # it before the early return on the latent-cached path).
            has_gt = getattr(batch, "depth_gt_list", None) is not None
            if recorder.current_record is not None and has_gt:
                recorder.current_record.depth_gt_present = True
            return orig_train_single(batch)

        process.train_single_accumulation = wrapped_train_single

    def summary(self) -> dict:
        return {
            "microbatches": self.microbatch_count,
            "optimizer_steps": self.optimizer_step_count,
            "tagged_param": self.tagged_param_name,
            "all_tagged_count": len(self.all_tagged_names),
            "depth_steps_engaged": sum(1 for r in self.records if r.depth_block_engaged),
            "steps_with_depth_gt": sum(1 for r in self.records if r.depth_gt_present),
            "steps_with_lora_grad": sum(1 for r in self.records if r.lora_grad_norm),
        }

    def release_live_params(self) -> None:
        self.tagged_param = None
