"""Instrumentation for Phase 1 noising hooks during real Krea 2 training.

Wraps the process's noising methods to capture before/after tensor snapshots
and invocation counts at each optimizer-step boundary. All captured tensors
use .detach().cpu().clone() to avoid retaining autograd graphs.
"""
from dataclasses import dataclass, field


def _clone(tensor):
    """Detach, move to CPU, and clone a tensor."""
    return tensor.detach().cpu().clone()


@dataclass
class StepRecord:
    """Per-optimizer-step capture."""

    optimizer_step: int = 0
    is_accumulation_step: bool = False
    microbatch_index: int = 0
    loss: float | None = None
    grad_noise_invocations: int = 0
    weight_noise_invocations: int = 0
    fisher_invocations: int = 0
    ema_invocations: int = 0
    grad_norm_pre_noise: float | None = None
    grad_noise_norm: float | None = None
    grad_noise_snr: float | None = None
    weight_norm_pre_noise: float | None = None
    weight_noise_norm: float | None = None
    fisher_trace: float | None = None
    tagged_before_optimizer: dict = field(default_factory=dict)
    tagged_after_optimizer: dict = field(default_factory=dict)
    tagged_after_weight_noise: dict = field(default_factory=dict)
    untagged_before_optimizer: dict = field(default_factory=dict)
    untagged_after_optimizer: dict = field(default_factory=dict)
    untagged_after_weight_noise: dict = field(default_factory=dict)
    ema_shadow_after_update: dict = field(default_factory=dict)
    live_param_before_ema: dict = field(default_factory=dict)


class NoisingRecorder:
    """Captures noising-hook activity during a training run.

    Install on a process instance via ``recorder.install(process)`` BEFORE
    calling ``job.run()``. The recorder wraps ``hook_before_train_loop`` to
    defer instrumentation until after the accelerator prepares the optimizer.
    """

    def __init__(self, tagged_param_name: str | None = None, untagged_param_name: str | None = None):
        self.records: list[StepRecord] = []
        self.current_record: StepRecord | None = None
        self.microbatch_count: int = 0
        self.optimizer_step_count: int = 0
        self.tagged_param = None
        self.tagged_param_name = tagged_param_name
        self.untagged_param = None
        self.untagged_param_name = untagged_param_name
        self.all_tagged_names: list[str] = []
        self.all_untagged_names: list[str] = []
        self._installed = False

    def _collect_params(self, process):
        """Identify tagged and untagged parameters from optimizer param groups."""
        tagged = []
        untagged = []
        for group in process.optimizer.param_groups:
            for p in group["params"]:
                name = getattr(p, "_param_name", f"param_id_{id(p)}")
                if getattr(p, "_is_lora", False):
                    tagged.append((name, p))
                    self.all_tagged_names.append(name)
                else:
                    untagged.append((name, p))
                    self.all_untagged_names.append(name)

        if self.tagged_param is None and tagged:
            self.tagged_param_name = tagged[0][0]
            self.tagged_param = tagged[0][1]
        if self.untagged_param is None and untagged:
            self.untagged_param_name = untagged[0][0]
            self.untagged_param = untagged[0][1]

        # If no untagged param in optimizer (e.g., pure LoRA training where
        # only LoRA params are trainable), scan the model for a frozen base param.
        if self.untagged_param is None and hasattr(process, "sd") and process.sd is not None:
            model = getattr(process.sd, "unet", None) or getattr(process.sd, "model", None)
            if model is not None:
                try:
                    unwrapped = model
                    if hasattr(unwrapped, "module"):
                        unwrapped = unwrapped.module
                    for pname, p in unwrapped.named_parameters():
                        if not getattr(p, "_is_lora", False):
                            self.untagged_param = p
                            self.untagged_param_name = f"base:{pname}"
                            self.all_untagged_names.append(self.untagged_param_name)
                            break
                except Exception:
                    pass

        return tagged, untagged

    def install(self, process):
        """Wrap process methods to capture noising activity.

        Must be called after get_job() returns but before job.run().
        Wraps hook_before_train_loop to defer the optimizer/ema-dependent
        instrumentation until after prepare_accelerator completes.
        """
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
        """Called after hook_before_train_loop completes (optimizer/ema ready)."""
        tagged, untagged = self._collect_params(process)

        recorder = self
        orig_grad_noise = process._inject_gradient_noise
        orig_weight_noise = process._inject_weight_noise
        orig_fisher = process._record_fisher_trace
        orig_train_single = process.train_single_accumulation
        orig_hook_train_loop = process.hook_train_loop

        def wrapped_hook_train_loop(batch):
            recorder.current_record = StepRecord(
                optimizer_step=process.step_num,
                is_accumulation_step=process.is_grad_accumulation_step,
            )
            result = orig_hook_train_loop(batch)
            if isinstance(result, dict) and recorder.current_record is not None:
                rec = recorder.current_record
                rec.loss = result.get("loss")
                rec.grad_noise_norm = result.get("grad_noise_norm")
                rec.grad_noise_snr = result.get("grad_noise_snr")
                rec.weight_noise_norm = result.get("weight_noise_norm")
                rec.weight_norm_pre_noise = result.get("weight_norm")
                rec.fisher_trace = result.get("fisher_trace")
            if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                recorder.records.append(recorder.current_record)
                recorder.optimizer_step_count += 1
            return result

        process.hook_train_loop = wrapped_hook_train_loop

        def wrapped_grad_noise():
            if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                if recorder.tagged_param is not None:
                    recorder.current_record.tagged_before_optimizer = {
                        recorder.tagged_param_name: _clone(recorder.tagged_param.data)
                    }
                if recorder.untagged_param is not None:
                    recorder.current_record.untagged_before_optimizer = {
                        recorder.untagged_param_name: _clone(recorder.untagged_param.data)
                    }
                if recorder.tagged_param is not None and recorder.tagged_param.grad is not None:
                    recorder.current_record.grad_norm_pre_noise = float(
                        recorder.tagged_param.grad.detach().norm().item()
                    )
            orig_grad_noise()
            if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                recorder.current_record.grad_noise_invocations += 1

        process._inject_gradient_noise = wrapped_grad_noise

        def wrapped_weight_noise():
            if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                if recorder.tagged_param is not None:
                    recorder.current_record.tagged_after_optimizer = {
                        recorder.tagged_param_name: _clone(recorder.tagged_param.data)
                    }
                    recorder.current_record.weight_norm_pre_noise = float(
                        recorder.tagged_param.data.detach().norm().item()
                    )
                if recorder.untagged_param is not None:
                    recorder.current_record.untagged_after_optimizer = {
                        recorder.untagged_param_name: _clone(recorder.untagged_param.data)
                    }
            orig_weight_noise()
            if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                recorder.current_record.weight_noise_invocations += 1
                if recorder.tagged_param is not None:
                    recorder.current_record.tagged_after_weight_noise = {
                        recorder.tagged_param_name: _clone(recorder.tagged_param.data)
                    }
                if recorder.untagged_param is not None:
                    recorder.current_record.untagged_after_weight_noise = {
                        recorder.untagged_param_name: _clone(recorder.untagged_param.data)
                    }

        process._inject_weight_noise = wrapped_weight_noise

        def wrapped_fisher():
            orig_fisher()
            if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                recorder.current_record.fisher_invocations += 1

        process._record_fisher_trace = wrapped_fisher

        def wrapped_train_single(batch):
            recorder.microbatch_count += 1
            if recorder.current_record is not None:
                recorder.current_record.microbatch_index += 1
            return orig_train_single(batch)

        process.train_single_accumulation = wrapped_train_single

        if process.ema is not None:
            orig_ema_update = process.ema.update

            def wrapped_ema_update(*args, **kwargs):
                if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                    if recorder.tagged_param is not None:
                        recorder.current_record.live_param_before_ema = {
                            recorder.tagged_param_name: _clone(recorder.tagged_param.data)
                        }
                result = orig_ema_update(*args, **kwargs)
                if recorder.current_record is not None and not recorder.current_record.is_accumulation_step:
                    recorder.current_record.ema_invocations += 1
                    if process.ema is not None and recorder.tagged_param is not None:
                        for ref, shadow in zip(
                            process.ema.shadow_params_refs if hasattr(process.ema, "shadow_params_refs") else [],
                            process.ema.shadow_params if hasattr(process.ema, "shadow_params") else [],
                        ):
                            pass
                        ema_obj = process.ema
                        target_id = id(recorder.tagged_param)
                        try:
                            for i, ref in enumerate(ema_obj._params_refs):
                                if ref() is recorder.tagged_param:
                                    recorder.current_record.ema_shadow_after_update = {
                                        recorder.tagged_param_name: _clone(ema_obj.shadow_params[i])
                                    }
                                    break
                        except (AttributeError, IndexError):
                            pass
                return result

            process.ema.update = wrapped_ema_update

    def summary(self) -> dict:
        return {
            "microbatches": self.microbatch_count,
            "optimizer_steps": self.optimizer_step_count,
            "grad_noise_calls": sum(r.grad_noise_invocations for r in self.records),
            "weight_noise_calls": sum(r.weight_noise_invocations for r in self.records),
            "fisher_calls": sum(r.fisher_invocations for r in self.records),
            "ema_calls": sum(r.ema_invocations for r in self.records),
            "tagged_param": self.tagged_param_name,
            "untagged_param": self.untagged_param_name,
            "all_tagged_count": len(self.all_tagged_names),
            "all_untagged_count": len(self.all_untagged_names),
        }

    def release_live_params(self) -> None:
        """Null out live GPU parameter references so GC can free model memory.

        Records retain .detach().cpu().clone() copies, so data is preserved.
        Call this after assertions are complete (or in the runner before GC).
        """
        self.tagged_param = None
        self.untagged_param = None
