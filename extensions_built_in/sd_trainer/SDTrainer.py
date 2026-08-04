import os
import random
from collections import OrderedDict
from typing import Union, Literal, List, Optional

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

import torch.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig, DepthConsistencyConfig, NormalIDConfig, BodyProportionConfig, FaceIDConfig, SubjectMaskConfig, BodyShapeConfig
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.memory_management import sync_grad_transfers
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss
import torch.nn.functional as F
from toolkit.unloader import unload_text_encoder
from PIL import Image
from torchvision.transforms import functional as TF
from toolkit.basic import flush


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


def depth_active_for_dataset(
    depth_consistency_config: Optional[DepthConsistencyConfig],
    dataset_config,
) -> bool:
    """Return whether one dataset needs Phase 2 depth processing.

    A dataset override wins when present; otherwise it inherits the process
    loss weight. ``preview_only`` is process-wide and still needs matched GT
    depth for every dataset even though it contributes no anchor loss.
    """
    if depth_consistency_config is not None and bool(
        getattr(depth_consistency_config, 'preview_only', False)
    ):
        return True
    dataset_weight = getattr(dataset_config, 'depth_loss_weight', None)
    if dataset_weight is None:
        dataset_weight = (
            depth_consistency_config.loss_weight
            if depth_consistency_config is not None
            else 0.0
        )
    return float(dataset_weight or 0.0) > 0.0


def preflight_depth_consistency(
    depth_consistency_config: Optional[DepthConsistencyConfig],
    dataset_configs: List,
    arch: Optional[str],
    low_vram: bool,
) -> Optional[DepthConsistencyConfig]:
    """Resolve the effective depth-consistency config and run backend preflight.

    Returns the config to use for training (possibly a disabled default built
    from a dataset-only activation), or ``None`` when depth is inactive. A
    dataset-only activation (``depth_loss_weight > 0`` with no process object)
    builds ``DepthConsistencyConfig(loss_weight=0.0, mask_source='none')`` so
    YAML/API jobs follow the same contract as migrated UI jobs.

    Raises ``ValueError`` when depth is active and:
      - ``arch == 'krea2'`` with ``low_vram`` true (tiled decode inside the
        autograd graph is unsupported), or
      - ``mask_source`` is not ``'none'`` (Phase 3 auto-masking is not ported), or
      - any depth-active dataset (``depth_loss_weight > 0``) uses
        ``random_crop`` / ``random_scale`` / a non-empty ``augments`` list (the
        GT depth cache requires the same deterministic bucket transform as the
        training tensor).

    Inert when depth is not configured and no dataset is depth-active.
    """
    _dataset_depth_active = any(
        depth_active_for_dataset(depth_consistency_config, dc)
        for dc in dataset_configs
    )

    config = depth_consistency_config
    if config is None and _dataset_depth_active:
        config = DepthConsistencyConfig(loss_weight=0.0, mask_source='none')

    if config is None:
        return None

    _depth_active = (
        config.loss_weight > 0
        or bool(getattr(config, 'preview_only', False))
        or _dataset_depth_active
    )

    if not _depth_active:
        return config

    if arch == 'krea2' and low_vram:
        raise ValueError('Depth consistency for Krea 2 requires model.low_vram: false.')
    # mask_source subject|body is now allowed (Phase 3 auto-masking). The
    # cross-check that subject_mask is actually enabled runs in
    # hook_before_train_loop where both configs are visible; the depth loss
    # gracefully degrades to full-image when a sample lacks its mask.

    for dc in dataset_configs:
        if not depth_active_for_dataset(config, dc):
            continue
        if getattr(dc, 'random_crop', False) or getattr(dc, 'random_scale', False):
            raise ValueError(
                "Depth-active datasets cannot use random_crop or random_scale in "
                "Phase 2; the GT depth cache requires deterministic bucket "
                "transforms (set random_crop: false and random_scale: false)."
            )
        _augments = getattr(dc, 'augments', None) or []
        if len(_augments) > 0:
            raise ValueError(
                "Depth-active datasets cannot use a non-empty augments list in "
                "Phase 2; the GT depth cache requires deterministic bucket "
                f"transforms (remove augments, got {_augments!r})."
            )

    return config


def normal_active_for_dataset(
    normal_config: Optional[NormalIDConfig],
    dataset_config,
) -> bool:
    """Return whether one dataset needs normal-anchor processing.

    A dataset override wins when present; otherwise it inherits the process
    loss weight. ``preview_only`` is process-wide and still needs matched GT
    normals for every dataset even though it contributes no anchor loss.
    """
    if normal_config is not None and bool(getattr(normal_config, 'preview_only', False)):
        return True
    dataset_weight = getattr(dataset_config, 'normal_loss_weight', None)
    if dataset_weight is None:
        dataset_weight = (
            normal_config.loss_weight if normal_config is not None else 0.0
        )
    return float(dataset_weight or 0.0) > 0.0


def preflight_normal_id(
    normal_config: Optional[NormalIDConfig],
    dataset_configs: List,
    arch: Optional[str],
    low_vram: bool,
) -> Optional[NormalIDConfig]:
    """Resolve the effective normal-id config. Returns None when normal is off.

    A dataset-only activation (``normal_loss_weight > 0`` with no process
    object) builds ``NormalIDConfig(loss_weight=0.0)`` so YAML/API jobs follow
    the same contract as migrated UI jobs. Normal GT is transform-independent
    (computed from the raw source image), so -- unlike depth -- there is no
    random_crop / random_scale / augments preflight and no VAE round-trip.

    Like depth, the live normal loss decodes x0 through the VAE with grad
    enabled, so Krea 2 with ``low_vram`` true (tiled decode inside the autograd
    graph) is rejected.
    """
    _dataset_normal_active = any(
        normal_active_for_dataset(normal_config, dc) for dc in dataset_configs
    )

    config = normal_config
    if config is None and _dataset_normal_active:
        config = NormalIDConfig(loss_weight=0.0)

    if config is None:
        return None

    _normal_active = (
        config.loss_weight > 0
        or bool(getattr(config, 'preview_only', False))
        or _dataset_normal_active
    )
    if not _normal_active:
        return config

    if arch == 'krea2' and low_vram:
        raise ValueError('Normal-anchor loss for Krea 2 requires model.low_vram: false.')

    return config


def body_proportion_active_for_dataset(
    body_proportion_config: Optional[BodyProportionConfig],
    dataset_config,
) -> bool:
    """Return whether one dataset needs body-proportion processing."""
    dataset_weight = getattr(dataset_config, 'body_proportion_loss_weight', None)
    if dataset_weight is None:
        dataset_weight = (
            body_proportion_config.loss_weight
            if body_proportion_config is not None
            else 0.0
        )
    return float(dataset_weight or 0.0) > 0.0


def preflight_body_proportion(
    body_proportion_config: Optional[BodyProportionConfig],
    dataset_configs: List,
    arch: Optional[str],
    low_vram: bool,
) -> Optional[BodyProportionConfig]:
    """Resolve the effective body-proportion config. Returns None when inactive.

    A dataset-only activation builds ``BodyProportionConfig(loss_weight=0.0)``.
    Like depth/normal, the live body-proportion loss decodes x0 through the VAE
    under gradient, so Krea 2 with ``low_vram`` true is rejected.
    """
    _dataset_bp_active = any(
        body_proportion_active_for_dataset(body_proportion_config, dc)
        for dc in dataset_configs
    )
    config = body_proportion_config
    if config is None and _dataset_bp_active:
        config = BodyProportionConfig(loss_weight=0.0)
    if config is None:
        return None
    _bp_active = config.loss_weight > 0 or _dataset_bp_active
    if not _bp_active:
        return config
    if arch == 'krea2' and low_vram:
        raise ValueError(
            'Body-proportion loss for Krea 2 requires model.low_vram: false.'
        )
    return config


def face_identity_active_for_dataset(
    face_id_config: Optional[FaceIDConfig],
    dataset_config,
) -> bool:
    """Return whether one dataset needs face-identity processing."""
    dataset_weight = getattr(dataset_config, 'identity_loss_weight', None)
    if dataset_weight is None:
        dataset_weight = (
            face_id_config.identity_loss_weight if face_id_config is not None else 0.0
        )
    return float(dataset_weight or 0.0) > 0.0


def preflight_face_id(
    face_id_config: Optional[FaceIDConfig],
    dataset_configs: List,
    arch: Optional[str],
    low_vram: bool,
) -> Optional[FaceIDConfig]:
    """Resolve the effective face-id config. Returns None when inactive.

    Like the other anchors, the live identity loss decodes x0 through the VAE
    under gradient, so Krea 2 with ``low_vram`` true is rejected. Dep availability
    (insightface/onnx2torch/onnxruntime-gpu) is checked at perceptor load via a
    lazy import, not here, so a missing-dep environment still imports cleanly.
    """
    _dataset_id_active = any(
        face_identity_active_for_dataset(face_id_config, dc) for dc in dataset_configs
    )
    config = face_id_config
    if config is None and _dataset_id_active:
        config = FaceIDConfig(identity_loss_weight=0.0)
    if config is None:
        return None
    _id_active = config.identity_loss_weight > 0 or _dataset_id_active
    if not _id_active:
        return config
    if arch == 'krea2' and low_vram:
        raise ValueError('Face-identity loss for Krea 2 requires model.low_vram: false.')
    return config


def body_shape_active_for_dataset(
    body_shape_config: Optional[BodyShapeConfig],
    dataset_config,
) -> bool:
    dataset_weight = getattr(dataset_config, 'body_shape_loss_weight', None)
    if dataset_weight is None:
        dataset_weight = (
            body_shape_config.loss_weight if body_shape_config is not None else 0.0
        )
    return float(dataset_weight or 0.0) > 0.0


def preflight_body_shape(
    body_shape_config: Optional[BodyShapeConfig],
    dataset_configs: List,
    arch: Optional[str],
    low_vram: bool,
) -> Optional[BodyShapeConfig]:
    _dataset_bs_active = any(
        body_shape_active_for_dataset(body_shape_config, dc) for dc in dataset_configs
    )
    config = body_shape_config
    if config is None and _dataset_bs_active:
        config = BodyShapeConfig(loss_weight=0.0)
    if config is None:
        return None
    _bs_active = config.loss_weight > 0 or _dataset_bs_active
    if not _bs_active:
        return config
    if arch == 'krea2' and low_vram:
        raise ValueError('Body-shape loss for Krea 2 requires model.low_vram: false.')
    return config


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        # if self.train_config.dtype in ["fp16", "float16"]:
        #     # patch the scaler to allow fp16 training
        #     org_unscale_grads = self.scaler._unscale_grads_
        #     def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        #         return org_unscale_grads(optimizer, inv_scale, found_inf, True)
        #     self.scaler._unscale_grads_ = _unscale_grads_replacer

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        # fallback class-only embeds for when the text encoder is unloaded and
        # per item DOP embeds were not cached to disk
        self.cached_dop_class_embeds: Optional[PromptEmbeds] = None
        
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None
        
        if self.train_config.diff_output_preservation:
            # datasets can have their own trigger words, the global one is copied to them if not set
            has_dataset_trigger = any(
                dataset.trigger_word is not None for dataset in self.dataset_configs
            )
            if self.trigger_word is None and not has_dataset_trigger:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
        
        if self.train_config.blank_prompt_preservation:
            if self.network_config is None:
                raise ValueError("blank_prompt_preservation requires a network to be set")
        
        if self.train_config.blank_prompt_preservation or self.train_config.diff_output_preservation:
            # always do a prior prediction when doing output preservation
            self.do_prior_prediction = True
        
        # store the loss target for a batch so we can use it in a loss
        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")

        _depth_consistency_raw = self.get_conf('depth_consistency', None)
        if _depth_consistency_raw is not None:
            self.depth_consistency_config: Optional[DepthConsistencyConfig] = DepthConsistencyConfig(
                **_depth_consistency_raw
            )
        else:
            self.depth_consistency_config = None
        # Depth-step counter for preview cadence. Decoupled from raw step
        # parity so previews render on DEPTH steps regardless of preview_every
        # being even/odd (see _depth_preview_due). Reset in hook_before_train_loop.
        self._depth_step_count = 0

        _normal_raw = self.get_conf('normal_id', None)
        if _normal_raw is not None:
            self.normal_config: Optional[NormalIDConfig] = NormalIDConfig(**_normal_raw)
        else:
            self.normal_config = None
        # Normal-step counter for preview cadence (mirrors _depth_step_count).
        self._normal_step_count = 0
        # Normal-anchor diagnostics (flushed into loss_dict when present).
        self._last_normal_loss: Optional[float] = None
        self._last_normal_loss_applied: Optional[float] = None
        self._last_normal_cos: Optional[float] = None

        _body_proportion_raw = self.get_conf('body_proportion', None)
        if _body_proportion_raw is not None:
            self.body_proportion_config: Optional[BodyProportionConfig] = BodyProportionConfig(
                **_body_proportion_raw
            )
        else:
            self.body_proportion_config = None
        # Body-proportion diagnostics (flushed into loss_dict when present).
        self._last_body_proportion_loss: Optional[float] = None
        self._last_body_proportion_loss_applied: Optional[float] = None

        _face_id_raw = self.get_conf('face_id', None)
        if _face_id_raw is not None:
            self.face_id_config: Optional[FaceIDConfig] = FaceIDConfig(**_face_id_raw)
        else:
            self.face_id_config = None
        # Face-identity diagnostics (flushed into loss_dict when present).
        self._last_identity_loss: Optional[float] = None
        self._last_identity_loss_applied: Optional[float] = None
        self._last_id_sim: Optional[float] = None

        _subject_mask_raw = self.get_conf('subject_mask', None)
        if _subject_mask_raw is not None:
            self.subject_mask_config: Optional[SubjectMaskConfig] = SubjectMaskConfig(**_subject_mask_raw)
        else:
            self.subject_mask_config = None

        _body_shape_raw = self.get_conf('body_shape', None)
        if _body_shape_raw is not None:
            self.body_shape_config: Optional[BodyShapeConfig] = BodyShapeConfig(**_body_shape_raw)
        else:
            self.body_shape_config = None
        self._last_body_shape_loss: Optional[float] = None
        self._last_body_shape_loss_applied: Optional[float] = None
        self._last_body_shape_cos: Optional[float] = None

    def before_model_load(self):
        pass

    def get_blank_control_image(self):
        # noise instead of a black image so the fallback does not read as a
        # meaningful (solid black) reference
        control_image = torch.rand((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
        if self.sd.has_multiple_control_images:
            control_image = [control_image]
        return control_image

    def encode_static_prompt(self, prompt, **kwargs):
        # static embeds (blank/trigger/uncond/DOP class) are always plain text.
        # Some models (edit models) cannot encode a prompt without control images
        # and raise, only then fall back to a blank control image. Real errors
        # surface on the fallback call.
        try:
            return self.sd.encode_prompt(prompt, **kwargs)
        except Exception:
            return self.sd.encode_prompt(prompt, control_images=self.get_blank_control_image(), **kwargs)
    
    def cache_sample_prompts(self):
        if self.train_config.disable_sampling:
            return
        if self.sample_config is not None and self.sample_config.samples is not None and len(self.sample_config.samples) > 0:
            # cache all the samples
            self.sd.sample_prompts_cache = []
            sample_folder = os.path.join(self.save_root, 'samples')
            output_path = os.path.join(sample_folder, 'test.jpg')
            for i in range(len(self.sample_config.prompts)):
                sample_item = self.sample_config.samples[i]
                prompt = self.sample_config.prompts[i]
                
                if self.trigger_word is not None:
                    prompt = self.sd.inject_trigger_into_prompt(
                        prompt, self.trigger_word, add_if_not_present=False
                    )

                # needed so we can autoparse the prompt to handle flags
                gen_img_config = GenerateImageConfig(
                    prompt=prompt, # it will autoparse the prompt
                    negative_prompt=sample_item.neg,
                    output_path=output_path,
                    ctrl_img=sample_item.ctrl_img,
                    ctrl_img_1=sample_item.ctrl_img_1,
                    ctrl_img_2=sample_item.ctrl_img_2,
                    ctrl_img_3=sample_item.ctrl_img_3,
                )
                
                has_control_images = False
                if gen_img_config.ctrl_img is not None or gen_img_config.ctrl_img_1 is not None or gen_img_config.ctrl_img_2 is not None or gen_img_config.ctrl_img_3 is not None:
                    has_control_images = True
                # see if we need to encode the control images
                if self.sd.encode_control_in_text_embeddings and has_control_images:
                    self.sd.prepare_sample_prompt_context(gen_img_config)
                    
                    video_exts = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']

                    def _is_ctrl_video(pth):
                        return os.path.splitext(str(pth))[1].lower() in video_exts

                    ctrl_img_list = []
                    
                    if gen_img_config.ctrl_img is not None and _is_ctrl_video(gen_img_config.ctrl_img):
                        # control VIDEO: pass the path through; models with
                        # supports_video_control_images handle it in get_prompt_embeds
                        ctrl_img_list.append(str(gen_img_config.ctrl_img))
                    elif gen_img_config.ctrl_img is not None:
                        ctrl_img = Image.open(gen_img_config.ctrl_img).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img = (
                            TF.to_tensor(ctrl_img)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img)
                    
                    if gen_img_config.ctrl_img_1 is not None and _is_ctrl_video(gen_img_config.ctrl_img_1):
                        # control VIDEO: pass the path through; models with
                        # supports_video_control_images handle it in get_prompt_embeds
                        ctrl_img_list.append(str(gen_img_config.ctrl_img_1))
                    elif gen_img_config.ctrl_img_1 is not None:
                        ctrl_img_1 = Image.open(gen_img_config.ctrl_img_1).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_1 = (
                            TF.to_tensor(ctrl_img_1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_1)
                    if gen_img_config.ctrl_img_2 is not None and _is_ctrl_video(gen_img_config.ctrl_img_2):
                        # control VIDEO: pass the path through; models with
                        # supports_video_control_images handle it in get_prompt_embeds
                        ctrl_img_list.append(str(gen_img_config.ctrl_img_2))
                    elif gen_img_config.ctrl_img_2 is not None:
                        ctrl_img_2 = Image.open(gen_img_config.ctrl_img_2).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_2 = (
                            TF.to_tensor(ctrl_img_2)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_2)
                    if gen_img_config.ctrl_img_3 is not None and _is_ctrl_video(gen_img_config.ctrl_img_3):
                        # control VIDEO: pass the path through; models with
                        # supports_video_control_images handle it in get_prompt_embeds
                        ctrl_img_list.append(str(gen_img_config.ctrl_img_3))
                    elif gen_img_config.ctrl_img_3 is not None:
                        ctrl_img_3 = Image.open(gen_img_config.ctrl_img_3).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_3 = (
                            TF.to_tensor(ctrl_img_3)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_3)
                    
                    if self.sd.has_multiple_control_images:
                        ctrl_img = ctrl_img_list
                    else:
                        ctrl_img = ctrl_img_list[0] if len(ctrl_img_list) > 0 else None
                    
                    
                    positive = self.sd.encode_prompt(
                        gen_img_config.prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                    negative = self.sd.encode_prompt(
                        gen_img_config.negative_prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                else:
                    positive = self.sd.encode_prompt(gen_img_config.prompt).to('cpu')
                    negative = self.sd.encode_prompt(gen_img_config.negative_prompt).to('cpu')
                
                self.sd.sample_prompts_cache.append({
                    'conditional': positive,
                    'unconditional': negative
                })
        

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

    # ------------------------------------------------------------------
    # Depth-consistency GT caching (Task 5a)
    # ------------------------------------------------------------------

    def _depth_should_cache(self) -> bool:
        """True when depth is active and the GT caching pass should run.

        Mirrors the activity test in ``preflight_depth_consistency`` so the
        pass stays fully inert (no DA2 import, no perceptor load, no
        file-item stamping) whenever depth is off.
        """
        cfg = self.depth_consistency_config
        if cfg is None:
            return False
        ds_depth = any(
            depth_active_for_dataset(cfg, dc)
            for dc in self.dataset_configs
        )
        return (
            cfg.loss_weight > 0
            or bool(getattr(cfg, 'preview_only', False))
            or ds_depth
        )

    def _depth_vae_id(self) -> str:
        """Stable VAE identity for the cache fingerprint.

        VAE class fully-qualified name + ``vae.config._name_or_path``, falling
        back to ``model.model_kwargs.vae_path``. Changing the VAE forces a
        cache miss because the round-trip pixels change.
        """
        vae = self.sd.vae
        class_fqn = f"{type(vae).__module__}.{type(vae).__name__}"
        vae_cfg = getattr(vae, 'config', None)
        name_or_path = getattr(vae_cfg, '_name_or_path', None)
        if name_or_path:
            return f"{class_fqn}:{name_or_path}"
        model_kwargs = getattr(self.model_config, 'model_kwargs', None)
        vae_path = getattr(model_kwargs, 'vae_path', None) if model_kwargs else None
        if vae_path:
            return f"{class_fqn}:{vae_path}"
        return class_fqn

    def _depth_vae_roundtrip(self, arr: torch.Tensor) -> torch.Tensor:
        """[0,1] pixels ``(1,3,H,W)`` -> VAE encode -> decode -> [0,1] pixels.

        Arch-agnostic: every target diffusion model implements
        ``encode_images`` / ``decode_latents``, each handling its own
        scaling/shift (or ``latents_mean``/``latents_std``) and 5D-frame
        logic internally. The cached GT depth is taken from these round-trip
        pixels so the live depth loss has a VAE-matched target. A direct
        ``vae.encode`` with a scalar ``scaling_factor`` is intentionally
        avoided -- it mis-normalizes AutoencoderKL / Qwen VAEs.
        """
        vae_dtype = self.sd.vae_torch_dtype
        pixels_m1 = (arr * 2.0 - 1.0).to(vae_dtype)
        # VAE residency guard: no-op when resident, protects the cached-latents
        # VAE-offload case (sec. 2.1).
        if next(self.sd.vae.parameters()).device != self.device_torch:
            self.sd.vae.to(self.device_torch)
        latents = self.sd.encode_images(
            [image for image in pixels_m1],
            device=self.device_torch,
            dtype=vae_dtype,
        )
        decoded = self.sd.decode_latents(
            latents,
            device=self.device_torch,
            dtype=vae_dtype,
        )
        return ((decoded.float() + 1.0) * 0.5).clamp(0, 1)

    def _load_depth_perceptor(self):
        """Lazily load the frozen DA2 perceptor; cached on the trainer.

        ``DepthAnythingForDepthEstimation`` imports inside the load so simply
        selecting Krea never downloads DA2 -- only enabling depth does.
        """
        if getattr(self, '_depth_perceptor', None) is not None:
            return self._depth_perceptor
        from toolkit.depth_perceptor import DifferentiableDepthEncoder
        cfg = self.depth_consistency_config
        self._depth_perceptor = DifferentiableDepthEncoder(
            model_id=cfg.model_id,
            input_size=int(cfg.input_size),
            device=self.device_torch,
            grad_checkpoint=bool(getattr(cfg, 'grad_checkpoint', True)),
        )
        return self._depth_perceptor

    def _cache_depth_gt_pass(self):
        """Iterate depth-active datasets and stamp + persist GT depth.

        Loads the perceptor once, computes a stable ``vae_id``, and stamps every
        depth-active file item with the cache path/key Task 3's mixin reads.
        """
        from toolkit.depth_perceptor import cache_depth_gt
        cfg = self.depth_consistency_config
        arch = self.sd.model_config.arch
        vae_id = self._depth_vae_id()
        encoder = self._load_depth_perceptor()
        def _ds_depth_active(dc):
            return depth_active_for_dataset(cfg, dc)

        def _cache_loader(loader):
            if loader is None:
                return
            for dataset in get_dataloader_datasets(loader):
                if not _ds_depth_active(dataset.dataset_config):
                    continue
                cache_depth_gt(
                    dataset.file_list, cfg,
                    encoder=encoder, arch=arch, vae_id=vae_id,
                    device=self.device_torch,
                    roundtrip_fn=self._depth_vae_roundtrip,
                )

        _cache_loader(self.data_loader)
        _cache_loader(self.data_loader_reg)

    # ------------------------------------------------------------------
    # Depth-anchor calculate_loss block (Task 5b)
    # ------------------------------------------------------------------

    def _depth_loss_active(self) -> bool:
        """True when the depth loss block should engage for this process.

        Mirrors ``_depth_should_cache``: depth is on when the process object
        has a positive ``loss_weight``, when ``preview_only`` is set, or when
        any dataset sets ``depth_loss_weight > 0``. When this is False the
        entire ``calculate_loss`` depth block (gating, decode, perceptor,
        diffusion-side masking) is unreachable, so a depth-inactive job runs
        the exact same path as before.
        """
        cfg = self.depth_consistency_config
        if cfg is None:
            return False
        if cfg.loss_weight > 0 or bool(getattr(cfg, 'preview_only', False)):
            return True
        return any(
            depth_active_for_dataset(cfg, dc)
            for dc in self.dataset_configs
        )

    def _resolve_depth_sample_gates(self, batch, timesteps, is_reg_per_sample):
        """Resolve per-sample depth-anchor gates for one optimizer microbatch.

        Returns a dict of ``(B,)`` tensors (and two python bools):

          * ``t``              -- flow-matching ratio ``timesteps / num_train_timesteps``.
          * ``eff_weight``     -- per-sample effective depth weight (dataset
            override else global), zeroed for alternating samples on diffusion
            steps so they skip the depth perceptor entirely.
          * ``in_band``        -- ``loss_min_t <= t <= loss_max_t`` and not a
            prior-preservation (reg) sample.
          * ``depth_objective``-- ``in_band & (eff_weight > 0)``: the samples
            that contribute a depth gradient this step.
          * ``preview_objective`` -- in-band samples processed without grad on
            preview cadence when ``preview_only`` is enabled.
          * ``diffusion_zero`` -- alternating samples on a depth step (these
            drop out of the diffusion loss so the two objectives alternate).
          * ``step_is_diffusion`` -- python bool (preview cadence is handled by
            ``_depth_preview_due``, which counts DEPTH steps).

        Per-sample split resolution uses ``resolve_loss_split`` (Task 2): a
        dataset ``'sum'`` forces the sample off (it sums every step); a dataset
        ``'diffusion_depth'`` forces it on; absent -> Auto (autodetect from the
        effective depth weight); an explicit global wins otherwise. Step parity
        (``self.step_num % 2``) selects the active objective for alternating
        samples -- batch size never affects it.
        """
        from toolkit.loss_split import resolve_loss_split

        cfg = self.depth_consistency_config
        nt = float(self.sd.noise_scheduler.config.num_train_timesteps)
        t = (timesteps.float() / nt).to(self.device_torch)
        if t.dim() == 0:
            t = t.view(1)
        b = t.shape[0]

        split_raw = getattr(batch, 'loss_split_list', None) or [None] * b
        global_split = getattr(self.train_config, 'loss_split', None)
        global_explicit = bool(getattr(self.train_config, '_loss_split_explicit', False))
        global_dc_w = float(cfg.loss_weight) if cfg is not None else 0.0
        per_sample_dc_w = getattr(batch, 'depth_loss_weight_list', None)

        def _eff_dc_weight(idx: int) -> float:
            if per_sample_dc_w is not None and idx < len(per_sample_dc_w):
                v = per_sample_dc_w[idx]
                if v is not None:
                    return float(v)
            return global_dc_w

        split_list = [
            resolve_loss_split(
                ds_value=v,
                global_value=global_split,
                global_explicit=global_explicit,
                effective_depth_weight=_eff_dc_weight(i),
            )
            for i, v in enumerate(split_raw)
        ]
        loss_split_diff_depth = torch.tensor(
            [s == 'diffusion_depth' for s in split_list],
            device=self.device_torch, dtype=torch.bool,
        )
        step_is_diffusion = (self.step_num % 2 == 0)

        min_list = getattr(batch, 'depth_loss_min_t_list', None) or [None] * b
        max_list = getattr(batch, 'depth_loss_max_t_list', None) or [None] * b
        t_min = torch.tensor(
            [float(v) if v is not None else float(cfg.loss_min_t) for v in min_list],
            device=self.device_torch, dtype=t.dtype,
        )
        t_max = torch.tensor(
            [float(v) if v is not None else float(cfg.loss_max_t) for v in max_list],
            device=self.device_torch, dtype=t.dtype,
        )
        # inclusive on both ends so loss_min_t=0, loss_max_t=1 spans the full
        # range, including t=1.0 (pure noise) when sampling is biased that way.
        in_band = (t >= t_min) & (t <= t_max)
        in_band = in_band & (~is_reg_per_sample.to(self.device_torch))

        w_list = getattr(batch, 'depth_loss_weight_list', None) or [None] * b
        eff_w = torch.tensor(
            [float(w) if w is not None else global_dc_w for w in w_list],
            device=self.device_torch, dtype=t.dtype,
        )
        # Loss-split gate: alternating samples skip depth on diffusion steps.
        if loss_split_diff_depth.any() and step_is_diffusion:
            split_mask = loss_split_diff_depth.to(self.device_torch, dtype=eff_w.dtype)
            eff_w = eff_w * (1.0 - split_mask)

        preview_only = bool(getattr(cfg, 'preview_only', False))
        preview_enabled = preview_only and int(getattr(cfg, 'preview_every', 0) or 0) > 0
        preview_objective = (
            in_band.clone() if preview_enabled else torch.zeros_like(in_band)
        )
        depth_objective = in_band & (eff_w > 0) & (not preview_only)
        # Mirrors the depth gate from the other side: alternating samples skip
        # diffusion on depth steps so the two objectives trade places. Gated on
        # depth_objective so a sample never drops diffusion without gaining
        # depth (e.g. preview_only with loss_weight=0, or out-of-band steps).
        diffusion_zero = loss_split_diff_depth & (not step_is_diffusion) & depth_objective

        return {
            't': t,
            'eff_weight': eff_w,
            'in_band': in_band,
            'depth_objective': depth_objective,
            'preview_objective': preview_objective,
            'diffusion_zero': diffusion_zero,
            'step_is_diffusion': step_is_diffusion,
        }

    def _depth_preview_due(self, cfg) -> bool:
        """Whether a depth preview tile should render on this DEPTH step.

        Cadence counts DEPTH steps (``self._depth_step_count``), not raw
        ``step_num``. Gating on raw step parity made every preview land on a
        diffusion step when ``preview_every`` was even, so the per-sample
        preview loop never executed. Returns False when previews are off
        (``preview_every <= 0``).
        """
        every = int(getattr(cfg, 'preview_every', 0) or 0)
        return every > 0 and (self._depth_step_count % every == 0)

    def _apply_diffusion_split_mask(self, loss_per_sample, diffusion_zero):
        """Mean of the per-sample diffusion loss, dropping depth-step samples.

        ``diffusion_zero`` is a ``(B,)`` bool tensor: True for alternating
        samples on a depth step (they contribute depth instead of diffusion).
        With nothing dropped this is the plain ``.mean()`` -- so a depth-off or
        non-alternating step reduces identically to the original path. With all
        samples dropped (pure depth step) returns ``0.0``.
        """
        if not torch.is_tensor(diffusion_zero):
            return loss_per_sample.mean()
        keep = (~diffusion_zero).to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            return loss_per_sample.sum() * 0.0
        if n_keep == loss_per_sample.shape[0]:
            return loss_per_sample.mean()
        return (loss_per_sample * keep).sum() / n_keep

    def _prune_preview_dir(self, directory, max_keep):
        """Keep the newest ``max_keep`` files in ``directory``; delete the rest."""
        try:
            files = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
            if len(files) <= max_keep:
                return
            files.sort(key=lambda p: os.path.getmtime(p))
            for p in files[: len(files) - max_keep]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        except OSError:
            pass

    def _compute_depth_anchor_loss(self, noise_pred, noisy_latents, timesteps,
                                   batch, gates):
        """Run the unified live decode + DA2 forward + SSI/grad loss.

        Adds depth-consistency loss only on ``gates['depth_objective']`` samples
        (in timestep band, positive weight, not reg, and not an alternating
        sample on a diffusion step). Returns the weighted-mean contribution to
        add to the total loss (a python ``0.0`` when no sample qualifies), and
        records ``self._last_depth_processed_indices`` for diagnostics.

        Krea2 Turbo note: Turbo merges the training adapter at +1.0 for training
        and inverts it at sample time (``krea2.py:301``). Depth previews are
        decoded from the training-merged base, so they may differ from final
        sampled images -- a training-time diagnostic, not a sample preview.
        """
        from toolkit.depth_loss import compute_depth_consistency_loss, render_depth_preview
        from toolkit.depth_perceptor import gaussian_blur_2d

        cfg = self.depth_consistency_config
        # Reuse the perceptor the caching pass (Task 5a) loaded and cached; only
        # fall back to the loader if it was never created.
        encoder = getattr(self, '_depth_perceptor', None)
        if encoder is None:
            encoder = self._load_depth_perceptor()
        depth_objective = gates['depth_objective']
        preview_objective = gates.get(
            'preview_objective', torch.zeros_like(depth_objective)
        )
        process_objective = depth_objective | preview_objective
        preview_only = bool(getattr(cfg, 'preview_only', False))
        t = gates['t']
        eff_w = gates['eff_weight']

        processed = []
        if not process_objective.any():
            self._last_depth_processed_indices = processed
            self._last_depth_consistency_loss = 0.0
            return 0.0

        # Advance the processing-step counter after the objective gate. Normal
        # training counts depth steps; preview-only counts eligible preview
        # evaluation steps. Both avoid raw parity collisions.
        self._depth_step_count += 1
        preview_due = self._depth_preview_due(cfg)
        if preview_only and not preview_due:
            self._last_depth_processed_indices = processed
            self._last_depth_consistency_loss = 0.0
            return 0.0

        # Preview-only evaluates the decode and perceptor without constructing
        # an autograd graph. Normal depth anchoring keeps the full graph.
        with torch.set_grad_enabled(not preview_only):
            # x0 recovery. Flow-matching (Krea): x0 = noisy - t * noise_pred.
            t_b = t.view(-1, 1, 1, 1)
            if getattr(self.sd, 'is_flow_matching', True):
                x0 = noisy_latents - t_b * noise_pred
            else:
                _ac = self.sd.noise_scheduler.alphas_cumprod.to(
                    device=timesteps.device, dtype=noisy_latents.dtype
                )[timesteps.long()].view(-1, 1, 1, 1)
                _sa = _ac.sqrt()
                _s1ma = (1.0 - _ac).sqrt()
                if getattr(self.sd, 'prediction_type', None) == 'v_prediction':
                    x0 = _sa * noisy_latents - _s1ma * noise_pred
                else:
                    x0 = (noisy_latents - _s1ma * noise_pred) / _sa.clamp(min=1e-8)

            # Unified live decode. The model owns scaling, shifting, and frame
            # dimensions; the residency guard covers cached-latent offload.
            if next(self.sd.vae.parameters()).device != self.device_torch:
                self.sd.vae.to(self.device_torch)
            decoded = self.sd.decode_latents(
                x0, device=self.device_torch, dtype=self.sd.vae_torch_dtype,
            )
            pixels = ((decoded.float() + 1.0) * 0.5).clamp(0, 1)

        # Pre-DA2 blur: the SAME gaussian_blur_2d the GT caching pass (Task 5a)
        # applied before DA2, so pred-depth and GT-depth stay apples-to-apples.
        # pixels itself is left untouched so preview tiles show the real decode.
        blur_sigma = float(getattr(cfg, 'pixel_blur_sigma', 0.0) or 0.0)
        pixels_for_da2 = gaussian_blur_2d(pixels, blur_sigma) if blur_sigma > 0 else pixels

        # mask_source: select the cached subject/body mask per sample (Phase 3
        # auto-masking). 'none' -> full-image (None). Graceful degrade: a sample
        # whose mask is missing falls back to full-image loss.
        _dc_mask_source = getattr(cfg, 'mask_source', 'none')
        _dc_subject_masks = getattr(batch, 'subject_masks', None)
        _dc_body_masks = getattr(batch, 'body_masks', None)
        total = pixels.new_zeros(())
        weighted_total = pixels.new_zeros(())
        ssi_sum = 0.0
        grad_sum = 0.0
        n = 0
        depth_gt_list = getattr(batch, 'depth_gt_list', None) or []
        for i in range(pixels.shape[0]):
            if not process_objective[i]:
                continue
            if i >= len(depth_gt_list) or depth_gt_list[i] is None:
                continue
            gt_t = depth_gt_list[i].to(pixels.device, dtype=torch.float32)
            # Resolve this sample's spatial mask from mask_source.
            _dc_mask_t = None
            if _dc_mask_source == 'subject' and _dc_subject_masks is not None and i < _dc_subject_masks.shape[0]:
                _dc_mask_t = _dc_subject_masks[i].float().to(pixels.device)
            elif _dc_mask_source == 'body' and _dc_body_masks is not None and i < _dc_body_masks.shape[0]:
                _dc_mask_t = _dc_body_masks[i].float().to(pixels.device)
            if _dc_mask_t is not None and _dc_mask_t.dim() == 3:
                _dc_mask_t = _dc_mask_t.squeeze(0)
            if preview_only:
                with torch.no_grad():
                    loss_i, ssi_i, grad_i, dpred_i, dgt_i = compute_depth_consistency_loss(
                        encoder, pixels_for_da2[i:i + 1].detach(), gt_t, _dc_mask_t,
                        ssi_weight=cfg.ssi_weight,
                        grad_weight=cfg.grad_weight,
                        grad_scales=cfg.grad_scales,
                    )
            else:
                loss_i, ssi_i, grad_i, dpred_i, dgt_i = compute_depth_consistency_loss(
                    encoder, pixels_for_da2[i:i + 1], gt_t, _dc_mask_t,
                    ssi_weight=cfg.ssi_weight,
                    grad_weight=cfg.grad_weight,
                    grad_scales=cfg.grad_scales,
                )
                total = total + loss_i
                weighted_total = weighted_total + loss_i * eff_w[i]
                ssi_sum += float(ssi_i)
                grad_sum += float(grad_i)
                n += 1
            processed.append(i)

            # Preview: four-panel Krea tile every preview_every DEPTH steps.
            # Cadence keys off _depth_step_count (advanced above), not raw step
            # parity, so an even preview_every no longer collides exclusively
            # with diffusion steps. Best-effort -- a failed render never aborts
            # the training step.
            if (preview_due and self.save_root is not None
                    and i < len(batch.file_items)):
                try:
                    from PIL import Image as _PILImage
                    from PIL.ImageOps import exif_transpose as _exif
                    from torchvision.transforms import functional as _TF
                    pred_rgb = pixels[i].detach().clamp(0, 1).cpu()
                    pred_pil = _TF.to_pil_image(pred_rgb)
                    ref_path = batch.file_items[i].path
                    ref_pil = _exif(_PILImage.open(ref_path)).convert('RGB')
                    combo = render_depth_preview(
                        pred_pil, ref_pil,
                        dpred_i.squeeze(0) if dpred_i.dim() == 3 else dpred_i,
                        dgt_i.squeeze(0) if dgt_i.dim() == 3 else dgt_i,
                    )
                    dc_preview_dir = os.path.join(self.save_root, 'depth_previews')
                    os.makedirs(dc_preview_dir, exist_ok=True)
                    _t_val = float(t[i].item())
                    _dc_val = float(loss_i.detach())
                    src_name = os.path.splitext(os.path.basename(ref_path))[0]
                    _h_px, _w_px = int(pred_rgb.shape[-2]), int(pred_rgb.shape[-1])
                    combo.save(os.path.join(
                        dc_preview_dir,
                        f'{src_name}_step{self.step_num:06d}_'
                        f't{_t_val:.2f}_dc{_dc_val:.4f}_s{_w_px}x{_h_px}.jpg'
                    ))
                    self._prune_preview_dir(
                        dc_preview_dir, int(getattr(cfg, 'preview_max_keep', 500))
                    )
                except Exception as e:  # noqa: BLE001
                    print_acc(f"  depth preview failed: {e}")

        self._last_depth_processed_indices = processed
        if n == 0:
            self._last_depth_consistency_loss = 0.0
            return 0.0
        applied = weighted_total / n
        self._last_depth_consistency_loss = (total / n).detach().item()
        self._last_depth_consistency_ssi = ssi_sum / n
        self._last_depth_consistency_grad = grad_sum / n
        return applied

    # ------------------------------------------------------------------
    # Normal-anchor (Sapiens surface normals) -- Phase 3
    # ------------------------------------------------------------------

    def _normal_should_cache(self) -> bool:
        """True when normal is active and the GT caching pass should run."""
        cfg = self.normal_config
        if cfg is None:
            return False
        ds_normal = any(
            normal_active_for_dataset(cfg, dc) for dc in self.dataset_configs
        )
        return (
            cfg.loss_weight > 0
            or bool(getattr(cfg, 'preview_only', False))
            or ds_normal
        )

    def _normal_loss_active(self) -> bool:
        """True when the normal loss block should engage. Mirrors _normal_should_cache."""
        return self._normal_should_cache()

    def _load_normal_perceptor(self):
        """Lazily load the frozen Sapiens normal perceptor; cached on the trainer.

        ``SapiensNormal._load_pretrained`` imports ``huggingface_hub`` and
        downloads the weights inside the load, so selecting Krea never downloads
        Sapiens -- only enabling normal loss does.
        """
        if getattr(self, '_normal_perceptor', None) is not None:
            return self._normal_perceptor
        from toolkit.normal_id import DifferentiableNormalEncoder
        cfg = self.normal_config
        self._normal_perceptor = DifferentiableNormalEncoder(
            device=self.device_torch,
            grad_checkpoint=bool(getattr(cfg, 'grad_checkpoint', True)),
        )
        return self._normal_perceptor

    def _cache_normal_gt_pass(self):
        """Iterate normal-active datasets and stamp + persist GT normals.

        GT normals are computed from the RAW source image (transform-
        independent), so there is no VAE round-trip and no bucket-transform
        fingerprint (unlike depth). The resident tensor is never held on the
        file-list item; the worker re-reads lazily via get_normal_gt().
        """
        from toolkit.normal_id import cache_normal_gt
        cfg = self.normal_config
        encoder = self._load_normal_perceptor()

        def _ds_normal_active(dc):
            return normal_active_for_dataset(cfg, dc)

        def _cache_loader(loader):
            if loader is None:
                return
            for dataset in get_dataloader_datasets(loader):
                if not _ds_normal_active(dataset.dataset_config):
                    continue
                cache_normal_gt(dataset.file_list, cfg, encoder=encoder)

        _cache_loader(self.data_loader)
        _cache_loader(self.data_loader_reg)

    def _compute_normal_anchor_loss(self, noise_pred, noisy_latents, timesteps, batch):
        """Run the unified live decode + Sapiens forward + cosine/L1 normal loss.

        Normal loss does NOT participate in the diffusion/depth ``loss_split``
        alternation: it fires every step on samples inside the timestep window
        with a positive weight, a cached GT normal, and not a reg sample.
        Returns the weighted-mean contribution to add to the total loss (a
        python ``0.0`` when no sample qualifies).

        Note: when depth is also active this performs a SECOND grad-enabled VAE
        decode of x0 (one per perceptor). On the 96 GB target that is acceptable
        headroom; the shared-decode optimization is deferred.
        """
        from toolkit.normal_id_loss import compute_normal_loss, render_normal_preview

        cfg = self.normal_config
        normal_gt_list = getattr(batch, 'normal_gt_list', None) or []
        if len(normal_gt_list) == 0:
            return 0.0

        encoder = getattr(self, '_normal_perceptor', None)
        if encoder is None:
            encoder = self._load_normal_perceptor()

        num_t = self.sd.noise_scheduler.config.num_train_timesteps
        t_ratio = (timesteps.float() / num_t)
        is_reg = batch.get_is_reg_list()

        per_ds_w = getattr(batch, 'normal_loss_weight_list', None)
        has_per_ds = per_ds_w is not None and any(w is not None for w in per_ds_w)
        global_w = cfg.loss_weight
        min_t_list = getattr(batch, 'normal_loss_min_t_list', None)
        max_t_list = getattr(batch, 'normal_loss_max_t_list', None)

        B = noise_pred.shape[0]
        weights = []
        valid = torch.zeros(B, dtype=torch.bool, device=self.device_torch)
        for i in range(B):
            w = per_ds_w[i] if (has_per_ds and per_ds_w[i] is not None) else global_w
            weights.append(float(w or 0.0))
            if is_reg[i] or weights[i] <= 0.0:
                continue
            if i >= len(normal_gt_list) or normal_gt_list[i] is None:
                continue
            lo = float(min_t_list[i]) if (min_t_list and min_t_list[i] is not None) else cfg.loss_min_t
            hi = float(max_t_list[i]) if (max_t_list and max_t_list[i] is not None) else cfg.loss_max_t
            tr = float(t_ratio[i].item())
            if not (lo <= tr <= hi):
                continue
            valid[i] = True

        if not valid.any():
            self._last_normal_loss = 0.0
            return 0.0

        self._normal_step_count += 1
        preview_only = bool(getattr(cfg, 'preview_only', False))
        preview_due = (
            int(getattr(cfg, 'preview_every', 0)) > 0
            and self._normal_step_count % int(cfg.preview_every) == 0
        )
        if preview_only and not preview_due:
            self._last_normal_loss = 0.0
            return 0.0

        with torch.set_grad_enabled(not preview_only):
            t_b = t_ratio.view(-1, 1, 1, 1)
            if getattr(self.sd, 'is_flow_matching', True):
                x0 = noisy_latents - t_b * noise_pred
            else:
                _ac = self.sd.noise_scheduler.alphas_cumprod.to(
                    device=timesteps.device, dtype=noisy_latents.dtype
                )[timesteps.long()].view(-1, 1, 1, 1)
                _sa = _ac.sqrt()
                _s1ma = (1.0 - _ac).sqrt()
                if getattr(self.sd, 'prediction_type', None) == 'v_prediction':
                    x0 = _sa * noisy_latents - _s1ma * noise_pred
                else:
                    x0 = (noisy_latents - _s1ma * noise_pred) / _sa.clamp(min=1e-8)

            if next(self.sd.vae.parameters()).device != self.device_torch:
                self.sd.vae.to(self.device_torch)
            decoded = self.sd.decode_latents(
                x0, device=self.device_torch, dtype=self.sd.vae_torch_dtype,
            )
            pixels = ((decoded.float() + 1.0) * 0.5).clamp(0, 1)

        idx_map = [i for i in range(B) if valid[i]]
        gt_tensor = torch.stack(
            [normal_gt_list[i].to(pixels.device, dtype=torch.float32) for i in idx_map],
            dim=0,
        )
        # Optional body-region restriction (Phase 3 auto-masking). Built on the
        # full batch then sliced to the valid subset; None when no item opts in.
        from toolkit.normal_id import NORMAL_SIZE
        _body_restrict = self._build_body_restrict_mask(batch, (B, NORMAL_SIZE, NORMAL_SIZE))
        if _body_restrict is not None:
            _body_restrict = _body_restrict.to(pixels.device, dtype=torch.float32)[idx_map]
        if preview_only:
            with torch.no_grad():
                cos_loss, l1_loss, gen_det, ref_det = compute_normal_loss(
                    encoder, pixels[idx_map].detach(), gt_tensor, mask=_body_restrict,
                )
            self._last_normal_loss = 0.0
            self._last_normal_loss_applied = 0.0
        else:
            cos_loss, l1_loss, gen_det, ref_det = compute_normal_loss(
                encoder, pixels[idx_map], gt_tensor, mask=_body_restrict,
            )
            per_sample = cos_loss + l1_loss
            w_tensor = torch.tensor(
                [weights[i] for i in idx_map],
                device=pixels.device, dtype=torch.float32,
            )
            applied = (per_sample * w_tensor).mean()
            self._last_normal_loss_applied = float(applied.detach().item())
            with torch.no_grad():
                self._last_normal_loss = float(l1_loss.mean().item())
                self._last_normal_cos = float((1.0 - cos_loss).mean().item())

        if (preview_due and self.save_root is not None
                and getattr(batch, 'file_items', None) is not None):
            try:
                from PIL import Image as _PILImage
                from PIL.ImageOps import exif_transpose as _exif_t
                nrm_preview_dir = os.path.join(self.save_root, 'normal_previews')
                os.makedirs(nrm_preview_dir, exist_ok=True)
                for j, i in enumerate(idx_map):
                    if i >= len(batch.file_items):
                        break
                    pred_pil = _PILImage.fromarray(
                        (pixels[i].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                        .astype("uint8")
                    )
                    try:
                        ref_pil = _exif_t(_PILImage.open(batch.file_items[i].path)).convert("RGB")
                    except Exception:  # noqa: BLE001
                        ref_pil = pred_pil
                    combo = render_normal_preview(
                        pred_pil, ref_pil,
                        gen_det[j].unsqueeze(0) if gen_det[j].dim() == 3 else gen_det[j:j + 1],
                        ref_det[j].unsqueeze(0) if ref_det[j].dim() == 3 else ref_det[j:j + 1],
                    )
                    src_name = os.path.splitext(os.path.basename(batch.file_items[i].path))[0]
                    combo.save(os.path.join(
                        nrm_preview_dir,
                        f'{src_name}_step{self.step_num:06d}_t{float(t_ratio[i]):.2f}.jpg',
                    ))
                self._prune_preview_dir(
                    nrm_preview_dir, int(getattr(cfg, 'preview_max_keep', 500))
                )
            except Exception as e:  # noqa: BLE001
                print_acc(f"  normal preview failed: {e}")

        return applied if not preview_only else 0.0

    # ------------------------------------------------------------------
    # Body-proportion anchor (ViTPose) -- Phase 3
    # ------------------------------------------------------------------

    def _body_proportion_should_cache(self) -> bool:
        cfg = self.body_proportion_config
        if cfg is None:
            return False
        return cfg.loss_weight > 0 or any(
            body_proportion_active_for_dataset(cfg, dc) for dc in self.dataset_configs
        )

    def _body_proportion_loss_active(self) -> bool:
        return self._body_proportion_should_cache()

    def _load_body_proportion_perceptor(self):
        """Lazily load the frozen ViTPose perceptor; cached on the trainer.

        The transformers ViTPose import + HF download live inside __init__, so
        selecting Krea never downloads ViTPose -- only enabling body-proportion
        loss does.
        """
        if getattr(self, '_body_proportion_perceptor', None) is not None:
            return self._body_proportion_perceptor
        from toolkit.body_proportion import DifferentiableBodyProportionEncoder
        self._body_proportion_perceptor = DifferentiableBodyProportionEncoder(
            device=self.device_torch,
        )
        return self._body_proportion_perceptor

    def _cache_body_proportion_gt_pass(self):
        """Iterate body-proportion-active datasets and stamp + persist GT ratios."""
        from toolkit.body_proportion import cache_body_proportion
        cfg = self.body_proportion_config
        encoder = self._load_body_proportion_perceptor()

        def _ds_bp_active(dc):
            return body_proportion_active_for_dataset(cfg, dc)

        def _cache_loader(loader):
            if loader is None:
                return
            for dataset in get_dataloader_datasets(loader):
                if not _ds_bp_active(dataset.dataset_config):
                    continue
                cache_body_proportion(dataset.file_list, cfg, encoder=encoder)

        _cache_loader(self.data_loader)
        _cache_loader(self.data_loader_reg)

    def _compute_body_proportion_anchor_loss(self, noise_pred, noisy_latents, timesteps, batch):
        """Run the unified live decode + ViTPose forward + ratio L1 loss.

        Body-proportion does NOT participate in diffusion/depth loss_split; it
        fires every step on samples in its timestep window with a positive
        weight, a cached GT ratio vector, and not a reg sample. Returns the
        weighted-mean contribution (a python 0.0 when no sample qualifies).

        Like normal, this performs its own grad-enabled x0 decode (a second one
        when depth/normal are also active) -- acceptable headroom on 96 GB.
        """
        from toolkit.body_proportion_loss import compute_body_proportion_loss

        cfg = self.body_proportion_config
        ref_bp = getattr(batch, 'body_proportion_gt', None)
        if ref_bp is None:
            return 0.0

        encoder = getattr(self, '_body_proportion_perceptor', None)
        if encoder is None:
            encoder = self._load_body_proportion_perceptor()

        num_t = self.sd.noise_scheduler.config.num_train_timesteps
        t_ratio = (timesteps.float() / num_t)
        is_reg = batch.get_is_reg_list()

        per_ds_w = getattr(batch, 'body_proportion_loss_weight_list', None)
        has_per_ds = per_ds_w is not None and any(w is not None for w in per_ds_w)
        global_w = cfg.loss_weight
        min_t_list = getattr(batch, 'body_proportion_loss_min_t_list', None)
        max_t_list = getattr(batch, 'body_proportion_loss_max_t_list', None)
        include_head = bool(getattr(cfg, 'include_head', False))

        B = noise_pred.shape[0]
        weights = []
        valid = torch.zeros(B, dtype=torch.bool, device=self.device_torch)
        for i in range(B):
            w = per_ds_w[i] if (has_per_ds and per_ds_w[i] is not None) else global_w
            weights.append(float(w or 0.0))
            if is_reg[i] or weights[i] <= 0.0:
                continue
            # ref_bp is (B, 2N); a zero row means no body detected for that item
            if ref_bp[i].abs().sum().item() == 0.0:
                continue
            lo = float(min_t_list[i]) if (min_t_list and min_t_list[i] is not None) else cfg.loss_min_t
            hi = float(max_t_list[i]) if (max_t_list and max_t_list[i] is not None) else cfg.loss_max_t
            tr = float(t_ratio[i].item())
            if not (lo <= tr <= hi):
                continue
            valid[i] = True

        if not valid.any():
            self._last_body_proportion_loss = 0.0
            return 0.0

        # x0 recovery + decode (grad-enabled)
        t_b = t_ratio.view(-1, 1, 1, 1)
        if getattr(self.sd, 'is_flow_matching', True):
            x0 = noisy_latents - t_b * noise_pred
        else:
            _ac = self.sd.noise_scheduler.alphas_cumprod.to(
                device=timesteps.device, dtype=noisy_latents.dtype
            )[timesteps.long()].view(-1, 1, 1, 1)
            _sa = _ac.sqrt()
            _s1ma = (1.0 - _ac).sqrt()
            if getattr(self.sd, 'prediction_type', None) == 'v_prediction':
                x0 = _sa * noisy_latents - _s1ma * noise_pred
            else:
                x0 = (noisy_latents - _s1ma * noise_pred) / _sa.clamp(min=1e-8)

        if next(self.sd.vae.parameters()).device != self.device_torch:
            self.sd.vae.to(self.device_torch)
        decoded = self.sd.decode_latents(
            x0, device=self.device_torch, dtype=self.sd.vae_torch_dtype,
        )
        pixels = ((decoded.float() + 1.0) * 0.5).clamp(0, 1)

        idx_map = [i for i in range(B) if valid[i]]
        ref_subset = ref_bp[idx_map].to(pixels.device, dtype=torch.float32)
        n = ref_subset.shape[-1] // 2
        ref_ratios = ref_subset[:, :n]
        ref_vis = ref_subset[:, n:]

        # Linear timestep weight (higher noise -> more weight, per source).
        bp_weight = t_ratio[idx_map]
        gen_ratios, gen_vis = encoder(pixels[idx_map], ref_ratios=ref_ratios, include_head=include_head)
        loss_per_sample, _missing = compute_body_proportion_loss(
            gen_ratios, gen_vis, ref_ratios, ref_vis,
        )
        loss_per_sample = loss_per_sample * bp_weight
        w_tensor = torch.tensor(
            [weights[i] for i in idx_map], device=pixels.device, dtype=torch.float32,
        )
        applied = (loss_per_sample * w_tensor).mean()
        self._last_body_proportion_loss_applied = float(applied.detach().item())
        with torch.no_grad():
            self._last_body_proportion_loss = float(loss_per_sample.mean().item())
        return applied

    # ------------------------------------------------------------------
    # Face-identity anchor (ArcFace) -- Phase 3
    # ------------------------------------------------------------------

    def _face_identity_should_cache(self) -> bool:
        cfg = self.face_id_config
        if cfg is None:
            return False
        return cfg.identity_loss_weight > 0 or any(
            face_identity_active_for_dataset(cfg, dc) for dc in self.dataset_configs
        )

    def _face_identity_loss_active(self) -> bool:
        return self._face_identity_should_cache()

    def _load_face_identity_perceptor(self):
        """Lazily load ArcFace + the SCRFD quality-gate detector; cached on trainer.

        Also computes the ArcFace bias direction: the mean embedding of 200 noise
        images. Subtracting + renormalizing collapses ArcFace's ~0.5 cluster bias
        for non-faces so the loss only rewards genuine identity similarity. The
        lazy imports surface a clean ImportError if insightface/onnx2torch/
        onnxruntime-gpu are missing.
        """
        if getattr(self, '_id_loss_model', None) is not None:
            return self._id_loss_model
        from toolkit.face_id import DifferentiableFaceEncoder
        cfg = self.face_id_config
        self._id_loss_model = DifferentiableFaceEncoder(
            model_name=getattr(cfg, 'face_model', 'buffalo_l'),
            device=self.device_torch,
        )
        # ArcFace bias correction direction.
        print_acc("  Computing ArcFace bias direction from noise...")
        with torch.no_grad():
            noise_embeds = []
            for _ in range(200):
                noise_img = (torch.randn(1, 3, 112, 112, device=self.device_torch) * 0.3 + 0.5).clamp(0, 1)
                noise_embeds.append(self._id_loss_model(noise_img))
            self._identity_mean_embed = torch.cat(noise_embeds, dim=0).mean(dim=0).cpu()
        # SCRFD detector for the x0 quality gate (skip generated non-face blobs).
        try:
            from insightface.app import FaceAnalysis
            self._id_face_detector = FaceAnalysis(
                name=getattr(cfg, 'face_model', 'buffalo_l'),
                allowed_modules=['detection'],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
            self._id_face_detector.prepare(ctx_id=0, det_size=(160, 160))
        except Exception as e:  # noqa: BLE001 -- gate is best-effort
            print_acc(f"  [face_id] SCRFD quality-gate detector unavailable ({e}); skipping gate")
            self._id_face_detector = None
        return self._id_loss_model

    def _cache_face_identity_gt_pass(self):
        from toolkit.face_id import cache_face_identity
        cfg = self.face_id_config
        encoder = self._load_face_identity_perceptor()

        def _ds_id_active(dc):
            return face_identity_active_for_dataset(cfg, dc)

        def _cache_loader(loader):
            if loader is None:
                return
            for dataset in get_dataloader_datasets(loader):
                if not _ds_id_active(dataset.dataset_config):
                    continue
                cache_face_identity(dataset.file_list, cfg, encoder=encoder)

        _cache_loader(self.data_loader)
        _cache_loader(self.data_loader_reg)

    def _compute_face_identity_anchor_loss(self, noise_pred, noisy_latents, timesteps, batch):
        """Run the unified live decode + ArcFace forward + bias-corrected cosine loss.

        Face-identity does NOT participate in diffusion/depth loss_split; it fires
        every step on samples in its timestep window with a cached GT embedding and
        a detected face. The SCRFD quality gate skips generated blobs that ArcFace
        would otherwise score spuriously. Returns the weighted-mean contribution.
        """
        from toolkit.face_id_loss import compute_identity_loss, bias_corrected_cosine

        cfg = self.face_id_config
        ref_emb = getattr(batch, 'identity_embedding', None)
        if ref_emb is None:
            return 0.0

        encoder = getattr(self, '_id_loss_model', None)
        if encoder is None:
            encoder = self._load_face_identity_perceptor()
        mean_emb = getattr(self, '_identity_mean_embed', None)
        face_detector = getattr(self, '_id_face_detector', None)

        num_t = self.sd.noise_scheduler.config.num_train_timesteps
        t_ratio = (timesteps.float() / num_t)
        is_reg = batch.get_is_reg_list()

        per_ds_w = getattr(batch, 'identity_loss_weight_list', None)
        has_per_ds = per_ds_w is not None and any(w is not None for w in per_ds_w)
        global_w = cfg.identity_loss_weight
        min_t_list = getattr(batch, 'identity_loss_min_t_list', None)
        max_t_list = getattr(batch, 'identity_loss_max_t_list', None)
        min_cos_list = getattr(batch, 'identity_loss_min_cos_list', None)
        face_bboxes = getattr(batch, 'face_bboxes', None)

        B = noise_pred.shape[0]
        weights = []
        valid = torch.zeros(B, dtype=torch.bool, device=self.device_torch)
        for i in range(B):
            w = per_ds_w[i] if (has_per_ds and per_ds_w[i] is not None) else global_w
            weights.append(float(w or 0.0))
            if is_reg[i] or weights[i] <= 0.0:
                continue
            if ref_emb[i].abs().sum().item() == 0.0:  # no face cached for this item
                continue
            lo = float(min_t_list[i]) if (min_t_list and min_t_list[i] is not None) else cfg.identity_loss_min_t
            hi = float(max_t_list[i]) if (max_t_list and max_t_list[i] is not None) else cfg.identity_loss_max_t
            tr = float(t_ratio[i].item())
            if not (lo <= tr <= hi):
                continue
            valid[i] = True

        if not valid.any():
            self._last_identity_loss = 0.0
            return 0.0

        # x0 recovery + decode (grad-enabled)
        t_b = t_ratio.view(-1, 1, 1, 1)
        if getattr(self.sd, 'is_flow_matching', True):
            x0 = noisy_latents - t_b * noise_pred
        else:
            _ac = self.sd.noise_scheduler.alphas_cumprod.to(
                device=timesteps.device, dtype=noisy_latents.dtype
            )[timesteps.long()].view(-1, 1, 1, 1)
            _sa = _ac.sqrt()
            _s1ma = (1.0 - _ac).sqrt()
            if getattr(self.sd, 'prediction_type', None) == 'v_prediction':
                x0 = _sa * noisy_latents - _s1ma * noise_pred
            else:
                x0 = (noisy_latents - _s1ma * noise_pred) / _sa.clamp(min=1e-8)

        if next(self.sd.vae.parameters()).device != self.device_torch:
            self.sd.vae.to(self.device_torch)
        decoded = self.sd.decode_latents(
            x0, device=self.device_torch, dtype=self.sd.vae_torch_dtype,
        )
        pixels = ((decoded.float() + 1.0) * 0.5).clamp(0, 1)

        # Scale cached normalized bboxes to x0_pixels coords; None where no face.
        _, _, px_h, px_w = pixels.shape
        scaled_bboxes = []
        for i in range(B):
            if valid[i] and face_bboxes is not None and i < len(face_bboxes) and face_bboxes[i] is not None:
                nb = face_bboxes[i]
                # zero bbox (all zeros) means no face
                if nb.abs().sum().item() > 0:
                    scaled_bboxes.append([
                        float(nb[0]) * px_w, float(nb[1]) * px_h,
                        float(nb[2]) * px_w, float(nb[3]) * px_h,
                    ])
                else:
                    scaled_bboxes.append(None)
            else:
                scaled_bboxes.append(None)

        idx_map = [i for i in range(B) if valid[i]]
        gen_emb, arcface_crops = encoder(pixels[idx_map], bboxes=[scaled_bboxes[i] for i in idx_map], return_crops=True)
        ref_subset = ref_emb[idx_map].to(pixels.device, dtype=torch.float32)

        # SCRFD quality gate: skip generated crops where no face is detected.
        face_detected = torch.ones(len(idx_map), dtype=torch.bool, device=pixels.device)
        if face_detector is not None and arcface_crops is not None:
            import cv2 as _cv2
            with torch.no_grad():
                for ci in range(arcface_crops.shape[0]):
                    crop_np = (arcface_crops[ci].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                    crop_bgr = _cv2.cvtColor(crop_np, _cv2.COLOR_RGB2BGR)
                    faces = face_detector.get(crop_bgr)
                    if len(faces) == 0:
                        face_detected[ci] = False

        cos_sim = bias_corrected_cosine(gen_emb, ref_subset, mean_emb)

        # Build the per-index mask in idx_map order.
        ref_valid = ref_subset.abs().sum(dim=-1) > 0
        cos_threshold = torch.tensor(
            [float(min_cos_list[i]) if (min_cos_list and min_cos_list[i] is not None) else cfg.identity_loss_min_cos
             for i in idx_map],
            device=pixels.device, dtype=torch.float32,
        )
        loss_mask = ref_valid & (cos_sim.detach() > cos_threshold) & face_detected
        id_weight = t_ratio[idx_map]

        per_sample_loss = (1.0 - cos_sim) * id_weight * loss_mask.float()
        w_tensor = torch.tensor(
            [weights[i] for i in idx_map], device=pixels.device, dtype=torch.float32,
        )
        if loss_mask.any():
            applied = (per_sample_loss * w_tensor).sum() / max(int(loss_mask.sum().item()), 1)
            self._last_identity_loss_applied = float(applied.detach().item())
            with torch.no_grad():
                self._last_id_sim = float((cos_sim * loss_mask.float()).sum().item() / max(int(loss_mask.sum().item()), 1))
                self._last_identity_loss = float(((1.0 - cos_sim) * loss_mask.float()).sum().item() / max(int(loss_mask.sum().item()), 1))
            return applied
        self._last_identity_loss = 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Subject-mask region weighting (auto-masking) -- Phase 3
    # ------------------------------------------------------------------

    def _build_subject_mask_weight(self, batch, latent_shape, dtype=None):
        """Build a per-latent region weight map from cached subject masks.

        Returns ``None`` (no-op) when auto-masking is disabled, no masks are
        present, or all of background/clothing/body loss weights resolve to
        None. Otherwise a ``(B, C_lat, lat_h, lat_w)`` float tensor composed
        multiplicatively: weight_map = ones; *= where(person, 1, bg_w);
        *= where(clothing, clothing_w, 1); *= where(body, body_w, 1). Per-item
        weight resolution: non-None per-dataset override > non-None global > None.
        """
        import torch.nn.functional as _F
        smc = self.subject_mask_config
        if smc is None or not smc.enabled:
            return None
        person = getattr(batch, 'subject_masks', None)
        body = getattr(batch, 'body_masks', None)
        clothing = getattr(batch, 'clothing_masks', None)
        if person is None and body is None and clothing is None:
            return None

        B = len(batch.file_items)
        _, C, lat_h, lat_w = latent_shape
        device = self.device_torch

        def _per_item(global_val, attr_name):
            lst = getattr(batch, attr_name + '_list', None)
            if lst is None:
                return [global_val] * B
            return [(v if v is not None else global_val) for v in lst]

        bg = _per_item(smc.background_loss_weight, 'background_loss_weight')
        cl = _per_item(smc.clothing_loss_weight, 'clothing_loss_weight')
        bd = _per_item(smc.body_loss_weight, 'body_loss_weight')
        if all(w is None for w in bg) and all(w is None for w in cl) and all(w is None for w in bd):
            return None

        def _resize_mask(stacked):
            if stacked is None:
                return None
            m = stacked.to(device=device, dtype=torch.float32)
            return _F.interpolate(m, size=(lat_h, lat_w), mode='nearest')

        person_lat = _resize_mask(person)
        body_lat = _resize_mask(body)
        clothing_lat = _resize_mask(clothing)
        weight_map = torch.ones((B, 1, lat_h, lat_w), device=device, dtype=torch.float32)

        for i in range(B):
            if bg[i] is not None and person_lat is not None:
                weight_map[i] = weight_map[i] * torch.where(
                    person_lat[i] > 0.5, torch.ones_like(person_lat[i]), torch.full_like(person_lat[i], float(bg[i]))
                )
            if cl[i] is not None and clothing_lat is not None:
                weight_map[i] = weight_map[i] * torch.where(
                    clothing_lat[i] > 0.5, torch.full_like(clothing_lat[i], float(cl[i])), torch.ones_like(clothing_lat[i])
                )
            if bd[i] is not None and body_lat is not None:
                weight_map[i] = weight_map[i] * torch.where(
                    body_lat[i] > 0.5, torch.full_like(body_lat[i], float(bd[i])), torch.ones_like(body_lat[i])
                )

        if dtype is not None:
            weight_map = weight_map.to(dtype)
        return weight_map.expand(B, C, lat_h, lat_w).contiguous()

    def _build_body_restrict_mask(self, batch, spatial_shape):
        """Per-sample body-restriction mask for perceptual anchors.

        Returns ``None`` when disabled or no item opts into
        ``perceptual_restrict_to_body``. Otherwise ``(B, H, W)`` float: 1.0
        inside the body, 0.0 outside; items that didn't opt in are all-ones.
        """
        import torch.nn.functional as _F
        smc = self.subject_mask_config
        if smc is None or not smc.enabled:
            return None
        body = getattr(batch, 'body_masks', None)
        if body is None:
            return None
        g_restrict = bool(getattr(smc, 'perceptual_restrict_to_body', False))
        per_item = [
            (v if v is not None else g_restrict)
            for v in getattr(batch, 'perceptual_restrict_to_body_list', [])
        ]
        if not any(per_item):
            return None
        _, H, W = spatial_shape
        B = len(per_item)
        body_f = body.to(torch.float32)
        body_f = _F.interpolate(body_f, size=(H, W), mode='nearest').squeeze(1)  # (B, H, W)
        restrict_vec = torch.tensor(
            [1.0 if v else 0.0 for v in per_item], dtype=torch.float32, device=body_f.device
        ).view(B, 1, 1)
        return restrict_vec * body_f + (1.0 - restrict_vec) * torch.ones_like(body_f)

    # ------------------------------------------------------------------
    # Body-shape anchor (HybrIK SMPL betas) -- Phase 3
    # ------------------------------------------------------------------

    def _body_shape_should_cache(self) -> bool:
        cfg = self.body_shape_config
        if cfg is None:
            return False
        return cfg.loss_weight > 0 or any(
            body_shape_active_for_dataset(cfg, dc) for dc in self.dataset_configs
        )

    def _body_shape_loss_active(self) -> bool:
        return self._body_shape_should_cache()

    def _load_body_shape_perceptor(self):
        if getattr(self, '_body_shape_perceptor', None) is not None:
            return self._body_shape_perceptor
        from toolkit.body_shape import DifferentiableBodyShapeEncoder
        self._body_shape_perceptor = DifferentiableBodyShapeEncoder(device=self.device_torch)
        return self._body_shape_perceptor

    def _cache_body_shape_gt_pass(self):
        from toolkit.body_shape import cache_body_shape
        cfg = self.body_shape_config
        encoder = self._load_body_shape_perceptor()

        def _ds_bs_active(dc):
            return body_shape_active_for_dataset(cfg, dc)

        for loader in (self.data_loader, self.data_loader_reg):
            if loader is None:
                continue
            for dataset in get_dataloader_datasets(loader):
                if not _ds_bs_active(dataset.dataset_config):
                    continue
                cache_body_shape(dataset.file_list, cfg, encoder=encoder)

    def _compute_body_shape_anchor_loss(self, noise_pred, noisy_latents, timesteps, batch):
        """Live decode + HybrIK forward + L1 on 10 SMPL betas (cosine-gated)."""
        from toolkit.body_shape_loss import compute_body_shape_loss

        cfg = self.body_shape_config
        ref_bs = getattr(batch, 'body_shape_gt', None)
        if ref_bs is None:
            return 0.0
        encoder = getattr(self, '_body_shape_perceptor', None)
        if encoder is None:
            encoder = self._load_body_shape_perceptor()

        num_t = self.sd.noise_scheduler.config.num_train_timesteps
        t_ratio = (timesteps.float() / num_t)
        is_reg = batch.get_is_reg_list()
        per_ds_w = getattr(batch, 'body_shape_loss_weight_list', None)
        has_per_ds = per_ds_w is not None and any(w is not None for w in per_ds_w)
        global_w = cfg.loss_weight
        min_t_list = getattr(batch, 'body_shape_loss_min_t_list', None)
        max_t_list = getattr(batch, 'body_shape_loss_max_t_list', None)
        min_cos_list = getattr(batch, 'body_shape_loss_min_cos_list', None)

        B = noise_pred.shape[0]
        weights, valid = [], torch.zeros(B, dtype=torch.bool, device=self.device_torch)
        for i in range(B):
            w = per_ds_w[i] if (has_per_ds and per_ds_w[i] is not None) else global_w
            weights.append(float(w or 0.0))
            if is_reg[i] or weights[i] <= 0.0:
                continue
            if ref_bs[i].abs().sum().item() == 0.0:
                continue
            lo = float(min_t_list[i]) if (min_t_list and min_t_list[i] is not None) else cfg.loss_min_t
            hi = float(max_t_list[i]) if (max_t_list and max_t_list[i] is not None) else cfg.loss_max_t
            tr = float(t_ratio[i].item())
            if not (lo <= tr <= hi):
                continue
            valid[i] = True

        if not valid.any():
            self._last_body_shape_loss = 0.0
            return 0.0

        t_b = t_ratio.view(-1, 1, 1, 1)
        if getattr(self.sd, 'is_flow_matching', True):
            x0 = noisy_latents - t_b * noise_pred
        else:
            _ac = self.sd.noise_scheduler.alphas_cumprod.to(
                device=timesteps.device, dtype=noisy_latents.dtype
            )[timesteps.long()].view(-1, 1, 1, 1)
            _sa = _ac.sqrt()
            _s1ma = (1.0 - _ac).sqrt()
            if getattr(self.sd, 'prediction_type', None) == 'v_prediction':
                x0 = _sa * noisy_latents - _s1ma * noise_pred
            else:
                x0 = (noisy_latents - _s1ma * noise_pred) / _sa.clamp(min=1e-8)

        if next(self.sd.vae.parameters()).device != self.device_torch:
            self.sd.vae.to(self.device_torch)
        decoded = self.sd.decode_latents(x0, device=self.device_torch, dtype=self.sd.vae_torch_dtype)
        pixels = ((decoded.float() + 1.0) * 0.5).clamp(0, 1)

        idx_map = [i for i in range(B) if valid[i]]
        ref_subset = ref_bs[idx_map].to(pixels.device, dtype=torch.float32)
        gen_betas = encoder(pixels[idx_map])  # (V, 10)
        l1, cos = compute_body_shape_loss(gen_betas, ref_subset)

        cos_threshold = torch.tensor(
            [float(min_cos_list[i]) if (min_cos_list and min_cos_list[i] is not None) else cfg.loss_min_cos
             for i in idx_map],
            device=pixels.device, dtype=torch.float32,
        )
        gate = (cos > cos_threshold)
        bs_weight = t_ratio[idx_map]
        per_sample = l1 * bs_weight * gate.float()
        w_tensor = torch.tensor(
            [weights[i] for i in idx_map], device=pixels.device, dtype=torch.float32,
        )
        n_valid = max(int(gate.sum().item()), 1)
        applied = (per_sample * w_tensor).sum() / n_valid
        self._last_body_shape_loss_applied = float(applied.detach().item())
        with torch.no_grad():
            self._last_body_shape_cos = float((cos * gate.float()).sum().item() / n_valid)
            self._last_body_shape_loss = float((l1 * gate.float()).sum().item() / n_valid)
        return applied

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.depth_consistency_config = preflight_depth_consistency(
            self.depth_consistency_config,
            self.dataset_configs,
            getattr(self.sd.model_config, 'arch', None),
            self.model_config.low_vram,
        )
        # Fresh start for a re-run / reattempt: preview cadence counts DEPTH
        # steps, so a stale counter would desync the tile cadence.
        self._depth_step_count = 0
        # Depth-GT caching pass: runs AFTER preflight and AFTER dataloaders
        # exist. Fully inert when depth is inactive (no DA2 import, no perceptor
        # load, no file-item stamping). The VAE round-trip pulls the VAE back
        # onto the device when latents were cached.
        if self._depth_should_cache():
            self._cache_depth_gt_pass()
        # Normal-anchor preflight + GT caching pass. Fully inert when normal is
        # inactive (no Sapiens import, no perceptor load, no stamping). Normal
        # GT is computed from the raw source image -- no VAE round-trip.
        self.normal_config = preflight_normal_id(
            self.normal_config, self.dataset_configs,
            getattr(self.sd.model_config, 'arch', None),
            self.model_config.low_vram,
        )
        self._normal_step_count = 0
        if self._normal_should_cache():
            self._cache_normal_gt_pass()
        # Body-proportion preflight + GT caching pass. Fully inert when inactive.
        self.body_proportion_config = preflight_body_proportion(
            self.body_proportion_config, self.dataset_configs,
            getattr(self.sd.model_config, 'arch', None),
            self.model_config.low_vram,
        )
        if self._body_proportion_should_cache():
            self._cache_body_proportion_gt_pass()
        # Face-identity preflight + GT caching pass. Fully inert when inactive;
        # the lazy import raises a clean ImportError if deps are missing.
        self.face_id_config = preflight_face_id(
            self.face_id_config, self.dataset_configs,
            getattr(self.sd.model_config, 'arch', None),
            self.model_config.low_vram,
        )
        if self._face_identity_should_cache():
            self._cache_face_identity_gt_pass()
        # Body-shape preflight + GT caching pass.
        self.body_shape_config = preflight_body_shape(
            self.body_shape_config, self.dataset_configs,
            getattr(self.sd.model_config, 'arch', None),
            self.model_config.low_vram,
        )
        if self._body_shape_should_cache():
            self._cache_body_shape_gt_pass()
        # Auto-masking (YOLO + SAM 2 + SegFormer): cache person/body/clothing
        # masks when subject_mask is enabled. Fully inert (no model downloads,
        # no extraction) when disabled.
        if self.subject_mask_config is not None and self.subject_mask_config.enabled:
            from toolkit.subject_mask import cache_subject_masks
            print_acc("Auto-masking: Extracting and caching subject masks...")
            _sm_preview_dir = os.path.join(self.save_root, 'subject_mask_previews')
            for loader in (self.data_loader, self.data_loader_reg):
                if loader is None:
                    continue
                for dataset in get_dataloader_datasets(loader):
                    cache_subject_masks(dataset.file_list, self.subject_mask_config,
                                        preview_dir=_sm_preview_dir)
        # Cross-check: depth mask_source subject|body requires subject_mask enabled.
        if (self.depth_consistency_config is not None
                and self.depth_consistency_config.loss_weight > 0
                and self.depth_consistency_config.mask_source != 'none'
                and not (self.subject_mask_config is not None and self.subject_mask_config.enabled)):
            raise ValueError(
                f"depth_consistency.mask_source={self.depth_consistency_config.mask_source!r} "
                "requires subject_mask.enabled: true (enable the auto-masking section)."
            )
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to('cpu')
        
        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            self.unconditional_embeds = self.encode_static_prompt(
                [self.train_config.unconditional_prompt],
                long_prompts=self.do_long_prompts,
            ).to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            # check if we have regs and using adapter and caching clip embeddings
            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                # we need a list of unconditional clip image embeds from other datasets to handle regs
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache

                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")

                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder
        if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
            print_acc("Caching embeddings and unloading text encoder")
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings
                self.sd.text_encoder_to(self.device_torch)
                self.cached_blank_embeds = self.encode_static_prompt("")
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.encode_static_prompt(self.trigger_word)
                if self.train_config.diff_output_preservation:
                    self.cached_dop_class_embeds = self.encode_static_prompt(self.train_config.diff_output_preservation_class)
                    self.diff_output_preservation_embeds = self.cached_dop_class_embeds
                
                self.cache_sample_prompts()
                
                print_acc("\n***** UNLOADING TEXT ENCODER *****")
                if self.is_caching_text_embeddings:
                    print_acc("Embeddings cached to disk. We dont need the text encoder anymore")
                else:
                    print_acc("This will train only with a blank prompt or trigger word, if set")
                    print_acc("If this is not what you want, remove the unload_text_encoder flag")
                print_acc("***********************************")
                print_acc("")

                # unload the text encoder
                if self.is_caching_text_embeddings:
                    unload_text_encoder(self.sd)
                else:
                    # todo once every model is tested to work, unload properly. Though, this will all be merged into one thing.
                    # keep legacy usage for now. 
                    self.sd.text_encoder_to("cpu")
                flush()
        
        if self.train_config.blank_prompt_preservation and self.cached_blank_embeds is None:
            # make sure we have this if not unloading
            self.cached_blank_embeds = self.sd.encode_prompt("").to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.diffusion_feature_extractor_path is not None:
            vae = self.sd.vae
            # if not (self.model_config.arch in ["flux"]) or self.sd.vae.__class__.__name__ == "AutoencoderPixelMixer":
            #     vae = self.sd.vae
            self.dfe = load_dfe(
                self.train_config.diffusion_feature_extractor_path, 
                vae=vae,
                sd=self.sd
            )
            self.dfe.to(self.device_torch)
            if hasattr(self.dfe, 'vision_encoder') and self.train_config.gradient_checkpointing:
                # must be set to train for gradient checkpointing to work
                self.dfe.vision_encoder.train()
                self.dfe.vision_encoder.gradient_checkpointing = True
            elif hasattr(self.dfe, 'model') and self.train_config.gradient_checkpointing:
                if hasattr(self.dfe.model, 'enable_gradient_checkpointing'): 
                    self.dfe.model.train()
                    self.dfe.model.enable_gradient_checkpointing()
                if hasattr(self.dfe.model, 'gradient_checkpointing_enable'): 
                    self.dfe.model.train()
                    self.dfe.model.gradient_checkpointing_enable()
                elif hasattr(self.dfe.model, 'gradient_checkpointing'):
                    self.dfe.model.train()
                    self.dfe.model.gradient_checkpointing = True
                else:
                    print_acc("Warning: Could not enable gradient checkpointing on diffusion feature extractor model.")
            else:
                self.dfe.eval()
                
            # enable gradient checkpointing on the vae
            if vae is not None and self.train_config.gradient_checkpointing:
                try:
                    vae.enable_gradient_checkpointing()
                    vae.train()
                except:
                    pass


    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        # to process turbo learning, we make one big step from our current timestep to the end
        # we then denoise the prediction on that remaining step and target our loss to our target latents
        # this currently only works on euler_a (that I know of). Would work on others, but needs to be coded to do so.
        # needs to be done on each item in batch as they may all have different timesteps
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            # set the timesteps to 1000 so we can capture them to calculate the sigmas
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()

            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()

            # set the scheduler to one timestep, we build the step and sigmas for each item in batch for the partial step
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        target_pred_chunks = []

        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            latents_item = latent_chunks[i]
            noise_item = noise_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                # extract the sigma idx for our midpoint timestep
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)

                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)

                # add noise to our target

                # build the big sigma step. The to step will now be to 0 giving it a full remaining denoising half step
                # self.sd.noise_scheduler.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas)]).detach()
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                # set our single timstep
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)

                # set the step index to None so it will be recalculated on first step
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]

            residual_noise = (noise_item * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            # remove the residual noise from the denoised latents. Output should be a clean prediction (theoretically)
            denoised_latent = denoised_latent - residual_noise

            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        # set the scheduler back to the original timesteps
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = denoised_latents / self.sd.vae.config['scaling_factor']
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            # since we are completely denoising, we can show them here
            with torch.no_grad():
                show_tensors(output)

        # we return our big partial step denoised latents as our pred and our untouched latents as our target.
        # you can do mse against the two here  or run the denoised through the vae for pixel space loss against the
        # input tensor images.

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    # you can expand these in a child class to make customization easier
    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        _is_reg_list = batch.get_is_reg_list()
        is_reg = any(_is_reg_list)
        additional_loss = 0.0

        # Depth-anchor sample gating (Task 5b). Fully inert when depth is off:
        # _depth_gates stays None, so no depth code is reachable and the loss
        # path below is byte-for-byte the original.
        _depth_gates = None
        if (self._depth_loss_active()
                and getattr(batch, 'depth_gt_list', None) is not None
                and len(noise_pred.shape) == 4):
            _is_reg_per_sample = torch.tensor(
                [bool(v) for v in _is_reg_list],
                device=self.device_torch, dtype=torch.bool,
            )
            _depth_gates = self._resolve_depth_sample_gates(
                batch, timesteps, _is_reg_per_sample,
            )

        # Normal-anchor gate. Fully inert when normal is off: _normal_gates
        # stays False, so no normal code is reachable. Normal does not interact
        # with depth loss_split -- it runs independently on its own samples.
        _normal_active = (
            self._normal_loss_active()
            and getattr(batch, 'normal_gt_list', None) is not None
            and len(noise_pred.shape) == 4
        )
        # Body-proportion gate. Independent of depth/normal; fires on its own
        # timestep-window samples with cached GT ratios.
        _body_proportion_active = (
            self._body_proportion_loss_active()
            and getattr(batch, 'body_proportion_gt', None) is not None
            and len(noise_pred.shape) == 4
        )
        # Face-identity gate. Independent of the other anchors.
        _face_identity_active = (
            self._face_identity_loss_active()
            and getattr(batch, 'identity_embedding', None) is not None
            and len(noise_pred.shape) == 4
        )
        # Body-shape gate.
        _body_shape_active = (
            self._body_shape_loss_active()
            and getattr(batch, 'body_shape_gt', None) is not None
            and len(noise_pred.shape) == 4
        )

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    # this only works if doing a prior pred
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)

                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std

                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier

                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust

                        eps = 1e-5
                        # match the noise to the prior
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()

            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W
                        lat_height = batch.latents.shape[3]
                        lat_width = batch.latents.shape[4]
                    else: 
                        lat_height = batch.latents.shape[2]
                        lat_width = batch.latents.shape[3]
                    # resize to size of noise_pred
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
                    # stack first channel to match channels of noise_pred
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
                    
                    if len(noise_pred.shape) == 5:
                        prior_mask = prior_mask.unsqueeze(2)  # add time dimension back for video
                        prior_mask = prior_mask.repeat(1, 1, noise_pred.shape[2], 1, 1) 

                    prior_mask_multiplier = 1.0 - prior_mask
                    
                    # scale so it is a mean of 1
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if hasattr(self.sd, 'get_loss_target'):
                    target = self.sd.get_loss_target(
                        noise=noise, 
                        batch=batch, 
                        timesteps=timesteps,
                    ).detach()
                elif self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        elif self.train_config.do_signal_amplification:
            if not self.sd.is_flow_matching:
                raise ValueError("Signal amplification is only supported for flow matching models")
            with torch.no_grad():
                nas = 1.0 - (timesteps / 1000).to(noise.device, dtype=noise.dtype)
                nas = nas * self.train_config.signal_amplification_strength
                while len(nas.shape) < len(noise.shape):
                    nas = nas.unsqueeze(-1)
                aug = batch.latents * nas
                target = noise - (batch.latents + aug)
                target = target.detach()
        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(
                noise=noise, 
                batch=batch, 
                timesteps=timesteps,
            ).detach()
            
        elif self.sd.is_flow_matching:
            # forward ODE
            target = (noise - batch.latents).detach()
            # reverse ODE
            # target = (batch.latents - noise).detach()
        else:
            target = noise
            
        if self.dfe is not None:
            if self.dfe.version == 1:
                model = self.sd
                if model is not None and hasattr(model, 'get_stepped_pred'):
                    stepped_latents = model.get_stepped_pred(noise_pred, noise)
                else:
                    # stepped_latents = noise - noise_pred
                    # first we step the scheduler from current timestep to the very end for a full denoise
                    bs = noise_pred.shape[0]
                    noise_pred_chunks = torch.chunk(noise_pred, bs)
                    timestep_chunks = torch.chunk(timesteps, bs)
                    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
                    stepped_chunks = []
                    for idx in range(bs):
                        model_output = noise_pred_chunks[idx]
                        timestep = timestep_chunks[idx]
                        self.sd.noise_scheduler._step_index = None
                        self.sd.noise_scheduler._init_step_index(timestep)
                        sample = noisy_latent_chunks[idx].to(torch.float32)
                        
                        sigma = self.sd.noise_scheduler.sigmas[self.sd.noise_scheduler.step_index]
                        sigma_next = self.sd.noise_scheduler.sigmas[-1] # use last sigma for final step
                        prev_sample = sample + (sigma_next - sigma) * model_output
                        stepped_chunks.append(prev_sample)
                    
                    stepped_latents = torch.cat(stepped_chunks, dim=0)
                    
                stepped_latents = stepped_latents.to(self.sd.vae.device, dtype=self.sd.vae.dtype)
                sl = stepped_latents
                if len(sl.shape) == 5:
                    # video B,C,T,H,W
                    sl = sl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                    b, t, c, h, w = sl.shape
                    sl = sl.reshape(b * t, c, h, w)
                pred_features = self.dfe(sl.float())
                with torch.no_grad():
                    bl = batch.latents
                    bl = bl.to(self.sd.vae.device)
                    if len(bl.shape) == 5:
                        # video B,C,T,H,W
                        bl = bl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                        b, t, c, h, w = bl.shape
                        bl = bl.reshape(b * t, c, h, w)
                    target_features = self.dfe(bl.float())
                    # scale dfe so it is weaker at higher noise levels
                    dfe_scaler = 1 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1).to(self.device_torch)
                
                dfe_loss = torch.nn.functional.mse_loss(pred_features, target_features, reduction="none") * \
                    self.train_config.diffusion_feature_extractor_weight * dfe_scaler
                additional_loss += dfe_loss.mean()
            elif self.dfe.version == 2:
                # version 2
                # do diffusion feature extraction on target
                with torch.no_grad():
                    rectified_flow_target = noise.float() - batch.latents.float()
                    target_feature_list = self.dfe(torch.cat([rectified_flow_target, noise.float()], dim=1))
                
                # do diffusion feature extraction on prediction
                pred_feature_list = self.dfe(torch.cat([noise_pred.float(), noise.float()], dim=1))
                
                dfe_loss = 0.0
                for i in range(len(target_feature_list)):
                    dfe_loss += torch.nn.functional.mse_loss(pred_feature_list[i], target_feature_list[i], reduction="mean")
                
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight * 100.0
            elif self.dfe.version in [3, 4, 5, 6, 7, 8, 9, 10]:
                dfe_loss = self.dfe(
                    noise=noise,
                    noise_pred=noise_pred,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    batch=batch,
                    scheduler=self.sd.noise_scheduler
                )
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight 
            else:
                raise ValueError(f"Unknown diffusion feature extractor version {self.dfe.version}")
        
        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                # we make cached blank prompt embeds that match the batch size
                unconditional_embeds = concat_prompt_embeds(
                    [self.unconditional_embeds] * noisy_latents.shape[0],
                )
                # joint audio models route this pass's audio pred to its own
                # slot so it cannot stomp the primary pred on the batch
                batch.audio_pred_slot = 'audio_pred_uncond'
                unconditional_target = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )
                batch.audio_pred_slot = None
                is_video = len(target.shape) == 5
                
                if self.train_config.do_guidance_loss_cfg_zero:
                    # zero cfg
                    # ref https://github.com/WeichenFan/CFG-Zero-star/blob/cdac25559e3f16cb95f0016c04c709ea1ab9452b/wan_pipeline.py#L557
                    batch_size = target.shape[0]
                    positive_flat = target.view(batch_size, -1)
                    negative_flat = unconditional_target.view(batch_size, -1)
                    # Calculate dot production
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    # Squared norm of uncondition
                    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
                    st_star = dot_product / squared_norm

                    alpha = st_star
                    
                    alpha = alpha.view(batch_size, 1, 1, 1) if not is_video else alpha.view(batch_size, 1, 1, 1, 1)
                else:
                    alpha = 1.0

                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)

                if self.train_config.guidance_loss_schedule == 'sigma':
                    # the (target - uncond) sample direction carries s * fresh_noise
                    # that nothing can predict at low sigma, so decay the
                    # extrapolation toward a plain flow target as sigma falls
                    sigma = (timesteps.to(target.device) / 1000.0).to(target.dtype)
                    sigma = sigma.view(-1, 1, 1, 1) if not is_video else sigma.view(-1, 1, 1, 1, 1)
                    guidance_scale = 1.0 + (guidance_scale - 1.0) * sigma

                unconditional_target = unconditional_target * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)

                # joint audio models (ltx2, minimax_h3, flux3) carry their audio
                # target/pred on the batch. Extrapolate the audio target the
                # same way so the audio stream trains contrastively as well.
                audio_uncond = getattr(batch, 'audio_pred_uncond', None)
                if batch.audio_target is not None and audio_uncond is not None:
                    audio_target = batch.audio_target.float()
                    audio_uncond = audio_uncond.float()
                    audio_dims = [1] * (audio_target.dim() - 1)
                    if self.train_config.do_guidance_loss_cfg_zero:
                        batch_size = audio_target.shape[0]
                        a_pos_flat = audio_target.view(batch_size, -1)
                        a_neg_flat = audio_uncond.view(batch_size, -1)
                        a_dot = torch.sum(a_pos_flat * a_neg_flat, dim=1, keepdim=True)
                        a_squared_norm = torch.sum(a_neg_flat ** 2, dim=1, keepdim=True) + 1e-8
                        audio_uncond = audio_uncond * (a_dot / a_squared_norm).view(-1, *audio_dims)

                    audio_guidance_scale = self._guidance_loss_target_batch
                    if isinstance(audio_guidance_scale, list):
                        audio_guidance_scale = torch.tensor(audio_guidance_scale).to(
                            audio_target.device, dtype=audio_target.dtype
                        ).view(-1, *audio_dims)

                    if self.train_config.guidance_loss_schedule == 'sigma':
                        # audio streams can run on their own remapped sigma
                        audio_sigma = getattr(batch, 'audio_sigma', None)
                        if audio_sigma is None:
                            audio_sigma = timesteps / 1000.0
                        audio_sigma = audio_sigma.to(
                            audio_target.device, dtype=audio_target.dtype
                        ).view(-1, *audio_dims)
                        audio_guidance_scale = 1.0 + (audio_guidance_scale - 1.0) * audio_sigma

                    batch.audio_target = (
                        audio_uncond + audio_guidance_scale * (audio_target - audio_uncond)
                    ).to(batch.audio_target.dtype).detach()

        if target is None:
            target = noise

        if self.train_config.do_differential_guidance:
            with torch.no_grad():
                guidance_scale = self.train_config.differential_guidance_scale
                target = noise_pred + guidance_scale * (target - noise_pred)

        pred = noise_pred

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            assert not self.train_config.train_turbo
            # ignore_snr = True
            if batch.sigmas is None:
                raise ValueError("Batch sigmas is None. This should not happen")

            # src https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1190
            denoised_latents = noise_pred * (-batch.sigmas) + noisy_latents
            weighing = batch.sigmas ** -2.0
            if loss_target == 'source':
                # denoise the latent and compare to the latent in the batch
                target = batch.latents
            elif loss_target == 'unaugmented':
                # we have to encode images into latents for now
                # we also denoise as the unaugmented tensor is not a noisy diffirental
                with torch.no_grad():
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor).to(self.device_torch, dtype=dtype)
                    unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                    target = unaugmented_latents.detach()

                # Get the target for loss depending on the prediction type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    target = target  # we are computing loss against denoise latents
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.sd.noise_scheduler.get_velocity(target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")

            # mse loss without reduction
            loss_per_element = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
            loss = loss_per_element
        else:
            local_loss_scale = 1.0
            if self.train_config.t0_loss_target or self.train_config.do_fft_loss:
                # do the loss on a stepped timestep 0 prediction
                # doto handle doing priors, preservations, masking, etc
                with torch.no_grad():
                    tv = timesteps.to(noise_pred.device).to(noise_pred.dtype) / 1000.0
                    # expand shape to match noise_pred
                    while len(tv.shape) < len(noise_pred.shape):
                        tv = tv.unsqueeze(-1)
                        # min 0.001
                        tv = torch.clamp(tv, min=0.001)
                
                # step latent, use here or with do_fft_loss
                if self.sd.x0_pred:
                    t0 = noise_pred
                else:
                    t0 = noisy_latents - tv * noise_pred
                
                if self.train_config.t0_loss_target:
                    # replace the loss targets and pred
                    target = batch.latents.detach()
                    pred = t0
                    # handle velocity equiv loss if set. This scales t0 loss to match velocity of flowmatchhing loss
                    if self.train_config.t0_velocity_equiv_weight:
                        velocity_equiv_weight = (1.0 / torch.clamp(tv, min=0.1) ** 2)
                        local_loss_scale = velocity_equiv_weight
                        
                if self.train_config.do_fft_loss:
                    with torch.no_grad():
                        target_mag = torch.fft.rfft2(batch.latents.to(t0.device).float(), norm="ortho").abs()
                    pred_mag = torch.fft.rfft2(t0.float(), norm="ortho").abs()
                    fft_loss = F.mse_loss(pred_mag, target_mag, reduction="none")
                    if self.train_config.do_fft_velocity_equiv_weight:
                        velocity_equiv_weight = (1.0 / torch.clamp(tv, min=0.1) ** 2)
                        fft_loss = fft_loss * velocity_equiv_weight
                    additional_loss += fft_loss.mean()
            if self.train_config.loss_type == "pseudo_huber":
                diff = pred.float() - target.float()
                c=0.01
                loss =(torch.sqrt(diff.pow(2) + c ** 2) - c)
            elif self.train_config.loss_type == "mae":
                loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            elif self.train_config.loss_type == "wavelet":
                loss = wavelet_loss(pred, batch.latents, noise)
            elif self.train_config.loss_type == "stepped":
                loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
                # the way this loss works, it is low, increase it to match predictable LR effects
                loss = loss * 10.0
            else:
                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
            
            loss = loss * local_loss_scale
            
            # apply model specific loss scaling
            loss = self.sd.scale_loss(loss)
                
            do_weighted_timesteps = False
            if self.sd.is_flow_matching:
                if self.train_config.linear_timesteps or self.train_config.linear_timesteps2:
                    do_weighted_timesteps = True
                if self.train_config.timestep_type == "weighted":
                    # use the noise scheduler to get the weights for the timesteps
                    do_weighted_timesteps = True

            # handle linear timesteps and only adjust the weight of the timesteps
            if do_weighted_timesteps:
                # calculate the weights for the timesteps
                timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    v2=self.train_config.linear_timesteps2,
                    timestep_type=self.train_config.timestep_type
                ).to(loss.device, dtype=loss.dtype)
                if len(loss.shape) == 4:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1).detach()
                elif len(loss.shape) == 5:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1, 1).detach()
                loss = loss * timestep_weight

        if self.train_config.do_prior_divergence and prior_pred is not None:
            loss = loss + (torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none") * -1.0)

        # Subject-mask region weighting (Phase 3 auto-masking). Composes
        # multiplicatively into mask_multiplier; no-op (None) when disabled.
        _subject_weight = self._build_subject_mask_weight(batch, noisy_latents.shape, dtype=dtype)
        if _subject_weight is not None:
            if not isinstance(mask_multiplier, torch.Tensor):
                mask_multiplier = _subject_weight
            else:
                mask_multiplier = mask_multiplier * _subject_weight

        if self.train_config.train_turbo:
            mask_multiplier = mask_multiplier[:, 3:, :, :]
            # resize to the size of the loss
            mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(pred.shape[2], pred.shape[3]), mode='nearest')

        # multiply by our mask
        try:
            if len(noise_pred.shape) == 5:
                # video B,C,T,H,W
                mask_multiplier = mask_multiplier.unsqueeze(2)  # add time dimension back for video
                mask_multiplier = mask_multiplier.repeat(1, 1, noise_pred.shape[2], 1, 1)
            loss = loss * mask_multiplier
        except Exception as e:
            # todo handle mask with video models
            print("Could not apply mask multiplier to loss")
            print(e)
            pass

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None and prior_mask_multiplier is not None:
            assert not self.train_config.train_turbo
            if self.train_config.loss_type == "mae":
                prior_loss = torch.nn.functional.l1_loss(pred.float(), prior_pred.float(), reduction="none")
            else:
                prior_loss = torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none")

            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if not torch.isfinite(prior_loss).all():
                print_acc("Prior loss is nan")
                prior_loss = None
            else:
                if len(noise_pred.shape) == 5:
                    # video B,C,T,H,W
                    prior_loss = prior_loss.mean([1, 2, 3, 4])
                else:
                    prior_loss = prior_loss.mean([1, 2, 3])
                # loss = loss + prior_loss
                # loss = loss + prior_loss
            # loss = loss + prior_loss
        if len(noise_pred.shape) == 5:
            loss = loss.mean([1, 2, 3, 4])
        else:
            loss = loss.mean([1, 2, 3])
        # per-image adaptive LR: record each item's raw per-sample loss (pre-multiplier) against
        # its timestep, keyed by file path. Model- and network-agnostic — this is the one shared
        # loss path for every architecture and both LoKr and LoRA. Never raises into training.
        if (getattr(self.train_config, 'per_image_adaptive_lr', False)
                or getattr(self.train_config, 'per_image_adaptive_lr_stats_only', False)) \
                and self.loss_watch is not None:
            try:
                loss_detached = loss.detach()
                ts_detached = timesteps.detach()
                window_steps = max(getattr(self, '_adaptive_lr_window_steps', 1), 1)
                window_idx = self.step_num // window_steps
                # timesteps come in on a [0, num_train_timesteps] scale for every arch here
                # (flow-matching schedulers included) — normalize to [0, 1] for the bucket index.
                num_train_timesteps = float(getattr(self.train_config, 'num_train_timesteps', 1000) or 1000)
                for idx, file_item in enumerate(batch.file_items):
                    # nominal configured resolution (e.g. 256/512/1024), NOT the post-aspect-crop
                    # dimensions — with bucketing, crop_width/crop_height vary per aspect ratio
                    # within the same resolution tier (256/288/336/... are all "256"), which would
                    # fragment one tier into several and defeat the point of grouping by it.
                    res = int(getattr(getattr(file_item, 'dataset_config', None), 'resolution', 0) or 0)
                    self.loss_watch.observe(
                        epoch=window_idx,
                        item_key=file_item.path,
                        timestep=float(ts_detached[idx].item()) / num_train_timesteps,
                        loss=float(loss_detached[idx].item()),
                        resolution=res,
                    )
            except Exception as e:
                print_acc(f"[adaptive-lr] observe failed: {e}")

        # apply loss multiplier before prior loss
        # multiply by our mask
        try:
            loss = loss * loss_multiplier
        except:
            # todo handle mask with video models
            pass
        if prior_loss is not None:
            loss = loss + prior_loss

        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                # add snr_gamma
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                # add snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma,
                                        fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                # add min_snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        # Depth-anchor alternation: alternating samples on a depth step drop out
        # of the diffusion mean. With no split active (or depth off) this is the
        # original plain .mean().
        if _depth_gates is not None and _depth_gates['diffusion_zero'].any():
            loss = self._apply_diffusion_split_mask(loss, _depth_gates['diffusion_zero'])
        else:
            loss = loss.mean()
        
        # check for audio loss
        if batch.audio_pred is not None and batch.audio_target is not None:
            audio_loss = torch.nn.functional.mse_loss(batch.audio_pred.float(), batch.audio_target.float(), reduction="mean")
            audio_loss = audio_loss * self.train_config.audio_loss_multiplier
            self.additional_logs['loss/img'] = loss.item()
            self.additional_logs['loss/audio'] = audio_loss.item()
            loss = loss + audio_loss

        # check for additional losses
        if self.adapter is not None and hasattr(self.adapter, "additional_loss") and self.adapter.additional_loss is not None:

            loss = loss + self.adapter.additional_loss.mean()
            self.adapter.additional_loss = None

        if self.train_config.target_norm_std:
            # seperate out the batch and channels
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss


        loss = loss + additional_loss

        # Depth-anchor loss: added only on depth-objective samples (in timestep
        # band, positive weight, not reg, not an alternating sample on a
        # diffusion step). Inert when depth is off (_depth_gates is None).
        if _depth_gates is not None:
            loss = loss + self._compute_depth_anchor_loss(
                noise_pred, noisy_latents, timesteps, batch, _depth_gates,
            )

        # Normal-anchor loss: fires on its own timestep-window samples,
        # independent of the diffusion/depth loss_split. Inert when normal is off.
        if _normal_active:
            loss = loss + self._compute_normal_anchor_loss(
                noise_pred, noisy_latents, timesteps, batch,
            )

        # Body-proportion loss: fires on its own timestep-window samples.
        if _body_proportion_active:
            loss = loss + self._compute_body_proportion_anchor_loss(
                noise_pred, noisy_latents, timesteps, batch,
            )

        # Face-identity loss: fires on its own timestep-window samples with a
        # cached GT embedding and a detected face (SCRFD quality gate).
        if _face_identity_active:
            loss = loss + self._compute_face_identity_anchor_loss(
                noise_pred, noisy_latents, timesteps, batch,
            )

        # Body-shape loss: L1 on SMPL betas, cosine-gated.
        if _body_shape_active:
            loss = loss + self._compute_body_shape_anchor_loss(
                noise_pred, noisy_latents, timesteps, batch,
            )

        if hasattr(self.sd, "get_additional_loss"):
            additional_model_loss = self.sd.get_additional_loss(pred, target)
            if additional_model_loss is not None:
                loss = loss + additional_model_loss
                self.additional_logs["additional_model_loss"] = additional_model_loss.item()

        if self.train_config.max_loss_debug and self.train_config.max_loss is not None:
            if loss.item() > self.train_config.max_loss:
                print_acc(f"Loss {loss.item()} is greater than max loss {self.train_config.max_loss}. Clipping to max loss.")
                print_acc(f"timesteps: {timesteps}")

        if self.train_config.max_loss is not None:
            loss = torch.clamp(loss, max=self.train_config.max_loss)
        
        return loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            train_config=self.train_config,
            **kwargs
        )

        return loss
    
    
    # ------------------------------------------------------------------
    #  Mean-Flow loss (Geng et al., “Mean Flows for One-step Generative
    #  Modelling”, 2025 – see Alg. 1 + Eq. (6) of the paper)
    # This version avoids jvp / double-back-prop issues with Flash-Attention
    # adapted from the work of lodestonerock
    # ------------------------------------------------------------------
    def get_mean_flow_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        total_steps = float(self.sd.noise_scheduler.config.num_train_timesteps)  # e.g. 1000
        base_eps = 1e-3
        min_time_gap = 1e-2
        
        with torch.no_grad():
            num_train_timesteps = self.sd.noise_scheduler.config.num_train_timesteps
            batch_size = batch.latents.shape[0]
            timestep_t_list = []
            timestep_r_list = []

            for i in range(batch_size):
                t1 = random.randint(0, num_train_timesteps - 1)
                t2 = random.randint(0, num_train_timesteps - 1)
                t_t = self.sd.noise_scheduler.timesteps[min(t1, t2)]
                t_r = self.sd.noise_scheduler.timesteps[max(t1, t2)]
                if (t_t - t_r).item() < min_time_gap * 1000:
                    scaled_time_gap = min_time_gap * 1000
                    if t_t.item() + scaled_time_gap > 1000:
                        t_r = t_r - scaled_time_gap
                    else:
                        t_t = t_t + scaled_time_gap
                timestep_t_list.append(t_t)
                timestep_r_list.append(t_r)

            timesteps_t = torch.stack(timestep_t_list, dim=0).float()
            timesteps_r = torch.stack(timestep_r_list, dim=0).float()

            t_frac = timesteps_t / total_steps  # [0,1]
            r_frac = timesteps_r / total_steps  # [0,1]

            latents_clean = batch.latents.to(dtype)
            noise_sample = noise.to(dtype)

            lerp_vector = latents_clean * (1.0 - t_frac[:, None, None, None]) + noise_sample * t_frac[:, None, None, None]

            eps = base_eps

            # concatenate timesteps as input for u(z, r, t)
            timesteps_cat = torch.cat([t_frac, r_frac], dim=0) * total_steps

        # model predicts u(z, r, t)
        u_pred = self.predict_noise(
            noisy_latents=lerp_vector.to(dtype),
            timesteps=timesteps_cat.to(dtype),
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            batch=batch,
            **pred_kwargs
        )

        with torch.no_grad():
            t_frac_plus_eps = (t_frac + eps).clamp(0.0, 1.0)
            lerp_perturbed = latents_clean * (1.0 - t_frac_plus_eps[:, None, None, None]) + noise_sample * t_frac_plus_eps[:, None, None, None]
            timesteps_cat_perturbed = torch.cat([t_frac_plus_eps, r_frac], dim=0) * total_steps

            u_perturbed = self.predict_noise(
                noisy_latents=lerp_perturbed.to(dtype),
                timesteps=timesteps_cat_perturbed.to(dtype),
                conditional_embeds=conditional_embeds,
                unconditional_embeds=unconditional_embeds,
                batch=batch,
                **pred_kwargs
            )

        # compute du/dt via finite difference (detached)
        du_dt = (u_perturbed - u_pred).detach() / eps
        # du_dt = (u_perturbed - u_pred).detach()
        du_dt = du_dt.to(dtype)
        
        
        time_gap = (t_frac - r_frac)[:, None, None, None].to(dtype)
        time_gap.clamp(min=1e-4)
        u_shifted = u_pred + time_gap * du_dt
        # u_shifted = u_pred + du_dt / time_gap
        # u_shifted = u_pred

        # a step is done like this:
        # stepped_latent = model_input + (timestep_next - timestep) * model_output
        
        # flow target velocity
        # v_target = (noise_sample - latents_clean) / time_gap
        # flux predicts opposite of velocity, so we need to invert it
        v_target = (latents_clean - noise_sample) / time_gap

        # compute loss
        loss = torch.nn.functional.mse_loss(
            u_shifted.float(),
            v_target.float(),
            reduction='none'
        )

        with torch.no_grad():
            pure_loss = loss.mean().detach()
            pure_loss.requires_grad_(True)

        loss = loss.mean()
        if loss.item() > 1e3:
            pass
        self.accelerator.backward(loss)
        return pure_loss



    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or
                                         isinstance(self.adapter, ReferenceAdapter) or
                                         (isinstance(self.adapter, CustomAdapter))
        ):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active
            self.adapter.is_active = False

        if self.train_config.unload_text_encoder and self.adapter is not None and not isinstance(self.adapter, CustomAdapter):
            raise ValueError("Prior predictions currently do not support unloading text encoder with adapter")
        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if (self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter)) or self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''

                triggers = ['[trigger]', '[name]']
                remove_tokens = []

                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name

                if self.adapter is not None:
                    triggers.append(self.adapter_config.trigger)
                    for i in range(1, self.adapter_config.num_tokens):
                        remove_tokens.append(f"{self.adapter_config.trigger}_{i}")
                    if self.adapter_config.trigger_class_name is not None:
                        class_name = self.adapter_config.trigger_class_name

                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt

                if batch.prompt_embeds is not None:
                    embeds_to_use = batch.prompt_embeds.clone().to(self.device_torch, dtype=dtype)
                else:
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    embeds_to_use = self.sd.encode_prompt(
                        prompt_list,
                        long_prompts=self.do_long_prompts).to(
                        self.device_torch,
                        dtype=dtype,
                        **prompt_kwargs
                    ).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not self.sd.is_flux and not self.sd.is_lumina2:
                # we need to remove the image embeds from the prompt except for flux
                embeds_to_use: PromptEmbeds = embeds_to_use.clone().detach()
                end_pos = embeds_to_use.text_embeds.shape[1] - self.adapter_config.num_tokens
                embeds_to_use.text_embeds = embeds_to_use.text_embeds[:, :end_pos, :]
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.clone().detach()
                    unconditional_embeds.text_embeds = unconditional_embeds.text_embeds[:, :end_pos]

            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
            
            guidance_embedding_scale = self.train_config.cfg_scale
            if self.train_config.do_guidance_loss:
                guidance_embedding_scale = self._guidance_loss_target_batch

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                batch=batch,
                **pred_kwargs  # adapter residuals in here
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def _mem_diag(self, label: str) -> None:
        # Phase 2 investigation tooling (env-gated, inert in normal runs).
        # Captures live vs reserved CUDA memory at step boundaries to
        # distinguish cross-step tensor retention from activation pressure.
        import os
        if os.environ.get('KREA2_MEM_DIAG', '0') != '1':
            return
        if not torch.cuda.is_available():
            return
        gb = 1024 ** 3
        alloc = torch.cuda.memory_allocated() / gb
        reserv = torch.cuda.memory_reserved() / gb
        peak_a = torch.cuda.max_memory_allocated() / gb
        peak_r = torch.cuda.max_memory_reserved() / gb
        free, _ = torch.cuda.mem_get_info()
        print_acc(
            f"[memdiag] step={self.step_num} {label:>14} | "
            f"alloc={alloc:6.2f}G reserv={reserv:6.2f}G | "
            f"peak_alloc={peak_a:6.2f}G peak_reserv={peak_r:6.2f}G | "
            f"gpu_free={free / gb:6.2f}G"
        )
    def end_of_training_loop(self):
        pass

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: Union[int, torch.Tensor] = 1,
        conditional_embeds: Union[PromptEmbeds, None] = None,
        unconditional_embeds: Union[PromptEmbeds, None] = None,
        batch: Optional['DataLoaderBatchDTO'] = None,
        is_primary_pred: bool = False,
        **kwargs,
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        guidance_embedding_scale = self.train_config.cfg_scale
        if self.train_config.do_guidance_loss:
            guidance_embedding_scale = self._guidance_loss_target_batch
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=guidance_embedding_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            batch=batch,
            **kwargs
        )
    

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        self._mem_diag('step_start')
        with torch.no_grad():
            self.timer.start('preprocess_batch')
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_raw(batch)
            batch = self.preprocess_batch(batch)
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_processed(batch)
            dtype = get_torch_dtype(self.train_config.dtype)
            # sanity check
            if self.sd.vae.dtype != self.sd.vae_torch_dtype:
                self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
            if isinstance(self.sd.text_encoder, list):
                for encoder in self.sd.text_encoder:
                    if encoder.dtype != self.sd.te_torch_dtype:
                        encoder.to(self.sd.te_torch_dtype)
            else:
                if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                    self.sd.text_encoder.to(self.sd.te_torch_dtype)

            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
            if self.train_config.do_cfg or self.train_config.do_random_cfg:
                # pick random negative prompts
                if self.negative_prompt_pool is not None:
                    negative_prompts = []
                    for i in range(noisy_latents.shape[0]):
                        num_neg = random.randint(1, self.train_config.max_negative_prompts)
                        this_neg_prompts = [random.choice(self.negative_prompt_pool) for _ in range(num_neg)]
                        this_neg_prompt = ', '.join(this_neg_prompts)
                        negative_prompts.append(this_neg_prompt)
                    self.batch_negative_prompt = negative_prompts
                else:
                    self.batch_negative_prompt = ['' for _ in range(batch.latents.shape[0])]

            if self.adapter and isinstance(self.adapter, CustomAdapter):
                # condition the prompt
                # todo handle more than one adapter image
                conditioned_prompts = self.adapter.condition_prompt(conditioned_prompts)

            network_weight_list = batch.get_network_weight_list()
            if self.train_config.single_item_batching:
                network_weight_list = network_weight_list + network_weight_list

            has_adapter_img = batch.control_tensor is not None
            has_clip_image = batch.clip_image_tensor is not None
            has_clip_image_embeds = batch.clip_image_embeds is not None
            # force it to be true if doing regs as we handle those differently
            if any([batch.file_items[idx].is_reg for idx in range(len(batch.file_items))]):
                has_clip_image = True
                if self._clip_image_embeds_unconditional is not None:
                    has_clip_image_embeds = True  # we are caching embeds, handle that differently
                    has_clip_image = False

            # do prior pred if prior regularization batch
            do_reg_prior = False
            if any([batch.file_items[idx].prior_reg for idx in range(len(batch.file_items))]):
                do_reg_prior = True

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not has_clip_image and has_adapter_img:
                raise ValueError(
                    "IPAdapter control image is now 'clip_image_path' instead of 'control_path'. Please update your dataset config ")

            match_adapter_assist = False

            # check if we are matching the adapter assistant
            if self.assistant_adapter:
                if self.train_config.match_adapter_chance == 1.0:
                    match_adapter_assist = True
                elif self.train_config.match_adapter_chance > 0.0:
                    match_adapter_assist = torch.rand(
                        (1,), device=self.device_torch, dtype=dtype
                    ) < self.train_config.match_adapter_chance

            self.timer.stop('preprocess_batch')

            is_reg = False
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    # todo move this to data loader
                    if batch.control_tensor is not None:
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
                        # match in channels
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                # we need to match the channels
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    # todo move this to data loader
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()

            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            if batch.mask_tensor is not None and self.sd.do_masked_loss:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    if len(noisy_latents.shape) == 5:
                        # video B,C,T,H,W
                        h = noisy_latents.shape[3]
                        w = noisy_latents.shape[4]
                    else:
                        h = noisy_latents.shape[2]
                        w = noisy_latents.shape[3]
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(h, w)
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.4
                # adapter_strength_max = 0.7
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding is not None:
                grad_on_text_encoder = True

            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                grad_on_text_encoder = True

            if self.adapter_config and self.adapter_config.type == 'te_augmenter':
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list

        # activate network if it exits

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):

            # if self.train_config.negative_prompt is not None:
            #     # add negative prompt
            #     conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
            #                                                  range(len(conditioned_prompts))]
            #     if prompt_2 is not None:
            #         prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with (network):
                # encode clip adapter here so embeds are active for tokenizer
                if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                    with self.timer('encode_clip_vision_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True
                            )
                        else:
                            # just do a blank one
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                torch.zeros(
                                    (noisy_latents.shape[0], 3, 512, 512),
                                    device=self.device_torch, dtype=dtype
                                ),
                                is_training=True,
                                has_been_preprocessed=True,
                                drop=True
                            )
                        # it will be injected into the tokenizer when called
                        self.adapter(conditional_clip_embeds)

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or is_reg):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    self.adapter.trigger_pre_te(
                        tensors_preprocessed=clip_images if not is_reg else None,  # on regs we send none to get random noise
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count,
                        batch_tensor=batch.tensor if not is_reg else None,
                        batch_size=noisy_latents.shape[0]
                    )

                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
                        with torch.set_grad_enabled(False):
                            if batch.prompt_embeds is not None:
                                # use the cached embeds
                                conditional_embeds = batch.prompt_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                            else:
                                embeds_to_use = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                if self.cached_trigger_embeds is not None and not is_reg:
                                    embeds_to_use = self.cached_trigger_embeds.clone().detach().to(
                                        self.device_torch, dtype=dtype
                                    )
                                conditional_embeds = concat_prompt_embeds(
                                    [embeds_to_use] * noisy_latents.shape[0]
                                )
                            if self.train_config.do_cfg:
                                unconditional_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                unconditional_embeds = concat_prompt_embeds(
                                    [unconditional_embeds] * noisy_latents.shape[0]
                                )

                            if self.train_config.diff_output_preservation:
                                if batch.dop_prompt_embeds is not None:
                                    # cached to disk with the trigger word replaced per dataset
                                    self.diff_output_preservation_embeds = batch.dop_prompt_embeds.clone().detach().to(
                                        self.device_torch, dtype=dtype
                                    )
                                else:
                                    # no per item cache, fall back to the class only embeds
                                    self.diff_output_preservation_embeds = self.cached_dop_class_embeds

                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False

                    elif grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)

                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                # todo only do one and repeat it
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                    else:
                        with torch.set_grad_enabled(False):
                            # make sure it is in eval mode
                            if isinstance(self.sd.text_encoder, list):
                                for te in self.sd.text_encoder:
                                    te.eval()
                            else:
                                self.sd.text_encoder.eval()
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            if self.sd.encode_control_in_text_embeddings and batch.control_tensor_list is not None:
                                prompt_kwargs['control_images'] = batch.control_tensor_list
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)
                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                            
                            if self.train_config.diff_output_preservation:
                                # datasets can have their own trigger words, replace per item
                                def replace_trigger_with_class(prompt, file_item):
                                    trigger = file_item.trigger_word if file_item.trigger_word is not None else self.trigger_word
                                    if trigger is None:
                                        return prompt
                                    return prompt.replace(trigger, self.train_config.diff_output_preservation_class)
                                dop_prompts = [replace_trigger_with_class(p, fi) for p, fi in zip(conditioned_prompts, batch.file_items)]
                                dop_prompts_2 = None
                                if prompt_2 is not None:
                                    dop_prompts_2 = [replace_trigger_with_class(p, fi) for p, fi in zip(prompt_2, batch.file_items)]
                                self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                    dop_prompts, dop_prompts_2,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()
                    
                    if self.decorator:
                        conditional_embeds.text_embeds = self.decorator(
                            conditional_embeds.text_embeds
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds.text_embeds = self.decorator(
                                unconditional_embeds.text_embeds, 
                                is_unconditional=True
                            )

                # flush()
                pred_kwargs = {}

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, T2IAdapter)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, T2IAdapter)):
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                down_block_additional_residuals = adapter(adapter_images)
                                if self.assistant_adapter:
                                    # not training. detach
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype).detach() * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]
                                else:
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype) * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]

                                pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals

                if self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter_embeds'):
                        # number of images to do if doing a quad image
                        quad_count = random.randint(1, 4)
                        image_size = self.adapter.input_size
                        if has_clip_image_embeds:
                            # todo handle reg images better than this
                            if is_reg:
                                # get unconditional image embeds from cache
                                embeds = [
                                    load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                    range(noisy_latents.shape[0])
                                ]
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    embeds,
                                    quad_count=quad_count
                                )

                                if self.train_config.do_cfg:
                                    embeds = [
                                        load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                        range(noisy_latents.shape[0])
                                    ]
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        embeds,
                                        quad_count=quad_count
                                    )

                            else:
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    batch.clip_image_embeds,
                                    quad_count=quad_count
                                )
                                if self.train_config.do_cfg:
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        batch.clip_image_embeds_unconditional,
                                        quad_count=quad_count
                                    )
                        elif is_reg:
                            # we will zero it out in the img embedder
                            clip_images = torch.zeros(
                                (noisy_latents.shape[0], 3, image_size, image_size),
                                device=self.device_torch, dtype=dtype
                            ).detach()
                            # drop will zero it out
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images,
                                drop=True,
                                is_training=True,
                                has_been_preprocessed=False,
                                quad_count=quad_count
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    torch.zeros(
                                        (noisy_latents.shape[0], 3, image_size, image_size),
                                        device=self.device_torch, dtype=dtype
                                    ).detach(),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=False,
                                    quad_count=quad_count
                                )
                        elif has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True,
                                quad_count=quad_count,
                                # do cfg on clip embeds to normalize the embeddings for when doing cfg
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    clip_images.detach().to(self.device_torch, dtype=dtype),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=True,
                                    quad_count=quad_count
                                )
                        else:
                            print_acc("No Clip Image")
                            print_acc([file_item.path for file_item in batch.file_items])
                            raise ValueError("Could not find clip image")

                    if not self.adapter_config.train_image_encoder:
                        # we are not training the image encoder, so we need to detach the embeds
                        conditional_clip_embeds = conditional_clip_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_clip_embeds = unconditional_clip_embeds.detach()

                    with self.timer('encode_adapter'):
                        self.adapter.train()
                        conditional_embeds = self.adapter(
                            conditional_embeds.detach(),
                            conditional_clip_embeds,
                            is_unconditional=False
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds = self.adapter(
                                unconditional_embeds.detach(),
                                unconditional_clip_embeds,
                                is_unconditional=True
                            )
                        else:
                            # wipe out unconsitional
                            self.adapter.last_unconditional = None

                if self.adapter and isinstance(self.adapter, ReferenceAdapter):
                    # pass in our scheduler
                    self.adapter.noise_scheduler = self.lr_scheduler
                    if has_clip_image or has_adapter_img:
                        img_to_use = clip_images if has_clip_image else adapter_images
                        # currently 0-1 needs to be -1 to 1
                        reference_images = ((img_to_use - 0.5) * 2).detach().to(self.device_torch, dtype=dtype)
                        self.adapter.set_reference_images(reference_images)
                        self.adapter.noise_scheduler = self.sd.noise_scheduler
                    elif is_reg:
                        self.adapter.set_blank_reference_images(noisy_latents.shape[0])
                    else:
                        self.adapter.set_reference_images(None)

                if self.train_config.do_guidance_loss and isinstance(self.train_config.guidance_loss_target, list):
                    batch_size = noisy_latents.shape[0]
                    # update the guidance value, random float between guidance_loss_target[0] and guidance_loss_target[1]
                    # sample before the prior prediction so the prior, main, uncond, and
                    # preservation passes all run at the same guidance values
                    self._guidance_loss_target_batch = [
                        random.uniform(
                            self.train_config.guidance_loss_target[0],
                            self.train_config.guidance_loss_target[1]
                        ) for _ in range(batch_size)
                    ]

                prior_pred = None

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                do_correct_pred_norm_prior = self.train_config.correct_pred_norm

                do_guidance_prior = False

                if batch.unconditional_latents is not None:
                    # for this not that, we need a prior pred to normalize
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True

                if ((
                        has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_guidance_prior or do_reg_prior or do_inverted_masked_prior or self.train_config.correct_pred_norm):
                    with self.timer('prior predict'):
                        prior_embeds_to_use = conditional_embeds
                        # use diff_output_preservation embeds if doing dfe
                        if self.train_config.diff_output_preservation:
                            prior_embeds_to_use = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        
                        if self.train_config.blank_prompt_preservation:
                            blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                self.device_torch, dtype=dtype
                            )
                            prior_embeds_to_use = concat_prompt_embeds(
                                [blank_embeds] * noisy_latents.shape[0]
                            )
                        
                        # joint audio models stash their audio pred on the batch.
                        # Give this pass its own slot so the preservation loss
                        # can pair it with the preservation pass below.
                        batch.audio_pred_slot = 'audio_pred_prior'
                        prior_pred = self.get_prior_prediction(
                            noisy_latents=noisy_latents,
                            conditional_embeds=prior_embeds_to_use,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            noise=noise,
                            batch=batch,
                            unconditional_embeds=unconditional_embeds,
                            conditioned_prompts=conditioned_prompts
                        )
                        batch.audio_pred_slot = None
                        if prior_pred is not None:
                            prior_pred = prior_pred.detach()

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or self.adapter_config.type in ['llm_adapter', 'text_encoder']):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    conditional_embeds = self.adapter.condition_encoded_embeds(
                        tensors_0_1=clip_images,
                        prompt_embeds=conditional_embeds,
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count
                    )
                    if self.train_config.do_cfg and unconditional_embeds is not None:
                        unconditional_embeds = self.adapter.condition_encoded_embeds(
                            tensors_0_1=clip_images,
                            prompt_embeds=unconditional_embeds,
                            is_training=True,
                            has_been_preprocessed=True,
                            is_unconditional=True,
                            quad_count=quad_count
                        )

                if self.adapter and isinstance(self.adapter, CustomAdapter) and batch.extra_values is not None:
                    self.adapter.add_extra_values(batch.extra_values.detach())

                    if self.train_config.do_cfg:
                        self.adapter.add_extra_values(torch.zeros_like(batch.extra_values.detach()),
                                                      is_unconditional=True)

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, ControlNetModel)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, ControlNetModel)):
                        if self.train_config.do_cfg:
                            raise ValueError("ControlNetModel is not supported with CFG")
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter: ControlNetModel = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                # add_text_embeds is pooled_prompt_embeds for sdxl
                                added_cond_kwargs = {}
                                if self.sd.is_xl:
                                    added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                                    added_cond_kwargs['time_ids'] = self.sd.get_time_ids_from_latents(noisy_latents)
                                down_block_res_samples, mid_block_res_sample = adapter(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=conditional_embeds.text_embeds,
                                    controlnet_cond=adapter_images,
                                    conditioning_scale=1.0,
                                    guess_mode=False,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )
                                pred_kwargs['down_block_additional_residuals'] = down_block_res_samples
                                pred_kwargs['mid_block_additional_residual'] = mid_block_res_sample
                
                self.before_unet_predict()
                
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
                with self.timer('condition_noisy_latents'):
                    # do it for the model
                    noisy_latents = self.sd.condition_noisy_latents(noisy_latents, batch)
                    if self.adapter and isinstance(self.adapter, CustomAdapter):
                        noisy_latents = self.adapter.condition_noisy_latents(noisy_latents, batch)
                
                if self.train_config.timestep_type == 'next_sample':
                    with self.timer('next_sample_step'):
                        with torch.no_grad():
                            
                            stepped_timestep_indicies = [self.sd.noise_scheduler.index_for_timestep(t) + 1 for t in timesteps]
                            stepped_timesteps = [self.sd.noise_scheduler.timesteps[x] for x in stepped_timestep_indicies]
                            stepped_timesteps = torch.stack(stepped_timesteps, dim=0)
                            
                            # do a sample at the current timestep and step it, then determine new noise
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pred_kwargs
                            )
                            stepped_latents = self.sd.step_scheduler(
                                next_sample_pred,
                                noisy_latents,
                                timesteps,
                                self.sd.noise_scheduler
                            )
                            # stepped latents is our new noisy latents. Now we need to determine noise in the current sample
                            noisy_latents = stepped_latents
                            original_samples = batch.latents.to(self.device_torch, dtype=dtype)
                            # todo calc next timestep, for now this may work as it
                            t_01 = (stepped_timesteps / 1000).to(original_samples.device)
                            if len(stepped_latents.shape) == 4:
                                t_01 = t_01.view(-1, 1, 1, 1)
                            elif len(stepped_latents.shape) == 5:
                                t_01 = t_01.view(-1, 1, 1, 1, 1)
                            else:
                                raise ValueError("Unknown stepped latents shape", stepped_latents.shape)
                            next_sample_noise = (stepped_latents - (1.0 - t_01) * original_samples) / t_01
                            noise = next_sample_noise
                            timesteps = stepped_timesteps
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    # do guided loss
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        mask_multiplier=mask_multiplier,
                        prior_pred=prior_pred,
                    )
                    
                elif self.train_config.loss_type == 'mean_flow':
                    loss = self.get_mean_flow_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        prior_pred=prior_pred,
                    )
                else:
                    with self.timer('predict_unet'):
                        noise_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            is_primary_pred=True,
                            **pred_kwargs
                        )
                    self.after_unet_predict()
                    self._mem_diag('after_forward')

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        prior_to_calculate_loss = prior_pred
                        # if we are doing diff_output_preservation and not noing inverted masked prior
                        # then we need to send none here so it will not target the prior
                        doing_preservation = self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation
                        if doing_preservation and not do_inverted_masked_prior:
                            prior_to_calculate_loss = None
                        
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_to_calculate_loss,
                        )
                    self._mem_diag('after_loss')

                    if self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation:
                        with torch.no_grad():
                            if self.train_config.diff_output_preservation:
                                preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                            elif self.train_config.blank_prompt_preservation:
                                blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                preservation_embeds = concat_prompt_embeds(
                                    [blank_embeds] * noisy_latents.shape[0]
                                )
                        batch.audio_pred_slot = 'audio_pred_preservation'
                        preservation_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            **pred_kwargs
                        )
                        batch.audio_pred_slot = None
                        multiplier = self.train_config.diff_output_preservation_multiplier if self.train_config.diff_output_preservation else self.train_config.blank_prompt_preservation_multiplier
                        preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier
                        self.additional_logs['loss/normal'] = loss.item()
                        self.additional_logs['loss/preservation'] = preservation_loss.item()

                        # preserve the audio stream of joint audio models too.
                        # Both passes ran on the same noisy audio, so this holds
                        # the audio branch to its base model output.
                        if batch.audio_pred_preservation is not None and batch.audio_pred_prior is not None:
                            audio_preservation_loss = torch.nn.functional.mse_loss(
                                batch.audio_pred_preservation.float(),
                                batch.audio_pred_prior.float(),
                            ) * multiplier * self.train_config.audio_loss_multiplier
                            self.additional_logs['loss/preservation_audio'] = audio_preservation_loss.item()
                            preservation_loss = preservation_loss + audio_preservation_loss

                        loss = loss + preservation_loss

                # check if nan
                if not torch.isfinite(loss):
                    print_acc("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    loss = loss * loss_multiplier.mean()
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    # if self.is_bfloat:
                    # loss.backward()
                    # else:
                    self.accelerator.backward(loss)
                    self._mem_diag('after_backward')

        return loss.detach()
        # flush()

    def _iter_lora_params_with_grad(self):
        """Yield tagged LoRA parameters whose gradients are populated."""
        groups = self.params
        if not groups:
            return
        if isinstance(groups[0], dict):
            iterable = (p for group in groups for p in group.get('params', []))
        else:
            iterable = iter(groups)
        for parameter in iterable:
            if not getattr(parameter, '_is_lora', False):
                continue
            if parameter.grad is None:
                continue
            yield parameter

    def _inject_gradient_noise(self) -> None:
        """Inject configured Gaussian noise into tagged LoRA gradients."""
        config = self.train_config.gradient_noise
        if not config.enabled:
            return

        step = max(0, int(getattr(self, 'step_num', 0)))
        should_log = config.log_every > 0 and step % config.log_every == 0
        gradient_sq = 0.0
        noise_sq = 0.0

        for parameter in self._iter_lora_params_with_grad():
            gradient = parameter.grad
            if config.mode == 'absolute':
                sigma = float(config.sigma)
            elif config.mode == 'relative':
                rms = float(gradient.detach().pow(2).mean().sqrt())
                sigma = float(config.sigma) * rms
            else:
                sigma = float(config.eta) / max(1.0, (1.0 + step) ** float(config.gamma))

            if sigma <= 0:
                continue
            noise = torch.randn_like(gradient) * sigma
            if should_log:
                gradient_sq += float(gradient.detach().pow(2).sum())
                noise_sq += float(noise.pow(2).sum())
            gradient.add_(noise)

        if should_log:
            gradient_norm = gradient_sq ** 0.5
            noise_norm = noise_sq ** 0.5
            self._last_grad_noise_norm = noise_norm
            self._last_grad_noise_snr = gradient_norm / noise_norm if noise_norm > 1e-12 else 0.0

    def _record_fisher_trace(self) -> None:
        """Record the diagonal Fisher trace when optimizer state provides it."""
        if self.optimizer is None:
            return
        total = 0.0
        found_state = False
        for group in self.optimizer.param_groups:
            for parameter in group.get('params', []):
                if not getattr(parameter, '_is_lora', False):
                    continue
                state = self.optimizer.state.get(parameter)
                if state is None or state.get('exp_avg_sq') is None:
                    continue
                total += float(state['exp_avg_sq'].sum())
                found_state = True
        if found_state:
            self._last_fisher_trace = total

    def _inject_weight_noise(self) -> None:
        """Inject configured Gaussian noise into tagged LoRA parameter values."""
        config = self.train_config.weight_noise
        if not config.enabled:
            return

        groups = self.params
        if not groups:
            return
        if isinstance(groups[0], dict):
            iterable = (p for group in groups for p in group.get('params', []))
        else:
            iterable = iter(groups)

        step = max(0, int(getattr(self, 'step_num', 0)))
        should_log = config.log_every > 0 and step % config.log_every == 0
        noise_sq = 0.0
        weight_sq = 0.0
        for parameter in iterable:
            if not getattr(parameter, '_is_lora', False):
                continue
            weight = parameter.data
            if should_log:
                weight_sq += float(weight.detach().pow(2).sum())
            if config.mode == 'absolute':
                sigma = float(config.sigma)
            else:
                rms = float(weight.detach().pow(2).mean().sqrt())
                sigma = float(config.sigma) * rms
            if sigma <= 0:
                continue
            pre_noise_norm = float(weight.norm()) if config.bound_norm else 0.0
            noise = torch.randn_like(weight) * sigma
            if should_log:
                noise_sq += float(noise.pow(2).sum())
            weight.add_(noise)
            if config.bound_norm and pre_noise_norm > 0.0:
                post_noise_norm = float(weight.norm())
                if post_noise_norm > 0.0:
                    weight.mul_(pre_noise_norm / post_noise_norm)

        if should_log:
            self._last_weight_noise_norm = noise_sq ** 0.5
            self._last_weight_norm = weight_sq ** 0.5

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        self.optimizer.zero_grad()
        for batch in batch_list:
            if self.sd.is_multistage:
                # handle multistage switching
                if self.steps_this_boundary >= self.train_config.switch_boundary_every or self.current_boundary_index not in self.sd.trainable_multistage_boundaries:
                    # iterate to make sure we only train trainable_multistage_boundaries
                    while True:
                        self.steps_this_boundary = 0
                        self.current_boundary_index += 1
                        if self.current_boundary_index >= len(self.sd.multistage_boundaries):
                            self.current_boundary_index = 0
                        if self.current_boundary_index in self.sd.trainable_multistage_boundaries:
                            # if this boundary is trainable, we can stop looking
                            break
            loss = self.train_single_accumulation(batch)
            self.steps_this_boundary += 1
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        if not self.is_grad_accumulation_step:
            # grads of memory-managed (offloaded) params are async D2H copies into
            # pinned tensors; join them before anything on the CPU reads .grad
            sync_grad_transfers()
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            self._inject_gradient_noise()
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.adapter and isinstance(self.adapter, CustomAdapter):
                    self.adapter.post_weight_update()
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
            self._record_fisher_trace()
            self._inject_weight_noise()
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': (total_loss / len(batch_list)).item()}
        )

        for metric_name in (
            '_last_grad_noise_norm',
            '_last_grad_noise_snr',
            '_last_weight_noise_norm',
            '_last_weight_norm',
            '_last_fisher_trace',
            '_last_normal_loss',
            '_last_normal_loss_applied',
            '_last_normal_cos',
            '_last_body_proportion_loss',
            '_last_body_proportion_loss_applied',
            '_last_identity_loss',
            '_last_identity_loss_applied',
            '_last_id_sim',
            '_last_body_shape_loss',
            '_last_body_shape_loss_applied',
            '_last_body_shape_cos',
        ):
            metric_value = getattr(self, metric_name, None)
            if metric_value is not None:
                loss_dict[metric_name.removeprefix('_last_')] = metric_value
                setattr(self, metric_name, None)

        self.end_of_training_loop()

        return loss_dict
