"""Render saved MOSS LoRA checkpoints at proper sampling settings.

MOSS-Image-Base + a trained LoRA checkpoint (frozen, strength configurable),
normal sampler — NO preview turbo lora — so twins/likeness can be judged
without the 8-step CFG-1 distill artifact clouding the read.

Mirrors the trainer's own preview path (same model class, same TE/VAE, same
uncond_skip_layers) with the turbo lora simply omitted, and attaches the
checkpoint through the same load_frozen_lora primitive the preview-turbo
machinery uses.

Usage (repo root as CWD, python_embeded):
  python Scripts/moss_ckpt_render.py \
      --ckpt output/moss_subject_isol/moss_subject_isol_000002000.safetensors \
      --out eval_out/nodop_2000_proper --guidance 3.5 --steps 25

  --prompts-file prompts.txt   one prompt per line, overrides the built-in set
  --quantize                   convrot8 the DiT (VRAM fallback; default bf16)
"""

import argparse
import os
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="trained LoRA .safetensors")
parser.add_argument("--out", required=True, help="output directory for renders")
parser.add_argument("--guidance", type=float, default=3.5)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--strength", type=float, default=1.0, help="LoRA strength")
parser.add_argument("--width", type=int, default=1536)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--prompts-file", default=None)
args = parser.parse_args()

# the twins-gate + likeness probe set from the ISOL runs (trigger literal)
DEFAULT_PROMPTS = [
    "Close-up portrait of the subject facing the camera, soft diffused window light from the left, calm expression, shallow depth of field, warm gray blurred background.",
    "Medium shot of the subject standing in a bright modern kitchen, wearing a charcoal knit sweater and holding a mug of coffee, turning toward the camera with a slight smile, natural daylight.",
    "The subject standing beside another woman, both facing the camera, soft diffused window light from the left, calm expressions, shallow depth of field, warm gray blurred background.",
    "The subject, standing in a bright modern kitchen beside another woman, both turned slightly toward the camera with slight smiles, natural daylight.",
    "The subject, standing next to another woman in a studio, both facing the camera, soft neutral lighting, simple gray backdrop.",
    "Two different women standing together in a studio, both facing the camera, distinct facial identities, soft neutral lighting, simple gray backdrop.",
    "Close-up portrait of a woman facing the camera, soft diffused window light from the left, calm expression, shallow depth of field, warm gray blurred background.",
]
NEG = "worst quality, bad anatomy, watermark, jpeg artifact"

if args.prompts_file:
    with open(args.prompts_file, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
else:
    prompts = DEFAULT_PROMPTS

os.makedirs(args.out, exist_ok=True)

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.preview_lora import load_frozen_lora
from toolkit.util.get_model import get_model_class

model_config = ModelConfig(
    name_or_path="OpenMOSS-Team/MOSS-Image-Base",
    arch="moss_image",
    quantize=args.quantize,
    qtype="convrot8",
    quantize_te=False,
    qtype_te="qfloat8",
    te_name_or_path="Comfy-Org/Qwen3.5/text_encoders/qwen3.5_9b_bf16.safetensors",
    vae_path="Comfy-Org/MOSS-Image/vae/flux2-vae_small.safetensors",
    model_kwargs={"uncond_skip_layers": [22]},
    low_vram=False,
)

print(f"[render] loading MOSS-Image-Base (quantize={args.quantize})")
ModelClass = get_model_class(model_config)
sd = ModelClass(torch.device("cuda"), model_config, "bf16")
sd.load_model()

print(f"[render] attaching checkpoint LoRA: {args.ckpt} (strength {args.strength})")
adapter = load_frozen_lora(
    sd.model,
    args.ckpt,
    args.strength,
    target_lin_modules=sd.target_lora_modules,
    label="checkpoint lora",
    tag="ckpt-lora",
)
print("[render] " + adapter.report.summary())
if adapter.report.matched == 0:
    print("[render] ERROR: no LoRA weights matched the DiT — aborting")
    sys.exit(1)
adapter.activate(sd.model, sd.device_torch, sd.torch_dtype)

gens = []
for i, prompt in enumerate(prompts):
    gen = GenerateImageConfig(
        prompt=prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        negative_prompt=NEG,
        seed=args.seed + i,
        output_folder=args.out,
        output_ext="jpg",
    )
    gens.append(gen)

print(f"[render] {len(gens)} prompts @ {args.width}x{args.height}, "
      f"cfg {args.guidance}, {args.steps} steps")
try:
    sd.generate_images(gens)
finally:
    adapter.deactivate()

print(f"[render] done -> {args.out}")
