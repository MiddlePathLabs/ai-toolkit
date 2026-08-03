# Krea 2 Phase 0 Decode-Gradient Probe

Date: 2026-08-03

## Result

The target embedded runtime loaded the cached Krea 2 Raw transformer, Qwen3-VL
text encoder, and Qwen-Image VAE, then decoded a 512 x 512 latent tensor with
`low_vram: false` and BF16 VAE precision. The decode output retained a graph
back to `noise_pred` and produced a finite, non-zero gradient.

| Measurement | Result |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU memory | 95.59 GiB |
| Model | Krea 2 Raw |
| Architecture | `krea2` |
| Latent shape | `(1, 16, 64, 64)` |
| Decoded shape | `(1, 3, 512, 512)` |
| Dtype | BF16 |
| `low_vram` | `false` |
| Decoded output `requires_grad` | `True` |
| Decoded grad function | `SqueezeBackward1` |
| `noise_pred` gradient norm | `0.004364013671875` |
| Gradient finite | `True` |
| Allocated peak | `38.698626048 GB` |
| Reserved peak | `39.323697152 GB` |

This is a direct model-level decode-contract probe using the same
`noisy_latents - t * noise_pred` relationship as the planned trainer check. It
is not an end-to-end training-step smoke test; a full training run remains a
separate Phase 2 acceptance check.

