# Identity-eval model files (user-supplied)

The Samples-screen identity evaluation (fork-only; see `docs/identity_eval.md`)
loads recognition models from this directory. **Nothing here is bundled or
downloaded automatically** — place the files yourself:

| File | Purpose | License | Source |
|---|---|---|---|
| `face_recognition_sface_2021dec.onnx` | SFace recognition (128-d) | Apache-2.0 | https://github.com/opencv/opencv_zoo (face_recognition_sface) |
| `face_detection_yunet_2023mar.onnx` | YuNet detector — only needed for SFace-only mode when `~/.insightface/models/buffalo_l` is absent | MIT | https://github.com/opencv/opencv_zoo (face_detection_yunet) |

ArcFace (`w600k_r50` + SCRFD `det_10g`) is NOT loaded from here — it comes
from the insightface home (`~/.insightface/models/buffalo_l`, non-commercial
research terms) and is never committed, bundled, or auto-downloaded.

If a file is missing, the feature degrades gracefully: with SFace absent it
runs ArcFace-only; with buffalo_l absent it runs SFace-only (needs YuNet).
The UI shows which models are active. The directory can be relocated via the
`IDENTITY_MODELS_DIR` settings key.
