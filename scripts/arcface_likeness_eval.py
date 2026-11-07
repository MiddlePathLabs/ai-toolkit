"""ArcFace (buffalo_l) likeness eval — generated images vs a reference dataset.

Computes the reference band (ref x ref pairwise cosine) from all detectable
faces in --refs-dir, then scores every image in --images: each detected face
gets mean/max cosine vs the reference set. Multi-person images report every
face (presumed subject = highest mean-over-refs face), so other-woman
separation and leak checks fall out of the same table.

Usage (repo root as CWD, python_embeded):
  python Scripts/arcface_likeness_eval.py \
      --refs-dir datasets/krea2_subject_refs --images eval_out/render_2000
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

parser = argparse.ArgumentParser()
parser.add_argument("--refs-dir", required=True)
parser.add_argument("--images", required=True, help="image file, or directory (jpg/png)")
parser.add_argument("--det-size", type=int, default=640)
parser.add_argument("--curate-loo", type=float, default=0.35,
                    help="drop refs whose leave-one-out mean sim is below this "
                         "(kills wrong-face picks from multi-person photos); 0 = off")
parser.add_argument("--out-csv", default=None)
args = parser.parse_args()

providers = ["CPUExecutionProvider"]  # ORT CUDA EP needs cudnn64_9.dll not present in python_embeded
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def faces_of(path):
    return app.get(load(path))


def emb_matrix(face_list):
    return np.stack([f.normed_embedding for f in face_list])


# ---- reference pass: most-central face per ref image ----
ref_files = sorted(
    f for f in glob.glob(os.path.join(args.refs_dir, "*"))
    if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")
)
ref_embs, ref_names, ref_no_face = [], [], []
for path in ref_files:
    fs = faces_of(path)
    if not fs:
        ref_no_face.append(os.path.basename(path))
        continue
    if len(fs) == 1:
        ref_embs.append(fs[0].normed_embedding)
    else:
        m = emb_matrix(fs)
        central = int(np.argmax((m @ m.T).sum(axis=1)))
        ref_embs.append(fs[central].normed_embedding)
    ref_names.append(os.path.basename(path))
R = np.stack(ref_embs)
n = len(R)
if n < 2:
    raise SystemExit("need >=2 reference faces")

if args.curate_loo > 0:
    S = R @ R.T
    loo = (S.sum(axis=1) - 1.0) / (n - 1)
    keep = loo >= args.curate_loo
    if 2 <= keep.sum() < n:
        dropped = [ref_names[i] for i, k in enumerate(keep) if not k]
        print(f"[curate] dropping {n - keep.sum()} outlier refs "
              f"(LOO < {args.curate_loo}): {', '.join(dropped)}")
        R, n = R[keep], int(keep.sum())
    else:
        print("[curate] nothing to drop")

sims = (R @ R.T)[np.triu_indices(n, k=1)]
band_mean, band_std = float(sims.mean()), float(sims.std())
print(f"[band] refs={n} faces from {len(ref_files)} files "
      f"({len(ref_no_face)} no-face)")
print(f"[band] ref x ref cosine: mean={band_mean:.3f} std={band_std:.3f} "
      f"min={sims.min():.3f} max={sims.max():.3f}")

# ---- target pass ----
if os.path.isdir(args.images):
    targets = sorted(
        f for f in glob.glob(os.path.join(args.images, "*"))
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
else:
    targets = [args.images]

rows, subject_scores, no_face = [], [], []
for path in targets:
    name = os.path.basename(path)
    fs = faces_of(path)
    if not fs:
        no_face.append(name)
        rows.append([name, 0, "", "", "", "NO-FACE"])
        continue
    per_face = []
    for f in fs:
        s = R @ f.normed_embedding
        per_face.append((float(s.mean()), float(s.max()),
                         tuple(f.bbox.round().astype(int).tolist())))
    per_face.sort(reverse=True)
    subj = per_face[0]
    subject_scores.append(subj[0])
    for i, (mean_s, max_s, bbox) in enumerate(per_face):
        tag = "subject?" if i == 0 else ("other" if len(per_face) > 1 else "solo")
        rows.append([name, i, str(bbox), f"{mean_s:.3f}", f"{max_s:.3f}", tag])

subject_scores_np = np.array(subject_scores)
print("\n[image]                       face  mean    max     role")
for r in rows:
    if r[-1] == "NO-FACE":
        print(f"{r[0]:<28} -- NO-FACE")
    else:
        print(f"{r[0]:<28} {r[1]:>4}  {r[3]}  {r[4]}  {r[5]}")

if len(subject_scores):
    print(f"\n[verdict] subject-face mean-of-means={subject_scores_np.mean():.3f} "
          f"best-frame={subject_scores_np.max():.3f} "
          f"({len(subject_scores)} scored, {len(no_face)} no-face)")
    print(f"[verdict] band-relative: {subject_scores_np.mean()/band_mean*100:.0f}% of band "
          f"(band mean {band_mean:.3f})")

if args.out_csv:
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "face_idx", "bbox", "mean_sim", "max_sim", "role"])
        w.writerows(rows)
    print(f"[csv] {args.out_csv}")
