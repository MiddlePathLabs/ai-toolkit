"""Face embedding backends for likeness eval — ArcFace (insightface) and AdaFace ViT.

AdaFace is CVLface's ViT-Base (github.com/mk-minchul/CVLface, MIT), trained on WebFace4M
(non-commercial research use, same class as InsightFace's own models). The architecture is
rebuilt here for inference so only the weights are downloaded:
minchul/cvlface_adaface_vit_base_webface4m (revision and sha256 pinned below). Input is a
112x112 face aligned to the 5-point ArcFace template (insightface.utils.face_align.norm_crop),
RGB, (x - 0.5) / 0.5; the embedding averages the image with its mirror, L2-normalized.

Chosen after Fizgig replaced ArcFace with it for likeness scoring on 12 Nov 2026 (their
best-epoch picks matched their eye picks on every run); swap in via --embedder adaface on
scripts/arcface_likeness_eval.py.
"""

import hashlib
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn

ADAFACE_WEIGHTS = {
    "url": (
        "https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/"
        "resolve/c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e/model.safetensors"
    ),
    "sha256": "5fafd6b7d599a3ede5fac5bd1d01ad05e9e93e89b39b7687d4a3bc93ff2aebc0",
    "filename": "adaface_vit_base_webface4m.safetensors",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.ReLU6(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.mlp = _Attn(dim, heads), _Mlp(dim, int(dim * ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaFaceViT(nn.Module):
    def __init__(self, out_dim=512, dim=512, depth=24, heads=16, ratio=3, patch=8, size=112):
        super().__init__()
        n = (size // patch) ** 2
        self.n, self.dim = n, dim
        self.patch_embed = _PatchEmbed(dim, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads, ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feature = nn.Sequential(nn.Linear(dim * n, dim, bias=False), nn.BatchNorm1d(dim, eps=2e-5),
                                     nn.Linear(dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, eps=2e-5))

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        return self.feature(self.norm(x).reshape(x.shape[0], self.n * self.dim))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_adaface_weights(path=None):
    """The pinned safetensors on disk, downloading once if missing. Returns the path.

    The sha256 pin is an integrity check, so any path supplied here must hold exactly
    the pinned checkpoint — this is not a loader for alternative AdaFace weights.
    """
    path = path or os.path.join(_REPO_ROOT, "models", "face", ADAFACE_WEIGHTS["filename"])
    if os.path.isdir(path):
        raise RuntimeError(f"--adaface-weights wants a file path, but {path} is a directory")
    if os.path.isfile(path) and _sha256(path) == ADAFACE_WEIGHTS["sha256"]:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        print(f"[adaface] downloading weights to {path} (~0.46 GB, once)")
        req = urllib.request.Request(ADAFACE_WEIGHTS["url"], headers={"User-Agent": "ai-toolkit-eval"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        digest = _sha256(tmp)
        if digest != ADAFACE_WEIGHTS["sha256"]:
            raise RuntimeError(
                f"AdaFace weights sha256 mismatch: got {digest}, expected {ADAFACE_WEIGHTS['sha256']}"
            )
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def load_into(net, sd):
    """Load CVLface 'model.net.'-prefixed tensors into net; missing/unexpected keys (beyond the
    BatchNorm num_batches_tracked the checkpoint omits) are an error."""
    sd = {k[len("model.net."):]: v for k, v in sd.items() if k.startswith("model.net.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"AdaFace weights don't fit the network (missing {missing[:4]}, unexpected {unexpected[:4]})"
        )
    return net.eval()


def load_adaface(path=None):
    """The network with the weights file's tensors (CVLface's 'model.net.' keys), eval mode, CPU float32."""
    from safetensors.torch import load_file

    path = ensure_adaface_weights(path)
    return load_into(AdaFaceViT(), load_file(path))


def embed_aligned(net, aligned_rgb):
    """Unit embedding of a 112x112x3 uint8 RGB aligned face: the face and its mirror, averaged."""
    x = torch.from_numpy(np.ascontiguousarray(aligned_rgb)).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)[None]
    with torch.no_grad():
        e = net(torch.cat([x, x.flip(3)])).sum(0)
    return (e / e.norm()).numpy()


def face_embeddings(net, img_rgb, faces):
    """Embeddings for every detected face: insightface 5-point norm_crop to 112x112, then AdaFace."""
    from insightface.utils import face_align

    return [embed_aligned(net, face_align.norm_crop(img_rgb, f.kps)) for f in faces]
