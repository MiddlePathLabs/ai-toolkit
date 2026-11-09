"""Training maps: where on a few fixed dataset pictures the model is still wrong,
and where the LoRA changes its prediction (port of Fizgig v9.3.0's training maps,
adapted to the toolkit trainer; see Fizgig docs/TRAINING_MAPS.md).

Probes are a few dataset pictures (spread evenly through the sorted paths, so the
same dataset picks the same ones), each at three fixed noise levels with fixed
seeded noise. Captured once before training and at every preview round
(BaseSDTrainProcess._sample cadence), only the weights change between captures:

  * "loss"   - per-position squared error of the model's prediction against the
               training target (sd.get_loss_target), channel-meaned. The viewer
               calls this "not learned yet".
  * "effect" - |prediction with the LoRA at 1 - prediction with it at 0| at the
               same noised input ("where the LoRA acts"), via the same
               network-multiplier toggle the DOP prior pass uses.
  * "learned"- loss(step 0) - loss(now) at the same noise, so the picture's own
               texture cancels; green where learning happened, magenta where it
               got worse. Rendered, not stored (recomputed from the two captures).
  * "attn"   - (krea2 only, when the run has a trigger word in the probe
               captions) the trigger's share of each image token's attention,
               mean over heads and blocks (toolkit/attn_probe.py), captured on
               the LoRA-on forward, run eagerly. Watch it settle on the subject;
               attention parked on a logo, sign or the background means the
               trigger is being tied to that instead.

Gated to still-image flow-matching models (krea2 / qwen_image_2 / moss_image,
anything with sd.is_flow_matching and straight-path add_noise); other families
are skipped with one console line. Forwards run no-grad, eval mode, with the
preview weights (the EMA shadow weights when EMA is on), inside _sample's
force-eager + fork_rng window when recurring. A failure is never fatal: one
console line and the maps are off for the rest of the run.

Output, in <save_root>/training_maps/: probes.json ({keys, timesteps, crops, attn
-- the keys carrying a trigger-word map}) and maps.jsonl, one line per capture
{"step", "maps": {key: {t: {"loss": grid, "effect": grid, "attn": grid}}}}
(grids pooled to <= MAX_SIDE, 4 significant figures; the same
step captured twice keeps the later render). Rendered composites (three labelled
panels over the exact bucket-cropped photo) land in samples/ as
MAP_<time>_<step>_p<N>.jpg so the job's sample gallery shows them next to the
previews of the same round.
"""
import contextlib
import json
import logging
import os
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DIRNAME = "training_maps"
TIMESTEPS = (0.25, 0.5, 0.75)    # flow-time noise levels: 0.25 detail, 0.75 layout
MAX_SIDE = 48                    # grids are pooled to at most this on the long side
SEED = 1234                      # base seed; per (probe, level) = SEED*1000003 + j*1009 + i


def _print(string):
    from toolkit.print import print_acc
    print_acc(string)


def _pick_keys(paths, n):
    """n keys spread evenly through the sorted list (the same dataset picks the same ones)."""
    keys = sorted(set(str(k) for k in paths))
    n = max(0, min(int(n), len(keys)))
    if n <= 1:
        return keys[len(keys) // 2:len(keys) // 2 + 1] if n == 1 else []
    return [keys[round(i * (len(keys) - 1) / (n - 1))] for i in range(n)]


def _display_name(path, used):
    """A probe's key: its file name, suffixed for uniqueness across datasets. Keys
    appear in probes.json / maps.jsonl, so absolute dataset paths never leak."""
    base = os.path.basename(str(path))
    name, ext = os.path.splitext(base)
    key, i = base, 2
    while key in used:
        key = f"{name}-{i}{ext}"
        i += 1
    used.add(key)
    return key


def _grid(m):
    """A (h, w) map -> {"w", "h", "v"}, pooled to <= MAX_SIDE on its long side, 4
    significant figures. Non-finite input raises (capture() then turns the maps
    off for the run -- broken weights shouldn't render as a map, and json.dumps
    must never emit NaN/Infinity)."""
    if not torch.isfinite(m).all():
        raise ValueError("non-finite map values")
    m = m.float()
    h, w = m.shape
    k = max(1, -(-max(h, w) // MAX_SIDE))
    if k > 1:
        m = F.avg_pool2d(m[None, None], k, ceil_mode=True)[0, 0]
    h, w = m.shape
    return {"w": int(w), "h": int(h), "v": [float(f"{x:.4g}") for x in m.flatten().tolist()]}


class _LossTargetBatch(SimpleNamespace):
    """The minimal batch get_loss_target touches on flow models (.latents)."""


class TrainingMaps:
    def __init__(self, process, images=3, timesteps=TIMESTEPS, seed=SEED):
        self.process = process
        self.ts = tuple(float(t) for t in timesteps)
        self.seed = int(seed)
        self.dir = os.path.join(process.save_root, DIRNAME)
        self.sample_dir = os.path.join(process.save_root, 'samples')
        self.failed = False
        self.last_rendered_step = None
        # one console line per run when the eager stance the trigger map needs
        # can't be engaged (its rows would silently render n/a otherwise)
        self._warned_stance = False
        # per (key, t) baseline loss grid {"key": {t_str: grid}}, for the
        # "learned" render; loaded from maps.jsonl on resume, else the step-0 capture
        self.first_loss = {}
        # running per-key scales so a colour fading really is the error falling
        self.scales = {}
        # probe key -> the trigger word's positions in the text the model sees
        # (empty when the family has no attention probe or the captions lack
        # the trigger); filled by _collect_probes
        self.trig = {}
        # path -> probe payload
        self.probes = self._collect_probes(images)
        self.keys = list(self.probes.keys())
        self.crops = {k: v["crop"] for k, v in self.probes.items() if v.get("crop")}

    # ------------------------------------------------------------------
    # gating + probe collection
    # ------------------------------------------------------------------

    @staticmethod
    def maybe_create(process):
        """A TrainingMaps for the run, or None (with one console line) when off or unsupported."""
        train_config = getattr(process, 'train_config', None)
        if train_config is None or not getattr(train_config, 'training_maps', False):
            return None
        sd = process.sd
        reasons = []
        if getattr(sd, 'is_llm', False):
            reasons.append("language models have no denoising maps")
        if getattr(sd, 'is_audio_model', False):
            reasons.append("audio models are not supported")
        if not getattr(sd, 'is_flow_matching', False):
            reasons.append(f"arch {getattr(sd, 'arch', '?')} is not flow-matching")
        if getattr(sd, 'is_edit', False):
            # edit variants ride reference/control latents through the batch;
            # probes pass batch=None, so their conditioning would differ from
            # training and the maps would describe the wrong thing
            reasons.append("edit variants condition on reference images the probes don't carry")
        if any(len(getattr(c, 'controls', []) or []) > 0 for c in getattr(process, 'dataset_configs', []) or []):
            reasons.append("control datasets condition on references the probes don't carry")
        if reasons:
            _print(f"[maps] training maps are off for this run ({'; '.join(reasons)}).")
            return None
        maps = TrainingMaps(
            process,
            images=getattr(train_config, 'training_maps_images', None) or 3,
        )
        if not maps.keys:
            _print("[maps] training maps are off for this run (no probe picture had both a "
                   "cached latent and a cached text embedding; enable cache_latents and "
                   "cache_text_embeddings).")
            return None
        wanted = getattr(train_config, 'training_maps_images', None) or 3
        if len(maps.probes) < wanted:
            _print(f"[maps] note: only {len(maps.probes)} of {wanted} "
                   "requested probe pictures were usable.")
        return maps

    def _collect_probes(self, images):
        from toolkit.data_loader import get_dataloader_datasets
        from toolkit.prompt_utils import PromptEmbeds

        items_by_path = {}
        for dataset in get_dataloader_datasets(self.process.data_loader):
            for item in getattr(dataset, 'file_list', []) or []:
                if getattr(item, 'is_reg', False):
                    continue
                if getattr(item, 'is_video', False) or getattr(item, 'is_audio_only', False):
                    continue
                if int(getattr(item, 'num_frames', 1) or 1) != 1:
                    continue
                items_by_path.setdefault(str(item.path), item)

        probes = {}
        used_names = set()
        picked = _pick_keys(items_by_path.keys(), images)
        trig_by_path = self._trigger_positions({p: items_by_path[p] for p in picked})
        for path in picked:
            item = items_by_path[path]
            try:
                # read the caches straight from disk: probing must never mutate
                # the shared dataset items (the dataloader deepcopies them per
                # batch, and a resident DTO latent cannot be deepcopied)
                latent, cached_uint8 = self._load_latent_from_cache(item)
                if latent is None:
                    continue
                if latent.dtype == torch.uint8:
                    # pixel-space latents cached as uint8, like get_latent decodes
                    from toolkit.dataloader_mixins import _latent_from_uint8
                    latent = _latent_from_uint8(latent)
                if latent.dim() == 5:      # (B, C, T, h, w): stills only
                    latent = latent[0, :, 0]
                elif latent.dim() == 4 and latent.shape[0] == 1:
                    latent = latent[0]
                if latent.dim() != 3:
                    continue
                latent = latent.float()
                te_path = item.get_text_embedding_path()
                if not te_path or not os.path.exists(te_path):
                    continue
                embeds = PromptEmbeds.load(te_path)
                if embeds is None:
                    continue
                photo = self._probe_photo(item, cached_uint8)
                if photo is None:
                    continue
                key = _display_name(path, used_names)
                if path in trig_by_path:
                    self.trig[key] = trig_by_path[path]
                probes[key] = {
                    "latent": latent,          # (C, h, w) fp32 on CPU
                    "embeds": embeds,
                    "photo": photo,            # PIL image, the exact trained view
                    "crop": self._crop_box(item),
                }
            except Exception as e:
                logger.debug(f"[maps] could not use probe {path}: {type(e).__name__}: {e}")
        return probes

    def _trigger_positions(self, items_by_path):
        """path -> the trigger's text positions, for the probe pictures whose
        caption contains it. Krea 2 only for now: the mapping needs the
        encoder's own template arithmetic (the instruction prefix's token count
        is sliced off the features the DiT sees). The trigger is resolved per
        item: a dataset can set its own trigger_word while the process one is
        None, and get_caption() injects the item's. One console line when the
        trigger map is off for the run; a bad tokenizer mapping disables it
        without touching the loss/effect maps."""
        sd = self.process.sd
        if getattr(sd, 'arch', '') != 'krea2':
            return {}
        out = {}
        try:
            from extensions_built_in.diffusion_models.krea2.src.text_encoder import (
                PROMPT_TEMPLATE_ENCODE_PREFIX,
                PROMPT_TEMPLATE_ENCODE_START_IDX,
            )
            from toolkit.attn_probe import trigger_positions
            tokenizer = getattr(sd, 'tokenizer', None)
            if tokenizer is None:
                return {}
            max_length = int(getattr(sd, 'max_text_length', 512) or 512)
            process_trigger = getattr(self.process, 'trigger_word', None)
            for path, item in items_by_path.items():
                trigger = getattr(item, 'trigger_word', None) or process_trigger
                if not str(trigger or '').strip():
                    continue
                caption = getattr(item, 'caption', None)
                if caption is None:
                    caption = item.get_caption()
                if not caption:
                    continue
                try:
                    pos = trigger_positions(
                        tokenizer, PROMPT_TEMPLATE_ENCODE_PREFIX, caption, trigger,
                        max_length=max_length, prefix_idx=PROMPT_TEMPLATE_ENCODE_START_IDX,
                    )
                except ValueError:
                    raise             # tokenizer-level: every item fails the same way
                except Exception as e:
                    # one broken item (a weird caption, say) must not sink the
                    # trigger map for the pictures that are fine
                    logger.debug(f"[maps] trigger positions failed for {path}: "
                                 f"{type(e).__name__}: {e}")
                    continue
                if pos:
                    out[path] = pos
        except ValueError as e:
            _print(f"[maps] the trigger-word map is off for this run ({str(e)[:160]})")
            return {}
        except Exception as e:
            _print(f"[maps] the trigger-word map is off for this run "
                   f"({type(e).__name__}: {str(e)[:160]})")
            return {}
        if not out:
            named = (f"'{str(process_trigger).strip()}'"
                     if str(process_trigger or '').strip() else "the trigger word")
            _print(f"[maps] no trigger-word map: {named} isn't in the "
                   "probe pictures' captions")
        return out

    @staticmethod
    def _load_latent_from_cache(item):
        """The item's latent (and its cached uint8 photo when present) read from
        the disk cache without touching the shared item. Falls back to
        get_latent() + cleanup for memory-cached datasets, restoring the item."""
        from safetensors.torch import load_file

        def decode(latent):
            if latent is not None and latent.dtype == torch.uint8:
                from toolkit.dataloader_mixins import _latent_from_uint8
                return _latent_from_uint8(latent)
            return latent

        path = item.get_latent_path()
        if path and os.path.exists(path):
            state_dict = load_file(path, device='cpu')
            return decode(state_dict.get('latent')), state_dict.get('tensor')
        latent = item.get_latent()
        if latent is None:
            return None, None
        try:
            from toolkit.dto import DTO
            if isinstance(latent, DTO):
                latent = latent.tensor
            latent = decode(latent).detach().float().cpu().clone()
            cached = getattr(item, '_cached_tensor_uint8', None)
            return latent, cached
        finally:
            # put the shared item back the way the dataloader expects it
            if getattr(item, 'is_caching_to_disk', False):
                item.cleanup_latent()

    @staticmethod
    def _probe_photo(item, cached_uint8=None):
        """The exact trained view of the picture: the cached pixel tensor when the
        latent cache carries one, else the original resized to its bucket and cropped
        with the item's own crop box (the same arithmetic the dataloader applies)."""
        cached = cached_uint8 if cached_uint8 is not None else getattr(item, '_cached_tensor_uint8', None)
        if cached is not None:
            from PIL import Image
            arr = cached
            if arr.dim() == 4:
                arr = arr[0]
            return Image.fromarray(arr.permute(1, 2, 0).contiguous().numpy())
        path = getattr(item, 'path', None)
        if not path or not os.path.exists(path):
            return None
        from PIL import Image
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            return None
        sw, sh = int(getattr(item, 'scale_to_width', 0) or 0), int(getattr(item, 'scale_to_height', 0) or 0)
        cw, ch = int(getattr(item, 'crop_width', 0) or 0), int(getattr(item, 'crop_height', 0) or 0)
        cx, cy = int(getattr(item, 'crop_x', 0) or 0), int(getattr(item, 'crop_y', 0) or 0)
        if min(sw, sh, cw, ch) <= 0 or cx < 0 or cy < 0:
            return None
        img = img.resize((sw, sh), Image.BICUBIC).crop((cx, cy, min(cx + cw, sw), min(cy + ch, sh)))
        if getattr(item, 'flip_x', False):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def _crop_box(item):
        """The trained crop as fractions of the bucket-scaled photo, for probes.json."""
        sw, sh = int(getattr(item, 'scale_to_width', 0) or 0), int(getattr(item, 'scale_to_height', 0) or 0)
        cw, ch = int(getattr(item, 'crop_width', 0) or 0), int(getattr(item, 'crop_height', 0) or 0)
        cx, cy = int(getattr(item, 'crop_x', 0) or 0), int(getattr(item, 'crop_y', 0) or 0)
        if min(sw, sh, cw, ch) <= 0:
            return None
        return [round(cx / sw, 5), round(cy / sh, 5), round(cw / sw, 5), round(ch / sh, 5)]

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------

    def start(self, resume=False):
        """Write probes.json; a fresh run clears old maps, a resume keeps appending
        and reloads its step-0 loss grids for the learned-since-the-start render."""
        os.makedirs(self.dir, exist_ok=True)
        if not resume:
            try:
                os.remove(os.path.join(self.dir, "maps.jsonl"))
            except OSError:
                pass
        else:
            self._load_first_loss()
        with open(os.path.join(self.dir, "probes.json"), "w", encoding="utf-8") as f:
            json.dump({
                "keys": self.keys,
                "timesteps": list(self.ts),
                "crops": self.crops,
                "attn": sorted(self.trig),
            }, f)
        _print(f"[maps] training maps on: {len(self.keys)} picture(s) x {len(self.ts)} noise level(s), "
               "captured before training and at every preview round "
               f"(output: {self.dir}, rendered into samples/)"
               + (f"; the trigger word's attention on {len(self.trig)} of them" if self.trig else ""))

    def _load_first_loss(self):
        """The earliest capture's loss grids from maps.jsonl, if one exists. Streams
        the file (a long run's jsonl is large) and keeps only the minimum-step row;
        a trailing partial line -- a run crashed mid-write -- is skipped."""
        path = os.path.join(self.dir, "maps.jsonl")
        best_step, best_maps = None, None
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue                     # still being written, or a torn tail
                    step = int(row.get("step", 0))
                    if best_step is None or step < best_step:
                        best_step, best_maps = step, (row.get("maps") or {})
        except OSError:
            return
        if best_maps is not None:
            self.first_loss = best_maps

    def capture_baseline(self):
        """The step-0 capture (weights as loaded; the EMA shadow is identical here)."""
        self.capture(step=0)

    def capture_in_sample(self, step):
        """Recurring capture, called from inside _sample's force-eager + fork_rng +
        EMA-eval window so probes see exactly the weights the previews see. The
        pre-loop baseline sample also lands here at step 0; the baseline capture
        already made that row, so don't write it twice."""
        if step is None:
            return
        if int(step) == 0 and self.first_loss:
            return
        self.capture(step=int(step))

    def capture(self, step):
        if self.failed or not self.keys:
            return
        t0 = time.time()
        try:
            maps = self._run(step)
            os.makedirs(self.dir, exist_ok=True)   # start() usually made it; don't depend on it
            with open(os.path.join(self.dir, "maps.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": int(step), "maps": maps}) + "\n")
            if not self.first_loss:
                self.first_loss = maps
        except Exception as e:
            # never fatal: a failed forward OR a failed maps.jsonl write (full
            # disk, permissions) only turns the maps off for this run
            self.failed = True
            _print(f"[maps] the training maps failed (step {step}), so they're off for the rest of "
                   f"this run. Training carries on. ({type(e).__name__}: {str(e)[:200]})")
            logger.debug("[maps] error:", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        try:
            self._render(step, maps)
        except Exception as e:
            # rendering failing must not stop future captures
            _print(f"[maps] rendering the maps failed (step {step}); the raw maps stay in "
                   f"maps.jsonl. ({type(e).__name__}: {str(e)[:160]})")
            logger.debug("[maps] render error:", exc_info=True)
        _print(f"[maps] step {step}: {len(self.keys)} picture(s), {time.time() - t0:.1f}s")

    def _run(self, step):
        process = self.process
        sd = process.sd
        device = process.device_torch
        dtype = sd.torch_dtype
        unet = getattr(sd, 'unet', None)
        was_training = unet.training if isinstance(unet, torch.nn.Module) else None
        network = getattr(process, 'network', None)
        had_network = network is not None
        net_multiplier = getattr(network, 'multiplier', None) if had_network else None
        net_ctx = network if had_network else contextlib.nullcontext()
        out = {}
        try:
            if isinstance(unet, torch.nn.Module):
                unet.eval()
            with torch.no_grad():
                for j, key in enumerate(self.keys):
                    probe = self.probes[key]
                    latents = probe["latent"][None].to(device=device, dtype=dtype)   # (1, C, h, w)
                    # PromptEmbeds.to() mutates in place, so never move the stored
                    # probe embeds themselves: clone first (AdvancedPromptEmbeds.to
                    # already returns a new object, but cloning it too is harmless)
                    embeds = probe["embeds"]
                    if hasattr(embeds, "clone"):
                        embeds = embeds.clone()
                    if hasattr(embeds, "to"):
                        embeds = embeds.to(device=device, dtype=dtype)
                    per = {}
                    for i, t in enumerate(self.ts):
                        gen = torch.Generator(device=device)
                        gen.manual_seed(self.seed * 1000003 + j * 1009 + i)
                        noise = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float32)
                        noise = noise.to(dtype)
                        timestep = torch.tensor([t * 1000.0], device=device, dtype=torch.float32)
                        noisy = sd.add_noise(latents, noise, timestep)
                        # the training LoRA only mounts inside the network context
                        # (the same `with network:` every training step uses);
                        # the multiplier toggle alone does nothing outside it
                        if had_network:
                            network.multiplier = 1.0
                        attn = None
                        if key in self.trig:
                            # the trigger's attention, on this same forward, run
                            # eagerly (a compiled block doesn't call the probe)
                            from toolkit import attn_probe
                            from toolkit.util.eager_stance import force_eager_stance
                            with force_eager_stance() as eager, \
                                    attn_probe.capturing(self.trig[key]) as trig_probe:
                                with net_ctx:
                                    pred = sd.predict_noise(
                                        latents=noisy,
                                        conditional_embeddings=embeds,
                                        timestep=timestep,
                                        batch=None,
                                    ).float()
                            if not eager and not self._warned_stance:
                                self._warned_stance = True
                                _print("[maps] torch's force-eager stance is unavailable "
                                       "here, so the trigger-attention map can't be "
                                       "captured (a compiled block never calls the probe); "
                                       "the other maps are unaffected.")
                            attn = trig_probe.result()
                        else:
                            with net_ctx:
                                pred = sd.predict_noise(
                                    latents=noisy,
                                    conditional_embeddings=embeds,
                                    timestep=timestep,
                                    batch=None,
                                ).float()
                        batch_shim = _LossTargetBatch(latents=latents)
                        if hasattr(sd, 'get_loss_target'):
                            target = sd.get_loss_target(
                                noise=noise, batch=batch_shim, timesteps=timestep,
                            ).float().detach()
                        else:
                            # mirrors the calculate_loss fallback
                            target = (noise - latents).float()
                        loss_map = (pred - target).pow(2).mean(dim=1)[0]             # (h, w)
                        maps = {"loss": _grid(loss_map)}
                        if attn is not None:
                            maps["attn"] = _grid(self._tokens_to_map(
                                attn, latents, patch=getattr(sd, 'patch_size', None)))
                        if step > 0 and had_network and net_multiplier is not None:
                            network.multiplier = 0.0
                            with net_ctx:
                                base = sd.predict_noise(
                                    latents=noisy,
                                    conditional_embeddings=embeds,
                                    timestep=timestep,
                                    batch=None,
                                ).float()
                            maps["effect"] = _grid((pred - base).abs().mean(dim=1)[0])
                            del base
                        per[f"{t:g}"] = maps
                        del noise, timestep, noisy, pred, target, loss_map
                    out[key] = per
                    del latents
        finally:
            if had_network and net_multiplier is not None and getattr(network, 'multiplier', None) != net_multiplier:
                network.multiplier = net_multiplier
            if isinstance(unet, torch.nn.Module) and was_training is not None:
                unet.train(was_training)
        return out

    @staticmethod
    def _tokens_to_map(attn, latents, patch=None):
        """The probe's per-image-token attention share (N,) laid out row by row
        on the latent's patch grid (N = (h/p) * (w/p); krea2 packs its image
        tokens row-major in prepare()). `patch` is the model's configured patch
        size when the caller knows it (krea2 exposes sd.patch_size); otherwise
        it is inferred from the token count, which only holds while the
        sequence is image tokens alone."""
        h, w = latents.shape[-2], latents.shape[-1]
        n = attn.numel()
        p = int(patch) if patch and int(patch) >= 1 \
            else max(1, round((h * w / max(1, n)) ** 0.5))
        gh, gw = -(-h // p), -(-w // p)
        if gh * gw != n:
            gh, gw = h // p, w // p
        if gh * gw != n:
            raise RuntimeError(f"can't lay {n} attention tokens on a {h}x{w} latent grid")
        return attn.detach().float().cpu().view(gh, gw)

    # ------------------------------------------------------------------
    # rendering (the job gallery lists samples/, so composites land there)
    # ------------------------------------------------------------------

    def _scale(self, key, kind, grid_max):
        """One colour scale per (probe, map) across all steps: the running max,
        so a colour fading really is the error falling."""
        s = self.scales.setdefault(key, {})
        if kind not in s or grid_max > s[kind]:
            s[kind] = grid_max
        return s[kind]

    def _render(self, step, maps):
        from PIL import Image, ImageDraw

        for j, key in enumerate(self.keys):
            probe = self.probes[key]
            photo = probe["photo"]
            pw, ph = max(photo.size[0], 64), max(photo.size[1], 64)
            if photo.size != (pw, ph):
                # the panels and overlays are sized (pw, ph); a photo smaller than
                # the clamp must be resized to match or the overlay indexing breaks
                photo = photo.resize((pw, ph))

            def map_of(capture, i, kind):
                t = f"{self.ts[i]:g}"
                grid = (capture.get(key) or {}).get(t, {}).get(kind)
                return self._grid_to_tensor(grid) if grid is not None else None

            def learned_of(i):
                # the same fixed noise each capture, so the picture's own texture
                # cancels and only the learning is left
                now, first = map_of(maps, i, 'loss'), map_of(self.first_loss, i, 'loss')
                if now is None or first is None or now.shape != first.shape:
                    return None
                return first - now

            def render_seq(grids, kind, diverging=False):
                imgs = []
                for m in grids:
                    if m is None:
                        imgs.append(None)
                        continue
                    up = self._upsample(m, pw, ph)
                    peak = (float(up.abs().max()) if diverging else float(up.max())) \
                        if up.numel() else 0.0
                    imgs.append(self._overlay(
                        photo, up,
                        scale=self._scale(key, kind, peak),
                        diverging=diverging,
                    ))
                return imgs

            # reading order follows Fizgig: learned first, then not-learned, then
            # the effect, then (when the run has a trigger in the probe captions)
            # the trigger word's attention
            rows = [
                (render_seq([learned_of(i) for i in range(len(self.ts))], 'learned', diverging=True),
                 "Learned since the start", "green=better  magenta=worse"),
                (render_seq([map_of(maps, i, 'loss') for i in range(len(self.ts))], 'loss'),
                 "Not learned yet", "red worst / yellow / blue / none best"),
                (render_seq([map_of(maps, i, 'effect') for i in range(len(self.ts))], 'effect'),
                 "Where the LoRA acts", "red strongest / yellow / blue / none"),
            ]
            if key in self.trig or any(
                (maps.get(key) or {}).get(f"{t:g}", {}).get("attn") is not None for t in self.ts):
                rows.append((
                    render_seq([map_of(maps, i, 'attn') for i in range(len(self.ts))], 'attn'),
                    "Trigger word attention", "red most attention / yellow / blue / none",
                ))

            row_h = ph + 16          # panel row height incl. its level labels
            header, gap, label_w = 26, 6, 152
            n_levels = len(self.ts)
            canvas = Image.new('RGB', (
                label_w + n_levels * (pw + gap),
                header + len(rows) * (row_h + 26),
            ), (24, 24, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 7), f"{key}  step {step}", fill=(220, 220, 225))
            labels = [f"t={t:g}" for t in self.ts]
            if n_levels == 3:
                labels = ["low t={:g}".format(self.ts[0]), "mid t={:g}".format(self.ts[1]),
                          "high t={:g}".format(self.ts[2])]
            y = header
            for imgs, title, legend in rows:
                draw.text((8, y + ph // 2), title, fill=(200, 200, 210))
                panel = self._panel(imgs, pw, ph, labels)
                canvas.paste(panel, (label_w, y))
                draw.text((label_w, y + row_h), legend, fill=(150, 150, 160))
                y += row_h + 26
            filename = f"MAP_{int(time.time() * 1000)}_{str(step).zfill(9)}_p{j}.jpg"
            os.makedirs(self.sample_dir, exist_ok=True)
            canvas.save(os.path.join(self.sample_dir, filename), quality=88)

    @staticmethod
    def _grid_to_tensor(grid):
        w, h = grid["w"], grid["h"]
        return torch.tensor(grid["v"], dtype=torch.float32).view(h, w)

    @staticmethod
    def _upsample(m, w, h):
        """Nearest-neighbour upsample of an (h, w) fp32 grid to the photo's size."""
        import numpy as np
        from PIL import Image
        img = Image.fromarray(m.numpy(), mode='F').resize((w, h), Image.NEAREST)
        return torch.from_numpy(np.array(img, dtype=np.float32, copy=True))

    @staticmethod
    def _panel(imgs, w, h, labels):
        """A row of the noise levels; missing levels render as a placeholder."""
        from PIL import Image, ImageDraw
        gap = 4
        canvas = Image.new('RGB', (len(imgs) * (w + gap), h + 16), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)
        for i, img in enumerate(imgs):
            x = i * (w + gap)
            if img is not None:
                canvas.paste(img, (x, 0))
            else:
                draw.rectangle([x, 0, x + w - 1, h - 1], outline=(70, 70, 80))
                draw.text((x + 4, h // 2), "n/a", fill=(90, 90, 100))
            draw.text((x + 2, h + 1), labels[i], fill=(140, 140, 150))
        return canvas

    @staticmethod
    def _overlay(photo, m, scale, diverging=False):
        """Greyed photo under a banded heatmap: none/blue/yellow/red (or
        green/magenta around zero when diverging). m is an (h, w) fp32 tensor
        already the size of the photo."""
        import numpy as np
        from PIL import Image

        arr = m.numpy()
        base = np.asarray(photo.convert('L').convert('RGB'), dtype=np.float32)
        base = base * 0.45 + 20.0          # greyed and dimmed under the map
        out = base
        if diverging:
            norm = np.clip(arr / max(scale, 1e-8), -1.0, 1.0)
            pos, neg = norm >= 0.02, norm <= -0.02
            out = out.copy()
            if pos.any():
                out[pos] = np.stack([np.full(pos.sum(), 20.0),
                                     170 + 85 * norm[pos],
                                     np.full(pos.sum(), 30.0)], axis=-1)     # green
            if neg.any():
                out[neg] = np.stack([205 + 50 * (-norm[neg]),
                                     np.full(neg.sum(), 20.0),
                                     165 + 90 * (-norm[neg])], axis=-1)      # magenta
        else:
            norm = np.clip(arr / max(scale, 1e-8), 0.0, 1.0)
            out = out.copy()
            # ascending thresholds: the highest band a cell reaches wins
            for lo, color in (
                (0.02, (32, 42, 72)),    # barely: faint blue
                (0.35, (40, 90, 230)),   # blue
                (0.65, (240, 210, 40)),  # yellow
                (0.85, (235, 40, 30)),   # red
            ):
                out[norm >= lo] = np.array(color, dtype=np.float32)
        return Image.fromarray(np.clip(out, 0, 255).astype('uint8'))
