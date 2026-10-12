"""Manual focus masks (datasets[].mask_path) + DOP audit tests.

Covers the 2026-10-11 audit of masking Dataset-B (multi-person) images so the
diffusion loss supervises only the trained subject, while diff_output_preservation
keeps running full-frame.

What is production code here:
  * SDTrainer.calculate_loss driven end-to-end through a fake trainer (the
    test_depth_loss_gating.py pattern) with the REAL TrainConfig defaults --
    the same loss path krea2 (4D) and minimax_h3 (5D) execute in a real run.
  * MaskFileItemDTOMixin.__init__ + load_mask_image on real image files
    (bucket resize/crop, flip, value_map for mask_min_value, invert_mask).

What is a formula re-expression (documented, not production-driven):
  * the diff_output_preservation loss at SDTrainer.py:5689-5726 lives inside
    train_single_accumulation and needs a live network + predict_noise; its
    expression mse(preservation_pred, prior_pred) * multiplier is re-built
    here to assert full-frame (unmasked) gradients. The masking behavior of
    the production site is established by code citation, not this test.

Not covered (needs live model on GPU): real VAE latent-dim interpolation of
the mask (F.interpolate at SDTrainer.py:4951 runs on cached-latent batches in
the trainer loop), albumentations replay alignment (incompatible with latent
caching), H3 temporal packing beyond T-expand.
"""
import contextlib
import os
import tempfile

import pytest
import torch

from toolkit.config_modules import TrainConfig
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.dataloader_mixins import MaskFileItemDTOMixin


# ----------------------------------------------------------------------
# Fakes (pattern: testing/test_depth_loss_gating.py)
# ----------------------------------------------------------------------

class _FakeSchedulerConfig:
    num_train_timesteps = 1000


class _FakeScheduler:
    config = _FakeSchedulerConfig()


class _FakeSD:
    """krea2/H3-shaped model: flow-matching, get_loss_target = noise - latents."""

    is_flow_matching = True
    prediction_type = "epsilon"
    x0_pred = False
    do_masked_loss = True

    def __init__(self):
        self.noise_scheduler = _FakeScheduler()

    def scale_loss(self, loss):
        return loss

    def get_loss_target(self, *args, **kwargs):
        return (kwargs.get("noise") - kwargs.get("batch").latents).detach()


class _FakeFileItem:
    def __init__(self, path):
        self.path = path
        self.is_reg = False


class _FakeBatch:
    def __init__(self, n, latents, mask_tensor=None):
        self.file_items = [_FakeFileItem(f"/tmp/img_{i}.png") for i in range(n)]
        self.latents = latents
        self.mask_tensor = mask_tensor
        self.loss_multiplier_list = [1.0] * n

    def get_is_reg_list(self):
        return [False] * len(self.file_items)


def _make_trainer(train_config=None):
    tr = _FakeTrainer()
    tr.train_config = train_config or TrainConfig()
    tr.sd = _FakeSD()
    tr.device_torch = torch.device("cpu")
    tr.additional_logs = {}
    tr.adapter = None
    tr.dfe = None
    tr.unconditional_embeds = None
    tr.loss_watch = None
    tr.subject_mask_config = None
    tr.face_id_config = None
    tr.depth_consistency_config = None
    tr._completed_update_clock = lambda: 0
    tr.timer = lambda name: contextlib.nullcontext()
    # aux-loss gates: with every perceptual config None these return False in
    # production too; stubbed to skip their config-reading helper chains.
    for gate in (
        "_depth_loss_active", "_normal_loss_active",
        "_body_proportion_loss_active", "_face_identity_loss_active",
        "_body_shape_loss_active", "_vae_anchor_loss_active",
    ):
        setattr(tr, gate, lambda: False)
    for name in (
        "_set_objective_eligibility",
        "_build_subject_mask_weight", "_audio_loss_per_sample",
        "_apply_diffusion_split_mask",
    ):
        setattr(tr, name, getattr(SDTrainer, name).__get__(tr))
    # staticmethods: assign the plain function, no binding
    tr._mask_excluded_elements = SDTrainer._mask_excluded_elements
    tr._per_sample_weighted_mean = SDTrainer._per_sample_weighted_mean
    return tr


class _FakeTrainer:
    pass


def _loss_with_mask(latents, noise_pred_leaf, mask_4d, prior_leaf=None,
                    train_config=None, batch=None, noise=None):
    """Run production calculate_loss with a mask shaped as SDTrainer.py:4956
    delivers it: (B, C_lat, h, w), expanded across latent channels.
    Default noise=zeros makes the flow-matching target (noise - latents) fully
    deterministic; value assertions use latents/pred directly."""
    tr = _make_trainer(train_config)
    n = latents.shape[0]
    b = batch or _FakeBatch(n, latents, mask_tensor=mask_4d)
    if noise is None:
        noise = torch.zeros_like(noise_pred_leaf)
    ts = torch.full((n,), 500.0)
    loss = SDTrainer.calculate_loss(
        tr,
        noise_pred=noise_pred_leaf,
        noise=noise,
        noisy_latents=(noise * 0.5 + latents * 0.5).detach(),
        timesteps=ts,
        batch=b,
        mask_multiplier=mask_4d if mask_4d is not None else 1.0,
        prior_pred=prior_leaf,
    )
    return tr, loss


def _half_mask(n, c, h, w, min_value=0.0):
    """Left half 1.0 (subject), right half min_value (other woman)."""
    m = torch.full((n, c, h, w), float(min_value))
    m[..., : w // 2] = 1.0
    return m


# ----------------------------------------------------------------------
# 1. All-white mask == no mask (loss and gradients)
# ----------------------------------------------------------------------

def test_all_white_mask_equals_no_mask():
    torch.manual_seed(0)
    latents = torch.randn(2, 16, 8, 8)

    pred_a = torch.randn(2, 16, 8, 8)
    pred_b = pred_a.clone()
    pred_a.requires_grad_(True)
    pred_b.requires_grad_(True)
    torch.manual_seed(1)
    _, loss_a = _loss_with_mask(latents, pred_a, None)
    white = torch.ones(2, 16, 8, 8)
    torch.manual_seed(1)
    _, loss_b = _loss_with_mask(latents, pred_b, white)
    assert torch.allclose(loss_a, loss_b, rtol=1e-6)
    loss_a.backward()
    loss_b.backward()
    assert torch.allclose(pred_a.grad, pred_b.grad, rtol=1e-5)


# ----------------------------------------------------------------------
# 2. Half mask suppresses the excluded half's loss + gradient
# ----------------------------------------------------------------------

def test_half_mask_suppresses_excluded_half():
    latents = torch.zeros(1, 4, 4, 8)
    pred = torch.zeros(1, 4, 4, 8)
    pred[..., 4:] = 3.0          # error lives ONLY in the excluded right half
    pred.requires_grad_(True)
    mask = _half_mask(1, 4, 4, 8)
    _, loss = _loss_with_mask(latents, pred, mask)
    # excluded half carries the only raw error; loss must be ~0
    assert float(loss) < 1e-12
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[..., 4:].abs().max().item() == 0.0   # no grad outside subject
    # error inside the kept half DOES produce gradient
    pred2 = torch.zeros(1, 4, 4, 8)
    pred2[..., :4] = 3.0
    pred2.requires_grad_(True)
    _, loss2 = _loss_with_mask(latents, pred2, mask)
    loss2.backward()
    assert pred2.grad[..., :4].abs().max().item() > 0.0
    # weighted mean normalizes by mask mass: mean over kept half only
    assert float(loss2) == pytest.approx(9.0, rel=1e-5)


def test_half_mask_value_is_kept_region_mean():
    # left half error 1.0, right half error 4.0; mask keeps left only
    latents = torch.zeros(1, 1, 4, 8)
    pred = torch.zeros(1, 1, 4, 8)
    pred[..., :4] = 1.0
    pred[..., 4:] = 2.0
    mask = _half_mask(1, 1, 4, 8)
    _, loss = _loss_with_mask(latents, pred, mask)
    assert float(loss) == pytest.approx(1.0, rel=1e-5)    # not (1+4)/2=2.5


# ----------------------------------------------------------------------
# 3. mask_min_value retains a reduced contribution outside the subject
# ----------------------------------------------------------------------

def test_mask_min_value_retains_reduced_contribution():
    latents = torch.zeros(1, 1, 4, 8)
    pred = torch.zeros(1, 1, 4, 8)
    pred[..., 4:] = 2.0          # error only in the min-weight region
    pred.requires_grad_(True)
    mask = _half_mask(1, 1, 4, 8, min_value=0.05)
    _, loss = _loss_with_mask(latents, pred, mask)
    # weighted mean normalizes by TOTAL mask mass (denominator = 32*1.0 + 32*0.05),
    # so an excluded-region element contributes 0.05/1.05 ~ 4.8% of an in-mask
    # element instead of being fully dropped:
    #   (32*0.05*4.0) / (32*1.05)
    assert float(loss.detach()) == pytest.approx((32 * 0.05 * 4.0) / (32 * 1.05), rel=1e-5)

    pred2 = torch.zeros(1, 1, 4, 8)
    pred2[..., :4] = 1.0         # left: weight 1.0, err^2 1.0
    pred2[..., 4:] = 2.0         # right: weight 0.05, err^2 4.0
    mask2 = _half_mask(1, 1, 4, 8, min_value=0.05)
    _, loss2 = _loss_with_mask(latents, pred2, mask2)
    # (32*1.0*1.0 + 32*0.05*4.0) / (32*1.0 + 32*0.05) = (32+6.4)/33.6
    assert float(loss2) == pytest.approx((32 * 1.0 + 32 * 0.05 * 4.0) / (32 * 1.05), rel=1e-5)


def test_mask_min_value_gradient_ratio():
    latents = torch.zeros(1, 1, 4, 8)
    pred = torch.ones(1, 1, 4, 8)   # uniform error 1.0 everywhere
    pred.requires_grad_(True)
    mask = _half_mask(1, 1, 4, 8, min_value=0.05)
    _, loss = _loss_with_mask(latents, pred, mask)
    loss.backward()
    g = pred.grad[0, 0, 0]
    # per-element grad ratio excluded:kept == 0.05:1 (weights multiply the
    # numerator; the denominator is the shared mask mass)
    assert g[0].item() == pytest.approx(g[7].item() / 0.05, rel=1e-4) or \
           g[7].item() == pytest.approx(g[0].item() * 0.05, rel=1e-4)


# ----------------------------------------------------------------------
# 4. All-black masks: zero loss, no NaN, step-ineligible
# ----------------------------------------------------------------------

def test_all_black_mask_zero_loss_no_nan():
    latents = torch.randn(1, 4, 4, 8)
    pred = torch.randn(1, 4, 4, 8, requires_grad=True)
    black = torch.zeros(1, 4, 4, 8)
    tr, loss = _loss_with_mask(latents, pred, black)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    loss.backward()
    assert pred.grad.abs().max().item() == 0.0
    # the batch assembly zero-fill (data_loader.py:722-735) turns unmasked
    # Dataset-A items into all-black rows; a fully-black batch marks the
    # objective ineligible so the lifecycle skips the optimizer update.
    assert tr._last_objective_eligible is False


def test_mixed_white_black_batch_black_row_contributes_nothing():
    # Dataset A row (zero-filled mask) + Dataset B row (half mask)
    latents = torch.zeros(2, 1, 4, 8)
    pred = torch.ones(2, 1, 4, 8)
    pred.requires_grad_(True)
    mask = torch.zeros(2, 1, 4, 8)
    mask[1] = _half_mask(1, 1, 4, 8)[0]
    _, loss = _loss_with_mask(latents, pred, mask)
    loss.backward()
    assert pred.grad[0].abs().max().item() == 0.0     # unmasked row: no grad
    assert pred.grad[1].abs().max().item() > 0.0      # masked row trains


# ----------------------------------------------------------------------
# 5. Numerical safety: nonfinite values inside excluded regions are inert
# ----------------------------------------------------------------------

def test_nan_in_excluded_region_is_inert():
    latents = torch.zeros(1, 1, 4, 8)
    pred = torch.zeros(1, 1, 4, 8)
    pred[..., 4:] = float("nan")   # poisoned outside the subject
    pred[..., :4] = 1.0
    pred.requires_grad_(True)
    mask = _half_mask(1, 1, 4, 8)
    _, loss = _loss_with_mask(latents, pred, mask)
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(1.0, rel=1e-5)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


# ----------------------------------------------------------------------
# 6. H3-shaped 5D latents mask the same way
# ----------------------------------------------------------------------

def test_5d_video_latents_masked_across_frames():
    latents = torch.zeros(1, 4, 3, 4, 8)      # B,C,T,H,W (H3 packs T=1 for images)
    pred = torch.zeros(1, 4, 3, 4, 8)
    pred[..., 4:] = 3.0
    pred.requires_grad_(True)
    mask4d = _half_mask(1, 4, 4, 8)           # arrives as (B,C,H,W)
    _, loss = _loss_with_mask(latents, pred, mask4d)
    assert float(loss) < 1e-12
    pred2 = pred.detach().clone()
    pred2[..., :, :, :4] = 2.0
    pred2.requires_grad_(True)
    _, loss2 = _loss_with_mask(latents, pred2, mask4d)
    loss2.backward()
    assert pred2.grad[..., :, :, :4].abs().max().item() > 0.0
    assert pred2.grad[..., :, :, 4:].abs().max().item() == 0.0


# ----------------------------------------------------------------------
# 7. do_prior_divergence term (NOT diff_output_preservation) is masked too
# ----------------------------------------------------------------------

def test_prior_divergence_term_masked():
    # The -MSE(pred, prior) term at SDTrainer.py:4092-4106 runs through the
    # same _mask_excluded_elements; prior differences confined to the excluded
    # half must not move the loss.
    tc = TrainConfig()
    tc.do_prior_divergence = True
    latents = torch.zeros(1, 1, 4, 8)
    mask = _half_mask(1, 1, 4, 8)

    pred = torch.zeros(1, 1, 4, 8, requires_grad=True)
    prior_outside = torch.zeros(1, 1, 4, 8)
    prior_outside[..., 4:] = 5.0        # prior differs ONLY outside subject
    _, loss_a = _loss_with_mask(latents, pred, mask, prior_leaf=prior_outside.detach(), train_config=tc)

    pred_b = torch.zeros(1, 1, 4, 8, requires_grad=True)
    prior_same = torch.zeros(1, 1, 4, 8)
    _, loss_b = _loss_with_mask(latents, pred_b, mask, prior_leaf=prior_same.detach(), train_config=tc)
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-9)  # masked out


# ----------------------------------------------------------------------
# 8. Per-sample independence (gradient accumulation parity)
# ----------------------------------------------------------------------

def test_per_sample_independence_for_accumulation():
    # per-sample weighted mean + batch mean == same per-sample values computed
    # alone; microbatch chunking (SDTrainer.py:5044-5072) cannot change what a
    # mask does to a sample.
    latents = torch.zeros(2, 1, 4, 8)
    pred = torch.ones(2, 1, 4, 8)
    pred.requires_grad_(True)
    mask = torch.zeros(2, 1, 4, 8)
    mask[0] = _half_mask(1, 1, 4, 8)[0]
    mask[1] = _half_mask(1, 1, 4, 8, min_value=0.05)[0]
    _, loss_batch = _loss_with_mask(latents, pred, mask)

    vals = []
    for i in range(2):
        p = pred[i:i + 1].detach().clone().requires_grad_(True)
        _, l = _loss_with_mask(latents[i:i + 1], p, mask[i:i + 1])
        vals.append(float(l))
    assert float(loss_batch) == pytest.approx(sum(vals) / 2, rel=1e-5)


# ----------------------------------------------------------------------
# 9. diff_output_preservation loss: full-frame, differentiable (formula
#    re-expression of SDTrainer.py:5700-5726 -- see module docstring)
# ----------------------------------------------------------------------

def test_dop_preservation_loss_full_frame_and_differentiable():
    prior_pred = torch.randn(1, 16, 8, 8).detach()   # get_prior_prediction: no_grad, LoRA off
    assert not prior_pred.requires_grad
    pres_data = torch.zeros(1, 16, 8, 8)             # LoRA-on class pass
    pres_data[..., 4:] = 2.0                         # error ONLY outside the "subject"
    preservation_pred = pres_data.clone().requires_grad_(True)
    multiplier = 1.0
    # production expression (SDTrainer.py:5709): no mask_multiplier anywhere
    preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier
    assert torch.isfinite(preservation_loss)
    preservation_loss.backward()
    # gradients exist in BOTH halves: the preservation loss is full-frame
    assert preservation_pred.grad[..., :4].abs().sum() > 0.0
    assert preservation_pred.grad[..., 4:].abs().sum() > 0.0


# ----------------------------------------------------------------------
# 9b. Identity subject weights compose with the default ones multiplier
#     (regression: 2026-10-12 SM1 crash -- the old channel-only expansion
#     turned the ones focus multiplier into (B,C,1,1) and the flattening
#     weighted mean raised "expanded size 63936 must match 16")
# ----------------------------------------------------------------------

def _subject_batch(n, latents, subj_half="left", weight=0.1):
    from toolkit.config_modules import SubjectMaskConfig  # noqa: F401
    C = latents.shape[1]
    H, W = latents.shape[-2:]
    subj = torch.zeros(n, 1, H, W)
    subj[..., : W // 2 if subj_half == "left" else W // 2:] = 1.0

    class _SubjectBatch(_FakeBatch):
        subject_person_masks = subj
        others_person_masks = torch.zeros(n, 1, H, W)
        subject_masks = torch.ones(n, 1, H, W)
        body_masks = None
        clothing_masks = None
        subject_person_loss_weight_list = [weight] * n
        others_person_loss_weight_list = [None] * n
        background_loss_weight_list = [None] * n
        clothing_loss_weight_list = [None] * n
        body_loss_weight_list = [None] * n

    return _SubjectBatch(n, latents, mask_tensor=None)


def test_subject_weight_with_default_multiplier_no_crash():
    from toolkit.config_modules import SubjectMaskConfig
    latents = torch.zeros(1, 16, 54, 74)
    tr = _make_trainer()
    tr.subject_mask_config = SubjectMaskConfig(
        enabled=True, subject_identity_dataset="X"
    )
    batch = _subject_batch(1, latents)
    sw = SDTrainer._build_subject_mask_weight(
        tr, batch, latents.shape, dtype=torch.float32
    )
    assert tuple(sw.shape) == (1, 16, 54, 74)

    pred = torch.ones(1, 16, 54, 74, requires_grad=True)
    loss = SDTrainer.calculate_loss(
        tr, noise_pred=pred, noise=torch.zeros_like(pred),
        noisy_latents=(latents * 0.5).detach(),
        timesteps=torch.full((1,), 500.0), batch=batch,
        mask_multiplier=torch.ones(1, 1, 1, 1), prior_pred=None,
    )
    loss.backward()
    # amplitude semantics: global-mean denominator (all ones), 0.1 amplitude
    # on the subject half -> loss = 0.55, grad ratio subject:elsewhere = 0.1
    assert float(loss.detach()) == pytest.approx(0.55, rel=1e-5)
    g = pred.grad[0, 0, 0]
    assert g[0].item() / g[-1].item() == pytest.approx(0.1, rel=1e-3)


# ----------------------------------------------------------------------
# 10-12. Production mask file loading: matching, transforms, min value
# ----------------------------------------------------------------------

class _KwBase:
    """Swallows cooperative-init kwargs the other FileItemDTO mixins would eat."""

    def __init__(self, *args, **kwargs):
        pass


class _MaskOnlyItem(MaskFileItemDTOMixin, _KwBase):
    """Just the mask mixin; geometry attrs set the way the bucketing code does."""

    def __init__(self, dataset_config, path):
        self.dataset_config = dataset_config
        super().__init__(dataset_config=dataset_config, path=path)


def _make_dataset_config(tmp, mask_dir, min_value=0.0, invert=False):
    class _DC:  # fields mirror DatasetConfig (config_modules.py:1683-1689)
        alpha_mask = False
        buckets = True
    dc = _DC()
    dc.mask_path = str(mask_dir)
    dc.mask_min_value = min_value
    dc.invert_mask = invert
    return dc


def _write_img(path, size, left_value, right_value):
    from PIL import Image
    w, h = size
    arr = torch.zeros(h, w)
    arr[:, : w // 2] = left_value
    arr[:, w // 2:] = right_value
    Image.fromarray((arr * 255).to(torch.uint8).numpy(), mode="L").save(path)


def _make_item(tmp, mask_name, min_value=0.0, invert=False, flip_x=False,
               scale=None, crop=None):
    img_dir = os.path.join(tmp, "images")
    mask_dir = os.path.join(tmp, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "photo.png")
    _write_img(img_path, (64, 64), 1.0, 0.5)
    if mask_name is not None:
        _write_img(os.path.join(mask_dir, mask_name), (64, 64), 1.0, 0.0)
    dc = _make_dataset_config(tmp, mask_dir, min_value, invert)
    item = _MaskOnlyItem(dc, img_path)
    # geometry exactly as the bucketing pass leaves it on a FileItemDTO
    item.scale_to_width, item.scale_to_height = scale or (64, 64)
    item.crop_width, item.crop_height = crop or (64, 64)
    item.crop_x, item.crop_y = 0, 0
    item.flip_x, item.flip_y = flip_x, False
    item.aug_replay_spatial_transforms = None
    return item


def test_mask_matched_by_filename():
    with tempfile.TemporaryDirectory() as tmp:
        item = _make_item(tmp, "photo.png")
        assert item.has_mask_image is True
        assert item.mask_path is not None
        item.load_mask_image()
        assert item.mask_tensor is not None
        assert item.mask_tensor.shape == (1, 64, 64)


def test_missing_mask_silently_unmasked():
    # name mismatch -> no error, no warning flag: masking silently disabled
    # for this item (audit risk #1)
    with tempfile.TemporaryDirectory() as tmp:
        item = _make_item(tmp, "wrong_name.png")
        assert item.has_mask_image is False
        assert item.mask_path is None


def test_mask_transforms_align_with_image_geometry():
    # resize 64->32, crop 8..40 (i.e. 32x32 window), flip_x: the expected mask
    # is computed independently with PIL, mirroring how the image transforms
    # use the SAME file-item geometry fields.
    from PIL import Image
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        item = _make_item(tmp, "photo.png", flip_x=True, scale=(32, 32), crop=(24, 24))
        item.crop_x, item.crop_y = 4, 8
        item.load_mask_image()
        got = item.mask_tensor[0].numpy()

        # independent expectation: original mask resized, cropped, flipped
        mask_dir = os.path.join(tmp, "masks")
        ref = Image.open(os.path.join(mask_dir, "photo.png")).convert("L")
        ref = ref.resize((32, 32), Image.BICUBIC)
        ref = ref.crop((4, 8, 4 + 24, 8 + 24))
        ref = ref.transpose(Image.FLIP_LEFT_RIGHT)
        expected = torch.from_numpy(np.array(ref, dtype=np.float32) / 255.0)
        # blur is 0 at this size (radius = int(32*rand*0.005) == 0)
        assert torch.allclose(torch.from_numpy(got).float(), expected, atol=0.02)


def test_mask_min_value_maps_black_to_floor():
    with tempfile.TemporaryDirectory() as tmp:
        item = _make_item(tmp, "photo.png", min_value=0.05)
        item.load_mask_image()
        t = item.mask_tensor
        assert t.min().item() == pytest.approx(0.05, abs=0.01)
        assert t.max().item() == pytest.approx(1.0, abs=0.01)


def test_invert_mask_swaps_polarity():
    with tempfile.TemporaryDirectory() as tmp:
        item = _make_item(tmp, "photo.png", invert=True)
        item.load_mask_image()
        t = item.mask_tensor
        # left was white -> now black; right was black -> now white
        assert t[0, :, :32].max().item() < 0.1
        assert t[0, :, 32:].min().item() > 0.9
