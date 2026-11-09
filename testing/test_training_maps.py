"""Pure-function tests for toolkit/training_maps.py (no GPU / model needed)."""
import json
import os
import tempfile

import pytest
import torch
from PIL import Image

from toolkit.training_maps import TrainingMaps, _grid, _pick_keys


def test_pick_keys_even_spread_dedupes_and_sorts():
    assert _pick_keys(['b', 'a', 'a', 'c', 'd', 'e'], 3) == ['a', 'c', 'e']
    assert _pick_keys(['only.jpg'], 3) == ['only.jpg']
    assert _pick_keys([], 3) == []
    # deterministic across shuffles
    shuffled = ['e', 'a', 'c', 'b', 'd']
    assert _pick_keys(shuffled, 5) == _pick_keys(sorted(shuffled), 5)


def test_grid_pools_to_max_side_and_roundtrips():
    m = torch.rand(37, 71)
    g = _grid(m)
    assert g['w'] <= 48 and g['h'] <= 48
    t = TrainingMaps._grid_to_tensor(g)
    assert t.shape == (g['h'], g['w'])
    assert all(isinstance(v, float) for v in g['v'])


def test_grid_small_grid_untouched():
    m = torch.rand(10, 10)
    g = _grid(m)
    assert (g['w'], g['h']) == (10, 10)
    assert len(g['v']) == 100


def _bare_maps(tmpdir):
    """A TrainingMaps with just the render/capture bookkeeping fields."""
    obj = object.__new__(TrainingMaps)
    obj.ts = (0.25, 0.5, 0.75)
    obj.keys = ['x']
    obj.probes = {'x': {'name': 'probe_x.jpg', 'photo': Image.new('RGB', (160, 96), (110, 80, 60)), 'crop': None}}
    obj.scales = {}
    obj.first_loss = {}
    obj.sample_dir = tmpdir
    return obj


def _fake_capture(step):
    per = {}
    for t in (0.25, 0.5, 0.75):
        h, w = 6, 10
        base = torch.rand(h, w) * 0.02
        loss = base + (0.5 if step > 0 else 0.9) * torch.exp(
            -((torch.arange(h)[:, None] - 3) ** 2 + (torch.arange(w)[None, :] - 5) ** 2) / 8)
        per[f'{t:g}'] = {'loss': _grid(loss)}
        if step > 0:
            eff = 0.4 * torch.exp(
                -((torch.arange(h)[:, None] - 2) ** 2 + (torch.arange(w)[None, :] - 3) ** 2) / 4)
            per[f'{t:g}']['effect'] = _grid(eff)
    return {'x': per}


def test_render_baseline_and_step(tmp_path):
    maps = _bare_maps(str(tmp_path))
    maps.first_loss = _fake_capture(0)
    TrainingMaps._render(maps, 60, _fake_capture(60))
    files = [f for f in os.listdir(str(tmp_path)) if f.startswith('MAP_')]
    assert len(files) == 1
    img = Image.open(os.path.join(str(tmp_path), files[0]))
    assert img.mode == 'RGB' and img.size[0] > 0

    # a fresh baseline render (no first_loss) must not raise and marks panels n/a
    maps.first_loss = {}
    TrainingMaps._render(maps, 0, _fake_capture(0))
    assert len([f for f in os.listdir(str(tmp_path)) if f.startswith('MAP_')]) == 2


def test_render_missing_grids_placeholders(tmp_path):
    maps = _bare_maps(str(tmp_path))
    # a capture with neither effect nor baseline: everything renders with placeholders
    TrainingMaps._render(maps, 10, _fake_capture(0))
    assert any(f.startswith('MAP_') for f in os.listdir(str(tmp_path)))


def test_crop_box_fractions():
    class Item:
        scale_to_width = 1000
        scale_to_height = 800
        crop_width = 500
        crop_height = 400
        crop_x = 250
        crop_y = 200
    assert TrainingMaps._crop_box(Item()) == [0.25, 0.25, 0.5, 0.5]

    class Bad:
        scale_to_width = 0
    assert TrainingMaps._crop_box(Bad()) is None


def test_jsonl_row_shape_matches_fizgig_schema(tmp_path):
    # {"step", "maps": {key: {t: {"loss", "effect"}}}} with pooled grids
    row = {"step": 5, "maps": _fake_capture(5)}
    text = json.dumps(row)
    loaded = json.loads(text)
    assert loaded['step'] == 5
    one = loaded['maps']['x']['0.5']['loss']
    assert {'w', 'h', 'v'} <= set(one.keys())
