"""Tests for toolkit/training_maps.py (pure functions + the never-fatal capture path)."""
import json
import os

import pytest
import torch
from PIL import Image

from toolkit.training_maps import (
    TrainingMaps, _COMPOSITE_MARKER, _display_name, _grid, _pick_keys, _pick_probes,
)


def test_pick_keys_even_spread_dedupes_and_sorts():
    assert _pick_keys(['b', 'a', 'a', 'c', 'd', 'e'], 3) == ['a', 'c', 'e']
    assert _pick_keys(['only.jpg'], 3) == ['only.jpg']
    assert _pick_keys([], 3) == []
    # deterministic across shuffles
    shuffled = ['e', 'a', 'c', 'b', 'd']
    assert _pick_keys(shuffled, 5) == _pick_keys(sorted(shuffled), 5)


def _is_composite(path):
    return _COMPOSITE_MARKER in os.path.basename(path)


def test_pick_probes_single_dataset_is_the_plain_spread():
    paths = [f'ds0/a{i}.jpg' for i in range(9)]
    group_of = {p: 'ds000' for p in paths}
    assert _pick_probes(paths, group_of, 3) == _pick_keys(paths, 3)


def test_pick_probes_one_per_dataset_composites_last_more_than_n():
    # two subjects' datasets + their two side-by-side composite datasets
    subj1 = [f's1/own{i}.jpg' for i in range(5)]
    subj2 = [f's2/mine{i}.jpg' for i in range(5)]
    comp1 = [f'cache1/{_COMPOSITE_MARKER}{i:02d}.png' for i in range(3)]
    comp2 = [f'cache2/{_COMPOSITE_MARKER}{i:02d}.png' for i in range(3)]
    paths = subj1 + subj2 + comp1 + comp2
    group_of = {}
    for i, group in enumerate([subj1, subj2, comp1, comp2]):
        for p in group:
            group_of[p] = f'ds{i:03d}'
    # n=3 but four datasets: one per dataset, more than n, composites after subjects
    picked = _pick_probes(paths, group_of, 3)
    assert len(picked) == 4
    assert not any(_is_composite(p) for p in picked[:2])
    assert all(_is_composite(p) for p in picked[2:])
    assert {group_of[p] for p in picked} == set(group_of.values())
    # with room to fill (n=6): the fill comes from the subjects' own pictures only
    picked6 = _pick_probes(paths, group_of, 6)
    assert len(picked6) == 6
    assert sum(_is_composite(p) for p in picked6) == 2
    # deterministic
    assert _pick_probes(list(reversed(paths)), group_of, 3) == picked


def test_pick_probes_mixed_dataset_prefers_its_composite_representative():
    own = [f'ds0/own{i}.jpg' for i in range(5)]
    stray = [f'ds0/{_COMPOSITE_MARKER}07.png']      # a hand-built mixed folder
    other = [f'ds1/mine{i}.jpg' for i in range(5)]
    paths = own + stray + other
    group_of = {p: ('ds000' if p.startswith('ds0/') else 'ds001') for p in paths}
    picked = _pick_probes(paths, group_of, 2)
    assert len(picked) == 2
    # the mixed dataset stands for its composite; the fill never adds composites
    assert picked[1] == stray[0]                    # composite datasets sort last
    assert picked[0] in other


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


def test_grid_rejects_nonfinite():
    m = torch.rand(6, 6)
    m[2, 2] = float('nan')
    with pytest.raises(ValueError):
        _grid(m)
    m[2, 2] = float('inf')
    with pytest.raises(ValueError):
        _grid(m)


def test_display_name_dedupes_and_leaks_no_paths():
    used = set()
    a = _display_name('E:/datasets/foo/img.jpg', used)
    b = _display_name('E:/other/bar/img.jpg', used)
    assert a == 'img.jpg'
    assert b != a and b.startswith('img-') and b.endswith('.jpg')
    assert '/' not in b and '\\' not in b


def _bare_maps(tmpdir, photo_size=(160, 96)):
    """A TrainingMaps with just the render/capture bookkeeping fields."""
    obj = object.__new__(TrainingMaps)
    obj.ts = (0.25, 0.5, 0.75)
    obj.keys = ['x.jpg']
    obj.probes = {'x.jpg': {
        'latent': torch.zeros(4, 6, 10),
        'photo': Image.new('RGB', photo_size, (110, 80, 60)),
        'crop': None,
    }}
    obj.scales = {}
    obj.first_loss = {}
    obj.trig = {}
    obj.sample_dir = str(tmpdir)
    obj.dir = str(tmpdir)
    obj.failed = False
    obj.process = None
    return obj


def _fake_maps(step):
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
    return {'x.jpg': per}


def test_capture_writes_valid_jsonl_and_renders(tmp_path):
    maps = _bare_maps(tmp_path / 'job')
    maps._run = lambda step: _fake_maps(step)          # instance attr shadows the method
    maps.capture(0)
    maps.capture(50)
    rows = [json.loads(l) for l in open(tmp_path / 'job' / 'maps.jsonl')]
    assert [r['step'] for r in rows] == [0, 50]
    one = rows[1]['maps']['x.jpg']['0.5']
    assert {'loss', 'effect'} <= set(one.keys()) and {'w', 'h', 'v'} <= set(one['loss'].keys())
    # first_loss locked to the earliest capture, MAP renders for both steps
    assert set(maps.first_loss.keys()) == {'x.jpg'}
    maps_files = [f for f in os.listdir(tmp_path / 'job') if '__' in f and f.endswith('.png')]
    assert len(maps_files) == 2


def test_capture_write_failure_never_fatal(tmp_path):
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    maps = _bare_maps(tmp_path / 'job')
    maps.dir = str(blocker)                            # maps.jsonl under a file: open fails
    maps._run = lambda step: _fake_maps(step)
    maps.capture(10)                                   # must not raise
    assert maps.failed is True


def test_capture_run_failure_never_fatal(tmp_path):
    maps = _bare_maps(tmp_path / 'job')

    def boom(step):
        raise RuntimeError('gpu said no')

    maps._run = boom
    maps.capture(10)                                   # must not raise
    assert maps.failed is True
    assert not os.path.exists(tmp_path / 'job' / 'maps.jsonl')


def test_capture_skips_duplicate_step_zero(tmp_path):
    maps = _bare_maps(tmp_path / 'job')
    calls = []
    maps._run = lambda step: (calls.append(step), _fake_maps(step))[1]
    maps.first_loss = {'x.jpg': {}}                    # baseline already captured
    maps.capture_in_sample(0)                          # the pre-loop sample also fires step 0
    assert calls == [] and not os.path.exists(tmp_path / 'job' / 'maps.jsonl')
    maps.capture_in_sample(250)
    assert calls == [250]


def test_load_first_loss_streams_min_step_and_tolerates_torn_tail(tmp_path):
    d = tmp_path / 'job'
    d.mkdir()
    with open(d / 'maps.jsonl', 'w') as f:
        f.write(json.dumps({'step': 100, 'maps': {'x.jpg': {}}}) + '\n')
        f.write(json.dumps({'step': 50, 'maps': {'x.jpg': {'0.5': {'loss': _grid(torch.rand(3, 3))}}}}) + '\n')
        f.write('{"step": 40, "maps": {"x.jpg"')       # torn tail from a crashed write
    maps = _bare_maps(tmp_path / 'job')
    maps._load_first_loss()
    assert '0.5' in maps.first_loss['x.jpg']           # the minimum-step row won


def test_render_baseline_and_step(tmp_path):
    maps = _bare_maps(tmp_path)
    maps.first_loss = _fake_maps(0)
    TrainingMaps._render(maps, 60, _fake_maps(60))
    files = [f for f in os.listdir(str(tmp_path)) if f.endswith('.png')]
    assert len(files) == 1
    img = Image.open(os.path.join(str(tmp_path), files[0]))
    assert img.mode == 'RGB' and img.size[0] > 0

    # a fresh baseline render (no first_loss) must not raise and marks panels n/a
    maps.first_loss = {}
    TrainingMaps._render(maps, 0, _fake_maps(0))
    assert len([f for f in os.listdir(str(tmp_path)) if f.endswith('.png')]) == 2


def test_render_tiny_photo(tmp_path):
    # a photo below the 64px panel clamp must render, not IndexError
    maps = _bare_maps(tmp_path, photo_size=(20, 30))
    TrainingMaps._render(maps, 10, _fake_maps(0))
    assert any(f.endswith('.png') for f in os.listdir(str(tmp_path)))


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


def test_start_writes_probes_json_without_paths(tmp_path):
    maps = _bare_maps(tmp_path / 'job')
    maps.keys = list(maps.probes.keys())
    maps.crops = {}
    maps.trig = {'x.jpg': {'fyxo': [3], 'zwxem': [7]}}   # two subjects, one caption
    maps.start(resume=False)
    p = json.load(open(tmp_path / 'job' / 'probes.json'))
    assert p['keys'] == ['x.jpg'] and p['timesteps'] == [0.25, 0.5, 0.75]
    assert p['attn'] == ['x.jpg']
    assert p['attn_words'] == {'x.jpg': ['fyxo', 'zwxem']}
    assert '/' not in json.dumps(p) and '\\' not in json.dumps(p)
    # a fresh start cleared any stale jsonl
    assert not os.path.exists(tmp_path / 'job' / 'maps.jsonl')


def test_tokens_to_map_row_major():
    # h=6, w=10 latent with patch 2 -> 15 tokens on a (3, 5) grid, row-major
    attn = torch.arange(15, dtype=torch.float32)
    m = TrainingMaps._tokens_to_map(attn, torch.zeros(1, 4, 6, 10))
    assert m.shape == (3, 5)
    assert m[0, 0] == 0 and m[0, 4] == 4 and m[1, 0] == 5 and m[2, 4] == 14
    # the model's configured patch is honored when given
    assert TrainingMaps._tokens_to_map(
        attn, torch.zeros(1, 4, 6, 10), patch=2).shape == (3, 5)
    # a token count that can't tile the grid raises
    with pytest.raises(RuntimeError):
        TrainingMaps._tokens_to_map(torch.arange(17, dtype=torch.float32), torch.zeros(1, 4, 6, 10))


def test_render_trigger_row_only_when_present(tmp_path):
    maps = _bare_maps(tmp_path)
    caps = _fake_maps(0)
    TrainingMaps._render(maps, 0, caps)          # no trig: 3 rows
    maps2 = _bare_maps(tmp_path)
    maps2.trig = {'x.jpg': {'fyxo': [3]}}
    for t in (0.25, 0.5, 0.75):
        caps['x.jpg'][f'{t:g}']['attn'] = _grid(torch.rand(3, 5))
    TrainingMaps._render(maps2, 50, caps)        # trig + attn grids: taller canvas
    files = sorted(f for f in os.listdir(str(tmp_path)) if f.endswith('.png'))
    a = Image.open(os.path.join(str(tmp_path), files[0]))
    b = Image.open(os.path.join(str(tmp_path), files[1]))
    assert b.size[1] > a.size[1]


def test_render_per_trigger_rows_when_several_words(tmp_path):
    # a caption naming two subjects: the union attention row plus one row per
    # word (each trigger on its own). Row heights are uniform, so 6 rows beats
    # 4 (union only) by exactly twice what 4 beats 3 (no trigger rows).
    def render(dirname, trig, with_attn_by):
        d = tmp_path / dirname
        maps = _bare_maps(d)
        maps.trig = trig
        caps = _fake_maps(50)
        if trig:
            # attn grids alone light the union row, so only add them when a
            # trigger is in play
            for t in (0.25, 0.5, 0.75):
                caps['x.jpg'][f'{t:g}']['attn'] = _grid(torch.rand(3, 5))
                if with_attn_by:
                    caps['x.jpg'][f'{t:g}']['attn_by'] = {
                        'fyxo': _grid(torch.rand(3, 5)), 'zwxem': _grid(torch.rand(3, 5))}
        TrainingMaps._render(maps, 50, caps)
        f = next(f for f in os.listdir(str(d)) if f.endswith('.png'))
        return Image.open(os.path.join(str(d), f)).size[1]

    h6 = render('six', {'x.jpg': {'fyxo': [3], 'zwxem': [7]}}, True)
    h4 = render('four', {'x.jpg': {'fyxo': [3]}}, False)
    h3 = render('three', {}, False)
    assert h4 - h3 > 0                              # the union row
    assert h6 - h4 == 2 * (h4 - h3)                 # then one row per trigger word

