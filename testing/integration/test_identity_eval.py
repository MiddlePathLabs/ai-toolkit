"""Identity evaluation (fork-only) tests.

Pure-logic tests always run: scoring math, filename parsing, reference
discovery, selection persistence, cache invalidation, device policy and
the YuNet row conversion. Real-model tests are gated on locally available
weights/fixtures (buffalo_l in ~/.insightface, dataset images under
datasets/) and skip with a clear reason when absent. Fixture images are
local only and never committed.
"""

import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from toolkit.identity_eval import (
    EmbeddingCache,
    FaceDetector,
    IdentityEvalRunner,
    choose_device,
    discover_training_refs,
    load_selection,
    loo_band,
    max_cosine,
    mean_cosine,
    parse_sample_filename,
    save_selection,
)
from toolkit.identity_eval.cache import CACHE_VERSION
from toolkit.identity_eval.device import parse_gpu_ids, read_job_gpu_state
from toolkit.identity_eval.detector import DetectedFace, ort_cuda_providers
from toolkit.identity_eval.models import find_arcface, find_sface
from toolkit.identity_eval.refs import canonical_ref_key, prune_selection
from toolkit.identity_eval.runner import try_load
from toolkit.identity_eval.sface import yunet_row

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, 'models', 'face')

# Real-data fixtures point at LOCAL subject data and are never committed.
# Configure them via env vars or identity_fixtures.local.json next to this
# file (see the committed identity_fixtures.local.json.example). Tests that
# need them skip with a pointer to that file when they are unset.
_LOCAL_FIXTURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identity_fixtures.local.json')


def _local_fixtures() -> dict:
    try:
        with open(_LOCAL_FIXTURES_PATH, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


_LOCAL = _local_fixtures()
_SUBJECT = os.environ.get('AI_TOOLKIT_IDENTITY_SUBJECT_DATASET') or _LOCAL.get('subject_dataset') or ''
_IMPOSTOR = os.environ.get('AI_TOOLKIT_IDENTITY_IMPOSTOR_DATASET') or _LOCAL.get('impostor_dataset') or ''
_SAMPLES_DIR = os.environ.get('AI_TOOLKIT_IDENTITY_SAMPLES_DIR') or _LOCAL.get('samples_dir') or ''
SUBJECT_DATASET = _SUBJECT if os.path.isdir(_SUBJECT) else ''
IMPOSTOR_DATASET = _IMPOSTOR if os.path.isdir(_IMPOSTOR) else ''
FIXTURES_HINT = (
    'set AI_TOOLKIT_IDENTITY_SUBJECT_DATASET / AI_TOOLKIT_IDENTITY_IMPOSTOR_DATASET '
    '/ AI_TOOLKIT_IDENTITY_SAMPLES_DIR or create testing/integration/identity_fixtures.local.json '
    '(see identity_fixtures.local.json.example)'
)

pytestmark = pytest.mark.integration


def _images_in(folder, limit):
    files = []
    for ext in ('.jpg', '.jpeg', '.png', '.webp'):
        files.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    unique = []
    seen = set()
    for path in sorted(files):
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique[:limit]


@pytest.fixture(scope='session')
def subject_images():
    if not SUBJECT_DATASET:
        pytest.skip(f'subject fixture dataset not configured; {FIXTURES_HINT}')
    images = _images_in(SUBJECT_DATASET, 6)
    if len(images) < 4:
        pytest.skip(f'not enough fixture images in {SUBJECT_DATASET}')
    return images


@pytest.fixture(scope='session')
def impostor_images():
    if not IMPOSTOR_DATASET:
        pytest.skip(f'impostor fixture dataset not configured; {FIXTURES_HINT}')
    images = _images_in(IMPOSTOR_DATASET, 6)
    if len(images) < 4:
        pytest.skip(f'not enough fixture images in {IMPOSTOR_DATASET}')
    return images


@pytest.fixture(scope='session')
def arcface_ready():
    status = find_arcface()
    if not status.available:
        pytest.skip(f'ArcFace weights absent: {status.reason}')
    return status


@pytest.fixture(scope='session')
def detector_cpu():
    status = find_arcface()
    if not status.available:
        pytest.skip(f'ArcFace weights absent: {status.reason}')
    return FaceDetector(device='cpu')


# ---------------------------------------------------------------- scoring math


class TestScoring:
    def test_mean_and_max_cosine_synthetic(self):
        refs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        refs /= np.linalg.norm(refs, axis=1, keepdims=True)
        g = np.array([1.0, 0.0])
        # cosines: 1, 0, 1/sqrt(2)
        assert mean_cosine(refs, g) == pytest.approx((1 + 0 + 2 ** -0.5) / 3, abs=1e-9)
        assert max_cosine(refs, g) == pytest.approx(1.0, abs=1e-9)

    def test_loo_band_synthetic(self):
        refs = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
        refs /= np.linalg.norm(refs, axis=1, keepdims=True)
        sims = refs @ refs.T
        expected = float(
            np.mean(
                [
                    np.mean([sims[i, j] for j in range(3) if j != i])
                    for i in range(3)
                ]
            )
        )
        assert loo_band(refs) == pytest.approx(expected, abs=1e-9)

    def test_loo_band_needs_two_refs(self):
        assert loo_band(np.array([[1.0, 0.0]])) is None
        assert loo_band(None) is None

    def test_parse_sample_filename(self):
        parsed = parse_sample_filename('E:/out/samples/1791073894128__000001500_03_seed-42_steps-20.jpg')
        assert parsed == {'filename': '1791073894128__000001500_03_seed-42_steps-20.jpg', 'step': 1500, 'prompt_idx': 3, 'seed': 42, 'is_raw': False}
        raw = parse_sample_filename('RAW_1000__000000250_01.jpg')
        assert raw['is_raw'] is True and raw['step'] == 250 and raw['seed'] is None
        assert parse_sample_filename('photo.jpg') is None
        assert parse_sample_filename('1000__10_x.png') is None


# ------------------------------------------------------------- ref discovery


def _make_dataset_tree(root):
    train = os.path.join(root, 'train_ds')
    reg = os.path.join(root, 'reg_ds')
    control = os.path.join(root, 'control')
    masks = os.path.join(root, 'masks')
    for folder in (train, reg, control, masks):
        os.makedirs(os.path.join(folder, 'sub'), exist_ok=True)
    for name in ('a.jpg', 'b.PNG', 'c.webp', 'd.jpeg'):
        open(os.path.join(train, name), 'wb').write(b'x')
    open(os.path.join(train, 'sub', 'e.jpg'), 'wb').write(b'x')
    open(os.path.join(train, '.hidden.jpg'), 'wb').write(b'x')
    open(os.path.join(train, 'caption.txt'), 'wb').write(b'x')
    open(os.path.join(reg, 'r.jpg'), 'wb').write(b'x')
    open(os.path.join(control, 'c1.png'), 'wb').write(b'x')
    open(os.path.join(masks, 'm1.png'), 'wb').write(b'x')
    return train, reg, control, masks


class TestDiscovery:
    def test_exclusions_and_precedence(self, tmp_path):
        train, reg, control, masks = _make_dataset_tree(str(tmp_path))
        config = {
            'config': {
                'process': [
                    {
                        'datasets': [
                            {'folder_path': train},
                            {'folder_path': reg, 'is_reg': True},
                            {'folder_path': reg, 'prior_reg': True},
                            {'folder_path': control, 'control_path': control, 'mask_path': masks},
                        ]
                    }
                ]
            }
        }
        result = discover_training_refs(config)
        names = [os.path.basename(p) for p in result['images']]
        assert sorted(names) == ['a.jpg', 'b.PNG', 'c.webp', 'd.jpeg', 'e.jpg']
        skipped = [d['skipped'] for d in result['datasets']]
        assert skipped == [None, 'is_reg', 'prior_reg', None]
        # control/mask folders excluded even though the dataset folder is the control dir itself
        assert result.get('excluded_by_control_mask', 0) == 1

    def test_dataset_path_preferred_and_divergence_surfaced(self, tmp_path):
        train, _, _, _ = _make_dataset_tree(str(tmp_path))
        other = os.path.join(tmp_path, 'other')
        os.makedirs(other)
        open(os.path.join(other, 'z.jpg'), 'wb').write(b'x')
        config = {
            'config': {
                'process': [
                    {
                        'datasets': [
                            {'folder_path': other, 'dataset_path': train},
                            {'folder_path': other, 'dataset_path': os.path.join(tmp_path, 'nowhere')},
                        ]
                    }
                ]
            }
        }
        result = discover_training_refs(config)
        names = [os.path.basename(p) for p in result['images']]
        assert 'a.jpg' in names and 'z.jpg' not in names  # dataset_path wins
        warnings = ' '.join(w for d in result['datasets'] for w in d['warnings'])
        assert 'dataset_path' in warnings  # divergence warning surfaced, not raised

    def test_missing_folder_and_json_dataset_reported(self, tmp_path):
        json_ds = os.path.join(tmp_path, 'ds.json')
        open(json_ds, 'wb').write(b'{}')
        config = {'config': {'process': [{'datasets': [
            {'folder_path': os.path.join(tmp_path, 'gone')},
            {'dataset_path': json_ds},
            {'is_reg': False, 'prior_reg': False},
        ]}]}}
        result = discover_training_refs(config)
        assert result['images'] == []
        all_warnings = ' '.join(w for d in result['datasets'] for w in (d.get('warnings') or []))
        assert 'not found' in all_warnings
        assert 'json dataset' in all_warnings

    def test_bad_configs_return_warnings(self):
        assert discover_training_refs(None)['warnings']
        assert discover_training_refs({'config': {'process': []}})['warnings']


class TestSelection:
    def test_round_trip_and_prune(self, tmp_path):
        path = str(tmp_path / 'selection.json')
        selection = {
            'version': 1,
            'training_refs': {r'C:\x\1.jpg': {'excluded': True}, '/y/2.jpg': {'face_index': 1}},
            'external_refs': {'abc': {'excluded': False}},
        }
        save_selection(path, selection)
        loaded = load_selection(path)
        assert loaded['training_refs']['/y/2.jpg']['face_index'] == 1
        pruned = prune_selection(loaded, ['/y/2.jpg'], ['abc'])
        assert set(pruned['training_refs']) == {'/y/2.jpg'}
        assert set(pruned['external_refs']) == {'abc'}

    def test_canonical_key_matches_ui_contract(self):
        # the panel writes / selectionForImage reads the same form in TS
        # (normalizeRefPath); the worker must look up the identical string
        assert canonical_ref_key(r'E:\Data\MixedCase\Img.JPG') == 'e:/data/mixedcase/img.jpg'
        assert canonical_ref_key('/home/user/refs/pic.png') == '/home/user/refs/pic.png'
        assert canonical_ref_key('/odd//trail/') == '/odd/trail'

    def test_corrupt_resets(self, tmp_path):
        path = str(tmp_path / 'selection.json')
        open(path, 'w').write('{not json')
        assert load_selection(path) == {'version': 1, 'training_refs': {}, 'external_refs': {}}


# -------------------------------------------------------------------- cache


class TestCache:
    def _entry(self, sha='a' * 64, det_sig='v2:scrfd:h:640x640'):
        return {
            'version': CACHE_VERSION,
            'image_sha': sha,
            'det_sig': det_sig,
            'faces': [
                {'index': 0, 'bbox': [0, 0, 10, 10], 'kps': [[1, 1]] * 5, 'det_score': 0.9, 'embeddings': {}},
                {'index': 1, 'bbox': [20, 20, 5, 5], 'kps': [[1, 1]] * 5, 'det_score': 0.8, 'embeddings': {}},
            ],
        }

    def test_hit_and_content_invalidation(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        entry = self._entry()
        cache.save(entry)
        loaded = cache.load('a' * 64, 'v2:scrfd:h:640x640')
        assert loaded is not None and len(loaded['faces']) == 2
        assert cache.load('b' * 64, 'v2:scrfd:h:640x640') is None  # different image content
        assert cache.load('a' * 64, 'v2:scrfd:h:320x320') is None  # different det size
        assert cache.load('a' * 64, 'v2:scrfd:h2:640x640') is None  # different detector file

    def test_model_sig_invalidation_and_face_index_scoping(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        entry = self._entry()
        face0, face1 = entry['faces']
        cache.set_embedding(face0, 'arcface:hash1', np.ones(4))
        cache.set_embedding(face1, 'arcface:hash1', np.full(4, 2.0))
        cache.save(entry)
        loaded = cache.load(entry['image_sha'], entry['det_sig'])
        assert np.allclose(cache.get_embedding(loaded['faces'][0], 'arcface:hash1'), np.ones(4))
        assert np.allclose(cache.get_embedding(loaded['faces'][1], 'arcface:hash1'), np.full(4, 2.0))
        # face embeddings are scoped per face entry: swapping indices swaps vectors
        loaded['faces'].reverse()
        assert np.allclose(cache.get_embedding(loaded['faces'][0], 'arcface:hash1'), np.full(4, 2.0))
        # a different model hash must be recomputed
        assert cache.get_embedding(loaded['faces'][0], 'arcface:hash2') is None

    def test_corrupt_entry_ignored(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        import hashlib

        key = hashlib.sha256(f"{'a'*64}|sig".encode()).hexdigest()
        open(os.path.join(str(tmp_path), f'{key}.json'), 'w').write('nope')
        assert cache.load('a' * 64, 'sig') is None


# ------------------------------------------------------------- device policy


class TestDevice:
    def test_choose_device_matrix(self):
        assert choose_device('auto', False, 8000) == 'cuda'
        assert choose_device('auto', False, 4096) == 'cpu'  # needs > 4 GB free
        assert choose_device('auto', True, 24000) == 'cpu'  # training wins
        assert choose_device('auto', None, None) == 'cpu'  # unknown state is conservative
        assert choose_device('auto', False, None) == 'cpu'
        assert choose_device('gpu', True, 0) == 'cuda'  # forced
        assert choose_device('cpu', False, 24000) == 'cpu'

    def test_parse_gpu_ids(self):
        assert parse_gpu_ids('0') == [0]
        assert parse_gpu_ids('1,2') == [1, 2]
        assert parse_gpu_ids(' 0 ; 1 ') == [0, 1]
        assert parse_gpu_ids('') == []

    def test_running_job_detected_via_db(self, tmp_path):
        db = str(tmp_path / 'aitk.db')
        con = sqlite3.connect(db)
        con.execute('CREATE TABLE Job (name TEXT, gpu_ids TEXT, status TEXT)')
        con.executemany(
            'INSERT INTO Job VALUES (?, ?, ?)',
            [('mine', '0', 'completed'), ('other', '0', 'running'), ('solo', '1', 'completed'), ('selfrun', '2', 'running')],
        )
        con.commit()
        con.close()
        running, gpus = read_job_gpu_state(db, 'mine')
        assert running is True and gpus == [0]  # 'other' runs on the same gpu
        running, gpus = read_job_gpu_state(db, 'solo')
        assert running is False and gpus == [1]  # the only running job is on gpu 2
        running, gpus = read_job_gpu_state(db, 'selfrun')
        assert running is True  # the queried job itself counts: eval of a training job's samples must use CPU

    def test_no_running_job_is_not_busy(self, tmp_path):
        db = str(tmp_path / 'aitk.db')
        con = sqlite3.connect(db)
        con.execute('CREATE TABLE Job (name TEXT, gpu_ids TEXT, status TEXT)')
        con.execute("INSERT INTO Job VALUES ('mine', '0', 'completed')")
        con.commit()
        con.close()
        running, gpus = read_job_gpu_state(db, 'mine')
        assert running is False and gpus == [0]

    def test_unreadable_db_is_unknown(self):
        running, gpus = read_job_gpu_state('Z:/nope/missing.db', 'x')
        assert running is None and gpus == []

    def test_forced_cuda_failure_falls_back_with_warning(self):
        warnings = []

        def loader(device):
            if device == 'cuda':
                raise RuntimeError('CUDA EP init failed')
            return 'model-on-cpu'

        model, error = try_load(loader, 'ArcFace', 'cuda', warnings.append)
        assert model == 'model-on-cpu' and error is None
        assert any('CUDA' in w for w in warnings)

    def test_cuda_providers_target_selected_gpu(self):
        providers = ort_cuda_providers(1)
        assert providers[0][0] == 'CUDAExecutionProvider'
        assert providers[0][1]['device_id'] == 1  # matches the GPU free VRAM was checked on
        assert providers[0][1]['gpu_mem_limit'] == 2 * 1024 ** 3
        assert providers[0][1]['arena_extend_strategy'] == 'kSameAsRequested'
        assert providers[-1] == 'CPUExecutionProvider'

    def test_cpu_failure_is_error_not_fallback(self):
        def loader(device):
            raise RuntimeError('broken')

        model, error = try_load(loader, 'ArcFace', 'cpu', lambda w: None)
        assert model is None and 'broken' in error

    def test_cuda_device_order_env_pinned(self):
        assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'


# ------------------------------------------------------------- sface geometry


class TestYuNetRow:
    def test_row_layout(self):
        face = DetectedFace(
            index=0,
            bbox=(10.0, 20.0, 110.0, 220.0),
            kps=np.array([[30, 40], [50, 40], [40, 50], [35, 60], [45, 60]], dtype=np.float32),
            det_score=0.87,
        )
        row = yunet_row(face)
        assert row.shape == (15,)
        assert row[0:4].tolist() == pytest.approx([10, 20, 100, 200])  # x, y, w, h
        assert row[4:6].tolist() == pytest.approx([30, 40])  # right eye (kps[0])
        assert row[6:8].tolist() == pytest.approx([50, 40])  # left eye (kps[1])
        assert row[8:10].tolist() == pytest.approx([40, 50])  # nose (kps[2])
        assert row[10:12].tolist() == pytest.approx([35, 60])  # right mouth (kps[3])
        assert row[12:14].tolist() == pytest.approx([45, 60])  # left mouth (kps[4])
        assert row[14] == pytest.approx(0.87)


# ------------------------------------------------------- real-model behaviors


def _single_face_embeddings(embedder, detector, paths, swap):
    """Embed every single-face image; swapped input feeds RGB (the legacy
    scripts' channel-order bug) instead of cv2.imread BGR."""
    import cv2

    vectors = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        feed = img[:, :, ::-1] if swap else img
        faces = detector.detect(feed)
        if faces.count != 1:
            continue
        vectors.append(embedder.embed(feed, faces.faces[0]))
    return np.stack(vectors) if len(vectors) >= 3 else None


def _bgr_vs_rgb_gaps(embedder, detector, subject_images, impostor_images):
    gaps = {}
    for swap in (False, True):
        same = _single_face_embeddings(embedder, detector, subject_images, swap)
        impostors = _single_face_embeddings(embedder, detector, impostor_images, swap)
        if same is None or impostors is None:
            pytest.skip('not enough single-face fixture images')
        gaps[swap] = loo_band(same) - float(np.mean(impostors @ same.T))
    return gaps


class TestRealModels:
    def test_arcface_embedding_dims_and_norm(self, arcface_ready, subject_images):
        import cv2

        from toolkit.identity_eval.arcface import ArcFaceEmbedder

        embedder = ArcFaceEmbedder(device='cpu')
        img = cv2.imread(subject_images[0])
        faces = FaceDetector(device='cpu').detect(img)
        if not faces.count:
            pytest.skip('no face in fixture image')
        emb = embedder.embed(img, faces.faces[0])
        assert emb.shape == (512,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-4)

    def test_sface_embedding_dims_and_norm(self, subject_images):
        if not find_sface(MODELS_DIR).available:
            pytest.skip('SFace weights not installed (models/face); see docs/identity_eval.md')
        import cv2

        from toolkit.identity_eval.sface import SFaceEmbedder

        embedder = SFaceEmbedder(MODELS_DIR, device='cpu')
        img = cv2.imread(subject_images[0])
        faces = FaceDetector(device='cpu').detect(img)
        if not faces.count:
            pytest.skip('no face in fixture image')
        emb = embedder.embed(img, faces.faces[0])
        assert emb.shape == (128,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-4)

    def test_arcface_bgr_input_beats_rgb_gap(self, arcface_ready, subject_images, impostor_images):
        """Section 1 regression guard: feeding cv2.imread BGR straight in
        must separate same-id from impostors better than channel-swapped
        RGB input (the legacy eval scripts' bug)."""
        from toolkit.identity_eval.arcface import ArcFaceEmbedder

        gaps = _bgr_vs_rgb_gaps(ArcFaceEmbedder(device='cpu'), FaceDetector(device='cpu'), subject_images, impostor_images)
        assert gaps[False] > gaps[True]

    def test_sface_bgr_input_beats_rgb_gap(self, subject_images, impostor_images):
        """Same channel-order guard for SFace: OpenCV DNN models are fed
        cv2.imread BGR here; a swapped feed must separate worse."""
        if not find_sface(MODELS_DIR).available:
            pytest.skip('SFace weights not installed (models/face); see docs/identity_eval.md')
        from toolkit.identity_eval.sface import SFaceEmbedder

        gaps = _bgr_vs_rgb_gaps(SFaceEmbedder(MODELS_DIR, device='cpu'), FaceDetector(device='cpu'), subject_images, impostor_images)
        assert gaps[False] > gaps[True]

    def test_sface_alignment_matches_arcface_alignment(self, tmp_path):
        status_arc = find_arcface()
        status_sface = find_sface(os.path.join(REPO_ROOT, 'models', 'face'))
        if not status_arc.available or not status_sface.available:
            pytest.skip('SFace weights not installed (models/face); landmark-order check needs both models')
        import cv2

        from toolkit.identity_eval.arcface import ArcFaceEmbedder, _norm_crop
        from toolkit.identity_eval.sface import SFaceEmbedder

        images = _images_in(SUBJECT_DATASET, 3) if SUBJECT_DATASET else []
        if not images:
            pytest.skip(f'no fixture images; {FIXTURES_HINT}')
        detector = FaceDetector(device='cpu')
        arcface = ArcFaceEmbedder(device='cpu')
        sface = SFaceEmbedder(os.path.join(REPO_ROOT, 'models', 'face'), device='cpu')
        img = cv2.imread(images[0])
        faces = detector.detect(img)
        if faces.count != 1:
            pytest.skip('fixture image does not contain exactly one face')
        face = faces.faces[0]
        arc_aligned = _norm_crop(img, face.kps, image_size=112)
        sface_aligned = sface._recognizer.alignCrop(img, yunet_row(face))
        emb_a = arcface.embed(arc_aligned, _face_from_kps(face.kps))
        emb_b = arcface.embed(sface_aligned, _face_from_kps(face.kps))
        similarity = float(np.dot(emb_a, emb_b))
        # A wrongly ordered landmark template mirrors/rotates the crop and
        # tanks the ArcFace similarity; equivalent alignment stays high.
        assert similarity > 0.8

    def test_zero_face_and_multiface_statuses(self, arcface_ready, subject_images, tmp_path):
        import cv2

        detector = FaceDetector(device='cpu')
        noise = (np.random.default_rng(0).uniform(0, 255, (640, 640, 3))).astype(np.uint8)
        noise_path = str(tmp_path / 'noise.jpg')
        cv2.imwrite(noise_path, noise)
        assert detector.detect(cv2.imread(noise_path)).count == 0

        # two real faces side by side -> multi-face detection
        crops = []
        for path in subject_images[:4]:
            img = cv2.imread(path)
            faces = detector.detect(img)
            if faces.count == 1:
                x1, y1, x2, y2 = [int(v) for v in faces.faces[0].bbox]
                pad_x, pad_y = int((x2 - x1) * 0.4), int((y2 - y1) * 0.4)
                crops.append(
                    img[max(0, y1 - pad_y):y2 + pad_y, max(0, x1 - pad_x):x2 + pad_x]
                )
            if len(crops) == 2:
                break
        if len(crops) < 2:
            pytest.skip('could not build a two-face composite from fixtures')
        height = min(c.shape[0] for c in crops)
        crops = [c[:height] for c in crops]
        composite = np.hstack([crops[0], np.zeros((height, 20, 3), np.uint8), crops[1]])
        composite_path = str(tmp_path / 'two_faces.jpg')
        cv2.imwrite(composite_path, composite)
        assert detector.detect(cv2.imread(composite_path)).count >= 2


class _KpsFace:
    def __init__(self, kps):
        self.kps = kps


def _face_from_kps(kps):
    return _KpsFace(np.asarray(kps, dtype=np.float32))


# ---------------------------------------------------- model-change resilience


class _FakeEmbedder:
    def __init__(self, sig):
        self.model_sig = sig


class TestModelChangeResilience:
    """Installing a recognition model after discovery must refresh, not crash."""

    def _runner(self, tmp_path):
        runner = IdentityEvalRunner(job_name='j', output_root=str(tmp_path))
        runner._embedders = lambda: {'arcface': _FakeEmbedder('arcface:a'), 'sface': _FakeEmbedder('sface:b')}
        return runner

    def test_refs_missing_models_detection(self, tmp_path):
        runner = self._runner(tmp_path)
        refs_doc = {
            'training': {'images': [{'faces': [{'embeddings': {'arcface:a': [0.1]}}]}]},
            'external': {'images': []},
        }
        assert runner._refs_missing_models(refs_doc) is True  # sface installed after discovery
        refs_doc['training']['images'][0]['faces'][0]['embeddings']['sface:b'] = [0.2]
        assert runner._refs_missing_models(refs_doc) is False

    def test_refset_skips_faces_missing_a_model(self, tmp_path):
        runner = self._runner(tmp_path)
        face_both = {'index': 0, 'bbox': [0, 0, 1, 1], 'det_score': 0.9,
                     'embeddings': {'arcface:a': [1.0, 0.0], 'sface:b': [0.0, 1.0]}}
        face_arc_only = {'index': 0, 'bbox': [0, 0, 1, 1], 'det_score': 0.9,
                         'embeddings': {'arcface:a': [1.0, 0.0]}}
        images = [
            {'path': 'x/a.jpg', 'name': 'a.jpg', 'sha': 's1', 'status': 'ok', 'faces': [face_both]},
            {'path': 'x/b.jpg', 'name': 'b.jpg', 'sha': 's2', 'status': 'ok', 'faces': [face_arc_only]},
        ]
        refset = runner._build_refset(images, {}, 'training')
        assert refset is not None
        assert refset['n'] == 1
        assert refset['skipped']['error'] == 1  # skipped, not a KeyError
        assert set(refset['matrices']) == {'arcface', 'sface'}

    def test_selection_exclusion_round_trip_canonical_key(self, tmp_path):
        """An exclusion saved under the canonical key must be honoured by the
        worker lookup (regression guard for the P1 path-key mismatch)."""
        runner = self._runner(tmp_path)
        face = {'index': 0, 'bbox': [0, 0, 1, 1], 'det_score': 0.9,
                'embeddings': {'arcface:a': [1.0, 0.0], 'sface:b': [0.0, 1.0]}}
        images = [
            {'path': r'E:\Data\MixedCase\a.JPG', 'name': 'a.JPG', 'sha': 's1', 'status': 'ok', 'faces': [face]},
        ]
        # key as the UI writes it (normalizeRefPath form)
        selection = {canonical_ref_key(r'E:\Data\MixedCase\a.JPG'): {'excluded': True}}
        refset = runner._build_refset(images, selection, 'training')
        assert refset is None  # excluded -> empty reference set, not silently included
        refset = runner._build_refset(images, {}, 'training')
        assert refset is not None and refset['n'] == 1


# ------------------------------------------------------- worker end-to-end


class _ScratchJob:
    """A scratch output tree with external refs + samples, no DB needed."""

    def __init__(self, root, subject_images, sample_images):
        self.root = root
        self.name = 'identity_eval_test_job'
        self.external_dir = os.path.join(root, self.name, 'identity_eval', 'external_refs')
        self.samples_dir = os.path.join(root, self.name, 'samples')
        os.makedirs(self.external_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        for src in subject_images[:3]:
            shutil.copy(src, os.path.join(self.external_dir, f'{os.path.basename(src)}'))
        self.samples = []
        for src in sample_images[:4]:
            dst = os.path.join(self.samples_dir, os.path.basename(src))
            shutil.copy(src, dst)
            self.samples.append(dst)


@pytest.fixture(scope='session')
def sample_images():
    if not _SAMPLES_DIR or not os.path.isdir(_SAMPLES_DIR):
        pytest.skip(f'samples fixture dir not configured; {FIXTURES_HINT}')
    found = sorted(glob.glob(os.path.join(_SAMPLES_DIR, '*_seed-*.jpg')))
    if len(found) < 4:
        pytest.skip(f'not enough sample images in {_SAMPLES_DIR}')
    return found


@pytest.fixture(scope='session')
def scratch_job(tmp_path_factory, subject_images, sample_images):
    return _ScratchJob(str(tmp_path_factory.mktemp('identity_eval')), subject_images, sample_images)


def _run_worker(args, timeout=240):
    return subprocess.run(
        [sys.executable, '-u', os.path.join(REPO_ROOT, 'ui_scripts', 'identity_eval.py'), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


class TestWorkerEndToEnd:
    def test_refs_then_run_arcface_only(self, scratch_job):
        status = find_arcface()
        if not status.available:
            pytest.skip(f'ArcFace weights absent: {status.reason}')
        result = _run_worker(['--command', 'refs', '--job-name', scratch_job.name, '--output-root', scratch_job.root, '--device-mode', 'cpu'])
        assert result.returncode == 0, result.stderr[-2000:]
        lines = [json.loads(l) for l in result.stdout.splitlines() if l.startswith('{')]
        assert lines[-1]['type'] == 'result'
        assert lines[-1]['refs']['counts']['external']['total'] == 3
        refs_doc = json.load(open(os.path.join(scratch_job.root, scratch_job.name, 'identity_eval', 'refs.json')))
        assert refs_doc['models']['arcface']['active'] is True
        assert refs_doc['models']['sface']['active'] == find_sface(os.path.join(REPO_ROOT, 'models', 'face')).available

        args_file = os.path.join(scratch_job.root, 'args.json')
        json.dump({'samples': scratch_job.samples}, open(args_file, 'w'))
        result = _run_worker(['--command', 'run', '--job-name', scratch_job.name, '--output-root', scratch_job.root, '--device-mode', 'cpu', '--sources', 'external', '--args-file', args_file])
        assert result.returncode == 0, result.stderr[-2000:]
        results = json.load(open(os.path.join(scratch_job.root, scratch_job.name, 'identity_eval', 'results.json')))
        assert results['completed'] is True
        assert results['refsets']['external']['n'] >= 1
        assert len(results['samples']) == len(scratch_job.samples)
        statuses = {s['status'] for s in results['samples']}
        assert statuses <= {'evaluated', 'ambiguous', 'failed'}
        for sample in results['samples']:
            if sample['status'] == 'evaluated':
                mean = sample['primary']['arcface']['external']['mean']
                assert mean is None or -1.0 <= mean <= 1.0

    def test_mixed_batch_statuses(self, scratch_job, tmp_path):
        """corrupt file + no-face image + good sample -> per-sample statuses."""
        import cv2

        status = find_arcface()
        if not status.available:
            pytest.skip(f'ArcFace weights absent: {status.reason}')
        corrupt = os.path.join(scratch_job.samples_dir, 'corrupt.jpg')
        open(corrupt, 'wb').write(b'\xff\xd8not really jpeg')
        noise = os.path.join(scratch_job.samples_dir, 'noise.jpg')
        cv2.imwrite(noise, np.random.default_rng(1).uniform(0, 255, (512, 512, 3)).astype(np.uint8))
        good = scratch_job.samples[0]
        args_file = os.path.join(scratch_job.root, 'mixed_args.json')
        json.dump({'samples': [corrupt, noise, good]}, open(args_file, 'w'))
        result = _run_worker(['--command', 'run', '--job-name', scratch_job.name, '--output-root', scratch_job.root, '--device-mode', 'cpu', '--sources', 'external', '--args-file', args_file])
        assert result.returncode == 0, result.stderr[-2000:]
        results = json.load(open(os.path.join(scratch_job.root, scratch_job.name, 'identity_eval', 'results.json')))
        by_name = {os.path.basename(s['path']): s for s in results['samples']}
        assert by_name['corrupt.jpg']['status'] == 'failed'
        assert 'unreadable' in by_name['corrupt.jpg']['error']
        assert by_name['noise.jpg']['status'] == 'failed'
        assert 'no face' in by_name['noise.jpg']['warnings'][0]
        assert by_name[os.path.basename(good)]['status'] in {'evaluated', 'ambiguous'}
        assert results['completed'] is True  # batch completes despite failures

    def test_cancel_mid_run_leaves_valid_partial_results(self, scratch_job):
        status = find_arcface()
        if not status.available:
            pytest.skip(f'ArcFace weights absent: {status.reason}')
        import cv2

        # unique content per entry so the cache can't shortcut the run; enough
        # variants that the run outlives the polling loop (CPU detection is
        # ~0.1-0.3 s/image here, so 40 gives comfortable margin)
        variants = []
        source = cv2.imread(scratch_job.samples[0])
        for i in range(40):
            variant = source.copy()
            variant[:, :, 0] = np.clip(variant[:, :, 0].astype(int) + i * 6, 0, 255).astype(np.uint8)
            path = os.path.join(scratch_job.root, f'variant_{i}.jpg')
            cv2.imwrite(path, variant)
            variants.append(path)
        args_file = os.path.join(scratch_job.root, 'cancel_args.json')
        json.dump({'samples': variants}, open(args_file, 'w'))

        process = subprocess.Popen(
            [sys.executable, '-u', os.path.join(REPO_ROOT, 'ui_scripts', 'identity_eval.py'),
             '--command', 'run', '--job-name', scratch_job.name, '--output-root', scratch_job.root,
             '--device-mode', 'cpu', '--sources', 'external', '--args-file', args_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_ROOT,
        )
        results_path = os.path.join(scratch_job.root, scratch_job.name, 'identity_eval', 'results.json')
        deadline = 120.0
        import time

        start = time.time()
        saw_partial = False
        while time.time() - start < deadline:
            if os.path.isfile(results_path):
                try:
                    partial = json.load(open(results_path))
                    if len(partial.get('samples', [])) >= 1 and not partial.get('completed'):
                        saw_partial = True
                        break
                except ValueError:
                    pass
            if process.poll() is not None:
                pytest.fail('worker finished before it could be cancelled')
            time.sleep(0.2)
        assert saw_partial, 'never observed a partial results.json'
        process.kill()
        process.wait(timeout=30)
        partial = json.load(open(results_path))
        assert partial['completed'] is False
        assert 1 <= len(partial['samples']) < len(variants)
        assert os.path.isfile(os.path.join(scratch_job.root, scratch_job.name, 'identity_eval', 'progress.json'))

    def test_no_models_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match='no detector available|no recognition model'):
            FaceDetector(device='cpu', insightface_root=str(tmp_path), models_dir=str(tmp_path))


# --------------------------------------------------- CUDA inference fallback


class TestCudaInferenceFallback:
    """Spec 5.4: ANY CUDA error (including first-inference cuDNN failures,
    which load-time try_load cannot see) degrades the run to CPU with a
    recorded warning instead of killing the worker."""

    def _runner(self, tmp_path):
        runner = IdentityEvalRunner(job_name='job_fallback', output_root=str(tmp_path), db_path=None)
        runner.device = 'cuda'
        return runner

    def test_cuda_inference_error_switches_to_cpu_and_retries(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)
        calls = []

        def fake_score(path, refsets, sources):
            calls.append(path)
            if runner.device == 'cuda':
                raise RuntimeError(
                    "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : cuDNN is unavailable or disabled "
                    "for CUDA Execution Provider: LoadLibrary failed for cudnn64_9.dll with error 2"
                )
            return {'filename': path, 'status': 'evaluated'}

        def fake_load_models():
            runner.device = 'cpu'
            runner.detector = object()

        monkeypatch.setattr(runner, '_score_sample', fake_score)
        monkeypatch.setattr(runner, '_load_models', fake_load_models)
        record = runner._score_with_cuda_fallback('a.jpg', {}, [])
        assert record['status'] == 'evaluated'
        assert calls == ['a.jpg', 'a.jpg']  # failed on CUDA, retried on CPU
        assert runner.device == 'cpu'
        assert runner.device_mode == 'cpu'
        assert any('CUDA inference failed' in w and 'cudnn64_9.dll' in w for w in runner.device_warnings)
        assert any('switched to CPU' in note for note in runner.model_notes)

    def test_cpu_errors_are_not_swallowed(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)
        runner.device = 'cpu'

        def fake_score(path, refsets, sources):
            raise RuntimeError('unreadable image file')

        monkeypatch.setattr(runner, '_score_sample', fake_score)
        with pytest.raises(RuntimeError, match='unreadable image file'):
            runner._score_with_cuda_fallback('a.jpg', {}, [])

    def test_non_cuda_errors_on_gpu_raise(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)

        def fake_score(path, refsets, sources):
            raise OSError('disk full')

        monkeypatch.setattr(runner, '_score_sample', fake_score)
        with pytest.raises(OSError, match='disk full'):
            runner._score_with_cuda_fallback('a.jpg', {}, [])


class TestEnsureCudaDlls:
    def test_registers_torch_lib_idempotently(self, tmp_path, monkeypatch):
        import toolkit.identity_eval.device as device_mod

        torch_lib = tmp_path / 'Lib' / 'site-packages' / 'torch' / 'lib'
        torch_lib.mkdir(parents=True)
        (torch_lib / 'cudnn64_9.dll').write_bytes(b'x')
        fake_python = tmp_path / 'python.exe'
        fake_python.write_bytes(b'x')
        monkeypatch.setattr(device_mod, '_CUDA_DLLS_REGISTERED', False)
        monkeypatch.setattr(device_mod.sys, 'executable', str(fake_python))
        monkeypatch.setattr(device_mod.os, 'add_dll_directory', lambda p: None)
        original_path = device_mod.os.environ['PATH']
        try:
            notes = device_mod.ensure_cuda_dlls()
            assert notes and str(torch_lib) in notes[0]
            assert device_mod.os.environ['PATH'].startswith(str(torch_lib) + os.pathsep)
            # second call is a no-op (PATH not prefixed twice)
            again = device_mod.ensure_cuda_dlls()
            assert again == []
            assert device_mod.os.environ['PATH'].count(str(torch_lib)) == 1
        finally:
            device_mod.os.environ['PATH'] = original_path

    def test_missing_torch_is_silent_noop(self, tmp_path, monkeypatch):
        import toolkit.identity_eval.device as device_mod

        monkeypatch.setattr(device_mod, '_CUDA_DLLS_REGISTERED', False)
        monkeypatch.setattr(device_mod.sys, 'executable', str(tmp_path / 'python.exe'))
        assert device_mod.ensure_cuda_dlls() == []


class TestWorkerErrorSurfacing:
    def test_cli_failure_lands_in_progress_json(self, tmp_path):
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        worker = os.path.join(repo_root, 'ui_scripts', 'identity_eval.py')
        result = subprocess.run(
            [
                sys.executable, '-u', worker,
                '--command', 'run',
                '--job-name', 'job_err',
                '--output-root', str(tmp_path),
                '--device-mode', 'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root,
        )
        assert result.returncode == 1
        progress_path = tmp_path / 'job_err' / 'identity_eval' / 'progress.json'
        assert progress_path.is_file()
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        assert progress.get('running') is False
        assert progress.get('error')
