"""Orchestration for identity evaluation: reference discovery and sample scoring.

One runner instance serves one CLI invocation (either ``refs`` or ``run``);
models load lazily inside :meth:`discover_refs` / :meth:`run` so nothing is
resident during normal sample viewing. Results are written incrementally
and atomically, so killing the worker mid-run leaves a valid partial
``results.json``.

Storage layout (all under ``output/<job>/identity_eval/``)::

    refs.json          discovery output (per-image face status)
    selection.json     user exclusions + face picks (edited via the API)
    external_refs/     user-added reference images, sha256-named
    emb_cache/         per-image detection + embedding cache
    results.json       per-sample scores (partial-valid)
    progress.json      live progress for the status route
"""

import datetime
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from . import refs as refs_mod
from .arcface import ArcFaceEmbedder
from .cache import EmbeddingCache
from .detector import DetectedFace, FaceDetector
from .device import ensure_cuda_dlls, resolve_device
from .models import find_arcface, find_sface, sha256_file
from .scoring import loo_band, parse_sample_filename, round3, score_face
from .sface import SFaceEmbedder

# v2: sample faces gained per-face ``kps`` (5-point landmarks) so the
# Phase 2d pose signal can run in post-processing. Nothing validates this
# on read; older documents simply lack the field and the pose signal skips.
RESULTS_VERSION = 2


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def _atomic_json(path: str, payload) -> None:
    tmp = path + '.tmp'
    import json

    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def _read_json(path: str):
    import json

    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def try_load(loader, what: str, device: str, on_warning: Callable[[str], None]):
    """Load a model on ``device``; a CUDA failure falls back to CPU.

    Returns ``(model, error)`` -- exactly one is None. Fallbacks are
    reported through ``on_warning`` so they land in the result's device
    warnings.
    """
    try:
        return loader(device), None
    except Exception as e:  # noqa: BLE001 -- CUDA init/OOM degrade to CPU
        if device == 'cpu':
            return None, f'{what} load failed: {e}'
        on_warning(f'{what} CUDA load failed ({e}); retrying on CPU')
        try:
            return loader('cpu'), None
        except Exception as e2:  # noqa: BLE001
            return None, f'{what} load failed on CPU too: {e2}'


class IdentityEvalRunner:
    def __init__(
        self,
        job_name: str,
        output_root: str,
        models_dir: Optional[str] = None,
        device_mode: str = 'auto',
        db_path: Optional[str] = None,
        insightface_root: Optional[str] = None,
        emit: Optional[Callable[[dict], None]] = None,
    ):
        self.job_name = job_name
        self.output_root = output_root
        self.models_dir = models_dir or os.path.join(os.getcwd(), 'models', 'face')
        self.device_mode = device_mode
        self.db_path = db_path
        self.insightface_root = insightface_root
        self.emit = emit or (lambda payload: None)

        self.job_dir = os.path.join(output_root, job_name)
        self.identity_dir = os.path.join(self.job_dir, 'identity_eval')
        self.external_dir = os.path.join(self.identity_dir, 'external_refs')
        self.cache_dir = os.path.join(self.identity_dir, 'emb_cache')
        self.refs_path = os.path.join(self.identity_dir, 'refs.json')
        self.results_path = os.path.join(self.identity_dir, 'results.json')
        self.progress_path = os.path.join(self.identity_dir, 'progress.json')
        os.makedirs(self.external_dir, exist_ok=True)
        self.cache = EmbeddingCache(self.cache_dir)

        # populated by _load_models
        self.device: Optional[str] = None
        self.gpu_id: int = 0
        self.device_warnings: List[str] = []
        self.detector: Optional[FaceDetector] = None
        self.arcface: Optional[ArcFaceEmbedder] = None
        self.sface: Optional[SFaceEmbedder] = None
        self.model_notes: List[str] = []

    # ------------------------------------------------------------------ models

    def _load_models(self) -> None:
        """Resolve the device, then load detector + active embedders.

        Any CUDA load failure falls back to CPU for that component with a
        recorded warning (never crashes the whole run).
        """
        if self.detector is not None:
            return
        device, warnings, gpu_id = resolve_device(self.device_mode, self.db_path, self.job_name)
        self.device_warnings = list(warnings)
        self.device = device
        self.gpu_id = gpu_id
        if device != 'cpu':
            # cuDNN lives in torch/lib (the worker never imports torch); ORT
            # sessions create fine without it and then die on the first Conv
            self.model_notes.extend(ensure_cuda_dlls())

        def make_detector(dev):
            return FaceDetector(
                device=dev,
                insightface_root=self.insightface_root,
                models_dir=self.models_dir,
                gpu_id=self.gpu_id,
            )

        detector, error = try_load(make_detector, 'face detector', device, self.device_warnings.append)
        if detector is None:
            raise RuntimeError(error or 'no face detector available')
        self.detector = detector
        if detector.backend == 'yunet':
            self.model_notes.append('detection via YuNet (buffalo_l absent)')

        arcface_status = find_arcface(self.insightface_root)
        if arcface_status.available:
            embedder, error = try_load(
                lambda dev: ArcFaceEmbedder(device=dev, insightface_root=self.insightface_root, gpu_id=self.gpu_id),
                'ArcFace',
                device,
                self.device_warnings.append,
            )
            if embedder is not None:
                self.arcface = embedder
            else:
                self.device_warnings.append(error)
        sface_status = find_sface(self.models_dir)
        if sface_status.available:
            embedder, error = try_load(
                lambda dev: SFaceEmbedder(self.models_dir, device=dev, gpu_id=self.gpu_id),
                'SFace',
                device,
                self.device_warnings.append,
            )
            if embedder is not None:
                self.sface = embedder
                self.device_warnings.extend(embedder.status.notes)
            else:
                self.device_warnings.append(error)
        if self.arcface is None and self.sface is None:
            raise RuntimeError(
                'no recognition model available: ArcFace needs ~/.insightface/models/buffalo_l, '
                f'SFace needs {os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")} '
                '(see docs/identity_eval.md)'
            )

    def _embedders(self) -> dict:
        out = {}
        if self.arcface is not None:
            out['arcface'] = self.arcface
        if self.sface is not None:
            out['sface'] = self.sface
        return out

    def _refs_missing_models(self, refs_doc: dict) -> bool:
        """True when any reference face lacks an embedding for a currently
        active model (discovery predates installing that model)."""
        signatures = [embedder.model_sig for embedder in self._embedders().values()]
        for source in ('training', 'external'):
            for record in refs_doc.get(source, {}).get('images', []):
                for face in record.get('faces', []):
                    embeddings = face.get('embeddings') or {}
                    if any(sig not in embeddings for sig in signatures):
                        return True
        return False

    def _models_report(self) -> dict:
        arcface_status = find_arcface(self.insightface_root)
        sface_status = find_sface(self.models_dir)
        return {
            'arcface': {**arcface_status.to_dict(), 'active': self.arcface is not None},
            'sface': {**sface_status.to_dict(), 'active': self.sface is not None},
            'detector': self.detector.backend if self.detector else None,
            'notes': self.model_notes,
        }

    # ------------------------------------------------------------------ images

    def _read_image(self, path: str):
        """(sha256, BGR ndarray or None) -- one file read for both."""
        import cv2

        try:
            with open(path, 'rb') as handle:
                data = handle.read()
        except OSError:
            return None, None
        image_sha = sha256_file(path)
        if not data:
            return image_sha, None
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)  # BGR, passthrough to models
        return image_sha, image

    def _face_vectors(self, face_entry: dict) -> Dict[str, np.ndarray]:
        return {
            name: vec
            for name, vec in (
                (name, self.cache.get_embedding(face_entry, embedder.model_sig))
                for name, embedder in self._embedders().items()
            )
            if vec is not None
        }

    def _process_image(self, path: str) -> dict:
        """Detection + embeddings for one image, cache-first.

        Returns ``{sha, status, error, entry}`` where status is one of
        ok / no_face / multi_face / error and entry is the saved cache entry.
        """
        image_sha, image = self._read_image(path)
        if image is None:
            return {'sha': image_sha, 'status': 'error', 'error': 'unreadable image file', 'entry': None}
        entry = self.cache.load(image_sha, self.detector.det_sig)
        fresh = entry is None
        if fresh:
            detection = self.detector.detect(image)
            entry = self.cache.entry_from_detection(image_sha, self.detector.det_sig, detection)
        recomputed = False
        for face_entry in entry.get('faces', []):
            missing = [
                name
                for name, embedder in self._embedders().items()
                if self.cache.get_embedding(face_entry, embedder.model_sig) is None
            ]
            if not missing:
                continue
            recomputed = True
            face = DetectedFace(
                index=face_entry['index'],
                bbox=tuple(face_entry['bbox']),
                kps=np.asarray(face_entry['kps'], dtype=np.float32),
                det_score=face_entry['det_score'],
            )
            for name in missing:
                embedding = self._embedders()[name].embed(image, face)
                self.cache.set_embedding(face_entry, self._embedders()[name].model_sig, embedding)
        if fresh or recomputed:
            self.cache.save(entry)
        face_count = len(entry.get('faces', [])) if entry else 0
        status = 'ok' if face_count == 1 else 'no_face' if face_count == 0 else 'multi_face'
        return {'sha': image_sha, 'status': status, 'error': None, 'entry': entry}

    # ------------------------------------------------------------------ refs

    def load_job_config(self) -> Optional[dict]:
        import sqlite3

        if not self.db_path or not os.path.isfile(self.db_path):
            return None
        try:
            connection = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True, timeout=5.0)
            try:
                row = connection.execute('SELECT job_config FROM Job WHERE name = ?', (self.job_name,)).fetchone()
            finally:
                connection.close()
        except Exception:  # noqa: BLE001
            return None
        if not row or not row[0]:
            return None
        import json

        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def discover_refs(self) -> dict:
        """Detect faces in every training + external reference image."""
        self._load_models()
        started = _now()
        self._write_progress({'phase': 'refs', 'done': 0, 'total': 0, 'running': True, 'started_at': started})

        config = self.load_job_config()
        training = refs_mod.discover_training_refs(config)
        training_paths = training['images']
        external_paths = [
            os.path.join(self.external_dir, name)
            for name in sorted(os.listdir(self.external_dir))
            if name.lower().endswith(refs_mod.IMAGE_EXTENSIONS)
        ]

        selection = refs_mod.load_selection(refs_mod.selection_path(self.identity_dir))
        selection = refs_mod.prune_selection(
            selection,
            training_paths,
            [os.path.basename(p) for p in external_paths],
        )
        refs_mod.save_selection(refs_mod.selection_path(self.identity_dir), selection)

        total = len(training_paths) + len(external_paths)
        self._write_progress({'phase': 'refs', 'done': 0, 'total': total, 'running': True, 'started_at': started})
        overall = [0]

        def run_batch(paths, source):
            images = []
            for path in paths:
                result = self._process_image(path)
                record = {
                    'path': os.path.abspath(path),
                    'name': os.path.basename(path),
                    'sha': result['sha'],
                    'status': {'ok': 'ok', 'multi_face': 'needs_review', 'no_face': 'no_face', 'error': 'error'}[result['status']],
                    'error': result['error'],
                    'faces': [face for face in (result['entry'] or {}).get('faces', [])],
                }
                images.append(record)
                overall[0] += 1
                self.emit({'type': 'progress', 'phase': 'refs', 'source': source, 'done': overall[0], 'total': total})
                self._write_progress(
                    {'phase': 'refs', 'done': overall[0], 'total': total, 'running': True, 'started_at': started}
                )
            return images

        training_images = run_batch(training_paths, 'training')
        external_images = run_batch(external_paths, 'external')
        document = {
            'version': RESULTS_VERSION,
            'job': self.job_name,
            'generated_at': _now(),
            'device': {'requested': self.device_mode, 'resolved': self.device, 'gpu': self.gpu_id, 'warnings': self.device_warnings},
            'models': self._models_report(),
            'training': {'datasets': training['datasets'], 'images': training_images, 'warnings': training['warnings']},
            'external': {'images': external_images},
            'warnings': list(self.device_warnings),
        }
        _atomic_json(self.refs_path, document)
        self._write_progress({'phase': 'refs', 'done': total, 'total': total, 'running': False, 'started_at': started})
        self.emit({'type': 'done', 'command': 'refs', 'refs_path': self.refs_path, 'training': len(training_images), 'external': len(external_images)})
        return document

    # ------------------------------------------------------------------ run

    def _build_refset(self, images: List[dict], selection_entries: dict, source: str) -> Optional[dict]:
        """Reference matrices per model from discovery records + selection.

        Training-ref selection keys use :func:`canonical_ref_key` -- the
        same lowercase/forward-slash form the UI writes and reads.
        """
        embedders = self._embedders()
        vectors: Dict[str, List[np.ndarray]] = {name: [] for name in embedders}
        included = 0
        skipped = {'excluded': 0, 'needs_review': 0, 'no_face': 0, 'error': 0}
        hash_parts = []
        for record in images:
            key = record['name'] if source == 'external' else refs_mod.canonical_ref_key(record['path'])
            choice = selection_entries.get(key) or {}
            if choice.get('excluded'):
                skipped['excluded'] += 1
                continue
            if record['status'] == 'error' or record['status'] == 'no_face':
                skipped[record['status']] += 1
                continue
            if record['status'] == 'needs_review':
                pick = choice.get('face_index')
                if not isinstance(pick, int) or pick < 0 or pick >= len(record['faces']):
                    skipped['needs_review'] += 1
                    continue
                face_index = pick
            else:
                face_index = 0
            face_entry = record['faces'][face_index]
            face_vectors = self._face_vectors(face_entry)
            # every active model must have an embedding for this face; a
            # record missing one is skipped, not crashed on (stale refs.json
            # after installing a model is refreshed before we get here, so
            # this only catches unreadable images)
            if not face_vectors or any(name not in face_vectors for name in embedders):
                skipped['error'] += 1
                continue
            for name in embedders:
                vectors[name].append(face_vectors[name])
            included += 1
            hash_parts.append(f"{record['sha']}:{face_index}")
        if included == 0:
            return None
        import hashlib
        import json as json_mod

        refset_hash = hashlib.sha256(json_mod.dumps(sorted(hash_parts)).encode('utf-8')).hexdigest()[:16]
        matrices = {name: np.stack(vecs) for name, vecs in vectors.items() if vecs}
        return {
            'n': included,
            'hash': refset_hash,
            'skipped': skipped,
            'matrices': matrices,
            'loo': {name: round3(loo_band(matrix)) for name, matrix in matrices.items()},
        }

    def _switch_to_cpu(self, reason: str) -> None:
        """Drop all loaded models and reload them on CPU.

        Used when CUDA fails at inference time (e.g. cuDNN DLL missing from
        the search path): session creation succeeds, so :meth:`try_load`'s
        load-time fallback never fires -- the error only surfaces on the
        first Conv node. Embedding cache keys carry no device component, so
        anything embedded before the switch stays valid.
        """
        self.device_warnings.append(reason)
        self.model_notes.append('device switched to CPU after a CUDA inference failure')
        self.device_mode = 'cpu'
        self.device = None
        self.detector = None
        self.arcface = None
        self.sface = None
        self._load_models()

    def _score_with_cuda_fallback(self, path: str, refsets: Dict[str, Optional[dict]], sources: List[str]) -> dict:
        """Score one sample, degrading to CPU if CUDA fails mid-inference."""
        try:
            return self._score_sample(path, refsets, sources)
        except Exception as e:  # noqa: BLE001 -- CUDA inference errors degrade, others raise
            text = str(e)
            if self.device == 'cpu' or not ('cuda' in text.lower() or 'onnxruntime' in text.lower()):
                raise
            self._switch_to_cpu(f'CUDA inference failed ({text.strip()[:200]}); continuing on CPU')
            return self._score_sample(path, refsets, sources)

    def run(self, sample_paths: List[str], sources: Optional[List[str]] = None) -> dict:
        """Score samples against the selected reference sets."""
        self._load_models()
        sources = [s for s in (sources or ['training']) if s in ('training', 'external')]
        if not sources:
            raise ValueError("sources must include 'training' and/or 'external'")
        refs_doc = _read_json(self.refs_path)
        if not refs_doc:
            raise RuntimeError('no reference discovery found; run refs discovery first')
        if self._refs_missing_models(refs_doc):
            # e.g. SFace was installed after discovery: reference records
            # predate the newly active model. Re-run discovery (cache-first,
            # only the missing model's embeddings are computed) instead of
            # crashing on the stale records.
            self.emit({'type': 'progress', 'phase': 'refs', 'done': 0, 'total': 0, 'reason': 'newly active model; refreshing reference embeddings'})
            refs_doc = self.discover_refs()
        if not sample_paths:
            raise ValueError('no samples to evaluate')

        started = _now()
        self._write_progress({'phase': 'samples', 'done': 0, 'total': len(sample_paths), 'running': True, 'started_at': started})
        self.emit({'type': 'started', 'phase': 'samples', 'total': len(sample_paths), 'device': self.device})

        selection = refs_mod.load_selection(refs_mod.selection_path(self.identity_dir))
        refsets: Dict[str, Optional[dict]] = {}
        for source in sources:
            images = refs_doc.get(source, {}).get('images', [])
            entries = selection.get('external_refs' if source == 'external' else 'training_refs', {})
            refsets[source] = self._build_refset(images, entries, source)
        if len(sources) == 2:
            combined_images = list(refs_doc.get('training', {}).get('images', [])) + list(refs_doc.get('external', {}).get('images', []))
            combined_entries = {**selection.get('training_refs', {}), **selection.get('external_refs', {})}
            refsets['combined'] = self._build_refset(combined_images, combined_entries, 'combined')
        else:
            refsets['combined'] = None

        active_sources = [s for s in sources if refsets.get(s)] or []
        if not active_sources and not refsets.get('combined'):
            raise RuntimeError(
                'reference set is empty after exclusions; enable references or pick faces for multi-face ones'
            )

        results = {
            'version': RESULTS_VERSION,
            'job': self.job_name,
            'created_at': started,
            'updated_at': started,
            'completed': False,
            'sources': active_sources,
            'device': {'requested': self.device_mode, 'resolved': self.device, 'gpu': self.gpu_id, 'warnings': self.device_warnings},
            'models': self._models_report(),
            'refsets': {
                name: (
                    {
                        'n': refset['n'],
                        'hash': refset['hash'],
                        'skipped': refset['skipped'],
                        'loo': refset['loo'],
                    }
                    if refset
                    else None
                )
                for name, refset in refsets.items()
            },
            'samples': [],
            'warnings': list(self.device_warnings),
        }

        for done, path in enumerate(sample_paths, start=1):
            results['samples'].append(self._score_with_cuda_fallback(path, refsets, active_sources))
            results['updated_at'] = _now()
            # a mid-run CUDA failure may have switched devices; keep the
            # recorded device + warnings in sync with what actually ran
            results['device'] = {
                'requested': self.device_mode,
                'resolved': self.device,
                'gpu': self.gpu_id,
                'warnings': self.device_warnings,
            }
            results['warnings'] = list(self.device_warnings)
            _atomic_json(self.results_path, results)
            self.emit({'type': 'progress', 'phase': 'samples', 'done': done, 'total': len(sample_paths)})
            self._write_progress(
                {'phase': 'samples', 'done': done, 'total': len(sample_paths), 'running': True, 'started_at': started}
            )

        results['completed'] = True
        results['updated_at'] = _now()
        _atomic_json(self.results_path, results)
        self._write_progress({'phase': 'samples', 'done': len(sample_paths), 'total': len(sample_paths), 'running': False, 'started_at': started})
        self.emit({'type': 'done', 'command': 'run', 'results_path': self.results_path, 'samples': len(sample_paths)})
        return results

    def _score_sample(self, path: str, refsets: Dict[str, Optional[dict]], sources: List[str]) -> dict:
        parsed = parse_sample_filename(path)
        record = {
            'path': os.path.abspath(path),
            'filename': os.path.basename(path),
            'file_sha': None,
            'step': parsed['step'] if parsed else None,
            'prompt_idx': parsed['prompt_idx'] if parsed else None,
            'seed': parsed['seed'] if parsed else None,
            'is_raw': parsed['is_raw'] if parsed else False,
            'status': 'failed',
            'face_count': 0,
            'error': None,
            'primary': None,
            'faces': [],
            'warnings': [],
        }
        processed = self._process_image(path)
        record['file_sha'] = processed['sha']
        entry = processed['entry']
        if processed['status'] == 'error':
            record['error'] = processed['error']
            record['warnings'].append(processed['error'])
            return record
        faces = entry.get('faces', [])
        record['face_count'] = len(faces)
        if processed['status'] == 'no_face':
            record['warnings'].append('no face detected')
            return record

        for face_entry in faces:
            vectors = self._face_vectors(face_entry)
            scores = {
                name: {
                    source: (score_face(refsets[source]['matrices'][name], vec) if refsets.get(source) and name in refsets[source]['matrices'] and vec is not None else {'mean': None, 'max': None})
                    for source in (sources + (['combined'] if refsets.get('combined') else []))
                }
                for name, vec in vectors.items()
            }
            record['faces'].append(
                {
                    'index': face_entry['index'],
                    'bbox': face_entry['bbox'],
                    # landmarks persist per sample face (Phase 2d pose signal);
                    # refs have always carried them, samples previously dropped them
                    'kps': face_entry.get('kps'),
                    'det_score': face_entry['det_score'],
                    'scores': scores,
                }
            )
        record['status'] = 'evaluated' if len(faces) == 1 else 'ambiguous'
        if len(faces) > 1:
            record['warnings'].append('multiple faces; scores shown per face (Phase 2 adds a subject face picker)')
        if record['status'] == 'evaluated':
            record['primary'] = record['faces'][0]['scores']
        return record

    # ------------------------------------------------------------------ misc

    def _write_progress(self, payload: dict) -> None:
        document = {**payload, 'job': self.job_name, 'updated_at': _now()}
        _atomic_json(self.progress_path, document)
