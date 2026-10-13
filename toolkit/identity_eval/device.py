"""Device selection policy for the identity-eval worker.

``auto`` (default): CPU whenever a job is running on this job's GPU, CUDA
when that GPU has more than 4 GB free, otherwise CPU. ``gpu``/``cpu``
force a choice; a forced CUDA that fails to initialize falls back to CPU
with a recorded warning. ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is set so
nvidia-smi indices match onnxruntime ``device_id`` values.

Job state comes from the toolkit SQLite DB (Job.status + Job.gpu_ids);
free VRAM from nvidia-smi. Both lookups are tolerant: if either can't be
read, auto degrades to CPU rather than risking a training OOM.
"""

import os
import subprocess
from typing import List, Optional, Tuple

# keep nvidia-smi indices aligned with ORT device ids (set before ORT init)
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

FREE_VRAM_MB_THRESHOLD = 4096


def choose_device(mode: str, job_running_on_gpu: Optional[bool], free_vram_mb: Optional[int]) -> str:
    """Pure policy: returns 'cuda' or 'cpu'.

    Unknown GPU state (None) is treated conservatively (CPU) because a
    wrong guess competes with training for VRAM.
    """
    mode = (mode or 'auto').lower()
    if mode == 'cpu':
        return 'cpu'
    if mode == 'gpu':
        return 'cuda'
    if job_running_on_gpu is None and free_vram_mb is None:
        return 'cpu'
    if job_running_on_gpu:
        return 'cpu'
    if free_vram_mb is not None and free_vram_mb > FREE_VRAM_MB_THRESHOLD:
        return 'cuda'
    return 'cpu'


def parse_gpu_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    ids = []
    for part in str(raw).replace(';', ',').split(','):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids


def read_job_gpu_state(db_path: Optional[str], job_name: str) -> Tuple[Optional[bool], List[int]]:
    """(job_running_on_job_gpu, job_gpu_ids) from the SQLite Job table.

    Returns (None, []) when the DB is unreadable so callers treat GPU
    state as unknown.
    """
    job_gpus: List[int] = []
    if not db_path or not os.path.isfile(db_path):
        return None, job_gpus
    try:
        import sqlite3

        connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=5.0)
        try:
            rows = connection.execute('SELECT name, gpu_ids, status FROM Job').fetchall()
        finally:
            connection.close()
    except Exception:  # noqa: BLE001 -- DB schema/lock issues degrade to unknown
        return None, job_gpus
    for name, gpu_ids, status in rows:
        if name == job_name:
            job_gpus = parse_gpu_ids(gpu_ids)
    if not job_gpus:
        job_gpus = [0]
    for name, gpu_ids, status in rows:
        if name == job_name:
            continue
        if str(status).lower() == 'running' and set(parse_gpu_ids(gpu_ids)) & set(job_gpus):
            return True, job_gpus
    return False, job_gpus


def query_free_vram_mb(gpu_id: int) -> Optional[int]:
    """Free VRAM (MiB) for one GPU via nvidia-smi; None when unavailable."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                f'--id={gpu_id}',
                '--query-gpu=memory.free',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'},
        )
        if result.returncode != 0:
            return None
        return int(float(result.stdout.strip().splitlines()[0]))
    except Exception:  # noqa: BLE001 -- no nvidia-smi / timeout / parse error
        return None


def resolve_device(mode: str, db_path: Optional[str], job_name: str) -> Tuple[str, List[str]]:
    """Apply the policy with live job/VRAM state. Returns (device, warnings)."""
    warnings: List[str] = []
    mode = (mode or 'auto').lower()
    job_running, job_gpus = read_job_gpu_state(db_path, job_name)
    free_vram = query_free_vram_mb(job_gpus[0]) if job_gpus else None
    device = choose_device(mode, job_running, free_vram)
    if mode == 'auto':
        if job_running:
            warnings.append('a training job is running on this GPU; using CPU')
        elif device == 'cpu' and free_vram is not None:
            warnings.append(f'only {free_vram} MiB VRAM free (need > {FREE_VRAM_MB_THRESHOLD}); using CPU')
        elif free_vram is None:
            warnings.append('GPU state unavailable; using CPU')
    if mode == 'gpu' and job_running:
        warnings.append('GPU forced while a training job is running on this GPU; training may run out of VRAM')
    return device, warnings
