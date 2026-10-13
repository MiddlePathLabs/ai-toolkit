"""Identity-eval worker CLI (fork-only feature).

A thin CLI over ``toolkit.identity_eval``. The UI routes spawn this script
(never a shell) and read two streams:

- stdout: JSON-lines progress ``{"type": "progress"|"warning"|"done"|"error", ...}``
- final stdout line: JSON result payload

All state lives under ``output/<job>/identity_eval/``; partial results stay
valid if the process is killed (cancel = kill, no graceful shutdown needed).

Usage:
  python ui_scripts/identity_eval.py --command refs  --job-name NAME --output-root DIR [...]
  python ui_scripts/identity_eval.py --command run   --job-name NAME --output-root DIR \
      --sources training,external --args-file run_args.json
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('HF_HUB_OFFLINE', '1')


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Identity evaluation worker')
    parser.add_argument('--command', choices=['refs', 'run'], required=True)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--output-root', required=True, help='training/output root')
    parser.add_argument('--models-dir', default=os.path.join(REPO_ROOT, 'models', 'face'))
    parser.add_argument('--device-mode', choices=['auto', 'gpu', 'cpu'], default='auto')
    parser.add_argument('--db', default=os.path.join(REPO_ROOT, 'aitk_db.db'), help='toolkit SQLite DB path')
    parser.add_argument('--sources', default='training', help='comma-separated: training,external')
    parser.add_argument('--args-file', default=None, help='JSON file: {"samples": [paths]} for run')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from toolkit.identity_eval import IdentityEvalRunner

    runner = IdentityEvalRunner(
        job_name=args.job_name,
        output_root=args.output_root,
        models_dir=args.models_dir,
        device_mode=args.device_mode,
        db_path=args.db,
        emit=emit,
    )
    try:
        if args.command == 'refs':
            document = runner.discover_refs()
            emit({'type': 'result', 'command': 'refs', 'refs': _summarize_refs(document)})
            return 0
        samples = []
        if args.args_file:
            with open(args.args_file, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            samples = [str(p) for p in payload.get('samples', []) if isinstance(p, str)]
        sources = [s.strip() for s in args.sources.split(',') if s.strip() in ('training', 'external')]
        results = runner.run(samples, sources)
        emit(
            {
                'type': 'result',
                'command': 'run',
                'completed': results['completed'],
                'samples': len(results['samples']),
                'refsets': results['refsets'],
            }
        )
        return 0
    except Exception as e:  # noqa: BLE001 -- worker reports errors as JSON, not tracebacks
        emit({'type': 'error', 'command': args.command, 'message': str(e)})
        return 1


def _summarize_refs(document: dict) -> dict:
    counts = {}
    for source in ('training', 'external'):
        images = document.get(source, {}).get('images', [])
        counts[source] = {
            'total': len(images),
            'ok': sum(1 for i in images if i['status'] == 'ok'),
            'needs_review': sum(1 for i in images if i['status'] == 'needs_review'),
            'no_face': sum(1 for i in images if i['status'] == 'no_face'),
            'error': sum(1 for i in images if i['status'] == 'error'),
        }
    return {'counts': counts, 'models': document.get('models')}


if __name__ == '__main__':
    sys.exit(main())
