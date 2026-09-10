#!/usr/bin/env python3
"""Write non-destructive V1.1 after-hashes and patch evidence."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path('/home/richard/simenv_official_clean')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    patches = sorted((ROOT / 'debug/patches').glob('a1_safe_startup_supervisor_v1_1_*'))
    if not patches:
        raise RuntimeError('baseline directory missing')
    out = patches[-1]
    targets = [
        ROOT / 'scripts/a1_safe_startup_supervisor/supervisor_core.py',
        ROOT / 'scripts/a1_safe_startup_supervisor/a1_safe_startup_supervisor.py',
        ROOT / 'scripts/a1_safe_startup_supervisor/test_supervisor_core.py',
        ROOT / 'scripts/a1_safe_startup_supervisor/run_acceptance.sh',
    ]
    data = {
        'target_sha256_after': {str(path.relative_to(ROOT)): sha256(path) for path in targets},
        'binary_sha256_after': {
            'devel/lib/unitree_guide/junior_ctrl': sha256(ROOT / 'devel/lib/unitree_guide/junior_ctrl'),
            'devel/lib/liblivox_laser_simulation.so': sha256(ROOT / 'devel/lib/liblivox_laser_simulation.so'),
        },
    }
    data['new_v1_1_files'] = [
        'scripts/a1_safe_startup_supervisor/collect_offline_truth.py',
        'scripts/a1_safe_startup_supervisor/summarize_v1_1.py',
        'scripts/a1_safe_startup_supervisor/finalize_v1_1_artifacts.py',
        'docs/a1_safe_startup_state_machine_v1_1.md',
        'audit_reports/a1_safe_startup_supervisor_v1_1_report.md',
    ]
    (out / 'after_manifest.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    chunks = []
    for target in targets:
        before = out / target.name
        result = subprocess.run(['diff', '-u', '--label', str(target.relative_to(ROOT)) + ' (before)', str(before),
                                 '--label', str(target.relative_to(ROOT)) + ' (after)', str(target)], text=True,
                                stdout=subprocess.PIPE, check=False)
        chunks.append(result.stdout)
    (out / 'v1_1_diff.patch').write_text(''.join(chunks), encoding='utf-8')


if __name__ == '__main__':
    main()
