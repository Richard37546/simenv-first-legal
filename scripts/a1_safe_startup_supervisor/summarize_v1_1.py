#!/usr/bin/env python3
"""Offline V1.1 archive aggregation.  It imports no ROS and controls nothing."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = pathlib.Path('/home/richard/simenv_official_clean')
OUT = ROOT / 'debug/a1_safe_startup_supervisor_v1_1'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def truth_summary(run_dir):
    path = run_dir / 'truth_offline.json'
    if not path.exists():
        return {'available': False}
    samples = load(path).get('truth', [])
    if not samples:
        return {'available': True, 'samples': 0}
    z = [item['position'][2] for item in samples]
    upright = [item['upright_score'] for item in samples]
    return {'available': True, 'samples': len(samples), 'height_min_m': min(z), 'height_max_m': max(z),
            'upright_min': min(upright), 'upright_max': max(upright),
            'truth_used_for_control': False}


def entry(manifest, events, timeline):
    run_dir = pathlib.Path(manifest['run_dir'])
    fixed = bool(manifest['resolved_parameters'].get('fixedstand_only'))
    delayed = manifest['resolved_parameters'].get('perception_delay_sim_sec', 0.0) > 0
    return {
        'run_id': manifest['run_id'], 'run_dir': str(run_dir), 'kind': 'fixedstand' if fixed else ('perception_delay' if delayed else 'full'),
        'final_state': manifest.get('final_state'), 'final_reason': manifest.get('final_reason'),
        'navigation_ready_final': manifest.get('navigation_ready_final'), 'fixedstand_only_passed': manifest.get('fixedstand_only_passed'),
        'events': events, 'truth_offline': truth_summary(run_dir), 'timeline_points': len(timeline),
        'nonzero_cmd_seen': bool(events.get('nonzero_cmd_events')), 'health_last': timeline[-1].get('health', {}) if timeline else {},
    }


def plot(runs):
    fig, axis = plt.subplots(figsize=(13, 4))
    for index, run in enumerate(runs):
        e = run['events']; start = e.get('controller_ready_sim'); end = e.get('first_fixedstand_effective_sim')
        if start is not None and end is not None:
            axis.barh(index, end - start, left=start, label=run['run_id'])
    axis.set_xlabel('simulation time (s)'); axis.set_ylabel('run'); axis.set_title('controller-ready to FixedStand-effective dwell')
    if runs: axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'passive_dwell_comparison.png', dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 4))
    for run in runs:
        truth_path = pathlib.Path(run['run_dir']) / 'truth_offline.json'
        if truth_path.exists():
            samples = load(truth_path).get('truth', [])
            if samples:
                axis.plot([item['sim_time'] for item in samples], [item['upright_score'] for item in samples], label=run['run_id'])
    axis.set_ylim(-1.05, 1.05); axis.set_xlabel('simulation time (s)'); axis.set_ylabel('truth upright score')
    if axis.lines: axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'offline_truth_upright_comparison.png', dpi=140); plt.close(fig)


def main():
    runs = []
    for manifest_path in sorted((OUT / 'runs').glob('*/manifest.json')):
        run_dir = manifest_path.parent; events_path = run_dir / 'events.json'; timeline_path = run_dir / 'timeline.json'
        if events_path.exists() and timeline_path.exists():
            runs.append(entry(load(manifest_path), load(events_path), load(timeline_path)))
    fixed = [item for item in runs if item['kind'] == 'fixedstand']
    full = [item for item in runs if item['kind'] == 'full']
    delayed = [item for item in runs if item['kind'] == 'perception_delay']
    write('fixedstand_runs.json', {'runs': fixed})
    write('full_startup_runs.json', {'runs': full})
    write('perception_delay_runs.json', {'runs': delayed})
    write('passive_dwell_metrics.json', {'policy': {'limit_sim_sec': 1.0, 'basis': 'successful evidence 0.227/0.503 s; historical risk about 2.77 s; controller-ready to mode-effective is gated below 1.0 s'}, 'runs': runs})
    plot(runs)


if __name__ == '__main__':
    main()
