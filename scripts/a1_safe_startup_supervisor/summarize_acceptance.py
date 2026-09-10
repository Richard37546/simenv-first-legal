#!/usr/bin/env python3
"""Offline-only aggregation of safe-startup evidence.  It imports no ROS."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path('/home/richard/simenv_official_clean')
OUT = ROOT / 'debug/a1_safe_startup_supervisor_v1'


def load(path): return json.loads(path.read_text(encoding='utf-8'))
def write(name, data): (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def chart_state_machine():
    labels = ['WAIT_GAZEBO', 'WAIT_CONTROLLER', 'START_CTRL', 'FIXEDSTAND', 'PERCEPTION', 'RL_ZERO_HOLD', 'NAV_READY']
    fig, ax = plt.subplots(figsize=(13, 2.5)); ax.axis('off')
    for i, label in enumerate(labels):
        ax.text(i, .5, label, ha='center', va='center', bbox={'boxstyle': 'round', 'fc': '#e9f2f6'})
        if i: ax.annotate('', (i-.18,.5), (i-.82,.5), arrowprops={'arrowstyle': '->'})
    ax.set_xlim(-.8, len(labels)-.2); ax.set_ylim(0, 1); fig.tight_layout(); fig.savefig(OUT / 'startup_state_machine.png', dpi=140); plt.close(fig)


def main():
    runs = []
    for manifest_path in sorted((OUT / 'runs').glob('*/manifest.json')):
        timeline_path = manifest_path.parent / 'timeline.json'
        if timeline_path.exists(): runs.append((load(manifest_path), load(timeline_path)))
    fixed = [(m, t) for m, t in runs if m['resolved_parameters'].get('fixedstand_only')]
    full = [(m, t) for m, t in runs if not m['resolved_parameters'].get('fixedstand_only')]
    write('fixedstand_runs.json', {'runs': [m for m, _ in fixed]})
    write('full_startup_runs.json', {'runs': [m for m, _ in full]})
    write('state_timelines.json', {'runs': [{'run_id': m['run_id'], 'run_dir': m['run_dir'], 'final_state': m['final_state'], 'final_reason': m['final_reason'], 'navigation_ready_final': m['navigation_ready_final'], 'timeline': t} for m, t in runs]})
    chart_state_machine()
    fig, ax = plt.subplots(figsize=(10, 4));
    for i, (m, t) in enumerate(fixed):
        start = next((x['sim_time'] for x in t if x['state'] == 'VERIFY_FIXEDSTAND'), None); end = t[-1]['sim_time'] if t else None
        if start is not None and end is not None: ax.barh(i, end-start, left=start, label=m['run_id'])
    ax.set_xlabel('simulation time (s)'); ax.set_ylabel('FixedStand run'); fig.tight_layout(); fig.savefig(OUT / 'passive_to_fixedstand_timelines.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for m, t in fixed:
        x=[p['sim_time'] for p in t]; imu=[p['health'].get('imu') or {} for p in t]; ax[0].plot(x, [a.get('roll') for a in imu], label=m['run_id'] + ' roll'); ax[0].plot(x, [a.get('pitch') for a in imu], ls='--', label=m['run_id'] + ' pitch'); ax[1].plot(x, [p['health'].get('max_joint_error_rad') for p in t], label=m['run_id'])
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); ax[0].set_ylabel('roll/pitch (rad)'); ax[1].set_ylabel('max joint error (rad)'); ax[1].set_xlabel('simulation time (s)'); fig.tight_layout(); fig.savefig(OUT / 'fixedstand_attitude_joint_error.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for m, t in full:
        x=[p['sim_time'] for p in t]; ax.step(x, [int(p['health']['scan_fresh'] and p['health']['filtered_fresh'] and p['health']['raw_odom_fresh'] and p['health']['gated_odom_fresh'] and p['health']['l3v_fresh']) for p in t], where='post', label=m['run_id'])
    ax.set_ylabel('perception fresh'); ax.set_xlabel('simulation time (s)'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / 'perception_freshness_timeline.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for m, t in full:
        x=[p['sim_time'] for p in t]; ax.step(x, [int(p['health']['mode'] == 'RL') for p in t], where='post', label=m['run_id'] + ' RL'); ax.step(x, [int(p['health']['rl_zero_hold_stable']) for p in t], where='post', ls='--', label=m['run_id'] + ' zero-hold')
    ax.legend(fontsize=7); ax.set_xlabel('simulation time (s)'); fig.tight_layout(); fig.savefig(OUT / 'rl_switch_zero_hold.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for m, t in full:
        ax.step([p['sim_time'] for p in t], [int(p['navigation_ready']) for p in t], where='post', label=m['run_id'])
    ax.set_ylabel('navigation-ready'); ax.set_xlabel('simulation time (s)'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / 'navigation_ready_ttl_revoke.png', dpi=140); plt.close(fig)


if __name__ == '__main__': main()
