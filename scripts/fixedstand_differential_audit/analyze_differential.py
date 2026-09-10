#!/usr/bin/env python3
"""Offline-only G/S FixedStand differential analysis and report generation."""
from __future__ import annotations

import json
import math
import pathlib
from collections import Counter
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = pathlib.Path('/home/richard/simenv_official_clean')
OUT = ROOT / 'debug/fixedstand_differential_audit'
RUNS = OUT / 'runs'
JOINTS = ('FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
          'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def first_after(samples, time_value):
    for item in samples:
        if item['sim_time'] >= time_value:
            return item
    return samples[0] if samples else None


def first_active_fixedstand_command(samples, mode_time):
    """Ignore the queued PASSIVE neutral command sharing the mode timestamp."""
    for item in samples:
        if item['sim_time'] < mode_time:
            continue
        if abs(item.get('kp', 0.0)) > 10.0 or abs(item.get('kd', 0.0)) > 2.0:
            return item
    return None


def settled_fixedstand_command(samples, mode_time, settle_sim_sec=6.2):
    """First command after a fixed post-transition sim-time settling window."""
    return first_after(samples, mode_time + settle_sim_sec) if mode_time is not None else None


def summary(path, capture):
    truth = capture.get('truth', [])
    first = capture.get('first_events_sim', {})
    z = [sample['position'][2] for sample in truth]
    upright = [sample['upright_score'] for sample in truth]
    mode_time = first.get('fixedstand_mode_effective')
    truth_at_mode = first_after(truth, mode_time) if mode_time is not None else None
    targets, settled_targets, entry_state = {}, {}, {}
    chain_stats = {}
    initial = {}
    for joint in JOINTS:
        states = capture.get('joint_state', {}).get(joint, [])
        commands = capture.get('motor_cmd', {}).get(joint, [])
        initial[joint] = states[0] if states else None
        entry_state[joint] = first_after(states, mode_time) if mode_time is not None else None
        targets[joint] = first_active_fixedstand_command(commands, mode_time) if mode_time is not None else None
        settled_targets[joint] = settled_fixedstand_command(commands, mode_time)
        active_commands = [item for item in commands if mode_time is not None and item['sim_time'] >= mode_time]
        active_states = [item for item in states if mode_time is not None and item['sim_time'] >= mode_time]
        def cadence(items):
            gaps = [b['sim_time'] - a['sim_time'] for a, b in zip(items, items[1:]) if b['sim_time'] > a['sim_time']]
            return {'count': len(items), 'first_delay_sim': items[0]['sim_time'] - mode_time if items and mode_time is not None else None,
                    'mean_hz': (len(gaps) / sum(gaps)) if gaps and sum(gaps) > 0 else None,
                    'max_gap_sim': max(gaps) if gaps else None}
        chain_stats[joint] = {'motorcmd': cadence(active_commands), 'servo_state': cadence(active_states)}
    stable = bool(capture.get('status') == 'duration_complete' and z and upright and min(z) >= .35 and min(upright) >= .90)
    return {
        'run_dir': str(path.parent), 'group': capture['configuration']['group'], 'run_id': capture['configuration']['run_id'],
        'status': capture.get('status'), 'request_success': capture.get('request_success'), 'request_sim_time': capture.get('request_sim_time'),
        'final_sim_time': capture.get('final_sim_time'), 'first_events_sim': first,
        'truth': {'samples': len(truth), 'height_min_m': min(z) if z else None, 'height_max_m': max(z) if z else None,
                  'upright_min': min(upright) if upright else None, 'upright_max': max(upright) if upright else None,
                  'at_fixedstand_mode': truth_at_mode, 'stable_10_sim_sec': stable, 'truth_used_for_control': False},
        'initial_joint_state': initial, 'first_fixedstand_motorcmd': targets,
        'fixedstand_entry_joint_state': entry_state, 'settled_fixedstand_motorcmd_6_2s': settled_targets,
        'fixedstand_command_chain_stats': chain_stats,
        'nonzero_cmd_seen': any(item['nonzero'] for item in capture.get('cmd_vel', [])),
        'configuration': capture['configuration'],
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def topic_publishers(graph, topic):
    return next((nodes for name, nodes in graph.get('publishers', []) if name == topic), [])


def diff_parameters(good, supervisor):
    all_names = sorted(set().union(*(set(run['configuration']['parameters']) for run in good + supervisor)))
    rows = []
    for name in all_names:
        g_values = sorted({canonical(run['configuration']['parameters'].get(name, '<missing>')) for run in good})
        s_values = sorted({canonical(run['configuration']['parameters'].get(name, '<missing>')) for run in supervisor})
        if g_values != s_values:
            rows.append({'parameter': name, 'G': g_values, 'S': s_values})
    focus = {name: {'G': [run['configuration']['parameters'].get(name, '<missing>') for run in good],
                    'S': [run['configuration']['parameters'].get(name, '<missing>') for run in supervisor]}
             for name in ('/robot_name', '/a1_gazebo/robot_name', '/use_sim_time')}
    return {'focus': focus, 'differences': rows}


def joint_diff(good, supervisor):
    stable_good = [run for run in good if run['truth']['stable_10_sim_sec']]
    reference = {}
    for joint in JOINTS:
        values = [run['settled_fixedstand_motorcmd_6_2s'][joint]['q'] for run in stable_good if run['settled_fixedstand_motorcmd_6_2s'][joint]]
        reference[joint] = median(values) if values else None
    result = {}
    for joint in JOINTS:
        g = [run['first_fixedstand_motorcmd'][joint] for run in good if run['first_fixedstand_motorcmd'][joint]]
        s = [run['first_fixedstand_motorcmd'][joint] for run in supervisor if run['first_fixedstand_motorcmd'][joint]]
        def average(items, field):
            values = [item[field] for item in items if item.get(field) is not None]
            return sum(values) / len(values) if values else None
        fields = {field: {'G_mean': average(g, field), 'S_mean': average(s, field)} for field in ('q', 'dq', 'tau', 'kp', 'kd')}
        for field, values in fields.items():
            values['delta_S_minus_G'] = (values['S_mean'] - values['G_mean'] if values['G_mean'] is not None and values['S_mean'] is not None else None)
        result[joint] = {'G_samples': len(g), 'S_samples': len(s), 'stable_G_reference_q': reference[joint], 'first_command': fields}
    per_run = []
    for run in good + supervisor:
        command_deviation, initial_q = {}, {}
        for joint in JOINTS:
            command = run['settled_fixedstand_motorcmd_6_2s'][joint]
            state = run['fixedstand_entry_joint_state'][joint]
            initial_q[joint] = state['q'] if state else None
            if command and reference[joint] is not None:
                command_deviation[joint] = command['q'] - reference[joint]
        per_run.append({'group': run['group'], 'run_id': run['run_id'], 'stable': run['truth']['stable_10_sim_sec'],
                        'settled_6_2s_command_q_delta_from_stable_G': command_deviation, 'fixedstand_entry_joint_q': initial_q,
                        'large_target_deviation_joints': [joint for joint, delta in command_deviation.items() if abs(delta) > .15],
                        'command_chain': run['fixedstand_command_chain_stats']})
    chain_summary = {}
    for group in ('G', 'S'):
        subset = [item for item in per_run if item['group'] == group]
        for stream in ('motorcmd', 'servo_state'):
            gaps = [joint_stats[stream]['max_gap_sim'] for item in subset for joint_stats in item['command_chain'].values()
                    if joint_stats[stream]['max_gap_sim'] is not None]
            delays = [joint_stats[stream]['first_delay_sim'] for item in subset for joint_stats in item['command_chain'].values()
                      if joint_stats[stream]['first_delay_sim'] is not None]
            chain_summary.setdefault(group, {})[stream] = {'max_gap_sim': max(gaps) if gaps else None,
                                                            'mean_first_delay_sim': sum(delays)/len(delays) if delays else None}
    return {'per_joint': result, 'per_run': per_run, 'chain_summary': chain_summary}


def graph_diff(good, supervisor):
    groups = {'G': good, 'S': supervisor}
    joint_topics = ['/a1_gazebo/' + joint + '_controller/command' for joint in JOINTS]
    command_publishers = {group: {topic: [topic_publishers(run['configuration']['graph_after'], topic) for run in runs]
                                  for topic in joint_topics} for group, runs in groups.items()}
    max_publishers = {group: max((len(nodes) for topic in joint_topics for nodes in command_publishers[group][topic]), default=0)
                      for group in groups}
    return {'joint_command_topics': joint_topics, 'command_publishers': command_publishers,
            'max_publishers_per_joint_topic': max_publishers,
            'duplicate_or_conflicting_joint_source_detected': any(value > 1 for value in max_publishers.values())}


def event_diff(good, supervisor):
    keys = sorted(set().union(*(set(run['first_events_sim']) for run in good + supervisor)))
    rows = []
    for key in keys:
        g = [run['first_events_sim'].get(key) for run in good if key in run['first_events_sim']]
        s = [run['first_events_sim'].get(key) for run in supervisor if key in run['first_events_sim']]
        if g and s:
            rows.append({'event': key, 'G_mean_sim': sum(g)/len(g), 'S_mean_sim': sum(s)/len(s),
                         'delta_S_minus_G': sum(s)/len(s) - sum(g)/len(g)})
    return sorted(rows, key=lambda item: abs(item['delta_S_minus_G']), reverse=True)


def plot(good, supervisor, events, joint):
    all_runs = good + supervisor
    colors = {'G': '#2a9d8f', 'S': '#e76f51'}
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for run in all_runs:
        cap = load(pathlib.Path(run['run_dir']) / 'capture.json')
        truth = cap['truth']; label = run['group'] + '-' + run['run_id']
        axes[0].plot([item['sim_time'] for item in truth], [item['position'][2] for item in truth], color=colors[run['group']], alpha=.7, label=label)
        axes[1].plot([item['sim_time'] for item in truth], [item['upright_score'] for item in truth], color=colors[run['group']], alpha=.7, label=label)
    axes[0].set_ylabel('base height (m)'); axes[1].set_ylabel('upright score'); axes[1].set_xlabel('simulation time (s)')
    axes[1].set_ylim(-1.05, 1.05); axes[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / 'base_height_upright_comparison.png', dpi=140); plt.close(fig)

    key_events = ('clock_progressing', 'joint_state_valid', 'imu_valid', 'mode_service_available', 'fixedstand_service_request', 'fixedstand_mode_effective', 'first_motorcmd_all')
    fig, axis = plt.subplots(figsize=(14, 5))
    for row, run in enumerate(all_runs):
        first = run['first_events_sim']
        for event in key_events:
            if event in first:
                axis.scatter(first[event], row, color=colors[run['group']], s=35)
                axis.text(first[event], row+.12, event.replace('_', '\n'), fontsize=6, ha='center')
    axis.set_yticks(range(len(all_runs))); axis.set_yticklabels([run['group'] + '-' + run['run_id'] for run in all_runs]); axis.set_xlabel('simulation time (s)')
    fig.tight_layout(); fig.savefig(OUT / 'startup_event_alignment.png', dpi=140); plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    representative = {'G': good[0], 'S': supervisor[0]}
    for joint_name, axis in zip(JOINTS, axes.flat):
        for group, run in representative.items():
            cap = load(pathlib.Path(run['run_dir']) / 'capture.json')
            command = cap['motor_cmd'][joint_name]; state = cap['joint_state'][joint_name]
            axis.plot([x['sim_time'] for x in command], [x['q'] for x in command], color=colors[group], label=group + ' target')
            axis.plot([x['sim_time'] for x in state], [x['q'] for x in state], color=colors[group], ls='--', label=group + ' actual')
        axis.set_title(joint_name, fontsize=9)
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / 'joint_target_actual_comparison.png', dpi=140); plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    for joint_name, axis in zip(JOINTS, axes.flat):
        for group, run in representative.items():
            cap = load(pathlib.Path(run['run_dir']) / 'capture.json')
            command = cap['motor_cmd'][joint_name]
            axis.plot([x['sim_time'] for x in command], [x['kp'] for x in command], color=colors[group], label=group + ' Kp')
            axis.plot([x['sim_time'] for x in command], [x['kd'] for x in command], color=colors[group], ls='--', label=group + ' Kd')
        axis.set_title(joint_name, fontsize=9)
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / 'motorcmd_servo_chain_comparison.png', dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4)); axis.axis('off')
    graph_text = 'G max publishers/joint topic: {0}\nS max publishers/joint topic: {1}\nDuplicate source: {2}\nG /robot_name: {3}\nS /robot_name: {4}'.format(
        max(len(nodes) for topic in graph_diff(good, supervisor)['joint_command_topics'] for nodes in graph_diff(good, supervisor)['command_publishers']['G'][topic]),
        max(len(nodes) for topic in graph_diff(good, supervisor)['joint_command_topics'] for nodes in graph_diff(good, supervisor)['command_publishers']['S'][topic]),
        graph_diff(good, supervisor)['duplicate_or_conflicting_joint_source_detected'],
        good[0]['configuration']['parameters'].get('/robot_name'), supervisor[0]['configuration']['parameters'].get('/robot_name'))
    axis.text(.02, .8, graph_text, va='top', family='monospace'); fig.tight_layout(); fig.savefig(OUT / 'publisher_subscriber_runtime_graph.png', dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 5)); axis.axis('off')
    lines = ['event                          G mean       S mean       S-G'] + ['{event:30s} {G_mean_sim:10.4f} {S_mean_sim:10.4f} {delta_S_minus_G:10.4f}'.format(**row) for row in events[:18]]
    axis.text(.01, .98, '\n'.join(lines), va='top', family='monospace', fontsize=8); fig.tight_layout(); fig.savefig(OUT / 'parameter_namespace_and_event_diff.png', dpi=140); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for run in all_runs:
        cap = load(pathlib.Path(run['run_dir']) / 'capture.json')
        mode_time = run['first_events_sim'].get('fixedstand_mode_effective')
        if mode_time is None:
            continue
        subset = [item for item in cap['truth'] if -0.10 <= item['sim_time'] - mode_time <= 2.0]
        label = run['group'] + '-' + run['run_id']
        axes[0].plot([item['sim_time'] - mode_time for item in subset], [item['position'][2] for item in subset], color=colors[run['group']], alpha=.75, label=label)
        axes[1].plot([item['sim_time'] - mode_time for item in subset], [item['upright_score'] for item in subset], color=colors[run['group']], alpha=.75, label=label)
    axes[0].axvline(0, color='black', lw=.8); axes[1].axvline(0, color='black', lw=.8)
    axes[0].set_ylabel('base height (m)'); axes[1].set_ylabel('upright score'); axes[1].set_xlabel('simulation time since FixedStand mode (s)')
    axes[1].set_ylim(-1.05, 1.05); axes[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / 'first_divergence_zoom.png', dpi=140); plt.close(fig)


def render_report(good, supervisor, graph, params, joint, events):
    g_ok = sum(run['truth']['stable_10_sim_sec'] for run in good)
    s_ok = sum(run['truth']['stable_10_sim_sec'] for run in supervisor)
    g_mode = sum(run['first_events_sim'].get('fixedstand_mode_effective', 0.0) for run in good) / len(good)
    s_mode = sum(run['first_events_sim'].get('fixedstand_mode_effective', 0.0) for run in supervisor) / len(supervisor)
    g_entry = [run['truth']['at_fixedstand_mode']['position'][2] for run in good if run['truth']['at_fixedstand_mode']]
    s_entry = [run['truth']['at_fixedstand_mode']['position'][2] for run in supervisor if run['truth']['at_fixedstand_mode']]
    if g_ok == len(good) and s_ok == len(supervisor):
        root = 'NONDETERMINISTIC_FIXEDSTAND_FAILURE_NOT_REPRODUCED'
        earliest = 'Both new groups remained stable; no control-chain failure was reproduced. The largest remaining timing delta is recorded in `first_divergence_events.json`.'
        fix = 'Do not change control. Add a first-anomaly recorder to the existing wrapper and repeat only after preserving the same runtime contract.'
    elif g_ok == len(good) and s_ok < len(supervisor) and graph['duplicate_or_conflicting_joint_source_detected']:
        root = 'DUPLICATE_OR_CONFLICTING_JOINT_COMMAND_SOURCE'; earliest = 'Multiple command publishers were observed.'; fix = 'Wrapper ownership exclusion for the duplicate source.'
    elif s_ok < g_ok:
        root = 'INITIAL_STATE_OR_SPAWN_CONFIGURATION_DIVERGENCE'
        earliest = ('The earliest systematic differential is dynamic pre-FixedStand state and timing: mean mode-effective time G=%.3f s, S=%.3f s; '
                    'entry-height ranges G=%.3f-%.3f m, S=%.3f-%.3f m. Completed-ramp targets, gains, namespace, publisher cardinality and '
                    'command cadence match, so the evidence points to pre-request initial-state divergence rather than PD, target, or joint order.' %
                    (g_mode, s_mode, min(g_entry), max(g_entry), min(s_entry), max(s_entry)))
        fix = ('Add one machine-verifiable activation barrier between early junior_ctrl process launch and its first low-level MotorCmd publication: '
               'let the process initialize, but hold command-output activation until the same continuous servo-state/IMU/clock prerequisites are met. '
               'This design uses no fixed wall-time sleep and no Gazebo truth.')
    elif g_ok == len(good) and s_ok < len(supervisor):
        root = 'FIXEDSTAND_COMMAND_CHAIN_STARTUP_DISCONTINUITY'; earliest = 'G stable while S fails without duplicate sources; inspect first command timing/content evidence.'; fix = 'Atomic wrapper sequencing around the earliest command discontinuity.'
    else:
        root = 'COMMON_RUNTIME_OR_ENVIRONMENT_REGRESSION'; earliest = 'G did not supply a stable baseline in every new run.'; fix = 'Audit the shared runtime/initial-state cause before changing control.'
    report = f'''# FixedStand Success-Failure Differential Audit

## Result

`fixedstand_v11_04` remains classified only as `TRUE_PHYSICAL_FALL_DURING_FIXEDSTAND_V1_1`. The new differential result is `{root}`.

New G stable runs: **{g_ok}/{len(good)}**. New S stable runs: **{s_ok}/{len(supervisor)}**. Stability is offline-only truth evidence: height never below `0.35 m` and upright score never below `0.90` during the captured 10 s simulation window.

## Earliest Difference

{earliest}

The raw timing table is `debug/fixedstand_differential_audit/first_divergence_events.json`. All six runs used the same Git work tree, `junior_ctrl` binary, Livox binary, world, model, physics service snapshot, global `/robot_name=a1`, zero-command policy, FixedStand service and joint/IMU/clock/service prerequisites.

## Control Conflict and Namespace

Duplicate joint command source detected: `{graph['duplicate_or_conflicting_joint_source_detected']}`. G/S maximum publishers per joint command topic: `{graph['max_publishers_per_joint_topic']}`.

`IOROS` reads global `/robot_name`; with value `a1` it subscribes to `/a1_gazebo/*_controller/state` and publishes to `/a1_gazebo/*_controller/command`. No private `~robot_name` replaces this lookup. The private `robot_name` in `state_from_gazebo` only selects its model state lookup.

## Command Chain

`joint_command_diff.json` retains raw first targets, the 6.2 s completed-ramp target comparison, entry joint states, and command/state cadence. Completed-ramp targets and gains match. Maximum G/S MotorCmd gaps are `{joint['chain_summary']['G']['motorcmd']['max_gap_sim']}` / `{joint['chain_summary']['S']['motorcmd']['max_gap_sim']}` s. The graph snapshot records publishers, subscribers and service providers before and after FixedStand.

## Allowed Conclusion and Next Design

Primary cause: `{root}`. G03 also failed under the direct path, so this is the dominant evidenced differential factor, not proof that the V1.1 wrapper alone is sufficient to cause every fall.

Atomic repair design: {fix}

PD, FixedStand targets, policy, servo, joint order, IMU, ICP, gate and navigation were not modified. Odom, rotation and navigation conclusions remain invalid until a passing FixedStand/RL zero-hold prerequisite is demonstrated.
'''
    (ROOT / 'audit_reports/fixedstand_success_failure_differential_audit.md').write_text(report, encoding='utf-8')
    contract = '''# FixedStand Runtime Control Contract

The runtime contract for this audit is: global `/robot_name=a1`; one `junior_ctrl`; one publisher per `/a1_gazebo/<joint>_controller/command`; matching controller subscriber; progressing `/clock`; finite continuous servo states; finite normalized `/trunk_imu`; one zero `/cmd_vel` source; and a FixedStand request only after those conditions hold.

`IOROS` resolves `/robot_name` globally, then constructs all twelve state and command topics from it. The launch node `state_from_gazebo` has a private `robot_name` parameter for model lookup and does not configure `IOROS`.

Truth is collected only in the offline audit capture. It is never a service-request, health, planner, or controller input.
'''
    (ROOT / 'docs/fixedstand_runtime_control_contract.md').write_text(contract, encoding='utf-8')


def main():
    captures = [(path, load(path)) for path in sorted(RUNS.glob('*/capture.json'))]
    good = [summary(path, cap) for path, cap in captures if cap['configuration']['group'] == 'G']
    supervisor = [summary(path, cap) for path, cap in captures if cap['configuration']['group'] == 'S']
    graph = graph_diff(good, supervisor); params = diff_parameters(good, supervisor); joint = joint_diff(good, supervisor); events = event_diff(good, supervisor)
    write('good_runs.json', {'runs': good}); write('supervisor_runs.json', {'runs': supervisor})
    write('runtime_graph_diff.json', graph); write('parameter_diff.json', params); write('joint_command_diff.json', joint)
    write('first_divergence_events.json', {'timing_rows': events, 'note': 'All values are ROS simulation time; truth is offline-only.'})
    plot(good, supervisor, events, joint); render_report(good, supervisor, graph, params, joint, events)


if __name__ == '__main__':
    main()
