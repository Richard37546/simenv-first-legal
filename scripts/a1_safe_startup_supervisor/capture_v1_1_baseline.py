#!/usr/bin/env python3
"""Capture a non-destructive V1.1 supervisor patch baseline."""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess


ROOT = pathlib.Path('/home/richard/simenv_official_clean')


def run(*args):
    return subprocess.check_output(args, cwd=str(ROOT), text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'debug' / 'patches' / ('a1_safe_startup_supervisor_v1_1_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    targets = [
        ROOT / 'scripts/a1_safe_startup_supervisor/supervisor_core.py',
        ROOT / 'scripts/a1_safe_startup_supervisor/a1_safe_startup_supervisor.py',
        ROOT / 'scripts/a1_safe_startup_supervisor/test_supervisor_core.py',
        ROOT / 'scripts/a1_safe_startup_supervisor/run_acceptance.sh',
    ]
    for target in targets:
        shutil.copy2(target, out / target.name)
    (out / 'git_status_short.txt').write_text(run('git', 'status', '--short') + '\n', encoding='utf-8')
    (out / 'git_diff.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=str(ROOT), text=True), encoding='utf-8')
    (out / 'git_diff_binary.patch').write_text(subprocess.check_output(['git', 'diff', '--binary'], cwd=str(ROOT), text=True), encoding='utf-8')
    manifest = {
        'created_utc': stamp,
        'git_head': run('git', 'rev-parse', 'HEAD'),
        'dirty_diff_sha256': sha256(out / 'git_diff_binary.patch'),
        'target_sha256_before': {str(path.relative_to(ROOT)): sha256(path) for path in targets},
        'binary_sha256_before': {
            'devel/lib/unitree_guide/junior_ctrl': sha256(ROOT / 'devel/lib/unitree_guide/junior_ctrl'),
            'devel/lib/liblivox_laser_simulation.so': sha256(ROOT / 'devel/lib/liblivox_laser_simulation.so'),
        },
        'environment': {key: os.environ.get(key, '') for key in ('GUI', 'UNITREE_CTRL_DT', 'GAZEBO_PLUGIN_PATH', 'ROS_PACKAGE_PATH', 'ROS_MASTER_URI')},
        'launch_contract': 'roslaunch unitree_guide multi_floor_gazeboSim.launch gui:=false paused:=false user_debug:=False rname:=a1 robot_x:=0.0 robot_y:=-2.2 robot_z:=0.6 robot_yaw:=1.5708',
        'resolved_supervisor_defaults': {'passive_limit_sim_sec': 1.0, 'fixedstand_window_sim_sec': 10.0, 'fixedstand_deadline_sim_sec': 20.0, 'rl_window_sim_sec': 20.0},
    }
    (out / 'baseline_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
