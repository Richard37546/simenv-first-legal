#!/usr/bin/env python3
"""Fail-closed, zero-velocity A1 V1.1 startup supervisor.

This process never subscribes to Gazebo truth.  A separate offline collector
may observe truth for acceptance, but no truth data is passed to this reducer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from unitree_legged_msgs.msg import MotorCmd, MotorState

from supervisor_core import Health, Reducer, State


ROOT = pathlib.Path('/home/richard/simenv_official_clean')
OUT = ROOT / 'debug/a1_safe_startup_supervisor_v1_1'
JUNIOR = ROOT / 'devel/lib/unitree_guide/junior_ctrl'
LIVOX = ROOT / 'devel/lib/liblivox_laser_simulation.so'
JOINTS = ('FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
          'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf')


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rpy(q):
    roll = math.atan2(2 * (q.w*q.x + q.y*q.z), 1 - 2 * (q.x*q.x + q.y*q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w*q.y - q.z*q.x))))
    yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
    return roll, pitch, yaw


class Cache:
    """Persistent receipts and validity counters for the minimum control chain."""

    def __init__(self):
        self.sim_time = 0.0
        self.previous_clock = None
        self.values, self.receipt, self.first_sim = {}, {}, {}
        self.joint_cmd, self.joint_state = {}, {}
        self.joint_state_valid_count = {joint: 0 for joint in JOINTS}
        self.servo_valid_count = {joint: 0 for joint in JOINTS}
        self.imu_valid_count = 0
        self.cmd_events = []
        self.subs = [
            rospy.Subscriber('/clock', Clock, self.clock_cb, queue_size=100),
            rospy.Subscriber('/trunk_imu', Imu, self.imu_cb, queue_size=100),
            rospy.Subscriber('/unitree/controller_mode', String, self.mode_cb, queue_size=20),
            rospy.Subscriber('/unitree/rl_mode_ready', Bool, self.rl_ready_cb, queue_size=20),
            rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb, queue_size=100),
            rospy.Subscriber('/scan', PointCloud, self.generic_cb, callback_args='scan', queue_size=5),
            rospy.Subscriber('/team/livox/scan_cloud_filtered', PointCloud2, self.generic_cb, callback_args='filtered', queue_size=5),
            rospy.Subscriber('/team/livox/icp_odom_raw', Odometry, self.generic_cb, callback_args='raw_odom', queue_size=20),
            rospy.Subscriber('/team/livox/icp_odom_gated', Odometry, self.generic_cb, callback_args='gated_odom', queue_size=20),
            rospy.Subscriber('/team/traversability_status', String, self.status_cb, queue_size=20),
        ]
        for joint in JOINTS:
            prefix = '/a1_gazebo/' + joint + '_controller'
            self.subs.append(rospy.Subscriber(prefix + '/command', MotorCmd, self.motor_cmd_cb, callback_args=joint, queue_size=20))
            self.subs.append(rospy.Subscriber(prefix + '/state', MotorState, self.motor_state_cb, callback_args=joint, queue_size=20))

    def put(self, name, value):
        self.values[name] = value
        self.receipt[name] = time.monotonic()

    def note_first(self, name, condition=True):
        if condition and name not in self.first_sim:
            self.first_sim[name] = self.sim_time

    def clock_cb(self, msg):
        now = msg.clock.to_sec()
        self.sim_time = now
        if self.previous_clock is not None and now > self.previous_clock + 1e-6:
            self.note_first('clock_progressing')
        self.previous_clock = now
        self.put('clock', now)
        self.note_first('clock_received')

    def imu_cb(self, msg):
        q = msg.orientation
        values = (q.x, q.y, q.z, q.w)
        norm = math.sqrt(sum(value * value for value in values)) if all(math.isfinite(value) for value in values) else float('nan')
        valid = all(math.isfinite(value) for value in values) and 0.90 <= norm <= 1.10
        if valid:
            self.imu_valid_count += 1
            self.note_first('imu_quaternion_valid')
        else:
            self.imu_valid_count = 0
        roll, pitch, yaw = rpy(q) if valid else (float('nan'),) * 3
        self.put('imu', {'quaternion_xyzw': [q.x, q.y, q.z, q.w], 'norm': norm,
                         'valid': valid, 'roll': roll, 'pitch': pitch, 'yaw': yaw,
                         'gyro': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]})
        self.note_first('imu_received')

    def mode_cb(self, msg):
        self.put('mode', msg.data)
        self.note_first('controller_mode_received')
        if msg.data == 'FIXEDSTAND':
            self.note_first('fixedstand_mode_effective')
        if msg.data == 'RL':
            self.note_first('rl_mode_effective')

    def rl_ready_cb(self, msg):
        self.put('rl_ready', bool(msg.data))

    def generic_cb(self, _msg, name):
        self.put(name, True)
        self.note_first(name + '_received')

    def status_cb(self, msg):
        try:
            self.put('l3v_status', json.loads(msg.data))
        except (TypeError, ValueError):
            self.put('l3v_status', {'parse_error': True})
        self.note_first('l3v_status_received')

    def cmd_cb(self, msg):
        fields = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z)
        nonzero = any(abs(value) > 1e-5 for value in fields)
        self.cmd_events.append({'sim_time': self.sim_time, 'wall_time': time.monotonic(), 'nonzero': nonzero,
                                'linear_x': msg.linear.x, 'angular_z': msg.angular.z})
        self.cmd_events = self.cmd_events[-500:]
        self.put('cmd', msg)
        self.note_first('zero_cmd_observed', not nonzero)
        self.note_first('nonzero_cmd_observed', nonzero)

    def motor_cmd_cb(self, msg, joint):
        values = (msg.q, msg.dq, msg.Kp, msg.Kd)
        self.joint_cmd[joint] = values
        self.put('joint_cmd_' + joint, True)
        if all(math.isfinite(value) for value in values):
            self.servo_valid_count[joint] += 1
            self.note_first('servo_update_started')
        else:
            self.servo_valid_count[joint] = 0

    def motor_state_cb(self, msg, joint):
        values = (msg.q, msg.dq, msg.tauEst)
        self.joint_state[joint] = values
        self.put('joint_state_' + joint, True)
        if all(math.isfinite(value) for value in values):
            self.joint_state_valid_count[joint] += 1
            self.note_first('joint_state_received')
        else:
            self.joint_state_valid_count[joint] = 0

    def fresh(self, name, ttl):
        return name in self.receipt and time.monotonic() - self.receipt[name] <= ttl

    def clock_progressing(self):
        return self.fresh('clock', 1.0) and 'clock_progressing' in self.first_sim

    def joints_valid_continuous(self):
        return all(self.fresh('joint_state_' + joint, 1.0) and self.joint_state_valid_count[joint] >= 3 for joint in JOINTS)

    def servo_updates_started(self):
        # In PASSIVE the controller has no FixedStand target yet, so MotorCmd
        # cannot be a prerequisite without creating a request cycle.  The
        # per-servo MotorState stream is the evidence that the controller and
        # Gazebo servo update chain are already running.  MotorCmd is still
        # required later by joints_healthy() while FixedStand is verified.
        return all(self.fresh('joint_state_' + joint, 1.0) and self.joint_state_valid_count[joint] >= 3 for joint in JOINTS)

    def joints_healthy(self):
        if not self.joints_valid_continuous() or not self.servo_updates_started():
            return False
        if not all(joint in self.joint_cmd and joint in self.joint_state for joint in JOINTS):
            return False
        errors = []
        for joint in JOINTS:
            values = self.joint_cmd[joint] + self.joint_state[joint]
            if not all(math.isfinite(value) for value in values):
                return False
            errors.append(abs(self.joint_cmd[joint][0] - self.joint_state[joint][0]))
        return max(errors, default=float('inf')) <= 0.8

    def max_joint_error(self):
        if not all(joint in self.joint_cmd and joint in self.joint_state for joint in JOINTS):
            return None
        return max(abs(self.joint_cmd[joint][0] - self.joint_state[joint][0]) for joint in JOINTS)

    def imu_quaternion_valid(self):
        return self.fresh('imu', .5) and self.imu_valid_count >= 3 and bool(self.values.get('imu', {}).get('valid'))

    def attitude_stable(self):
        imu = self.values.get('imu', {})
        return (self.imu_quaternion_valid() and abs(imu['roll']) <= .35 and abs(imu['pitch']) <= .35 and
                max(abs(value) for value in imu['gyro']) <= 1.0)

    def l3v_fresh(self):
        status = self.values.get('l3v_status', {})
        freshness = status.get('input_freshness', {}) if isinstance(status, dict) else {}
        return (self.fresh('l3v_status', 2.0) and freshness.get('all_required_inputs_fresh') is True and
                freshness.get('stale_reasons', []) == [])

    def final_cmd_zero(self):
        cmd = self.values.get('cmd')
        if cmd is None or not self.fresh('cmd', .75):
            return False
        return not any(abs(value) > 1e-5 for value in (cmd.linear.x, cmd.linear.y, cmd.linear.z, cmd.angular.x, cmd.angular.y, cmd.angular.z))

    def no_unexpired_nonzero_cmd(self, ttl=.75):
        cutoff = time.monotonic() - ttl
        return not any(event['nonzero'] and event['wall_time'] >= cutoff for event in self.cmd_events)

    def snapshot(self, junior_process_running, mode_service_available):
        perception_topics = all(self.fresh(name, 2.0) for name in ('scan', 'filtered', 'raw_odom', 'gated_odom', 'clock'))
        control_ready = self.joints_valid_continuous()
        joint_healthy = self.joints_healthy()
        final_zero = self.final_cmd_zero()
        no_residual = self.no_unexpired_nonzero_cmd()
        fixed = self.attitude_stable() and joint_healthy and not bool(self.values.get('rl_ready', False)) and final_zero and no_residual
        rl = self.attitude_stable() and joint_healthy and final_zero and no_residual
        health = Health(
            sim_time=self.sim_time,
            gazebo_spawned=self.fresh('clock', 1.0),
            clock_progressing=self.clock_progressing(),
            junior_process_running=junior_process_running,
            mode_service_available=mode_service_available,
            controller_ready=control_ready,
            joints_valid_continuous=control_ready,
            imu_quaternion_valid=self.imu_quaternion_valid(),
            servo_updates_started=self.servo_updates_started(),
            final_cmd_zero=final_zero,
            no_unexpired_nonzero_cmd=no_residual,
            mode=self.values.get('mode', ''),
            fixedstand_stable=fixed,
            perception_fresh=perception_topics and self.l3v_fresh(),
            rl_ready_fresh=self.fresh('rl_ready', .75) and bool(self.values.get('rl_ready', False)),
            rl_zero_hold_stable=rl,
            critical_fresh=perception_topics and self.l3v_fresh() and self.attitude_stable() and joint_healthy,
        )
        for name, value in {
            'junior_process_running': health.junior_process_running,
            'mode_service_available': health.mode_service_available,
            'continuous_joint_state_valid': health.joints_valid_continuous,
            'imu_quaternion_valid': health.imu_quaternion_valid,
            'servo_updates_started': health.servo_updates_started,
            'final_cmd_zero': health.final_cmd_zero,
            'no_unexpired_nonzero_cmd': health.no_unexpired_nonzero_cmd,
        }.items():
            self.note_first(name, value)
        return health

    def close(self):
        for sub in self.subs:
            sub.unregister()


def bash(command, log, env):
    handle = log.open('w', encoding='utf-8')
    proc = subprocess.Popen(['/bin/bash', '-lc', command], cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    proc._log_handle = handle
    return proc


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        deadline = time.monotonic() + 8
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    if proc:
        proc._log_handle.close()


def service_available(name):
    try:
        _code, _message, state = rospy.get_master().getSystemState()
        return any(service == name for service, _providers in state[2])
    except Exception:
        return False


def service_call(name):
    try:
        rospy.wait_for_service(name, timeout=.25)
        return bool(rospy.ServiceProxy(name, Trigger)().success)
    except (rospy.ROSException, rospy.ServiceException):
        return False


def render_timeline(run_dir, timeline):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        times = [item['sim_time'] for item in timeline]
        state_ids = {state.value: index for index, state in enumerate(State)}
        fig, axis = plt.subplots(figsize=(14, 4))
        axis.step(times, [state_ids[item['state']] for item in timeline], where='post')
        axis.set_yticks(list(state_ids.values())); axis.set_yticklabels(list(state_ids))
        axis.set_xlabel('simulation time (s)'); fig.tight_layout(); fig.savefig(run_dir / 'supervisor_state_timeline.png', dpi=140); plt.close(fig)
        fig, axis = plt.subplots(figsize=(14, 3))
        axis.plot(times, [item['health']['mode'] == 'FIXEDSTAND' for item in timeline], label='FIXEDSTAND')
        axis.plot(times, [item['health']['mode'] == 'RL' for item in timeline], label='RL')
        axis.plot(times, [item['navigation_ready'] for item in timeline], label='navigation_ready')
        axis.legend(); axis.set_xlabel('simulation time (s)'); fig.tight_layout(); fig.savefig(run_dir / 'mode_and_ready_timeline.png', dpi=140); plt.close(fig)
    except Exception as exc:
        (run_dir / 'plot_error.txt').write_text(str(exc) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--fixedstand-only', action='store_true')
    parser.add_argument('--offline-truth-capture', action='store_true')
    parser.add_argument('--perception-delay-sim-sec', type=float, default=0.0)
    parser.add_argument('--world-file', default=str(ROOT / 'generated_building/competition_scene.world'))
    parser.add_argument('--wall-watchdog-sec', type=float, default=1200.0)
    parser.add_argument('--passive-limit-sim-sec', type=float, default=1.0)
    parser.add_argument('--fixedstand-window-sim-sec', type=float, default=10.0)
    parser.add_argument('--fixedstand-deadline-sim-sec', type=float, default=20.0)
    parser.add_argument('--rl-window-sim-sec', type=float, default=20.0)
    args = parser.parse_args()
    if args.perception_delay_sim_sec < 0:
        parser.error('--perception-delay-sim-sec must be non-negative')

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    run_dir = OUT / 'runs' / (args.run_id + '_' + stamp + '_pid' + str(os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({'GUI': 'false', 'PAUSED': 'false', 'UNITREE_CTRL_DT': env.get('UNITREE_CTRL_DT', '0.006'),
                'BUILDING_WORLD_FILE': args.world_file})
    source = ('source /opt/ros/noetic/setup.bash; source /home/richard/simenv_official_clean/devel/setup.bash; '
              'export GAZEBO_MODEL_PATH=/home/richard/simenv_official_clean/generated_building:$(rospack find unitree_gazebo)/models:${GAZEBO_MODEL_PATH:-}; ')
    manifest = {
        'run_id': args.run_id, 'run_dir': str(run_dir), 'gui': False, 'state_machine_started': False,
        'nonzero_cmd_vel_published': False, 'supervisor_uses_gazebo_truth': False,
        'world_file_used_only_by_gazebo_launch': args.world_file, 'resolved_parameters': vars(args),
        'binary_sha256': {'junior_ctrl': sha256(JUNIOR), 'livox_plugin': sha256(LIVOX)},
        'junior_ctrl_bootstrap': 'wait for ROS master; rosparam set /robot_name a1; exec junior_ctrl',
        'launch_command': 'roslaunch unitree_guide multi_floor_gazeboSim.launch gui:=false paused:=false user_debug:=False rname:=a1 robot_x:=0.0 robot_y:=-2.2 robot_z:=0.6 robot_yaw:=1.5708',
        'complete': False,
    }
    dump(run_dir / 'manifest.json', manifest)
    reducer = Reducer(args.passive_limit_sim_sec, args.fixedstand_window_sim_sec,
                      args.rl_window_sim_sec, args.fixedstand_deadline_sim_sec)
    processes, cache, zero_timer = {}, None, None
    sent_fixed = sent_rl = started_perception = fixedstand_complete = False
    perception_delay_origin = None
    timeline, events = [], {'gazebo_launch_wall_elapsed_sec': 0.0, 'junior_ctrl_launch_wall_elapsed_sec': None}
    start_wall = time.monotonic()
    try:
        processes['gazebo'] = bash(source + 'exec roslaunch unitree_guide multi_floor_gazeboSim.launch gui:=false paused:=false user_debug:=False rname:=a1 robot_x:=0.0 robot_y:=-2.2 robot_z:=0.6 robot_yaw:=1.5708', run_dir / 'gazebo.log', env)
        # This Popen is intentionally adjacent to Gazebo launch.  It is not gated on
        # controller-manager, Livox, ICP, L3V, or navigation freshness.
        processes['junior_ctrl'] = bash(
            source + 'until rosparam set /robot_name a1 >/dev/null 2>&1; do sleep 0.05; done; exec ' + str(JUNIOR),
            run_dir / 'junior_ctrl.log', env)
        events['junior_ctrl_launch_wall_elapsed_sec'] = time.monotonic() - start_wall
        rospy.init_node('a1_safe_startup_supervisor', anonymous=True, disable_signals=True)
        cache = Cache()
        if args.offline_truth_capture:
            processes['offline_truth_collector'] = bash(
                source + 'exec python3 scripts/a1_safe_startup_supervisor/collect_offline_truth.py --output ' +
                str(run_dir / 'truth_offline.json'), run_dir / 'offline_truth_collector.log', env)
        zero_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        nav_pub = rospy.Publisher('/unitree/navigation_ready', Bool, queue_size=1, latch=False)
        zero_timer = rospy.Timer(rospy.Duration(.1), lambda _event: zero_pub.publish(Twist()))
        while not rospy.is_shutdown() and time.monotonic() - start_wall < args.wall_watchdog_sec:
            junior_running = processes['junior_ctrl'].poll() is None
            mode_service = service_available('/unitree/request_fixedstand')
            health = cache.snapshot(junior_running, mode_service)
            if reducer.state == State.REQUEST_FIXEDSTAND and not sent_fixed:
                health.fixedstand_service_accepted = service_call('/unitree/request_fixedstand')
                sent_fixed = health.fixedstand_service_accepted
                if sent_fixed:
                    cache.note_first('fixedstand_request_accepted')
            if reducer.state == State.START_OR_WAIT_PERCEPTION:
                if args.fixedstand_only:
                    fixedstand_complete = True
                    reducer.reason = 'fixedstand_only_acceptance_complete'
                elif perception_delay_origin is None:
                    perception_delay_origin = cache.sim_time
                elif cache.sim_time - perception_delay_origin >= args.perception_delay_sim_sec and not started_perception:
                    commands = {
                        'l1s': 'rosrun team_livox_scan_preprocess l1s_scan_clean_cloud_node.py __name:=safe_startup_l1s',
                        'icp': 'rosrun rtabmap_odom icp_odometry __name:=safe_startup_icp _frame_id:=base _odom_frame_id:=team_livox_odom _publish_tf:=false _wait_for_transform:=true _wait_for_transform_duration:=0.2 _subscribe_scan_cloud:=true scan:=/team/livox/unused_scan scan_cloud:=/team/livox/scan_cloud_filtered odom:=/team/livox/icp_odom_raw odom_info:=/team/livox/icp_odom_info',
                        'gate': 'python3 scripts/l2_livox_icp_rtabmap_readiness/l2_livox_odom_gate.py __name:=safe_startup_odom_gate _input_topic:=/team/livox/icp_odom_raw _output_topic:=/team/livox/icp_odom_gated _status_topic:=/team/livox/icp_odom_gate_status _output_frame_id:=team_livox_odom _output_child_frame_id:=base',
                        'l3v': 'python3 scripts/l3v_local_traversability_diagnostic_node/l3v_local_traversability_node.py __name:=safe_startup_l3v',
                    }
                    for name, command in commands.items():
                        processes[name] = bash(source + 'exec ' + command, run_dir / (name + '.log'), env)
                    started_perception = True
                    cache.note_first('perception_stack_started')
            if reducer.state == State.REQUEST_RL and not args.fixedstand_only and not sent_rl:
                health.rl_service_accepted = service_call('/unitree/request_rl')
                sent_rl = health.rl_service_accepted
                if sent_rl:
                    cache.note_first('rl_request_accepted')
            if not fixedstand_complete:
                reducer.step(health)
            nav_pub.publish(Bool(data=reducer.navigation_ready))
            timeline.append({
                'sim_time': cache.sim_time, 'wall_elapsed_sec': time.monotonic() - start_wall,
                'state': reducer.state.value, 'reason': reducer.reason, 'navigation_ready': reducer.navigation_ready,
                'health': {
                    'mode': health.mode, 'clock_progressing': health.clock_progressing,
                    'junior_process_running': health.junior_process_running, 'mode_service_available': health.mode_service_available,
                    'continuous_joint_state_valid': health.joints_valid_continuous, 'imu_quaternion_valid': health.imu_quaternion_valid,
                    'servo_updates_started': health.servo_updates_started, 'final_cmd_zero': health.final_cmd_zero,
                    'no_unexpired_nonzero_cmd': health.no_unexpired_nonzero_cmd, 'fixedstand_stable': health.fixedstand_stable,
                    'attitude_stable': cache.attitude_stable(), 'joints_healthy': cache.joints_healthy(),
                    'motor_commands_available': all(joint in cache.joint_cmd for joint in JOINTS),
                    'perception_fresh': health.perception_fresh, 'rl_ready_fresh': health.rl_ready_fresh,
                    'rl_zero_hold_stable': health.rl_zero_hold_stable, 'critical_fresh': health.critical_fresh,
                    'imu': cache.values.get('imu'), 'max_joint_error_rad': cache.max_joint_error(),
                    'scan_fresh': cache.fresh('scan', 2.0), 'filtered_fresh': cache.fresh('filtered', 2.0),
                    'raw_odom_fresh': cache.fresh('raw_odom', 2.0), 'gated_odom_fresh': cache.fresh('gated_odom', 2.0),
                    'l3v_fresh': cache.l3v_fresh(),
                },
            })
            if fixedstand_complete or reducer.state in (State.NAVIGATION_READY, State.FAIL_SAFE_HOLD):
                break
            time.sleep(.05)
        if not fixedstand_complete and reducer.state not in (State.NAVIGATION_READY, State.FAIL_SAFE_HOLD):
            reducer.fail('WALL_WATCHDOG_EXCEEDED')
    finally:
        if zero_timer:
            zero_timer.shutdown()
        if cache:
            events.update(cache.first_sim)
            events['final_sim_time'] = cache.sim_time
            events['first_fixedstand_effective_sim'] = reducer.fixedstand_effective_sim
            events['fixedstand_request_sim'] = reducer.fixedstand_request_sim
            events['controller_ready_sim'] = reducer.controller_ready_sim
            events['passive_dwell_controller_to_effective_sim'] = (
                reducer.fixedstand_effective_sim - reducer.controller_ready_sim
                if reducer.fixedstand_effective_sim is not None and reducer.controller_ready_sim is not None else None)
            events['passive_dwell_spawn_to_effective_sim'] = (
                reducer.fixedstand_effective_sim - events.get('clock_progressing')
                if reducer.fixedstand_effective_sim is not None and events.get('clock_progressing') is not None else None)
            events['nonzero_cmd_events'] = [event for event in cache.cmd_events if event['nonzero']]
            cache.close()
        dump(run_dir / 'timeline.json', timeline)
        dump(run_dir / 'events.json', events)
        render_timeline(run_dir, timeline)
        manifest.update({
            'complete': True, 'final_state': reducer.state.value, 'final_reason': reducer.reason,
            'navigation_ready_final': reducer.navigation_ready, 'fixedstand_only_passed': fixedstand_complete,
            'processes_started': list(processes), 'timeline_points': len(timeline),
        })
        dump(run_dir / 'manifest.json', manifest)
        for proc in reversed(list(processes.values())):
            stop(proc)
    return 0 if fixedstand_complete or reducer.state == State.NAVIGATION_READY else 2


if __name__ == '__main__':
    raise SystemExit(main())
