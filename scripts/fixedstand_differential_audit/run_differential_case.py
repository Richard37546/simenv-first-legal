#!/usr/bin/env python3
"""G/S FixedStand differential runner. It never publishes nonzero Twist."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import time

import rosgraph
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetPhysicsProperties
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger
from unitree_legged_msgs.msg import MotorCmd, MotorState


ROOT = pathlib.Path('/home/richard/simenv_official_clean')
OUT = ROOT / 'debug/fixedstand_differential_audit/runs'
JUNIOR = ROOT / 'devel/lib/unitree_guide/junior_ctrl'
LIVOX = ROOT / 'devel/lib/liblivox_laser_simulation.so'
JOINTS = ('FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
          'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def rpy(q):
    return (
        math.atan2(2 * (q.w*q.x + q.y*q.z), 1 - 2 * (q.x*q.x + q.y*q.y)),
        math.asin(max(-1.0, min(1.0, 2 * (q.w*q.y - q.z*q.x)))),
        math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z)),
    )


def upright(q):
    return 1.0 - 2.0 * (q.x*q.x + q.y*q.y)


def finite(values):
    return all(math.isfinite(value) for value in values)


def start(command, log, env):
    handle = log.open('w', encoding='utf-8')
    proc = subprocess.Popen(['/bin/bash', '-lc', command], cwd=str(ROOT), stdout=handle,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    proc._log_handle = handle
    return proc


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        deadline = time.monotonic() + 8.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    if proc:
        proc._log_handle.close()


class Capture:
    def __init__(self):
        self.sim_time = 0.0
        self.previous_clock = None
        self.first, self.events = {}, []
        self.joint_state = {joint: [] for joint in JOINTS}
        self.motor_cmd = {joint: [] for joint in JOINTS}
        self.truth, self.imu, self.mode, self.cmd_vel = [], [], [], []
        self.state_count = {joint: 0 for joint in JOINTS}
        self.imu_count = 0
        self.receipt = {}
        self.subscribers = [
            rospy.Subscriber('/clock', Clock, self.clock_cb, queue_size=2000),
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.truth_cb, queue_size=2000),
            rospy.Subscriber('/trunk_imu', Imu, self.imu_cb, queue_size=2000),
            rospy.Subscriber('/unitree/controller_mode', String, self.mode_cb, queue_size=200),
            rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb, queue_size=500),
        ]
        for joint in JOINTS:
            prefix = '/a1_gazebo/' + joint + '_controller'
            self.subscribers.append(rospy.Subscriber(prefix + '/state', MotorState, self.state_cb, callback_args=joint, queue_size=2000))
            self.subscribers.append(rospy.Subscriber(prefix + '/command', MotorCmd, self.command_cb, callback_args=joint, queue_size=2000))

    def note(self, name, condition=True):
        if condition and name not in self.first:
            self.first[name] = self.sim_time

    def put(self, name):
        self.receipt[name] = time.monotonic()

    def fresh(self, name, ttl):
        return name in self.receipt and time.monotonic() - self.receipt[name] <= ttl

    def clock_cb(self, msg):
        self.sim_time = msg.clock.to_sec()
        if self.previous_clock is not None and self.sim_time > self.previous_clock + 1e-6:
            self.note('clock_progressing')
        self.previous_clock = self.sim_time
        self.put('clock'); self.note('clock_received')

    def truth_cb(self, msg):
        if 'a1_gazebo' not in msg.name:
            return
        pose = msg.pose[msg.name.index('a1_gazebo')]
        self.truth.append({'sim_time': self.sim_time, 'position': [pose.position.x, pose.position.y, pose.position.z],
                           'rpy': list(rpy(pose.orientation)), 'upright_score': upright(pose.orientation)})
        self.note('spawn_truth_received')
        if pose.position.z < .35:
            self.note('base_height_first_below_0_35')
        if upright(pose.orientation) < .9:
            self.note('upright_first_below_0_9')
        if pose.position.z < .20 or upright(pose.orientation) < .5:
            self.note('first_irrecoverable_instability')

    def imu_cb(self, msg):
        q = msg.orientation
        values = (q.x, q.y, q.z, q.w)
        norm = math.sqrt(sum(value * value for value in values)) if finite(values) else float('nan')
        valid = finite(values) and .90 <= norm <= 1.10
        self.imu_count = self.imu_count + 1 if valid else 0
        self.imu.append({'sim_time': self.sim_time, 'frame_id': msg.header.frame_id, 'q_norm': norm,
                         'rpy': list(rpy(q)) if valid else [None, None, None],
                         'gyro': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]})
        self.put('imu'); self.note('imu_valid', valid)

    def mode_cb(self, msg):
        self.mode.append({'sim_time': self.sim_time, 'mode': msg.data})
        self.put('mode'); self.note('mode_received')
        self.note('fixedstand_mode_effective', msg.data == 'FIXEDSTAND')

    def cmd_cb(self, msg):
        fields = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z)
        self.cmd_vel.append({'sim_time': self.sim_time, 'values': list(fields), 'nonzero': any(abs(value) > 1e-5 for value in fields)})
        self.cmd_vel = self.cmd_vel[-2000:]
        self.put('cmd_vel'); self.note('zero_cmd_observed', not self.cmd_vel[-1]['nonzero'])
        self.note('nonzero_cmd_observed', self.cmd_vel[-1]['nonzero'])

    def state_cb(self, msg, joint):
        values = (msg.q, msg.dq, msg.tauEst)
        if finite(values):
            self.state_count[joint] += 1
        else:
            self.state_count[joint] = 0
        self.joint_state[joint].append({'sim_time': self.sim_time, 'q': msg.q, 'dq': msg.dq, 'tau_est': msg.tauEst})
        self.put('state_' + joint); self.note('joint_state_valid', all(count >= 3 for count in self.state_count.values()))

    def command_cb(self, msg, joint):
        values = (msg.q, msg.dq, msg.tau, msg.Kp, msg.Kd)
        self.motor_cmd[joint].append({'sim_time': self.sim_time, 'q': msg.q, 'dq': msg.dq, 'tau': msg.tau, 'kp': msg.Kp, 'kd': msg.Kd,
                                       'finite': finite(values)})
        self.put('command_' + joint); self.note('first_motorcmd_' + joint)
        self.note('first_motorcmd_all', all(self.motor_cmd[item] for item in JOINTS))

    def service_available(self):
        try:
            _code, _message, state = rospy.get_master().getSystemState()
            return any(name == '/unitree/request_fixedstand' for name, _providers in state[2])
        except Exception:
            return False

    def prereqs(self, junior_live):
        state_ok = all(self.fresh('state_' + joint, 1.0) and self.state_count[joint] >= 3 for joint in JOINTS)
        imu_ok = self.fresh('imu', .5) and self.imu_count >= 3
        final_zero = self.fresh('cmd_vel', .75) and bool(self.cmd_vel) and not self.cmd_vel[-1]['nonzero']
        no_residual = not any(item['nonzero'] and item['sim_time'] >= self.sim_time - .75 for item in self.cmd_vel)
        values = {'clock_progressing': self.fresh('clock', 1.0) and 'clock_progressing' in self.first,
                  'junior_process_running': junior_live, 'mode_service_available': self.service_available(),
                  'continuous_joint_state_valid': state_ok, 'imu_quaternion_valid': imu_ok,
                  'servo_updates_started': state_ok, 'final_cmd_zero': final_zero,
                  'no_unexpired_nonzero_cmd': no_residual}
        for name, value in values.items():
            self.note(name, value)
        return values, all(values.values())

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


def graph_snapshot():
    master = rospy.get_master()
    try:
        _code, _message, state = master.getSystemState()
        nodes = master.getPublishedTopics('/')
        return {'publishers': state[0], 'subscribers': state[1], 'services': state[2], 'published_topics': nodes,
                'nodes': rosgraph.Master(rospy.get_name()).getSystemState()[2]}
    except Exception as exc:
        return {'error': str(exc)}


def parameter_snapshot():
    result = {}
    for name in rospy.get_param_names():
        try:
            result[name] = rospy.get_param(name)
        except Exception as exc:
            result[name] = '<read_error: %s>' % exc
    return result


def physics_snapshot():
    try:
        rospy.wait_for_service('/gazebo/get_physics_properties', timeout=3.0)
        value = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)()
        return {'time_step': value.time_step, 'max_update_rate': value.max_update_rate,
                'gravity': [value.gravity.x, value.gravity.y, value.gravity.z], 'ode_config': str(value.ode_config)}
    except Exception as exc:
        return {'error': str(exc)}


def wait_master(deadline):
    master = rosgraph.Master('fixedstand_differential_audit')
    while time.monotonic() < deadline:
        try:
            master.getPid()
            return True
        except Exception:
            time.sleep(.05)
    return False


def service_call():
    try:
        rospy.wait_for_service('/unitree/request_fixedstand', timeout=.25)
        return bool(rospy.ServiceProxy('/unitree/request_fixedstand', Trigger)().success)
    except (rospy.ROSException, rospy.ServiceException):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('G', 'S'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--duration-sim-sec', type=float, default=10.0)
    parser.add_argument('--wall-watchdog-sec', type=float, default=900.0)
    args = parser.parse_args()
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    run_dir = OUT / (args.group + '_' + args.run_id + '_' + stamp + '_pid' + str(os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({'GUI': 'false', 'PAUSED': 'false', 'UNITREE_CTRL_DT': env.get('UNITREE_CTRL_DT', '0.006'),
                'BUILDING_WORLD_FILE': str(ROOT / 'generated_building/competition_scene.world')})
    source = ('source /opt/ros/noetic/setup.bash; source /home/richard/simenv_official_clean/devel/setup.bash; '
              'export GAZEBO_MODEL_PATH=/home/richard/simenv_official_clean/generated_building:$(rospack find unitree_gazebo)/models:${GAZEBO_MODEL_PATH:-}; ')
    processes, capture, timer = {}, None, None
    start_wall = time.monotonic()
    requested = False
    request_sim = None
    status = 'UNKNOWN'
    configuration = {'group': args.group, 'run_id': args.run_id, 'gui': False, 'duration_sim_sec': args.duration_sim_sec,
                     'truth_used_for_control': False, 'nonzero_cmd_published': False,
                     'environment': {name: env.get(name, '') for name in ('GUI', 'PAUSED', 'UNITREE_CTRL_DT', 'GAZEBO_PLUGIN_PATH', 'ROS_PACKAGE_PATH', 'ROS_MASTER_URI')},
                     'executables': {'junior_ctrl': {'path': str(JUNIOR), 'sha256': sha256(JUNIOR)},
                                     'livox_plugin': {'path': str(LIVOX), 'sha256': sha256(LIVOX)}},
                     'launch_command': 'roslaunch unitree_guide multi_floor_gazeboSim.launch gui:=false paused:=false user_debug:=False rname:=a1 robot_x:=0.0 robot_y:=-2.2 robot_z:=0.6 robot_yaw:=1.5708'}
    try:
        if args.group == 'G':
            processes['gazebo'] = start(source + 'exec roslaunch unitree_guide multi_floor_gazeboSim.launch gui:=false paused:=false user_debug:=False rname:=a1 robot_x:=0.0 robot_y:=-2.2 robot_z:=0.6 robot_yaw:=1.5708', run_dir / 'gazebo.log', env)
        else:
            processes['supervisor'] = start(source + 'exec python3 scripts/a1_safe_startup_supervisor/a1_safe_startup_supervisor.py --run-id differential_' + args.run_id + ' --fixedstand-only --offline-truth-capture', run_dir / 'supervisor.log', env)
        if not wait_master(time.monotonic() + 90.0):
            raise RuntimeError('ros_master_not_ready')
        rospy.init_node('fixedstand_differential_capture_' + args.group.lower(), anonymous=True, disable_signals=True)
        capture = Capture()
        zero = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        if args.group == 'G':
            subprocess.check_call(['/bin/bash', '-lc', source + 'rosparam set /robot_name a1'], cwd=str(ROOT), env=env)
            processes['junior_ctrl'] = start(source + 'exec ' + str(JUNIOR), run_dir / 'junior_ctrl.log', env)
        # Keep exactly one zero-command source in each group: this direct G
        # runner publishes it, while S relies on the supervisor's own timer.
        timer = rospy.Timer(rospy.Duration(.1), lambda _event: zero.publish(Twist())) if args.group == 'G' else None
        configuration['graph_before'] = graph_snapshot()
        configuration['parameters'] = parameter_snapshot()
        configuration['physics'] = physics_snapshot()
        target_sim = None
        while time.monotonic() - start_wall < args.wall_watchdog_sec and not rospy.is_shutdown():
            junior_live = (processes.get('junior_ctrl').poll() is None if 'junior_ctrl' in processes else
                           processes.get('supervisor').poll() is None)
            conditions, ready = capture.prereqs(junior_live)
            if args.group == 'G' and ready and not requested:
                requested = service_call()
                if requested:
                    request_sim = capture.sim_time
                    capture.note('fixedstand_service_request')
                    target_sim = request_sim + args.duration_sim_sec
            if args.group == 'S' and 'fixedstand_mode_effective' in capture.first and target_sim is None:
                target_sim = capture.first['fixedstand_mode_effective'] + args.duration_sim_sec
            if target_sim is not None and capture.sim_time >= target_sim:
                status = 'duration_complete'
                break
            if args.group == 'S' and processes['supervisor'].poll() is not None:
                status = 'supervisor_exited_before_duration'
                break
            time.sleep(.005)
        else:
            status = 'wall_watchdog_exceeded'
        configuration['graph_after'] = graph_snapshot()
        configuration['request_conditions'] = conditions if 'conditions' in locals() else {}
    finally:
        if timer:
            timer.shutdown()
        if capture:
            result = {'configuration': configuration, 'status': status, 'request_success': requested,
                      'request_sim_time': request_sim, 'final_sim_time': capture.sim_time,
                      'first_events_sim': capture.first, 'truth': capture.truth, 'imu': capture.imu,
                      'mode': capture.mode, 'joint_state': capture.joint_state, 'motor_cmd': capture.motor_cmd,
                      'cmd_vel': capture.cmd_vel, 'processes': {name: {'pid': proc.pid, 'exit_code_before_cleanup': proc.poll()} for name, proc in processes.items()}}
            dump(run_dir / 'capture.json', result)
            capture.close()
        for proc in reversed(list(processes.values())):
            stop(proc)
    return 0 if status == 'duration_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
