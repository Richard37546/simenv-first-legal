"""Pure, fail-closed reducer for the A1 V1.1 startup supervisor.

The reducer has no ROS/Gazebo imports.  Its adapter supplies health evidence;
simulator truth must remain outside this module's online decision path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    WAIT_GAZEBO_SPAWN = 'WAIT_GAZEBO_SPAWN'
    START_JUNIOR_CTRL_EARLY = 'START_JUNIOR_CTRL_EARLY'
    WAIT_CONTROL_PREREQUISITES = 'WAIT_CONTROL_PREREQUISITES'
    REQUEST_FIXEDSTAND = 'REQUEST_FIXEDSTAND'
    VERIFY_FIXEDSTAND = 'VERIFY_FIXEDSTAND'
    START_OR_WAIT_PERCEPTION = 'START_OR_WAIT_PERCEPTION'
    WAIT_PERCEPTION_FRESH = 'WAIT_PERCEPTION_FRESH'
    REQUEST_RL = 'REQUEST_RL'
    VERIFY_RL_ZERO_HOLD = 'VERIFY_RL_ZERO_HOLD'
    NAVIGATION_READY = 'NAVIGATION_READY'
    FAIL_SAFE_HOLD = 'FAIL_SAFE_HOLD'


@dataclass
class Health:
    sim_time: float = 0.0
    gazebo_spawned: bool = False
    clock_progressing: bool = False
    junior_process_running: bool = False
    mode_service_available: bool = False
    controller_ready: bool = False
    joints_valid_continuous: bool = False
    imu_quaternion_valid: bool = False
    servo_updates_started: bool = False
    final_cmd_zero: bool = False
    no_unexpired_nonzero_cmd: bool = False
    fixedstand_service_accepted: bool = False
    rl_service_accepted: bool = False
    mode: str = ''
    fixedstand_stable: bool = False
    perception_fresh: bool = False
    rl_ready_fresh: bool = False
    rl_zero_hold_stable: bool = False
    critical_fresh: bool = False

    @property
    def control_prerequisites_ready(self):
        return all((
            self.clock_progressing,
            self.junior_process_running,
            self.mode_service_available,
            self.controller_ready,
            self.joints_valid_continuous,
            self.imu_quaternion_valid,
            self.servo_updates_started,
            self.final_cmd_zero,
            self.no_unexpired_nonzero_cmd,
        ))


def classify_offline_attitude(supervisor_healthy, truth_upright_score):
    """Offline-only classification helper; never feed truth back to control."""
    if truth_upright_score is not None and truth_upright_score >= 0.90 and not supervisor_healthy:
        return 'SUPERVISOR_ATTITUDE_PREDICATE_MISMATCH'
    if truth_upright_score is not None and truth_upright_score < 0.50:
        return 'ROBOT_PHYSICALLY_UNSTABLE'
    return 'NO_ATTITUDE_MISMATCH_EVIDENCE'


class Reducer:
    """State transitions and reasons; the adapter owns all process/service effects."""

    def __init__(self, passive_limit_sim=1.0, fixedstand_window_sim=10.0,
                 rl_window_sim=20.0, fixedstand_deadline_sim=20.0):
        self.state = State.WAIT_GAZEBO_SPAWN
        self.reason = 'waiting_for_gazebo_spawn_and_clock'
        self.passive_limit_sim = passive_limit_sim
        self.fixedstand_window_sim = fixedstand_window_sim
        self.rl_window_sim = rl_window_sim
        self.fixedstand_deadline_sim = fixedstand_deadline_sim
        self.controller_ready_sim = None
        self.fixedstand_request_sim = None
        self.fixedstand_effective_sim = None
        self.fixedstand_ok_since = None
        self.rl_ok_since = None
        self.navigation_ready = False

    def fail(self, reason):
        self.state = State.FAIL_SAFE_HOLD
        self.reason = reason
        self.navigation_ready = False

    def _passive_dwell_exceeded(self, h):
        return (self.controller_ready_sim is not None and
                self.fixedstand_effective_sim is None and
                h.sim_time - self.controller_ready_sim > self.passive_limit_sim)

    def step(self, h: Health):
        if self.state in (State.FAIL_SAFE_HOLD, State.NAVIGATION_READY):
            if self.state == State.NAVIGATION_READY and (
                    not h.critical_fresh or h.mode != 'RL' or not h.rl_ready_fresh or
                    not h.final_cmd_zero or not h.no_unexpired_nonzero_cmd):
                self.fail('NAVIGATION_READY_REVOKED_CRITICAL_HEALTH_STALE')
            return self.state

        # Before the first observed command the adapter has no evidence of a
        # residual command.  Preconditions still prevent any service request
        # until a zero command is observed; once one is observed, a recent
        # nonzero command is an immediate fail-closed condition.
        if h.final_cmd_zero and not h.no_unexpired_nonzero_cmd:
            self.fail('NONZERO_COMMAND_PRESENT')
            return self.state

        if self.state == State.WAIT_GAZEBO_SPAWN:
            if h.gazebo_spawned and h.clock_progressing:
                self.state = State.START_JUNIOR_CTRL_EARLY
                self.reason = 'gazebo_spawn_detected_start_junior_early'
        elif self.state == State.START_JUNIOR_CTRL_EARLY:
            if h.junior_process_running:
                self.state = State.WAIT_CONTROL_PREREQUISITES
                self.reason = 'junior_ctrl_running_parallel_to_control_checks'
        elif self.state == State.WAIT_CONTROL_PREREQUISITES:
            if h.controller_ready and self.controller_ready_sim is None:
                self.controller_ready_sim = h.sim_time
            if self._passive_dwell_exceeded(h):
                self.fail('PASSIVE_DWELL_EXCEEDED')
            elif h.control_prerequisites_ready:
                self.state = State.REQUEST_FIXEDSTAND
                self.reason = 'all_control_prerequisites_satisfied'
        elif self.state == State.REQUEST_FIXEDSTAND:
            if self._passive_dwell_exceeded(h):
                self.fail('PASSIVE_DWELL_EXCEEDED')
            elif h.fixedstand_service_accepted:
                self.fixedstand_request_sim = h.sim_time
                self.state = State.VERIFY_FIXEDSTAND
                self.reason = 'fixedstand_request_accepted_waiting_for_effective_mode'
        elif self.state == State.VERIFY_FIXEDSTAND:
            if h.mode != 'FIXEDSTAND':
                if self._passive_dwell_exceeded(h):
                    self.fail('PASSIVE_DWELL_EXCEEDED')
                else:
                    self.reason = 'waiting_for_fixedstand_mode_effective'
            else:
                if self.fixedstand_effective_sim is None:
                    self.fixedstand_effective_sim = h.sim_time
                if h.sim_time - self.fixedstand_effective_sim > self.fixedstand_deadline_sim:
                    self.fail('FIXEDSTAND_STABILITY_TIMEOUT')
                elif not h.fixedstand_stable:
                    self.fixedstand_ok_since = None
                    self.reason = 'FIXEDSTAND_HEALTH_NOT_STABLE'
                elif self.fixedstand_ok_since is None:
                    self.fixedstand_ok_since = h.sim_time
                    self.reason = 'fixedstand_stability_window_started'
                elif h.sim_time - self.fixedstand_ok_since >= self.fixedstand_window_sim:
                    self.state = State.START_OR_WAIT_PERCEPTION
                    self.reason = 'fixedstand_verified'
        elif self.state == State.START_OR_WAIT_PERCEPTION:
            self.state = State.WAIT_PERCEPTION_FRESH
            self.reason = 'perception_stack_start_requested_after_fixedstand'
        elif self.state == State.WAIT_PERCEPTION_FRESH:
            if h.mode != 'FIXEDSTAND' or not h.fixedstand_stable:
                self.fail('FIXEDSTAND_LOST_DURING_PERCEPTION_WAIT')
            elif h.perception_fresh:
                self.state = State.REQUEST_RL
                self.reason = 'perception_fresh_after_fixedstand'
        elif self.state == State.REQUEST_RL:
            if h.rl_service_accepted:
                self.state = State.VERIFY_RL_ZERO_HOLD
                self.reason = 'rl_request_accepted'
        elif self.state == State.VERIFY_RL_ZERO_HOLD:
            if h.mode != 'RL' or not h.rl_ready_fresh:
                self.rl_ok_since = None
                self.reason = 'waiting_for_rl_mode_or_fresh_ready'
            elif not h.rl_zero_hold_stable:
                self.rl_ok_since = None
                self.reason = 'RL_ZERO_HOLD_NOT_STABLE'
            elif self.rl_ok_since is None:
                self.rl_ok_since = h.sim_time
                self.reason = 'rl_zero_hold_window_started'
            elif h.sim_time - self.rl_ok_since >= self.rl_window_sim and h.critical_fresh:
                self.state = State.NAVIGATION_READY
                self.reason = 'startup_verified'
                self.navigation_ready = True
        return self.state
