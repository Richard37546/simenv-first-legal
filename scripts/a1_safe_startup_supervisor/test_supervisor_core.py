#!/usr/bin/env python3
"""Offline V1.1 reducer tests: no ROS, Gazebo, truth subscription, or commands."""
import json
import pathlib
import unittest

from supervisor_core import Health, Reducer, State, classify_offline_attitude


def ready(sim_time=0.0, **overrides):
    values = dict(
        sim_time=sim_time, gazebo_spawned=True, clock_progressing=True,
        junior_process_running=True, mode_service_available=True,
        controller_ready=True, joints_valid_continuous=True,
        imu_quaternion_valid=True, servo_updates_started=True,
        final_cmd_zero=True, no_unexpired_nonzero_cmd=True,
    )
    values.update(overrides)
    return Health(**values)


def drive_to_request(reducer):
    reducer.step(ready(0.0))
    reducer.step(ready(0.1))
    reducer.step(ready(0.2))
    assert reducer.state == State.REQUEST_FIXEDSTAND


def drive_to_verify(reducer):
    drive_to_request(reducer)
    reducer.step(ready(0.3, fixedstand_service_accepted=True))
    assert reducer.state == State.VERIFY_FIXEDSTAND


class SupervisorCoreV11Tests(unittest.TestCase):
    def test_01_waits_for_spawn_and_clock(self):
        reducer = Reducer(); reducer.step(Health(gazebo_spawned=True))
        self.assertEqual(reducer.state, State.WAIT_GAZEBO_SPAWN)

    def test_02_requests_early_junior_before_control_ready(self):
        reducer = Reducer(); reducer.step(ready(0.0, controller_ready=False, joints_valid_continuous=False, servo_updates_started=False))
        self.assertEqual(reducer.state, State.START_JUNIOR_CTRL_EARLY)

    def test_03_controller_delay_does_not_block_junior_start_state(self):
        reducer = Reducer(); reducer.step(ready(0.0, controller_ready=False, joints_valid_continuous=False, servo_updates_started=False))
        reducer.step(ready(0.1, controller_ready=False, joints_valid_continuous=False, servo_updates_started=False))
        self.assertEqual(reducer.state, State.WAIT_CONTROL_PREREQUISITES)

    def test_04_livox_delay_does_not_block_fixedstand_request(self):
        reducer = Reducer(); drive_to_request(reducer)
        self.assertEqual(reducer.state, State.REQUEST_FIXEDSTAND)

    def test_05_icp_delay_does_not_block_fixedstand_request(self):
        reducer = Reducer(); drive_to_request(reducer)
        self.assertEqual(reducer.reason, 'all_control_prerequisites_satisfied')

    def test_06_l3v_delay_does_not_block_fixedstand_request(self):
        reducer = Reducer(); drive_to_request(reducer)
        self.assertNotEqual(reducer.state, State.WAIT_PERCEPTION_FRESH)

    def test_07_joint_delay_blocks_request(self):
        reducer = Reducer(); reducer.step(ready(0.0)); reducer.step(ready(0.1)); reducer.step(ready(0.2, joints_valid_continuous=False))
        self.assertEqual(reducer.state, State.WAIT_CONTROL_PREREQUISITES)

    def test_08_imu_delay_blocks_request(self):
        reducer = Reducer(); reducer.step(ready(0.0)); reducer.step(ready(0.1)); reducer.step(ready(0.2, imu_quaternion_valid=False))
        self.assertEqual(reducer.state, State.WAIT_CONTROL_PREREQUISITES)

    def test_09_service_delay_blocks_request(self):
        reducer = Reducer(); reducer.step(ready(0.0)); reducer.step(ready(0.1)); reducer.step(ready(0.2, mode_service_available=False))
        self.assertEqual(reducer.state, State.WAIT_CONTROL_PREREQUISITES)

    def test_10_passive_dwell_timeout(self):
        reducer = Reducer(passive_limit_sim=.5); reducer.step(ready(0.0)); reducer.step(ready(0.1)); reducer.step(ready(0.2, mode_service_available=False)); reducer.step(ready(0.8, mode_service_available=False))
        self.assertEqual(reducer.reason, 'PASSIVE_DWELL_EXCEEDED')

    def test_11_service_accept_without_mode_fails_passive_dwell(self):
        reducer = Reducer(passive_limit_sim=.5); drive_to_verify(reducer); reducer.step(ready(.8, mode='PASSIVE'))
        self.assertEqual(reducer.reason, 'PASSIVE_DWELL_EXCEEDED')

    def test_12_fixedstand_success(self):
        reducer = Reducer(fixedstand_window_sim=.2); drive_to_verify(reducer)
        reducer.step(ready(.4, mode='FIXEDSTAND', fixedstand_stable=True)); reducer.step(ready(.7, mode='FIXEDSTAND', fixedstand_stable=True))
        self.assertEqual(reducer.state, State.START_OR_WAIT_PERCEPTION)

    def test_13_perception_delay_keeps_fixedstand(self):
        reducer = Reducer(fixedstand_window_sim=.1); drive_to_verify(reducer)
        reducer.step(ready(.4, mode='FIXEDSTAND', fixedstand_stable=True)); reducer.step(ready(.6, mode='FIXEDSTAND', fixedstand_stable=True)); reducer.step(ready(.7, mode='FIXEDSTAND', fixedstand_stable=True, perception_fresh=False))
        self.assertEqual(reducer.state, State.WAIT_PERCEPTION_FRESH)

    def test_14_perception_fresh_reaches_rl_request(self):
        reducer = Reducer(); reducer.state = State.WAIT_PERCEPTION_FRESH
        reducer.step(ready(1.0, mode='FIXEDSTAND', fixedstand_stable=True, perception_fresh=True))
        self.assertEqual(reducer.state, State.REQUEST_RL)

    def test_15_rl_ready_ttl_blocks_navigation(self):
        reducer = Reducer(); reducer.state = State.VERIFY_RL_ZERO_HOLD
        reducer.step(ready(1.0, mode='RL', rl_ready_fresh=False))
        self.assertEqual(reducer.reason, 'waiting_for_rl_mode_or_fresh_ready')

    def test_16_nonzero_residual_fails_closed(self):
        reducer = Reducer(); reducer.step(ready(0.0, no_unexpired_nonzero_cmd=False))
        self.assertEqual(reducer.reason, 'NONZERO_COMMAND_PRESENT')

    def test_17_navigation_ready_revokes_on_critical_loss(self):
        reducer = Reducer(rl_window_sim=.1); reducer.state = State.VERIFY_RL_ZERO_HOLD
        reducer.step(ready(1.0, mode='RL', rl_ready_fresh=True, rl_zero_hold_stable=True, critical_fresh=True))
        reducer.step(ready(1.2, mode='RL', rl_ready_fresh=True, rl_zero_hold_stable=True, critical_fresh=True))
        reducer.step(ready(1.3, mode='RL', rl_ready_fresh=True, critical_fresh=False))
        self.assertEqual(reducer.reason, 'NAVIGATION_READY_REVOKED_CRITICAL_HEALTH_STALE')

    def test_18_navigation_ready_revokes_on_nonzero_command(self):
        reducer = Reducer(); reducer.state = State.NAVIGATION_READY; reducer.navigation_ready = True
        reducer.step(ready(1.0, mode='RL', rl_ready_fresh=True, critical_fresh=True, no_unexpired_nonzero_cmd=False))
        self.assertFalse(reducer.navigation_ready)

    def test_19_fixedstand_lost_during_perception_fails(self):
        reducer = Reducer(); reducer.state = State.WAIT_PERCEPTION_FRESH
        reducer.step(ready(1.0, mode='PASSIVE', fixedstand_stable=False))
        self.assertEqual(reducer.reason, 'FIXEDSTAND_LOST_DURING_PERCEPTION_WAIT')

    def test_20_offline_truth_mismatch_is_classified_not_controlled(self):
        self.assertEqual(classify_offline_attitude(False, .99), 'SUPERVISOR_ATTITUDE_PREDICATE_MISMATCH')

    def test_21_offline_truth_unstable_classification(self):
        self.assertEqual(classify_offline_attitude(True, .1), 'ROBOT_PHYSICALLY_UNSTABLE')

    def test_22_fixedstand_deadline_after_effective_mode(self):
        reducer = Reducer(fixedstand_deadline_sim=.2); drive_to_verify(reducer)
        reducer.step(ready(.4, mode='FIXEDSTAND', fixedstand_stable=False)); reducer.step(ready(.7, mode='FIXEDSTAND', fixedstand_stable=False))
        self.assertEqual(reducer.reason, 'FIXEDSTAND_STABILITY_TIMEOUT')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SupervisorCoreV11Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    out = pathlib.Path('/home/richard/simenv_official_clean/debug/a1_safe_startup_supervisor_v1_1')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'unit_tests.json').write_text(json.dumps({'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful()}, indent=2) + '\n')
    raise SystemExit(not result.wasSuccessful())
