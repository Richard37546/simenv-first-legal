#!/usr/bin/env python3
import math
import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "continuous_yaw_consistency.py"
SPEC = importlib.util.spec_from_file_location("continuous_yaw_consistency", str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContinuousYawConsistencyTest(unittest.TestCase):
    def test_constant_mounting_offset_is_not_an_error(self):
        monitor = MODULE.ContinuousYawConsistency(math.radians(2.9), 3)
        for degrees in (0.0, 15.0, 40.0):
            result = monitor.observe(math.radians(degrees), math.radians(degrees + 12.0))
        self.assertFalse(result["triggered"])
        self.assertAlmostEqual(result["yaw_error_rad"], 0.0, places=9)

    def test_persistent_drift_triggers_without_rebasing(self):
        monitor = MODULE.ContinuousYawConsistency(math.radians(2.9), 3)
        monitor.observe(0.0, 0.0)
        monitor.observe(math.radians(3.1), 0.0)
        monitor.observe(math.radians(3.2), 0.0)
        result = monitor.observe(math.radians(3.3), 0.0)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["consecutive_mismatch_samples"], 3)

    def test_small_error_resets_the_consecutive_window(self):
        monitor = MODULE.ContinuousYawConsistency(math.radians(2.9), 3)
        monitor.observe(0.0, 0.0)
        monitor.observe(math.radians(3.1), 0.0)
        result = monitor.observe(math.radians(1.0), 0.0)
        self.assertEqual(result["consecutive_mismatch_samples"], 0)
        self.assertFalse(result["triggered"])

    def test_wrapped_angles_remain_consistent(self):
        monitor = MODULE.ContinuousYawConsistency(math.radians(2.9), 2)
        monitor.observe(math.radians(179.0), math.radians(169.0))
        result = monitor.observe(math.radians(-179.0), math.radians(171.0))
        self.assertFalse(result["triggered"])
        self.assertAlmostEqual(result["yaw_error_rad"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
