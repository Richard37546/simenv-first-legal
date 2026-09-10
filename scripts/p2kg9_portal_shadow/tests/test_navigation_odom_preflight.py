#!/usr/bin/env python3
"""Unit tests for the read-only navigation gated-odom preflight."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/p2kg9_portal_shadow/wait_for_navigation_odom_ready.py"


def load_module():
    old_modules = {name: sys.modules.get(name) for name in ("rospy", "nav_msgs", "nav_msgs.msg")}
    fake_rospy = types.ModuleType("rospy")
    fake_rospy.core = types.SimpleNamespace(is_initialized=lambda: True)
    fake_rospy.init_node = lambda *args, **kwargs: None
    fake_rospy.wait_for_message = lambda *args, **kwargs: None
    fake_nav_msgs = types.ModuleType("nav_msgs")
    fake_nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    fake_nav_msgs_msg.Odometry = object
    sys.modules.update({"rospy": fake_rospy, "nav_msgs": fake_nav_msgs, "nav_msgs.msg": fake_nav_msgs_msg})
    try:
        spec = importlib.util.spec_from_file_location("navigation_odom_preflight_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class Stamp:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


def odom(x=1.0, y=2.0, z=0.0, stamp=3.0):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=Stamp(stamp)),
        pose=types.SimpleNamespace(
            pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=z))
        ),
    )


class NavigationOdomPreflightTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_valid_message_is_ready_and_has_no_command_authority(self):
        ready, payload = self.module.describe_usable_odom(odom(), "/team/livox/icp_odom_gated")
        self.assertTrue(ready)
        self.assertEqual(payload["status"], "NAVIGATION_ODOM_READY")
        self.assertFalse(payload["command_authority"])

    def test_zero_stamp_is_not_treated_as_ready(self):
        ready, payload = self.module.describe_usable_odom(odom(stamp=0.0), "/team/livox/icp_odom_gated")
        self.assertFalse(ready)
        self.assertEqual(payload["reason"], "missing_message_stamp")

    def test_wait_timeout_is_a_structured_non_actuating_failure(self):
        fake_rospy = types.SimpleNamespace(
            core=types.SimpleNamespace(is_initialized=lambda: True),
            wait_for_message=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("timeout")),
        )
        ready, payload = self.module.wait_for_usable_odom(fake_rospy, object, "/team/livox/icp_odom_gated", 45.0)
        self.assertFalse(ready)
        self.assertEqual(payload["status"], "NAVIGATION_ODOM_UNAVAILABLE")
        self.assertFalse(payload["command_authority"])

    def test_source_has_no_control_publisher_or_authority_path(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("Publisher(", source)
        self.assertNotIn("rostopic pub", source)
        self.assertIn('"command_authority": False', source)
