import pathlib
import re
import unittest


RUNTIME = pathlib.Path(__file__).resolve().parents[1] / "scripts/slam_bev_runtime/run_slam_bev_runtime.sh"


class SlamBevRuntimeImuPriorContractTests(unittest.TestCase):
    def _icp_invocation(self):
        text = RUNTIME.read_text(encoding="utf-8")
        match = re.search(
            r'start_bg slam_bev_icp_odometry_raw.*?\n\s*sleep 2\n',
            text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "production icp_odometry invocation was not found")
        return match.group(0)

    def test_imu_prior_is_bound_to_the_production_icp_node(self):
        invocation = self._icp_invocation()
        self.assertIn("rosrun rtabmap_odom icp_odometry", invocation)
        self.assertEqual(invocation.count("_wait_imu_to_init:=true"), 1)
        self.assertEqual(invocation.count("imu:=/trunk_imu"), 1)

    def test_existing_icp_and_authority_contract_is_retained(self):
        invocation = self._icp_invocation()
        for token in (
            "_frame_id:=base",
            "_odom_frame_id:=team_livox_odom",
            "_publish_tf:=false",
            "_Odom/GuessMotion:=true",
            "_Odom/ResetCountdown:=1",
            "_Icp/PointToPlane:=true",
            "_Icp/VoxelSize:=0.0",
            "_Icp/PMOutlierRatio:=0.65",
            "scan_cloud:=/team/livox/scan_cloud_filtered",
            "odom:=/team/livox/icp_odom_raw",
        ):
            self.assertIn(token, invocation)
        self.assertIn("_startup_stable_samples:=10", RUNTIME.read_text(encoding="utf-8"))
        self.assertNotIn("Icp/MaxTranslation:=0.25", invocation)
        self.assertNotIn("Icp/MaxTranslation:=0.20", invocation)


if __name__ == "__main__":
    unittest.main()
