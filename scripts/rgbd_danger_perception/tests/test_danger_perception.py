import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from danger_perception_node import (  # noqa: E402
    CAMERA_FRAME,
    TARGET_FRAME,
    CameraModel,
    DangerObservation,
    DangerTrackManager,
    DetectorConfig,
    GeometryEvidence,
    OdomSample,
    RedCandidate,
    ShadowValidationRecorder,
    adaptive_sync_tolerance,
    associated_candidate_points,
    atomic_write_json,
    camera_point_to_odom,
    classify_sphere_geometry,
    extract_red_candidates,
    multi_face_plane_evidence,
    nearest_sample,
    rounded_image_indices,
)


def red_image(shape=(120, 160), circle=True):
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if circle:
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        image[(xx - 80) ** 2 + (yy - 60) ** 2 <= 20 ** 2] = [255, 0, 0]
    else:
        image[45:75, 45:120] = [255, 0, 0]
    return image


def camera():
    return CameraModel(CAMERA_FRAME, 160, 120, [100.0, 0, 80.0, 0, 100.0, 60.0, 0, 0, 1.0])


def sphere_points(center=(0.0, 0.0, 2.0), radius=0.15):
    rows = []
    for x in np.linspace(-0.13, 0.13, 13):
        for y in np.linspace(-0.13, 0.13, 13):
            if x * x + y * y < 0.14 * 0.14:
                z = center[2] - math.sqrt(radius * radius - x * x - y * y)
                rows.append([center[0] + x, center[1] + y, z])
    return np.asarray(rows, dtype=float)


def planar_points():
    return np.asarray([[x, y, 2.0] for x in np.linspace(-0.15, 0.15, 15) for y in np.linspace(-0.15, 0.15, 15)], dtype=float)


def multi_face_cube_points():
    """Three visible cube faces with the support imbalance measured in R0126."""
    rows = []
    for y in np.linspace(-0.15, 0.15, 24):
        for z in np.linspace(1.85, 2.15, 24):
            rows.append([-0.15, y, z])
    for x in np.linspace(-0.15, 0.15, 19):
        for z in np.linspace(1.85, 2.15, 19):
            rows.append([x, -0.15, z])
    for x in np.linspace(-0.15, 0.15, 10):
        for y in np.linspace(-0.15, 0.15, 10):
            rows.append([x, y, 2.15])
    return np.asarray(rows, dtype=float)


def candidate_from_circle():
    candidates = extract_red_candidates(red_image(), DetectorConfig(min_component_area_px=20))
    assert candidates
    return candidates[0]


def evidence(position=(1.0, 2.0, 0.5), confidence=0.9):
    geometry = GeometryEvidence(True, confidence, (0.0, 0.0, 2.0), 0.15, 0.001, 0.01, 80, None)
    return DangerObservation(1.0, position, confidence, geometry)


class DangerPerceptionTests(unittest.TestCase):
    def test_t1_red_spherical_candidate_extracted(self):
        candidate = candidate_from_circle()
        self.assertGreater(candidate.area_px, 1000)
        self.assertGreater(candidate.circularity, 0.55)

    def test_t2_red_box_like_image_is_not_sphere_shape(self):
        candidates = extract_red_candidates(red_image(circle=False), DetectorConfig(min_component_area_px=20))
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].aspect_ratio, 1.45)

    def test_t3_non_red_rejected(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[20:60, 20:60] = [0, 255, 0]
        self.assertEqual(extract_red_candidates(image, DetectorConfig(min_component_area_px=5)), [])

    def test_t4_projection_associates_candidate_points(self):
        candidate = candidate_from_circle()
        points = np.vstack((np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]]), np.array([[1.0, 1.0, 2.0]])))
        associated = associated_candidate_points(candidate, points, camera(), DetectorConfig(min_range_m=0.1))
        self.assertEqual(len(associated), 3)

    def test_t5_nan_inf_points_ignored(self):
        candidate = candidate_from_circle()
        points = np.array([[0.0, 0.0, 2.0], [np.nan, 0.0, 2.0], [0.0, np.inf, 2.0], [0.0, 0.0, -1.0]])
        associated = associated_candidate_points(candidate, points, camera(), DetectorConfig(min_range_m=0.1))
        self.assertEqual(len(associated), 1)

    def test_t6_unorganized_points_array_supported(self):
        candidate = candidate_from_circle()
        points = sphere_points()
        associated = associated_candidate_points(candidate, points, camera(), DetectorConfig(min_range_m=0.1))
        self.assertGreater(len(associated), 24)

    def test_t7_sphere_geometry_scores_above_planar_box(self):
        candidate = candidate_from_circle()
        config = DetectorConfig(min_associated_points=24)
        sphere = classify_sphere_geometry(candidate, sphere_points(), config)
        plane = classify_sphere_geometry(candidate, planar_points(), config)
        self.assertTrue(sphere.sphere_like)
        self.assertAlmostEqual(sphere.fitted_radius_m, config.expected_sphere_radius_m, places=3)
        self.assertLessEqual(sphere.sphere_residual_m, config.max_sphere_residual_m)
        self.assertFalse(multi_face_plane_evidence(sphere_points(), config).multi_face_planar)
        self.assertFalse(plane.sphere_like)
        self.assertGreater(sphere.confidence, plane.confidence)

    def test_t7a_multi_face_cube_rejected_by_two_plane_evidence(self):
        # This deliberately lets the synthetic cube through the earlier RMS
        # gate so the strengthened C9 discriminator itself is exercised.
        config = DetectorConfig(min_associated_points=24, max_sphere_residual_m=0.10)
        points = multi_face_cube_points()
        evidence = multi_face_plane_evidence(points, config)
        result = classify_sphere_geometry(candidate_from_circle(), points, config)
        self.assertTrue(evidence.multi_face_planar)
        self.assertGreaterEqual(evidence.first_support_fraction, config.multi_plane_first_min_support_fraction)
        self.assertGreaterEqual(evidence.second_support_fraction, config.multi_plane_second_min_support_fraction)
        self.assertLessEqual(evidence.normal_abs_dot, config.multi_plane_max_abs_normal_dot)
        self.assertFalse(result.sphere_like)
        self.assertEqual(result.rejection_reason, "multi_face_box_like_geometry")

    def test_t7b_rotated_multi_face_cube_rejected(self):
        angle = math.radians(37.0)
        rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        points = multi_face_cube_points().dot(rotation.T)
        result = classify_sphere_geometry(
            candidate_from_circle(),
            points,
            DetectorConfig(min_associated_points=24, max_sphere_residual_m=0.10),
        )
        self.assertFalse(result.sphere_like)
        self.assertEqual(result.rejection_reason, "multi_face_box_like_geometry")

    def test_t8_sphere_center_estimate_near_known_center(self):
        candidate = candidate_from_circle()
        result = classify_sphere_geometry(candidate, sphere_points(center=(0.02, -0.01, 2.0)), DetectorConfig(min_associated_points=24))
        self.assertIsNotNone(result.center_camera_xyz_m)
        self.assertLess(np.linalg.norm(np.asarray(result.center_camera_xyz_m) - np.array([0.02, -0.01, 2.0])), 0.02)

    def test_t9_optical_to_base_and_mount_translation(self):
        odom = OdomSample(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(camera_point_to_odom((0.0, 0.0, 1.0), odom), (1.28, 0.0, 0.043))
        self.assertEqual(camera_point_to_odom((1.0, 0.0, 0.0), odom), (0.28, -1.0, 0.043))
        self.assertEqual(camera_point_to_odom((0.0, 1.0, 0.0), odom), (0.28, 0.0, -0.957))

    def test_t10_odom_composition_rotation(self):
        yaw90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
        odom = OdomSample(1.0, (5.0, 1.0, 0.0), yaw90)
        point = camera_point_to_odom((0.0, 0.0, 1.0), odom)
        self.assertAlmostEqual(point[0], 5.0, places=6)
        self.assertAlmostEqual(point[1], 2.28, places=6)
        self.assertAlmostEqual(point[2], 0.043, places=6)

    def test_t11_stale_odom_fails_closed(self):
        old = OdomSample(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.assertIsNone(nearest_sample([old], 2.0, 0.05))

    def test_t12_single_observation_is_tentative(self):
        manager = DangerTrackManager(DetectorConfig(confirmation_observations=2))
        track = manager.add_observation(evidence())
        self.assertEqual(track.state, "TENTATIVE")
        self.assertEqual(manager.snapshot(1.0)["tracks"], [])
        self.assertEqual(manager.hypotheses_snapshot(1.0)["hypotheses"][0]["hypothesis_id"], "danger-0001")

    def test_t13_consistent_observations_confirm(self):
        manager = DangerTrackManager(DetectorConfig(confirmation_observations=2))
        for stamp in (1.0, 1.1):
            obs = evidence((1.0, 2.0, 0.5))
            obs.stamp_sec = stamp
            manager.add_observation(obs)
        self.assertEqual(manager.tracks[0].state, "CONFIRMED")

    def test_t14_same_sphere_deduplicates(self):
        manager = DangerTrackManager(DetectorConfig(confirmation_observations=2, association_distance_m=0.3))
        first = evidence((1.0, 2.0, 0.5))
        second = evidence((1.1, 2.0, 0.5))
        second.stamp_sec = 1.2
        manager.add_observation(first)
        manager.add_observation(second)
        self.assertEqual(len(manager.tracks), 1)
        self.assertEqual(manager.tracks[0].track_id, "danger-0001")

    def test_t15_separate_spheres_create_tracks(self):
        manager = DangerTrackManager(DetectorConfig(association_distance_m=0.3))
        manager.add_observation(evidence((0.0, 0.0, 0.0)))
        second = evidence((1.0, 0.0, 0.0))
        second.stamp_sec = 1.1
        manager.add_observation(second)
        self.assertEqual([track.track_id for track in manager.tracks], ["danger-0001", "danger-0002"])

    def test_t16_uncertain_red_geometry_never_confirms(self):
        result = classify_sphere_geometry(candidate_from_circle(), planar_points(), DetectorConfig(min_associated_points=24))
        manager = DangerTrackManager(DetectorConfig(confirmation_observations=1))
        if result.sphere_like:
            manager.add_observation(evidence())
        self.assertEqual(manager.snapshot(1.0)["tracks"], [])

    def test_t17_public_json_schema_exact(self):
        manager = DangerTrackManager(DetectorConfig(confirmation_observations=1))
        manager.add_observation(evidence())
        second = evidence()
        second.stamp_sec = 1.1
        manager.add_observation(second)
        snapshot = manager.snapshot(1.0)
        self.assertEqual(set(snapshot), {"schema_version", "stamp_sec", "frame_id", "tracks"})
        self.assertEqual(snapshot["frame_id"], TARGET_FRAME)
        self.assertEqual(set(snapshot["tracks"][0]), {"track_id", "position_xyz_m", "state", "confidence", "confirmation_count", "last_observed_stamp_sec"})

    def test_t18_empty_snapshot_is_legal(self):
        snapshot = DangerTrackManager(DetectorConfig()).snapshot(0.0)
        self.assertEqual(snapshot, {"schema_version": 1, "stamp_sec": 0.0, "frame_id": TARGET_FRAME, "tracks": []})

    def test_t18a_tentative_survives_one_missed_frame_then_expires(self):
        manager = DangerTrackManager(DetectorConfig(tentative_hypothesis_ttl_sec=2.0))
        manager.add_observation(evidence())
        self.assertEqual(len(manager.hypotheses_snapshot(1.5)["hypotheses"]), 1)
        self.assertEqual(manager.hypotheses_snapshot(3.1)["hypotheses"], [])

    def test_t18b_hypothesis_schema_contains_tentative_only(self):
        manager = DangerTrackManager(DetectorConfig())
        manager.add_observation(evidence())
        payload = manager.hypotheses_snapshot(1.0)
        self.assertEqual(set(payload), {"schema_version", "stamp_sec", "frame_id", "hypotheses"})
        self.assertEqual(payload["hypotheses"][0]["state"], "TENTATIVE")
        self.assertEqual(set(payload["hypotheses"][0]), {"hypothesis_id", "state", "position_xyz_m", "confidence", "support_count", "last_observed_stamp_sec"})

    def test_t19_atomic_snapshot_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_danger_tracks.json"
            payload = DangerTrackManager(DetectorConfig()).snapshot(0.0)
            atomic_write_json(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_t20_no_superseded_detector_import(self):
        source = (MODULE_DIR / "danger_perception_node.py").read_text(encoding="utf-8")
        for forbidden in ("vision_scene_semantics", "room_frontier_viewpoint_selector", "room_scan_summary", "danger_source_visible", "urllib.request"):
            self.assertNotIn(forbidden, source)

    def test_t20a_confirmation_never_allows_one_observation(self):
        manager = DangerTrackManager(DetectorConfig(confirmation_observations=1))
        self.assertEqual(manager.add_observation(evidence()).state, "TENTATIVE")

    def test_t20b_rejection_diagnostics_are_lightweight_counters(self):
        source = (MODULE_DIR / "danger_perception_node.py").read_text(encoding="utf-8")
        for name in ("red_candidate", "image_shape_reject", "insufficient_3d_points", "radius_reject", "sphere_residual_reject", "plane_or_box_reject", "association_reject", "strong_sphere_support"):
            self.assertIn(name, source)

    def test_t21_overlay_rounding_never_indexes_outside_image(self):
        rows, cols, valid = rounded_image_indices(
            np.array([0.0, 159.4, 159.6, -0.4]),
            np.array([0.0, 119.4, 119.6, 0.0]),
            160,
            120,
        )
        self.assertEqual(valid.tolist(), [True, True, False, True])
        self.assertTrue(np.all((rows[valid] >= 0) & (rows[valid] < 120)))
        self.assertTrue(np.all((cols[valid] >= 0) & (cols[valid] < 160)))

    def test_adaptive_tolerance_requires_measured_cadence(self):
        self.assertIsNone(adaptive_sync_tolerance([1.0], [1.0], 0.0))
        self.assertAlmostEqual(adaptive_sync_tolerance([1.0, 1.05, 1.10], [1.0, 1.05, 1.10], 0.0), 0.0225)

    def test_shadow_validation_snapshot_is_compact_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ShadowValidationRecorder(Path(directory) / "shadow.json", Path(directory) / "samples", 5)
            recorder.set_metadata("pointcloud", {"frame_id": CAMERA_FRAME})
            recorder.increment("processed_frames")
            recorder.add_stage_ms("full_processed_frame", 0.004)
            recorder.event({"stamp_sec": 1.0, "sphere_like": False})
            recorder.persist_if_due(force=True)
            payload = json.loads((Path(directory) / "shadow.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["pointcloud"]["frame_id"], CAMERA_FRAME)
            self.assertEqual(payload["timing_ms"]["full_processed_frame"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
