#!/usr/bin/env python3
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
CONTRACT_DIR = HERE.parent / "local_subgoal_runner_mvp"
CONTRACT_FILE = CONTRACT_DIR / "local_grid_contract.py"
if not CONTRACT_FILE.is_file():
    raise ModuleNotFoundError(f"local_grid_contract not found at expected shared path: {CONTRACT_FILE}")
sys.path.insert(0, str(CONTRACT_DIR))
from local_grid_contract import (  # noqa: E402
    FORMAL_GRID_FRAME,
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M,
    flatten_index,
    grid_content_hash,
    grid_metadata,
    static_footprint_fits_map,
    validate_grid_metadata,
)
from motion_consistent_evidence import (  # noqa: E402
    MOTION_CONSISTENCY_CONTRACT,
    motion_is_valid,
    warp_array,
    warp_sensor_evidence,
)


SECTORS = ("front", "front_left", "front_right", "left_side", "right_side")
LOCAL_STATUSES = (
    "FREE_SUPPORTED",
    "OBSTACLE_SUPPORTED",
    "UNKNOWN_INSUFFICIENT_EVIDENCE",
    "CONFLICT_NEEDS_CAUTION",
)
CONFLICT_RATIO_THRESHOLD = 0.05
CONFLICT_PROVENANCE_SCHEMA_VERSION = "l3v_conflict_provenance_v1"

# These limits are the existing L3V height bands, named here so the
# ground-relative low-obstacle test reuses rather than broadens them.
OBSTACLE_MIN_BASE_Z_M = 0.15
OBSTACLE_MAX_BASE_Z_M = 1.5
NEAR_GROUND_MIN_BASE_Z_M = -0.45
NEAR_GROUND_MAX_BASE_Z_M = 0.20


def ratio(num, den):
    return float(num / den) if den else 0.0


def height_classes(base_z):
    """Return the unchanged absolute-height classes for one base-frame point."""
    return (
        OBSTACLE_MIN_BASE_Z_M < base_z <= OBSTACLE_MAX_BASE_Z_M,
        NEAR_GROUND_MIN_BASE_Z_M <= base_z <= NEAR_GROUND_MAX_BASE_Z_M,
    )


def low_obstacle_cells_from_current_cloud(near_ground_heights, resolution):
    """Find low solid relief from the current cloud's local floor support only.

    The support radius is the existing 0.20 m near-ground envelope expressed
    in Grid cells. A cell is newly obstacle-supported only when a current
    near-ground point rises by the existing 0.15 m obstacle prominence above
    the minimum current near-ground support in that local envelope.
    """
    if resolution <= 0.0:
        return set()
    radius_cells = max(1, int(math.ceil(NEAR_GROUND_MAX_BASE_Z_M / resolution)))
    low_obstacle_cells = set()
    for cell, heights in near_ground_heights.items():
        local_floor_samples = []
        for x_index in range(cell[0] - radius_cells, cell[0] + radius_cells + 1):
            for y_index in range(cell[1] - radius_cells, cell[1] + radius_cells + 1):
                local_floor_samples.extend(near_ground_heights.get((x_index, y_index), ()))
        if not local_floor_samples:
            continue
        local_floor_z = min(local_floor_samples)
        if any(height - local_floor_z > OBSTACLE_MIN_BASE_Z_M for height in heights):
            low_obstacle_cells.add(cell)
    return low_obstacle_cells


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def sector_of_point(x, y):
    if x <= 0.0:
        return None
    deg = math.degrees(math.atan2(y, x))
    if abs(deg) <= 20.0:
        return "front"
    if 20.0 < deg <= 60.0:
        return "front_left"
    if -60.0 <= deg < -20.0:
        return "front_right"
    if 60.0 < deg <= 120.0:
        return "left_side"
    if -120.0 <= deg < -60.0:
        return "right_side"
    return None


def bresenham(a, b):
    r0, c0 = a
    r1, c1 = b
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return points


class L3VLocalTraversabilityNode:
    def __init__(self):
        rospy.init_node("l3v_local_traversability_node")
        self.frame_id = rospy.get_param("~frame_id", "base")
        self.resolution = float(rospy.get_param("~resolution", 0.05))
        self.x_min = float(rospy.get_param("~x_min", -0.30))
        self.x_max = float(rospy.get_param("~x_max", 3.0))
        self.y_min = float(rospy.get_param("~y_min", -1.5))
        self.y_max = float(rospy.get_param("~y_max", 1.5))
        self.rows = int(round((self.x_max - self.x_min) / self.resolution))
        self.cols = int(round((self.y_max - self.y_min) / self.resolution))
        self.max_points_per_cloud = int(rospy.get_param("~max_points_per_cloud", 4000))
        self.depth_stride = int(rospy.get_param("~depth_stride", 15))
        self.heartbeat_rate_hz = float(rospy.get_param("~heartbeat_rate_hz", rospy.get_param("~publish_hz", 2.0)))
        self.input_stale_timeout_sec = float(rospy.get_param("~input_stale_timeout_sec", 2.0))
        self.decay_per_publish = float(rospy.get_param("~decay_per_publish", 0.96))
        self.motion_evidence_max_translation_m = float(
            rospy.get_param("~motion_evidence_max_translation_m", 1.0)
        )
        self.motion_evidence_max_yaw_delta_rad = float(
            rospy.get_param("~motion_evidence_max_yaw_delta_rad", math.pi / 2.0)
        )
        self.use_rgbd_obstacle_evidence = bool(rospy.get_param("~use_rgbd_obstacle_evidence", False))
        # R58: audit-only provenance for the small forward doorway region.  It
        # never participates in labels, Grid content, status, or navigation.
        self.doorway_provenance_enabled = bool(rospy.get_param("~doorway_provenance_enabled", True))
        self.doorway_provenance_x_min_m = float(rospy.get_param("~doorway_provenance_x_min_m", 0.25))
        self.doorway_provenance_x_max_m = float(rospy.get_param("~doorway_provenance_x_max_m", 0.70))
        self.doorway_provenance_abs_y_max_m = float(rospy.get_param("~doorway_provenance_abs_y_max_m", 0.25))
        # Diagnostic-only, bounded provenance.  It never changes labels, Grid
        # content, status thresholds, or navigation authority.
        self.conflict_provenance_enabled = bool(rospy.get_param("~conflict_provenance_enabled", True))
        self.conflict_provenance_max_cells = max(1, int(rospy.get_param("~conflict_provenance_max_cells", 8)))
        self.conflict_provenance_run_id = str(rospy.get_param(
            "~conflict_provenance_run_id",
            os.environ.get("RUNTIME_RUN_ID", os.environ.get("STATE_MACHINE_RUN_ID", "UNKNOWN_AT_L3V_PRODUCER")),
        ))
        self.lock = threading.Lock()
        self.listener = tf.TransformListener()
        self.reset_evidence()
        self.latest_odom = None
        self.odom_history = deque(maxlen=300)
        self.latest_depth_valid_ratio = 0.0
        self.latest_depth_near_valid_ratio = 0.0
        self.depth_image_count = 0
        self.depth_points_count = 0
        self.lidar_cloud_count = 0
        self.camera_info_seen = False
        self.last_cloud_time = rospy.Time(0)
        self.last_depth_time = rospy.Time(0)
        self.last_odom_wall_time = None
        self.last_lidar_wall_time = None
        self.last_depth_points_wall_time = None
        self.last_depth_image_wall_time = None
        self.grid_publish_seq = 0
        self.heartbeat_tick_count = 0
        self.last_publish_wall_time_sec = None
        self.last_publish_ros_time_sec = None
        self.producer_instance_id = uuid.uuid4().hex
        self.content_generation_id = 0
        self.content_dirty = True
        self.cached_grid = None
        self.cached_status = None
        self.freshness_revoked_content_generation = None
        self.latest_lidar_source_stamp = None
        self.latest_odom_source_stamp = None
        self.last_source_stamp = {"lidar": None, "rgbd": None, "odom": None}
        self.pending_tf_failure_reasons = []
        self.pending_time_regression_reasons = []
        self.motion_evidence_invalidated = False
        self.motion_evidence_reason = None
        self.doorway_lidar_occ_meta = {}
        self.doorway_current_direct_lidar = {}
        self.conflict_cell_lifecycle = {}
        self.cached_conflict_provenance = None
        self.pending_conflict_provenance_publication = None

        self.grid_pub = rospy.Publisher("/team/local_traversability_grid", OccupancyGrid, queue_size=1, latch=False)
        self.evidence_pub = rospy.Publisher("/team/local_traversability_evidence", String, queue_size=1, latch=False)
        self.front_pub = rospy.Publisher("/team/front_auxiliary_evidence", String, queue_size=1, latch=False)
        self.status_pub = rospy.Publisher("/team/traversability_status", String, queue_size=1, latch=False)
        self.doorway_provenance_pub = rospy.Publisher(
            "/audit/p2kg15/doorway_raw_occupancy_provenance", String, queue_size=10, latch=False
        )
        self.conflict_provenance_pub = rospy.Publisher(
            "/audit/l3v/conflict_provenance", String, queue_size=10, latch=False
        )

        rospy.Subscriber("/team/livox/icp_odom_gated", Odometry, self.odom_cb, queue_size=100)
        rospy.Subscriber("/team/livox/scan_cloud_filtered", PointCloud2, self.lidar_cb, queue_size=2)
        rospy.Subscriber("/real_sense/depth/points", PointCloud2, self.depth_points_cb, queue_size=2)
        rospy.Subscriber("/real_sense/depth/image_raw", Image, self.depth_image_cb, queue_size=2)
        rospy.Subscriber("/real_sense/rgb/camera_info", CameraInfo, self.camera_info_cb, queue_size=2)

    def reset_evidence(self):
        shape = (self.rows, self.cols)
        self.lidar_free = np.zeros(shape, dtype=np.float32)
        self.lidar_occ = np.zeros(shape, dtype=np.float32)
        self.rgbd_free = np.zeros(shape, dtype=np.float32)
        self.rgbd_occ = np.zeros(shape, dtype=np.float32)
        self.traversed = np.zeros(shape, dtype=np.float32)
        # Per-current-LiDAR-update support only.  These are not another map:
        # they are reset before every processed LiDAR cloud and are used only
        # to arbitrate a direct observation against retained lidar_occ.
        self.current_lidar_free_support = np.zeros(shape, dtype=bool)
        self.current_lidar_occ_support = np.zeros(shape, dtype=bool)
        self.sector = {
            "lidar": defaultdict(lambda: {"point_count": 0, "near_ground_count": 0, "obstacle_height_count": 0}),
            "rgbd": defaultdict(lambda: {"point_count": 0, "near_ground_count": 0, "obstacle_height_count": 0}),
        }

    def decay(self):
        """Apply one publish-timed evidence decay and report whether it changed content."""
        changed = False
        for arr in (self.lidar_free, self.lidar_occ, self.rgbd_free, self.rgbd_occ, self.traversed):
            if self.decay_per_publish != 1.0 and np.any(arr != 0.0):
                changed = True
            arr *= self.decay_per_publish
        return changed

    def sensor_evidence_locked(self):
        return {
            "lidar_free": self.lidar_free,
            "lidar_occ": self.lidar_occ,
            "rgbd_free": self.rgbd_free,
            "rgbd_occ": self.rgbd_occ,
        }

    def clear_sensor_evidence_locked(self):
        for array in self.sensor_evidence_locked().values():
            array.fill(0.0)
        self.current_lidar_free_support.fill(False)
        self.current_lidar_occ_support.fill(False)
        self.doorway_lidar_occ_meta.clear()
        self.doorway_current_direct_lidar.clear()
        self.conflict_cell_lifecycle.clear()
        self.cached_conflict_provenance = None

    def doorway_provenance_cell(self, cell):
        """Return true only for the bounded R58 audit ROI in base coordinates."""
        if not self.doorway_provenance_enabled or cell is None:
            return False
        x, y = cell
        base_x = self.x_min + (int(x) + 0.5) * self.resolution
        base_y = self.y_min + (int(y) + 0.5) * self.resolution
        return (
            self.doorway_provenance_x_min_m <= base_x <= self.doorway_provenance_x_max_m
            and abs(base_y) <= self.doorway_provenance_abs_y_max_m
        )

    def reproject_doorway_provenance_locked(self, old_pose, new_pose):
        """Move only audit metadata with the same nearest-cell SE(2) rule."""
        if not self.doorway_lidar_occ_meta:
            return
        old_c, old_s = math.cos(old_pose["yaw"]), math.sin(old_pose["yaw"])
        new_c, new_s = math.cos(new_pose["yaw"]), math.sin(new_pose["yaw"])
        moved = {}
        for (old_x_index, old_y_index), meta in self.doorway_lidar_occ_meta.items():
            local_x = self.x_min + (int(old_x_index) + 0.5) * self.resolution
            local_y = self.y_min + (int(old_y_index) + 0.5) * self.resolution
            world_x = old_pose["x"] + old_c * local_x - old_s * local_y
            world_y = old_pose["y"] + old_s * local_x + old_c * local_y
            dx, dy = world_x - new_pose["x"], world_y - new_pose["y"]
            new_x = new_c * dx + new_s * dy
            new_y = -new_s * dx + new_c * dy
            new_cell = self.local_to_cell(new_x, new_y)
            if new_cell is None:
                continue
            copied = dict(meta)
            copied["reprojection_count"] = int(copied.get("reprojection_count", 0)) + 1
            copied["last_reprojection_from_cell"] = [int(old_x_index), int(old_y_index)]
            copied["last_reprojection_to_cell"] = [int(new_cell[0]), int(new_cell[1])]
            previous = moved.get(new_cell)
            if previous is None or float(copied.get("oldest_source_stamp", float("inf"))) < float(
                previous.get("oldest_source_stamp", float("inf"))
            ):
                moved[new_cell] = copied
        self.doorway_lidar_occ_meta = moved

    def reproject_sensor_evidence_locked(self, old_pose, new_pose):
        old = (old_pose["x"], old_pose["y"], old_pose["yaw"])
        new = (new_pose["x"], new_pose["y"], new_pose["yaw"])
        valid, reason = motion_is_valid(
            old, new, self.motion_evidence_max_translation_m,
            self.motion_evidence_max_yaw_delta_rad,
        )
        if not valid:
            self.clear_sensor_evidence_locked()
            self.motion_evidence_invalidated = True
            self.motion_evidence_reason = reason
            return False
        warped = warp_sensor_evidence(
            self.sensor_evidence_locked(), old, new,
            self.resolution, self.x_min, self.y_min,
        )
        self.lidar_free = warped["lidar_free"]
        self.lidar_occ = warped["lidar_occ"]
        self.rgbd_free = warped["rgbd_free"]
        self.rgbd_occ = warped["rgbd_occ"]
        self.reproject_current_lidar_support_locked(old, new)
        self.reproject_doorway_provenance_locked(old_pose, new_pose)
        self.reconcile_current_lidar_support_locked()
        return True

    def reproject_current_lidar_support_locked(self, old_pose, new_pose):
        """Keep the most-recent LiDAR direct-support flags base-frame aligned.

        These masks retain boolean OR semantics: they describe one scan only,
        never accumulate evidence, and expire when the next LiDAR scan resets
        them.  Their spatial transform is deliberately the same production
        nearest-cell SE(2) transform used by the corresponding lidar evidence.
        """
        self.current_lidar_free_support = warp_array(
            self.current_lidar_free_support, old_pose, new_pose,
            self.resolution, self.x_min, self.y_min,
        )
        self.current_lidar_occ_support = warp_array(
            self.current_lidar_occ_support, old_pose, new_pose,
            self.resolution, self.x_min, self.y_min,
        )

    def local_to_cell(self, x, y):
        """Internal evidence cell `(x_index, y_index)`; public data is [y][x]."""
        if x < self.x_min or x >= self.x_max or y < self.y_min or y >= self.y_max:
            return None
        row = int(math.floor((x - self.x_min) / self.resolution))
        col = int(math.floor((y - self.y_min) / self.resolution))
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def robot_cell(self):
        return self.local_to_cell(0.0, 0.0)

    def lookup_to_base(self, source_frame):
        source = source_frame.lstrip("/")
        if source == self.frame_id:
            return None, None, True
        try:
            trans, rot = self.listener.lookupTransform(self.frame_id, source, rospy.Time(0))
            return trans, rot, True
        except Exception:
            return None, None, False

    def transform_to_base(self, point, source_frame):
        trans, rot, ok = self.lookup_to_base(source_frame)
        x, y, z = point
        if source_frame.lstrip("/") == self.frame_id:
            return (float(x), float(y), float(z)), None
        if ok and trans is not None:
            m = tf.transformations.quaternion_matrix(rot)
            v = np.dot(m, np.array([x, y, z, 1.0]))
            return (float(v[0] + trans[0]), float(v[1] + trans[1]), float(v[2] + trans[2])), None
        return None, f"tf_unavailable:{source_frame}"

    def record_source_stamp_locked(self, source, stamp):
        previous = self.last_source_stamp.get(source)
        if previous is not None and stamp < previous:
            self.pending_time_regression_reasons.append(f"{source}_source_stamp_regression")
            self.content_dirty = True
            return False
        self.last_source_stamp[source] = stamp
        return True

    def mark_ray(
        self,
        target_cell,
        free_arr,
        occ_arr=None,
        current_free_support=None,
        current_occ_support=None,
    ):
        source = self.robot_cell()
        if source is None or target_cell is None:
            return
        cells = bresenham(source, target_cell)
        for cell in cells[:-1]:
            free_arr[cell] += 1.0
            if current_free_support is not None:
                current_free_support[cell] = True
        if occ_arr is not None:
            occ_arr[target_cell] += 1.0
            if current_occ_support is not None:
                current_occ_support[target_cell] = True

    def begin_current_lidar_support_locked(self):
        """Start transient direct-support bookkeeping for exactly one LiDAR cloud."""
        self.current_lidar_free_support.fill(False)
        self.current_lidar_occ_support.fill(False)

    def reconcile_current_lidar_support_locked(self):
        """Apply the single current-LiDAR support invariant.

        A current obstacle endpoint always wins.  Unobserved cells retain the
        established SE(2) reprojection and decay lifecycle unchanged.  This is
        called after a complete LiDAR cloud and after any valid evidence
        reprojection, so neither callback ordering can reintroduce stale
        historical lidar_occ into a direct-free cell.
        """
        contradicted = self.current_lidar_free_support & ~self.current_lidar_occ_support
        self.lidar_occ[contradicted] = 0.0
        return contradicted

    def odom_cb(self, msg):
        pose = msg.pose.pose
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        current = {
            "stamp": stamp,
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": yaw_from_quat(pose.orientation),
        }
        with self.lock:
            self.last_odom_wall_time = time.monotonic()
            if not self.record_source_stamp_locked("odom", stamp):
                return
            if self.latest_odom is not None:
                self.reproject_sensor_evidence_locked(self.latest_odom, current)
            self.latest_odom_source_stamp = stamp
            self.latest_odom = current
            self.odom_history.append(current)
            self.update_traversed_locked()
            self.content_dirty = True

    def update_traversed_locked(self):
        if not self.latest_odom:
            return
        # Odom history is world/odom anchored. Rebuild it in the current base
        # frame instead of carrying stale base-indexed free support forward.
        self.traversed.fill(0.0)
        cur = self.latest_odom
        c = math.cos(-cur["yaw"])
        s = math.sin(-cur["yaw"])
        cells = []
        for pose in self.odom_history:
            dx = pose["x"] - cur["x"]
            dy = pose["y"] - cur["y"]
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            cell = self.local_to_cell(lx, ly)
            if cell:
                cells.append(cell)
        previous = None
        for cell in cells:
            if previous is None:
                self.traversed[cell] += 1.0
            else:
                for ccell in bresenham(previous, cell):
                    self.traversed[ccell] += 1.0
            previous = cell

    def process_cloud(self, msg, source):
        now = rospy.Time.now()
        if source == "lidar":
            if (now - self.last_cloud_time).to_sec() < 0.08:
                return
            self.last_cloud_time = now
        else:
            if (now - self.last_depth_time).to_sec() < 0.08:
                return
            self.last_depth_time = now
        processed = 0
        seen = 0
        cloud_tf_valid = True
        source_stamp = msg.header.stamp.to_sec() if msg.header.stamp else now.to_sec()
        with self.lock:
            if not self.record_source_stamp_locked(source, source_stamp):
                return
            if source == "lidar":
                self.latest_lidar_source_stamp = source_stamp
                self.doorway_current_direct_lidar = {}
                self.begin_current_lidar_support_locked()
            points = []
            for raw in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                seen += 1
                if source == "rgbd" and self.depth_stride > 1 and seen % self.depth_stride != 0:
                    continue
                p, tf_error = self.transform_to_base([float(raw[0]), float(raw[1]), float(raw[2])], msg.header.frame_id)
                if p is None:
                    self.pending_tf_failure_reasons.append(tf_error)
                    self.content_dirty = True
                    cloud_tf_valid = False
                    break
                dist = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                if dist < 0.10 or dist > 8.0:
                    continue
                sec = sector_of_point(p[0], p[1])
                if sec:
                    self.sector[source][sec]["point_count"] += 1
                    if -0.35 <= p[2] <= 0.15:
                        self.sector[source][sec]["near_ground_count"] += 1
                    if 0.15 < p[2] <= 1.5:
                        self.sector[source][sec]["obstacle_height_count"] += 1
                cell = self.local_to_cell(p[0], p[1])
                if cell is None:
                    continue
                obstacle, near_ground = height_classes(p[2])
                points.append((raw, p, cell, obstacle, near_ground))
                processed += 1
                if processed >= self.max_points_per_cloud:
                    break
            low_obstacle_cells = set()
            if source == "lidar":
                near_ground_heights = defaultdict(list)
                for _raw, point, cell, _obstacle, near_ground in points:
                    if near_ground:
                        near_ground_heights[cell].append(float(point[2]))
                low_obstacle_cells = low_obstacle_cells_from_current_cloud(
                    near_ground_heights, self.resolution,
                )
            for raw, p, cell, obstacle, near_ground in points:
                obstacle = bool(obstacle or (source == "lidar" and cell in low_obstacle_cells))
                if source == "lidar":
                    self.mark_ray(
                        cell,
                        self.lidar_free,
                        self.lidar_occ if obstacle else None,
                        self.current_lidar_free_support,
                        self.current_lidar_occ_support if obstacle else None,
                    )
                    if obstacle and self.doorway_provenance_cell(cell):
                        pose = self.latest_odom or {}
                        world_xy = None
                        if all(key in pose for key in ("x", "y", "yaw")):
                            c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
                            world_xy = [
                                float(pose["x"] + c * p[0] - s * p[1]),
                                float(pose["y"] + s * p[0] + c * p[1]),
                            ]
                        point = {
                            "sensor_xyz": [float(raw[0]), float(raw[1]), float(raw[2])],
                            "base_xyz": [float(p[0]), float(p[1]), float(p[2])],
                            "world_xy_at_capture": world_xy,
                            "odom_pose_at_capture": [pose.get("x"), pose.get("y"), pose.get("yaw")],
                        }
                        direct = self.doorway_current_direct_lidar.setdefault(
                            cell, {"source_stamp": float(source_stamp), "point_count": 0, "points": []}
                        )
                        direct["point_count"] += 1
                        if len(direct["points"]) < 4:
                            direct["points"].append(point)
                        prior = self.doorway_lidar_occ_meta.get(cell, {})
                        self.doorway_lidar_occ_meta[cell] = {
                            "oldest_source_stamp": float(prior.get("oldest_source_stamp", source_stamp)),
                            "latest_source_stamp": float(source_stamp),
                            "observation_count": int(prior.get("observation_count", 0)) + 1,
                            "representative_source_points": direct["points"],
                            "reprojection_count": int(prior.get("reprojection_count", 0)),
                            "last_reprojection_from_cell": prior.get("last_reprojection_from_cell"),
                            "last_reprojection_to_cell": prior.get("last_reprojection_to_cell"),
                        }
                    if near_ground and not obstacle:
                        self.lidar_free[cell] += 1.0
                else:
                    if near_ground and not obstacle:
                        self.mark_ray(cell, self.rgbd_free, None)
                        self.rgbd_free[cell] += 1.0
                    elif self.use_rgbd_obstacle_evidence and obstacle:
                        self.mark_ray(cell, self.rgbd_free, self.rgbd_occ)
            if source == "lidar":
                self.lidar_cloud_count += 1
                if cloud_tf_valid:
                    self.reconcile_current_lidar_support_locked()
                    # A current, transform-valid LiDAR frame is anchored in the
                    # current base frame and may safely re-establish evidence.
                    self.motion_evidence_invalidated = False
                    self.motion_evidence_reason = None
            else:
                self.depth_points_count += 1
            self.content_dirty = True

    def lidar_cb(self, msg):
        self.last_lidar_wall_time = time.monotonic()
        self.process_cloud(msg, "lidar")

    def depth_points_cb(self, msg):
        self.last_depth_points_wall_time = time.monotonic()
        self.process_cloud(msg, "rgbd")

    def depth_image_cb(self, msg):
        self.last_depth_image_wall_time = time.monotonic()
        if msg.encoding in ("16UC1", "mono16"):
            arr = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) * 0.001
        elif msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32)
        else:
            return
        finite = np.isfinite(arr)
        clean = np.where(finite, arr, 0.0)
        valid = finite & (clean > 0.15) & (clean < 8.0)
        near = valid & (clean < 3.0)
        with self.lock:
            self.latest_depth_valid_ratio = ratio(int(np.sum(valid)), int(arr.size))
            self.latest_depth_near_valid_ratio = ratio(int(np.sum(near)), int(arr.size))
            self.depth_image_count += 1

    def camera_info_cb(self, _msg):
        self.camera_info_seen = True

    def labels_locked(self):
        has_free = (self.lidar_free + self.rgbd_free + self.traversed) > 0.5
        has_occ = (self.lidar_occ + self.rgbd_occ) > 0.5
        labels = np.full((self.rows, self.cols), "unknown", dtype=object)
        labels[has_free] = "free"
        labels[has_occ & ~has_free] = "occupied"
        labels[has_occ & has_free] = "conflict"
        labels[(self.rgbd_free > 0.5) & (self.lidar_free <= 0.5) & ~has_occ] = "rgbd_supported_free"
        labels[(self.lidar_occ > 0.5) & ~has_free] = "lidar_supported_obstacle"
        return labels

    def occupancy_grid_locked(self, labels):
        # Evidence storage is [x][y]. ROS OccupancyGrid is public [y][x].
        internal = np.full((self.rows, self.cols), -1, dtype=np.int8)
        internal[np.isin(labels, ["free", "rgbd_supported_free"])] = 0
        internal[np.isin(labels, ["occupied", "lidar_supported_obstacle", "conflict"])] = 100
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.rows
        msg.info.height = self.cols
        msg.info.origin.position.x = self.x_min
        msg.info.origin.position.y = self.y_min
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        public_data = [-1] * (self.rows * self.cols)
        for x_index in range(self.rows):
            for y_index in range(self.cols):
                index = flatten_index(x_index, y_index, self.rows, self.cols)
                public_data[index] = int(internal[x_index, y_index])
        msg.data = public_data
        return msg

    def unknown_occupancy_grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.rows
        msg.info.height = self.cols
        msg.info.origin.position.x = self.x_min
        msg.info.origin.position.y = self.y_min
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = np.full((self.cols, self.rows), -1, dtype=np.int8).reshape(-1).astype(int).tolist()
        return msg

    def input_freshness_locked(self, now_wall):
        def age(last_wall):
            return None if last_wall is None else max(0.0, now_wall - last_wall)

        odom_age = age(self.last_odom_wall_time)
        lidar_age = age(self.last_lidar_wall_time)
        depth_points_age = age(self.last_depth_points_wall_time)
        depth_image_age = age(self.last_depth_image_wall_time)
        stale_reasons = []
        if self.latest_odom is None or odom_age is None or odom_age > self.input_stale_timeout_sec:
            stale_reasons.append("odom_stale_or_missing")
        lidar_fresh = lidar_age is not None and lidar_age <= self.input_stale_timeout_sec
        if not self.use_rgbd_obstacle_evidence:
            # RGB-D is diagnostic-only in this production configuration; it
            # must not keep formal collision navigation alive after LiDAR has
            # become stale.
            if not lidar_fresh:
                stale_reasons.append("lidar_stale_or_missing")
        else:
            sensor_fresh = (
                lidar_fresh
                or (depth_points_age is not None and depth_points_age <= self.input_stale_timeout_sec)
                or (depth_image_age is not None and depth_image_age <= self.input_stale_timeout_sec)
            )
            if not sensor_fresh:
                stale_reasons.append("lidar_or_depth_stale_or_missing")
        return {
            "odom_age_sec": odom_age,
            "lidar_cloud_age_sec": lidar_age,
            "depth_points_age_sec": depth_points_age,
            "depth_image_age_sec": depth_image_age,
            "all_required_inputs_fresh": not stale_reasons,
            "stale_reasons": stale_reasons,
        }

    def sector_summary_locked(self, labels):
        out = {}
        for sec in SECTORS:
            lidar = self.sector["lidar"][sec]
            rgbd = self.sector["rgbd"][sec]
            out[sec] = {
                "lidar_point_count": int(lidar["point_count"]),
                "lidar_near_ground_ratio": ratio(lidar["near_ground_count"], lidar["point_count"]),
                "lidar_obstacle_height_ratio": ratio(lidar["obstacle_height_count"], lidar["point_count"]),
                "rgbd_depth_valid_count": int(rgbd["point_count"]),
                "rgbd_depth_valid_ratio": self.latest_depth_valid_ratio,
                "rgbd_front_auxiliary_useful": bool(self.latest_depth_valid_ratio >= 0.20 and self.sector["rgbd"]["front"]["point_count"] > 1000),
                "odom_traversed_support": bool(np.sum(self.traversed > 0.5) > 0),
                "conflict_ratio": ratio(int(np.sum(labels == "conflict")), labels.size),
                "final_local_traversability_status": self.local_status_for_sector(sec, labels),
            }
        front = self.sector["lidar"]["front"]["point_count"]
        left = self.sector["lidar"]["front_left"]["point_count"] + self.sector["lidar"]["left_side"]["point_count"]
        right = self.sector["lidar"]["front_right"]["point_count"] + self.sector["lidar"]["right_side"]["point_count"]
        side_avg = (left + right) / 2.0
        for sec in SECTORS:
            out[sec]["lidar_front_to_side_density_ratio"] = ratio(front, side_avg)
        return out

    def local_status_for_sector(self, sec, labels):
        mask = np.zeros(labels.shape, dtype=bool)
        for r in range(self.rows):
            x = self.x_min + (r + 0.5) * self.resolution
            for c in range(self.cols):
                y = self.y_min + (c + 0.5) * self.resolution
                if sector_of_point(x, y) == sec:
                    mask[r, c] = True
        if not np.any(mask):
            return "UNKNOWN_INSUFFICIENT_EVIDENCE"
        conflict_ratio = ratio(int(np.sum(mask & (labels == "conflict"))), int(np.sum(mask)))
        occ_ratio = ratio(int(np.sum(mask & np.isin(labels, ["occupied", "lidar_supported_obstacle"]))), int(np.sum(mask)))
        free_ratio = ratio(int(np.sum(mask & np.isin(labels, ["free", "rgbd_supported_free"]))), int(np.sum(mask)))
        if conflict_ratio > 0.02:
            return "CONFLICT_NEEDS_CAUTION"
        if occ_ratio > 0.10:
            return "OBSTACLE_SUPPORTED"
        if free_ratio > 0.20:
            return "FREE_SUPPORTED"
        return "UNKNOWN_INSUFFICIENT_EVIDENCE"

    def doorway_raw_occupancy_provenance_locked(self, grid_msg, status):
        """Audit only raw=100 cells in the R58 ROI; no Grid input is changed."""
        if not self.doorway_provenance_enabled:
            return None
        cells = []
        now_stamp = float(grid_msg.header.stamp.to_sec())
        for x_index in range(self.rows):
            for y_index in range(self.cols):
                cell = (x_index, y_index)
                if not self.doorway_provenance_cell(cell):
                    continue
                raw_value = int(grid_msg.data[flatten_index(x_index, y_index, self.rows, self.cols)])
                if raw_value != 100:
                    continue
                meta = self.doorway_lidar_occ_meta.get(cell)
                direct = self.doorway_current_direct_lidar.get(cell)
                direct_free = bool(self.current_lidar_free_support[x_index, y_index])
                # This must report the same spatially reprojected support used
                # by lifecycle arbitration.  ROI point metadata is diagnostic
                # only and may originate outside the ROI before an odom warp.
                direct_current = bool(self.current_lidar_occ_support[x_index, y_index])
                oldest = meta.get("oldest_source_stamp") if meta else None
                latest = meta.get("latest_source_stamp") if meta else None
                cells.append(
                    {
                        "grid_cell": [int(x_index), int(y_index)],
                        "base_xy": [
                            self.x_min + (x_index + 0.5) * self.resolution,
                            self.y_min + (y_index + 0.5) * self.resolution,
                        ],
                        "raw_occupancy_value": raw_value,
                        "lidar_occ_evidence_strength": float(self.lidar_occ[x_index, y_index]),
                        "lidar_occ_observation_count": int(meta.get("observation_count", 0)) if meta else 0,
                        "latest_contributing_sensor_source_stamp": latest,
                        "oldest_retained_source_stamp": oldest,
                        "current_lidar_direct_support": direct_current,
                        "current_direct_lidar_point_count": int(direct.get("point_count", 0)) if direct else 0,
                        "current_lidar_direct_free_support": direct_free,
                        "representative_sensor_frame_points": direct.get("points", []) if direct else [],
                        "representative_retained_source_points": meta.get("representative_source_points", []) if meta else [],
                        "retained_historical_evidence": bool(meta and (int(meta.get("reprojection_count", 0)) > 0 or not direct_current)),
                        "reprojection_count": int(meta.get("reprojection_count", 0)) if meta else 0,
                        "last_reprojection_from_cell": meta.get("last_reprojection_from_cell") if meta else None,
                        "last_reprojection_to_cell": meta.get("last_reprojection_to_cell") if meta else None,
                        "evidence_age_sec": (now_stamp - float(oldest)) if oldest is not None else None,
                        "self_geometry_support": "NOT_ASSERTED_NO_SELF_GEOMETRY_EVIDENCE",
                    }
                )
        if not cells:
            return None
        pose = self.latest_odom or {}
        return {
            "schema": "r58_doorway_raw_occupancy_provenance_v1",
            "grid_stamp": now_stamp,
            "grid_content_generation_id": status.get("content_generation_id"),
            "grid_content_hash": status.get("grid_content_hash"),
            "source_scan_stamp": status.get("source_scan_stamp"),
            "robot_pose_odom": [pose.get("x"), pose.get("y"), pose.get("yaw")],
            "roi": {"x_m": [self.doorway_provenance_x_min_m, self.doorway_provenance_x_max_m], "abs_y_m": self.doorway_provenance_abs_y_max_m},
            "cells": cells,
        }

    def conflict_provenance_locked(self, labels, *, content_stamp_sec, status_transition):
        """Bounded evidence record for formal L3V conflict diagnosis only.

        Per-cell source timestamps are not retained by the production evidence
        arrays.  This record deliberately labels the available sensor stamps
        as global/latest rather than inventing cell-specific times.
        """
        if not self.conflict_provenance_enabled:
            return None
        traversed_mask = self.traversed > 0.5
        conflict_mask = traversed_mask & (labels == "conflict")
        conflict_indices = [tuple(int(value) for value in row) for row in np.argwhere(conflict_mask)]
        active = set(conflict_indices)
        for cell in tuple(self.conflict_cell_lifecycle):
            if cell not in active:
                del self.conflict_cell_lifecycle[cell]
        for cell in conflict_indices:
            lifecycle = self.conflict_cell_lifecycle.setdefault(cell, {
                "first_conflict_content_stamp_sec": float(content_stamp_sec),
            })
            lifecycle["latest_conflict_content_stamp_sec"] = float(content_stamp_sec)

        def source_row(source, strength, stamp, stamp_scope, direct_support):
            return {
                "source": source,
                "support": float(strength),
                "latest_source_stamp_sec": stamp,
                "source_stamp_scope": stamp_scope,
                "current_direct_support": bool(direct_support),
            }

        ranked = sorted(
            conflict_indices,
            key=lambda cell: (
                -float(self.lidar_occ[cell] + self.rgbd_occ[cell]),
                -float(self.lidar_free[cell] + self.rgbd_free[cell] + self.traversed[cell]),
                cell,
            ),
        )[:self.conflict_provenance_max_cells]
        cells = []
        for cell in ranked:
            x_index, y_index = cell
            free_sources = []
            occupied_sources = []
            if self.lidar_free[cell] > 0.5:
                free_sources.append(source_row(
                    "LIDAR_FREE_RAY_ACCUMULATED", self.lidar_free[cell], self.latest_lidar_source_stamp,
                    "LATEST_LIDAR_SOURCE_GLOBAL_NOT_CELL_SPECIFIC", self.current_lidar_free_support[cell],
                ))
            if self.rgbd_free[cell] > 0.5:
                free_sources.append(source_row(
                    "RGBD_FREE_RAY_ACCUMULATED", self.rgbd_free[cell], self.last_source_stamp.get("rgbd"),
                    "LATEST_RGBD_SOURCE_GLOBAL_NOT_CELL_SPECIFIC", False,
                ))
            if self.traversed[cell] > 0.5:
                free_sources.append(source_row(
                    "ODOM_TRAVERSED_HISTORY", self.traversed[cell], self.latest_odom_source_stamp,
                    "LATEST_ODOM_SOURCE_GLOBAL_NOT_CELL_SPECIFIC", False,
                ))
            if self.lidar_occ[cell] > 0.5:
                occupied_sources.append(source_row(
                    "LIDAR_OCC_ENDPOINT_ACCUMULATED", self.lidar_occ[cell], self.latest_lidar_source_stamp,
                    "LATEST_LIDAR_SOURCE_GLOBAL_NOT_CELL_SPECIFIC", self.current_lidar_occ_support[cell],
                ))
            if self.rgbd_occ[cell] > 0.5:
                occupied_sources.append(source_row(
                    "RGBD_OCC_ENDPOINT_ACCUMULATED", self.rgbd_occ[cell], self.last_source_stamp.get("rgbd"),
                    "LATEST_RGBD_SOURCE_GLOBAL_NOT_CELL_SPECIFIC", False,
                ))
            relevant_stamps = [
                value for value in (self.latest_lidar_source_stamp, self.last_source_stamp.get("rgbd"), self.latest_odom_source_stamp)
                if isinstance(value, (int, float))
            ]
            lifecycle = self.conflict_cell_lifecycle[cell]
            cells.append({
                "cell_index_xy": [x_index, y_index],
                "base_xy_m": [
                    float(self.x_min + (x_index + 0.5) * self.resolution),
                    float(self.y_min + (y_index + 0.5) * self.resolution),
                ],
                "label": "conflict",
                "free_evidence": free_sources or [{"availability": "UNAVAILABLE"}],
                "occupied_evidence": occupied_sources or [{"availability": "UNAVAILABLE"}],
                "traversed_history_support": float(self.traversed[cell]),
                "first_conflict_content_stamp_sec": lifecycle["first_conflict_content_stamp_sec"],
                "latest_relevant_evidence_stamp_sec": max(relevant_stamps) if relevant_stamps else None,
                "cell_time_limit": "PER_CELL_EVIDENCE_TIMESTAMPS_UNAVAILABLE_IN_CURRENT_PRODUCTION_ARRAYS",
            })
        traversed_count = int(np.sum(traversed_mask))
        return {
            "schema_version": CONFLICT_PROVENANCE_SCHEMA_VERSION,
            "authority": "AUDIT_ONLY",
            "run_id": self.conflict_provenance_run_id,
            "trigger": "LOCAL_STATUS_TO_CONFLICT_NEEDS_CAUTION" if status_transition else "CONFLICT_CONTENT_REFRESH",
            "grid_content_stamp_sec": float(content_stamp_sec),
            "conflict_cell_count": len(conflict_indices),
            "eligible_traversed_cell_count": traversed_count,
            "conflict_ratio": ratio(len(conflict_indices), traversed_count),
            "conflict_ratio_threshold": CONFLICT_RATIO_THRESHOLD,
            "representative_cell_limit": self.conflict_provenance_max_cells,
            "representative_conflict_cells": cells,
            "interpretation": "CELL_HAS_FREE_OR_TRAVERSED_SUPPORT_AND_OCCUPIED_SUPPORT; NO_PHYSICAL_CAUSE_INFERRED",
        }

    def build_content_locked(self, now_wall):
        """Commit one content generation. Heartbeats never enter this method."""
        labels = self.labels_locked()
        input_freshness = self.input_freshness_locked(now_wall)
        time_reasons = list(self.pending_time_regression_reasons)
        tf_reasons = list(dict.fromkeys(self.pending_tf_failure_reasons))
        motion_reasons = []
        if self.motion_evidence_invalidated:
            motion_reasons.append(self.motion_evidence_reason or "motion_evidence_reacquisition_required")
        rejection_reasons = list(input_freshness["stale_reasons"]) + time_reasons + motion_reasons
        if tf_reasons:
            rejection_reasons.append("tf_invalid")
        inputs_fresh = bool(input_freshness["all_required_inputs_fresh"]) and not time_reasons
        tf_valid = not tf_reasons
        motion_consistent = not self.motion_evidence_invalidated
        grid_msg = self.occupancy_grid_locked(labels) if inputs_fresh and tf_valid and motion_consistent else self.unknown_occupancy_grid()
        content_stamp = rospy.Time.now()
        self.content_generation_id += 1
        grid_msg.header.stamp = content_stamp
        grid_msg.header.frame_id = FORMAL_GRID_FRAME
        grid_msg.header.seq = self.content_generation_id
        sector = self.sector_summary_locked(labels)
        traversed = int(np.sum(self.traversed > 0.5))
        conflict = int(np.sum((self.traversed > 0.5) & (labels == "conflict")))
        free_supported = int(np.sum((self.traversed > 0.5) & np.isin(labels, ["free", "rgbd_supported_free", "conflict"])))
        conflict_ratio = ratio(conflict, traversed)
        computed_status = "CONFLICT_NEEDS_CAUTION" if conflict_ratio > CONFLICT_RATIO_THRESHOLD else "FREE_SUPPORTED"
        local_status = computed_status if inputs_fresh and tf_valid else "UNKNOWN_INSUFFICIENT_EVIDENCE"
        previous_status = (self.cached_status or {}).get("local_traversability_status")
        conflict_record = (
            self.conflict_provenance_locked(
                labels, content_stamp_sec=content_stamp.to_sec(),
                status_transition=previous_status != "CONFLICT_NEEDS_CAUTION",
            )
            if local_status == "CONFLICT_NEEDS_CAUTION" else None
        )
        metadata = grid_metadata(grid_msg)
        geometry_errors = validate_grid_metadata(grid_msg)
        footprint_fits_map = static_footprint_fits_map(metadata)
        inflation_cells = int(math.ceil(STATIC_PLANNING_FOOTPRINT_RADIUS_M / self.resolution))
        if geometry_errors:
            rejection_reasons.append("grid_geometry_invalid")
        if not footprint_fits_map:
            rejection_reasons.append("static_footprint_not_fully_covered")
        static_qualification = bool(
            inputs_fresh and tf_valid and motion_consistent and not time_reasons and not geometry_errors
            and footprint_fits_map and not rejection_reasons
        )
        diagnostic_only = not static_qualification
        footprint_qualification = (
            "QUALIFIED_FOR_DECLARED_ENVELOPE" if static_qualification else "UNQUALIFIED"
        )
        inflation_qualification = (
            "QUALIFIED_POINT_PLANNING_OBSTACLE_INFLATION" if static_qualification else "UNQUALIFIED"
        )
        grid_hash = grid_content_hash(
            grid_msg, self.producer_instance_id, self.content_generation_id, content_stamp.to_sec()
        )
        status = {
            "contract_version": GRID_CONTRACT_VERSION,
            "schema_version": GRID_STATUS_SCHEMA_VERSION,
            "node": "l3v_local_traversability_node",
            "producer_instance_id": self.producer_instance_id,
            "content_generation_id": self.content_generation_id,
            "grid_content_stamp": content_stamp.to_sec(),
            "grid_content_hash": grid_hash,
            "frame_id": FORMAL_GRID_FRAME,
            "origin": {"x": self.x_min, "y": self.y_min},
            "resolution": self.resolution,
            "width": self.rows,
            "height": self.cols,
            "occupancy_semantics": {"free": 0, "occupied": 100, "unknown": -1},
            "source_scan_stamp": self.latest_lidar_source_stamp,
            "source_odom_stamp": self.latest_odom_source_stamp,
            "latest_lidar_source_stamp": self.latest_lidar_source_stamp,
            "latest_odom_source_stamp": self.latest_odom_source_stamp,
            "source_time_domain": "ros_time",
            "motion_consistency_contract": MOTION_CONSISTENCY_CONTRACT,
            "motion_evidence_consistent": motion_consistent,
            "motion_evidence_reason": self.motion_evidence_reason,
            "motion_evidence_max_translation_m": self.motion_evidence_max_translation_m,
            "motion_evidence_max_yaw_delta_rad": self.motion_evidence_max_yaw_delta_rad,
            "input_time_monotonic": not time_reasons,
            "tf_valid": tf_valid,
            "tf_failure_reasons": tf_reasons,
            "all_required_inputs_fresh": inputs_fresh,
            "stale_reasons": input_freshness["stale_reasons"],
            "input_freshness": input_freshness,
            "input_freshness_window_sec": self.input_stale_timeout_sec,
            "footprint_qualification": footprint_qualification,
            "inflation_qualification": inflation_qualification,
            "static_footprint_radius_m": STATIC_PLANNING_FOOTPRINT_RADIUS_M,
            "static_footprint_fits_map": footprint_fits_map,
            "planning_collision_model": POINT_PLANNING_WITH_OBSTACLE_INFLATION,
            "inflation_radius_cells": inflation_cells,
            "inflation_radius_m_discrete": inflation_cells * self.resolution,
            "self_geometry_handling": "NONE",
            "diagnostic_only": diagnostic_only,
            "safe_for_navigation": bool(
                inputs_fresh and tf_valid and not time_reasons and not diagnostic_only
                and footprint_qualification == "QUALIFIED_FOR_DECLARED_ENVELOPE"
            ),
            "upstream_navigation_allowed": bool(static_qualification),
            "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
            "local_traversability_status": local_status,
            "computed_local_traversability_status_when_fresh": computed_status,
            "conflict_provenance": conflict_record,
            "grid": {
                "x_forward_m": [self.x_min, self.x_max],
                "y_left_m": [self.y_min, self.y_max],
                "resolution_m": self.resolution,
                "width": self.rows,
                "height": self.cols,
                "matrix_layout": "matrix[y_index][x_index]",
            },
        }
        self.cached_grid = grid_msg
        self.cached_status = status
        self.cached_conflict_provenance = conflict_record
        self.pending_conflict_provenance_publication = (
            conflict_record if conflict_record and conflict_record.get("trigger") == "LOCAL_STATUS_TO_CONFLICT_NEEDS_CAUTION" else None
        )
        self.content_dirty = False
        self.freshness_revoked_content_generation = None
        self.pending_tf_failure_reasons = []
        self.pending_time_regression_reasons = []

    def heartbeat_status_locked(self, now_wall):
        """Refresh time-dependent qualification without changing cached content identity."""
        status = dict(self.cached_status)
        input_freshness = self.input_freshness_locked(now_wall)
        generation = self.cached_status.get("content_generation_id")
        if not input_freshness["all_required_inputs_fresh"]:
            self.freshness_revoked_content_generation = generation
        cached_inputs_fresh = bool(self.cached_status.get("all_required_inputs_fresh"))
        inputs_fresh = (
            cached_inputs_fresh
            and bool(input_freshness["all_required_inputs_fresh"])
            and getattr(self, "freshness_revoked_content_generation", None) != generation
        )
        stale_reasons = list(input_freshness["stale_reasons"])
        if not inputs_fresh and not stale_reasons:
            stale_reasons = list(self.cached_status.get("stale_reasons", []))
            if not stale_reasons:
                stale_reasons = ["content_inputs_not_fresh_until_new_input"]
        rejection_reasons = list(self.cached_status.get("rejection_reasons", []))
        rejection_reasons.extend(stale_reasons)
        status["input_freshness"] = input_freshness
        status["all_required_inputs_fresh"] = inputs_fresh
        status["stale_reasons"] = stale_reasons
        status["safe_for_navigation"] = bool(self.cached_status.get("safe_for_navigation") and inputs_fresh)
        status["upstream_navigation_allowed"] = bool(self.cached_status.get("upstream_navigation_allowed") and inputs_fresh)
        status["rejection_reasons"] = list(dict.fromkeys(rejection_reasons))
        if not inputs_fresh:
            status["local_traversability_status"] = "UNKNOWN_INSUFFICIENT_EVIDENCE"
        return status

    def publish(self, _event):
        now_wall = time.monotonic()
        with self.lock:
            if self.content_dirty or self.cached_grid is None:
                self.build_content_locked(now_wall)
            self.heartbeat_tick_count += 1
            self.last_publish_wall_time_sec = now_wall
            self.last_publish_ros_time_sec = rospy.Time.now().to_sec()
            grid_msg = self.cached_grid
            status = self.heartbeat_status_locked(now_wall)
            status["status_heartbeat_stamp"] = self.last_publish_ros_time_sec
            status["publication_sequence"] = self.heartbeat_tick_count
            status["heartbeat"] = {
                "enabled": True,
                "rate_hz": self.heartbeat_rate_hz,
                "tick_count": self.heartbeat_tick_count,
                "last_publish_wall_time_sec": self.last_publish_wall_time_sec,
                "last_publish_ros_time_sec": self.last_publish_ros_time_sec,
            }
            evidence = {**status, "sector_summary": self.sector_summary_locked(self.labels_locked())}
            front = {
                "diagnostic_only": status["diagnostic_only"],
                "safe_for_navigation": status["safe_for_navigation"],
                "content_generation_id": status["content_generation_id"],
                "grid_content_stamp": status["grid_content_stamp"],
            }
            doorway_provenance = self.doorway_raw_occupancy_provenance_locked(grid_msg, status)
            conflict_provenance = self.pending_conflict_provenance_publication
            self.pending_conflict_provenance_publication = None
            # Preserve the pre-G13A cadence: this output uses pre-decay evidence.
            if self.decay():
                self.content_dirty = True
        self.grid_pub.publish(grid_msg)
        self.evidence_pub.publish(String(data=json.dumps(evidence, sort_keys=True)))
        self.front_pub.publish(String(data=json.dumps(front, sort_keys=True)))
        self.status_pub.publish(String(data=json.dumps(status, sort_keys=True)))
        if doorway_provenance is not None:
            self.doorway_provenance_pub.publish(String(data=json.dumps(doorway_provenance, sort_keys=True)))
        if conflict_provenance is not None:
            self.conflict_provenance_pub.publish(String(data=json.dumps(conflict_provenance, sort_keys=True)))

    def spin(self):
        def heartbeat_loop():
            interval = 1.0 / max(self.heartbeat_rate_hz, 0.1)
            while not rospy.is_shutdown():
                self.publish(None)
                time.sleep(interval)

        thread = threading.Thread(target=heartbeat_loop, name="l3v_grid_heartbeat")
        thread.daemon = True
        thread.start()
        rospy.spin()


if __name__ == "__main__":
    L3VLocalTraversabilityNode().spin()
