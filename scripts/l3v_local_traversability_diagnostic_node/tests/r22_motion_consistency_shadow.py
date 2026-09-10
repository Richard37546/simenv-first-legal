#!/usr/bin/env python3
"""Read-only R22 replay for the R31 stale-doorway regression."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELPER_PATH = HERE.parent / "motion_consistent_evidence.py"
R29_PATH = ROOT / "audit_tools/p2kg15_stair_moving_turn_r29_offline.py"
STATE_SUMMARY = ROOT / "debug/state_machine_navigation/run_archives/run_0072_20260809_184340_361324836_pid2600123/state_machine_navigation_summary.json"

SPEC = importlib.util.spec_from_file_location("motion_consistent_evidence", HELPER_PATH)
MOTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOTION)
R29_SPEC = importlib.util.spec_from_file_location("r29_offline", R29_PATH)
R29 = importlib.util.module_from_spec(R29_SPEC)
R29_SPEC.loader.exec_module(R29)

X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION = -0.30, 3.0, -1.5, 1.5, 0.05
ROWS, COLS = 66, 60
RADIUS_M = 0.2641935843278561
KAPPA_M_INV = 0.6620444444444447
TARGET_GRID_STAMP = 41.598
TARGET_CELLS = ((32, 51), (33, 51))
CANDIDATE = (15.483678817749023, 0.16314192116260529, 0.2799291242590431)


def euler_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def local_cell(x, y):
    row = int(math.floor((x - X_MIN) / RESOLUTION))
    col = int(math.floor((y - Y_MIN) / RESOLUTION))
    return (row, col) if 0 <= row < ROWS and 0 <= col < COLS else None


def bresenham(a, b):
    r0, c0 = a
    r1, c1 = b
    out = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dc - dr
    while True:
        out.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return out
        twice = 2 * err
        if twice > -dr:
            err -= dr
            c0 += sc
        if twice < dc:
            err += dc
            r0 += sr


def laser_to_base(point):
    # Recorded /tf_static base <- laser_livox transform.
    x, y, z = point
    qy, qw = 0.38249949727600974, 0.9239556994702722
    return (
        (1.0 - 2.0 * qy * qy) * x + 2.0 * qy * qw * z + 0.2,
        y,
        -2.0 * qy * qw * x + (1.0 - 2.0 * qy * qy) * z + 0.08,
    )


def mark_cloud(evidence, cloud):
    origin = local_cell(0.0, 0.0)
    processed = 0
    for raw in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = laser_to_base(tuple(map(float, raw)))
        if math.sqrt(x * x + y * y + z * z) < 0.10 or math.sqrt(x * x + y * y + z * z) > 8.0:
            continue
        target = local_cell(x, y)
        if target is None:
            continue
        obstacle = 0.15 < z <= 1.5
        near_ground = -0.45 <= z <= 0.20
        for cell in bresenham(origin, target)[:-1]:
            evidence["lidar_free"][cell] += 1.0
        if obstacle:
            evidence["lidar_occ"][target] += 1.0
        if near_ground and not obstacle:
            evidence["lidar_free"][target] += 1.0
        processed += 1
        if processed >= 4000:
            break


def warp_latest_timestamps(array, old_pose, new_pose):
    output = np.zeros_like(array)
    old_c, old_s = math.cos(old_pose[2]), math.sin(old_pose[2])
    new_c, new_s = math.cos(new_pose[2]), math.sin(new_pose[2])
    for row, column in zip(*np.nonzero(array)):
        old_x = X_MIN + (int(row) + 0.5) * RESOLUTION
        old_y = Y_MIN + (int(column) + 0.5) * RESOLUTION
        world_x = old_pose[0] + old_c * old_x - old_s * old_y
        world_y = old_pose[1] + old_s * old_x + old_c * old_y
        dx, dy = world_x - new_pose[0], world_y - new_pose[1]
        new_x = new_c * dx + new_s * dy
        new_y = -new_s * dx + new_c * dy
        target = local_cell(new_x, new_y)
        if target:
            output[target] = max(output[target], array[row, column])
    return output


def mark_latest_occurrence(latest, cloud, stamp):
    processed = 0
    for raw in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = laser_to_base(tuple(map(float, raw)))
        if math.sqrt(x * x + y * y + z * z) < 0.10 or math.sqrt(x * x + y * y + z * z) > 8.0:
            continue
        target = local_cell(x, y)
        if target is None:
            continue
        if not (0.15 < z <= 1.5):
            processed += 1
            if processed >= 4000:
                break
            continue
        latest[target] = float(stamp)
        processed += 1
        if processed >= 4000:
            break


def labels(evidence):
    free = evidence["lidar_free"] > 0.5
    occupied = evidence["lidar_occ"] > 0.5
    return np.where(occupied, 100, np.where(free, 0, -1)).astype(np.int8)


def counts(grid):
    return {"occupied": int((grid == 100).sum()), "free": int((grid == 0).sum()), "unknown": int((grid == -1).sum())}


def target_value(grid, cell):
    # OccupancyGrid public layout is [y][x]; shadow evidence is [x][y].
    return int(grid[cell[0], cell[1]])


def frozen_target(value):
    if isinstance(value, dict):
        if value.get("P_through_odom") and value.get("frozen_geometry", {}).get("geometry_valid"):
            return value
        for child in value.values():
            found = frozen_target(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = frozen_target(child)
            if found:
                return found
    return None


def arc_points():
    target = frozen_target(json.loads(STATE_SUMMARY.read_text(encoding="utf-8")))
    centre = tuple(target["frozen_geometry"]["portal_center_odom"])
    normal_raw = tuple(target["frozen_geometry"]["portal_normal_odom"])
    length = math.hypot(*normal_raw)
    normal = (normal_raw[0] / length, normal_raw[1] / length)
    half = float(target["portal_width_m"]) / 2.0 - RADIUS_M
    crossing = R29.maximum_curvature_crossing(CANDIDATE, centre, normal, half, KAPPA_M_INV)
    normal_yaw = math.atan2(normal[1], normal[0])
    x, y, alpha = CANDIDATE[0], CANDIDATE[1], crossing["initial_heading_error_to_normal_rad"]
    progress, points = 0.0, [(x, y)]
    while progress < crossing["normal_distance_to_plane_m"] + 0.30:
        next_alpha = alpha + crossing["turn_sign"] * KAPPA_M_INV * 0.01
        x += 0.5 * (math.cos(normal_yaw + alpha) + math.cos(normal_yaw + next_alpha)) * 0.01
        y += 0.5 * (math.sin(normal_yaw + alpha) + math.sin(normal_yaw + next_alpha)) * 0.01
        progress += 0.5 * (math.cos(alpha) + math.cos(next_alpha)) * 0.01
        alpha = next_alpha
        points.append((x, y))
    return crossing, points


def sweep(grid, points, centre, normal):
    c, s = math.cos(CANDIDATE[2]), math.sin(CANDIDATE[2])
    for index, point in enumerate(points):
        dx, dy = point[0] - CANDIDATE[0], point[1] - CANDIDATE[1]
        bx, by = c * dx + s * dy, -s * dx + c * dy
        covered = []
        for row in range(ROWS):
            x = X_MIN + (row + 0.5) * RESOLUTION
            for col in range(COLS):
                y = Y_MIN + (col + 0.5) * RESOLUTION
                if math.hypot(x - bx, y - by) <= RADIUS_M:
                    covered.append((int(grid[row, col]), x, y))
        if not covered:
            return {"safe": False, "state": "OUT_OF_GRID", "path_distance_m": index * 0.01}
        if any(value == 100 for value, _x, _y in covered):
            tangent = (-normal[1], normal[0])
            trigger = []
            for row in range(ROWS):
                x = X_MIN + (row + 0.5) * RESOLUTION
                for col in range(COLS):
                    y = Y_MIN + (col + 0.5) * RESOLUTION
                    if math.hypot(x - bx, y - by) > RADIUS_M or int(grid[row, col]) != 100:
                        continue
                    world_x = CANDIDATE[0] + math.cos(CANDIDATE[2]) * x - math.sin(CANDIDATE[2]) * y
                    world_y = CANDIDATE[1] + math.sin(CANDIDATE[2]) * x + math.cos(CANDIDATE[2]) * y
                    dx, dy = world_x - centre[0], world_y - centre[1]
                    trigger.append({
                        "grid_index": [row, col],
                        "base_cell_center_m": [x, y],
                        "world_odom_m": [world_x, world_y],
                        "portal_inward_normal_m": dx * normal[0] + dy * normal[1],
                        "portal_tangent_m": dx * tangent[0] + dy * tangent[1],
                    })
            return {"safe": False, "state": "OCCUPIED", "path_distance_m": index * 0.01, "point_base": [bx, by], "triggering_cells": trigger}
        occupied_centres = [
            (X_MIN + (row + 0.5) * RESOLUTION, Y_MIN + (col + 0.5) * RESOLUTION)
            for row, col in zip(*np.where(grid == 100))
        ]
        for value, x, y in covered:
            if value == 0:
                continue
            inside_start = x * x + y * y <= RADIUS_M * RADIUS_M
            occupied_inflated = any(math.hypot(x - ox, y - oy) <= RADIUS_M for ox, oy in occupied_centres)
            if value == -1 and inside_start and not occupied_inflated:
                continue
            return {"safe": False, "state": "UNKNOWN", "path_distance_m": index * 0.01}
    return {"safe": True, "state": "FREE"}


def empty_evidence():
    return {"lidar_free": np.zeros((ROWS, COLS), dtype=np.float32), "lidar_occ": np.zeros((ROWS, COLS), dtype=np.float32)}


def replay(bag_path):
    old, fixed = empty_evidence(), empty_evidence()
    latest_fixed_occ = np.zeros((ROWS, COLS), dtype=np.float64)
    old_pose = fixed_pose = None
    snapshots = None
    recorded_grid = None
    window_counts = []
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, message, stamp in bag.read_messages(topics=[
            "/team/livox/icp_odom_gated", "/team/livox/scan_cloud_filtered", "/team/local_traversability_grid",
        ]):
            t = stamp.to_sec()
            if t > 43.7:
                break
            if topic == "/team/livox/icp_odom_gated":
                pose = (float(message.pose.pose.position.x), float(message.pose.pose.position.y), euler_yaw(message.pose.pose.orientation))
                if fixed_pose is not None:
                    fixed = MOTION.warp_sensor_evidence(fixed, fixed_pose, pose, RESOLUTION, X_MIN, Y_MIN)
                    latest_fixed_occ = warp_latest_timestamps(latest_fixed_occ, fixed_pose, pose)
                old_pose = pose
                fixed_pose = pose
            elif topic == "/team/livox/scan_cloud_filtered":
                if old_pose is not None:
                    mark_cloud(old, message)
                    mark_cloud(fixed, message)
                    mark_latest_occurrence(latest_fixed_occ, message, t)
            else:
                old_grid, fixed_grid = labels(old), labels(fixed)
                if 37.5 <= t <= 43.7:
                    window_counts.append({"bag_stamp": t, "old": counts(old_grid), "corrected": counts(fixed_grid)})
                if abs(message.header.stamp.to_sec() - TARGET_GRID_STAMP) < 1e-6:
                    snapshots = (old_grid.copy(), fixed_grid.copy(), t, old_pose, fixed_pose, latest_fixed_occ.copy())
                    recorded_grid = message
                for evidence in (old, fixed):
                    evidence["lidar_free"] *= 0.96
                    evidence["lidar_occ"] *= 0.96
    if snapshots is None:
        raise RuntimeError("target R22 Grid snapshot not found")
    old_grid, fixed_grid, grid_bag_stamp, old_pose, fixed_pose, latest_fixed_occ = snapshots
    crossing, points = arc_points()
    raw_recorded = np.asarray(recorded_grid.data, dtype=np.int8).reshape((COLS, ROWS)).T
    target = frozen_target(json.loads(STATE_SUMMARY.read_text(encoding="utf-8")))
    centre = tuple(target["frozen_geometry"]["portal_center_odom"])
    normal_raw = tuple(target["frozen_geometry"]["portal_normal_odom"])
    normal_length = math.hypot(*normal_raw)
    normal = (normal_raw[0] / normal_length, normal_raw[1] / normal_length)
    gazebo_rows = []
    with rosbag.Bag(str(bag_path)) as bag:
        for _topic, message, stamp in bag.read_messages(topics=["/gazebo/model_states"]):
            if abs(stamp.to_sec() - TARGET_GRID_STAMP) > 0.20:
                continue
            name = "a1_gazebo" if "a1_gazebo" in message.name else "a1"
            if name in message.name:
                index = message.name.index(name)
                pose = message.pose[index]
                gazebo_rows.append((
                    stamp.to_sec(), float(pose.position.x), float(pose.position.y), euler_yaw(pose.orientation)
                ))
    gazebo_pose = min(gazebo_rows, key=lambda row: abs(row[0] - TARGET_GRID_STAMP)) if gazebo_rows else None
    gazebo_trigger = []
    if gazebo_pose:
        for cell in ((36, 43),):
            base_x = X_MIN + (cell[0] + 0.5) * RESOLUTION
            base_y = Y_MIN + (cell[1] + 0.5) * RESOLUTION
            world_x = gazebo_pose[1] + math.cos(gazebo_pose[3]) * base_x - math.sin(gazebo_pose[3]) * base_y
            world_y = gazebo_pose[2] + math.sin(gazebo_pose[3]) * base_x + math.cos(gazebo_pose[3]) * base_y
            gazebo_trigger.append({
                "grid_index": list(cell),
                "gazebo_world_m": [world_x, world_y],
                "nearest_upper_jamb_y_distance_m": abs(world_y - 15.465),
                "inside_wall_x_slab": -1.28 <= world_x <= -1.10,
                "footprint_radius_m": RADIUS_M,
                "footprint_intersects_upper_jamb_by_distance_bound": abs(world_y - 15.465) <= RADIUS_M,
            })
    return {
        "target_grid": {"content_stamp": TARGET_GRID_STAMP, "bag_stamp": grid_bag_stamp, "source_odom_pose": list(old_pose)},
        "red_test": {
            "recorded_grid_target_cells": {str(cell): target_value(raw_recorded, cell) for cell in TARGET_CELLS},
            "old_shadow_target_cells": {str(cell): target_value(old_grid, cell) for cell in TARGET_CELLS},
            "old_shadow_reproduces_raw_occupied": all(target_value(old_grid, cell) == 100 for cell in TARGET_CELLS),
        },
        "corrected": {
            "target_cells": {str(cell): target_value(fixed_grid, cell) for cell in TARGET_CELLS},
            "target_cells_not_raw_occupied": all(target_value(fixed_grid, cell) != 100 for cell in TARGET_CELLS),
            "old_counts": counts(old_grid),
            "corrected_counts": counts(fixed_grid),
            "old_sweep": sweep(raw_recorded, points, centre, normal),
            "corrected_sweep": sweep(fixed_grid, points, centre, normal),
            "corrected_trigger_cell_occupancy_last_seen": {
                str(cell): float(latest_fixed_occ[tuple(cell)])
                for cell in ((36, 43),)
            },
            "new_blocker_gazebo_ground_truth": {
                "nearest_model_state": list(gazebo_pose) if gazebo_pose else None,
                "trigger_cells": gazebo_trigger,
                "classification": "REAL_UPPER_DOOR_JAMB_FOOTPRINT_INTERSECTION" if gazebo_trigger and gazebo_trigger[0]["footprint_intersects_upper_jamb_by_distance_bound"] else "GROUND_TRUTH_UNRESOLVED",
            },
            "r30_geometry": {
                "portal_aperture_safe": crossing["aperture_safe"],
                "p_through_goal_region": abs(-0.2262277856393867) <= 0.30,
                "path_length_to_portal_m": crossing["path_length_to_plane_m"],
            },
        },
        "window_sample_count": len(window_counts),
        "window_first_last_counts": [window_counts[0], window_counts[-1]] if window_counts else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = replay(Path(args.bag))
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
