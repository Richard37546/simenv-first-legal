#!/usr/bin/env python3
"""Compact read-only Gazebo geometry evidence sidecar.

It never publishes commands and never participates in collision checking.  The
topic contains raw `/gazebo/model_states` poses plus SDF collision metadata so
an offline audit can distinguish a zero-motion command from a nearby geometry
constraint without recording point clouds or video.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def numeric_vector(text: Optional[str], count: int) -> Optional[List[float]]:
    try:
        values = [float(item) for item in str(text or "").split()]
    except ValueError:
        return None
    return values if len(values) == count else None


def geometry_summary(collision: ET.Element) -> Dict[str, Any]:
    geometry = collision.find("geometry")
    if geometry is None:
        return {"type": "UNKNOWN"}
    box = geometry.find("box/size")
    if box is not None:
        return {"type": "box", "size_xyz_m": numeric_vector(box.text, 3)}
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return {
            "type": "cylinder",
            "radius_m": float(cylinder.findtext("radius") or 0.0),
            "length_m": float(cylinder.findtext("length") or 0.0),
        }
    sphere = geometry.find("sphere/radius")
    if sphere is not None:
        return {"type": "sphere", "radius_m": float(sphere.text or 0.0)}
    mesh = geometry.find("mesh/uri")
    return {"type": "mesh" if mesh is not None else "UNKNOWN", "uri": mesh.text if mesh is not None else None}


def load_collision_metadata(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    root = ET.parse(str(path)).getroot()
    model = root.find("model")
    if model is None:
        raise ValueError("model_missing")
    entries: List[Dict[str, Any]] = []
    for link in model.findall("link"):
        link_pose = numeric_vector(link.findtext("pose"), 6) or [0.0] * 6
        for collision in link.findall("collision"):
            collision_pose = numeric_vector(collision.findtext("pose"), 6) or [0.0] * 6
            # This intentionally retains local SDF coordinates rather than
            # claiming a collision-engine clearance calculation.
            entries.append({
                "link_name": str(link.get("name") or ""),
                "collision_name": str(collision.get("name") or ""),
                "link_pose_model_local": link_pose,
                "collision_pose_link_local": collision_pose,
                "geometry": geometry_summary(collision),
            })
    return str(model.get("name") or ""), entries


def position_of(pose: Any) -> Tuple[float, float, float]:
    position = pose.position
    return float(position.x), float(position.y), float(position.z)


class GeometrySidecar:
    def __init__(self, rospy: Any, building_sdf: Path, robot_model: str, nearby_radius_m: float,
                 rate_hz: float, static_limit: int) -> None:
        from gazebo_msgs.msg import ModelStates  # type: ignore
        from std_msgs.msg import String  # type: ignore

        self.rospy = rospy
        self.robot_model = robot_model
        self.nearby_radius_m = nearby_radius_m
        self.static_limit = static_limit
        self.building_model_name, self.static_collisions = load_collision_metadata(building_sdf)
        self.publisher = rospy.Publisher("/audit/nearby_model_geometry", String, queue_size=2)
        self._string_type = String
        self.latest: Optional[Any] = None
        self.subscriber = rospy.Subscriber("/gazebo/model_states", ModelStates, self._callback, queue_size=2)
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._publish)

    def _callback(self, msg: Any) -> None:
        self.latest = msg

    def _publish(self, _event: Any) -> None:
        if self.latest is None:
            return
        msg = self.latest
        poses = dict(zip(msg.name, msg.pose))
        robot_pose = poses.get(self.robot_model)
        if robot_pose is None:
            return
        robot_xyz = position_of(robot_pose)
        nearby: List[Dict[str, Any]] = []
        for name, pose in poses.items():
            if name == self.robot_model:
                continue
            xyz = position_of(pose)
            distance_xy = math.hypot(xyz[0] - robot_xyz[0], xyz[1] - robot_xyz[1])
            if distance_xy <= self.nearby_radius_m:
                nearby.append({"model_name": str(name), "pose_xyz": list(xyz), "center_distance_xy_m": distance_xy})
        nearby.sort(key=lambda entry: float(entry["center_distance_xy_m"]))
        # Static collision coordinates remain raw model-local metadata.  An
        # offline reader combines them with the recorded building model pose;
        # this avoids a brittle substitute for Gazebo contact/clearance truth.
        static_near = sorted(
            self.static_collisions,
            key=lambda entry: math.hypot(
                float(entry["link_pose_model_local"][0]) - robot_xyz[0],
                float(entry["link_pose_model_local"][1]) - robot_xyz[1],
            ),
        )[:self.static_limit]
        building_pose = poses.get(self.building_model_name)
        payload = {
            "schema_version": 1,
            "sim_stamp": float(self.rospy.Time.now().to_sec()),
            "audit_only": True,
            "robot_model": self.robot_model,
            "robot_pose_xyz": list(robot_xyz),
            "nearby_dynamic_models": nearby[:16],
            "building_model_name": self.building_model_name,
            "building_model_pose_xyz": list(position_of(building_pose)) if building_pose is not None else None,
            "static_collision_coordinates": "SDF_MODEL_LOCAL_RAW",
            "nearest_static_collision_metadata": static_near,
            "contact_truth": "NO_EXISTING_RUNTIME_CONTACT_TOPIC_FOUND",
        }
        self.publisher.publish(self._string_type(data=json.dumps(payload, sort_keys=True, separators=(",", ":"))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only compact Gazebo geometry evidence sidecar")
    parser.add_argument("--building-sdf", default="generated_building/model.sdf")
    parser.add_argument("--robot-model", default="a1_gazebo")
    parser.add_argument("--nearby-radius-m", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    parser.add_argument("--static-collision-limit", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="parse SDF only; does not import ROS or start a node")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    building_sdf = Path(args.building_sdf)
    if not building_sdf.is_file():
        raise SystemExit("building SDF missing: {0}".format(building_sdf))
    try:
        model_name, collisions = load_collision_metadata(building_sdf)
    except (ET.ParseError, ValueError) as exc:
        raise SystemExit("building SDF parse failed: {0}".format(exc))
    if args.check:
        print("geometry sidecar check passed: model={0} collisions={1}".format(model_name, len(collisions)))
        return 0
    if args.rate_hz <= 0.0:
        raise SystemExit("rate-hz must be positive")
    try:
        import rospy  # type: ignore
    except ImportError as exc:
        raise SystemExit("ROS Noetic environment required: {0}".format(exc))
    rospy.init_node("minimum_zero_progress_geometry_sidecar", anonymous=False)
    GeometrySidecar(rospy, building_sdf, str(args.robot_model), max(0.1, float(args.nearby_radius_m)),
                    float(args.rate_hz), max(1, int(args.static_collision_limit)))
    rospy.loginfo("[minimum_zero_progress_geometry_sidecar] passive audit-only sidecar started")
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
