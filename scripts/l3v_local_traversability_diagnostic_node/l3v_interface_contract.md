# L3V Local Traversability Diagnostic Node Interface Contract

## Scope

This module is diagnostic-only. It does not publish `/cmd_vel`, does not send goals, does not call `move_base`, does not connect to frontier exploration, and must not be treated as planner-ready evidence.

## Inputs

Allowed subscribed topics:

- `/team/livox/icp_odom_gated` (`nav_msgs/Odometry`)
- `/team/livox/scan_cloud_filtered` (`sensor_msgs/PointCloud2`)
- `/real_sense/depth/image_raw` (`sensor_msgs/Image`)
- `/real_sense/depth/points` (`sensor_msgs/PointCloud2`)
- `/real_sense/rgb/camera_info` (`sensor_msgs/CameraInfo`)
- `/tf`
- `/tf_static`

Explicitly forbidden:

- `/Odometry_gazebo`
- `/ground_truth/*`
- `/gazebo/model_states`
- `/gazebo/link_states`
- `/livox/Pointcloud2`
- `danger_truth.json`
- generated building runtime layout metadata

## Outputs

### `/team/local_traversability_grid`

Type: `nav_msgs/OccupancyGrid`

Frame: `base` by default.

The grid is robot-centric:

- x forward: `0.0m` to `3.0m`
- y left/right: `-1.5m` to `1.5m`
- resolution: `0.05m/cell`
- width: `60`
- height: `60`

OccupancyGrid semantics:

- `0`: free evidence
- `100`: occupied or conflict evidence
- `-1`: unknown / insufficient evidence

Detailed semantic categories are not encoded into the `OccupancyGrid`; they are reported in JSON.

### `/team/local_traversability_evidence`

Type: `std_msgs/String`

Payload: JSON.

Contains:

- diagnostic-only flags
- grid configuration
- sensor input counters
- sector summary
- semantic detail labels:
  - `conflict`
  - `rgbd_supported_free`
  - `lidar_supported_obstacle`
- navigation/frontier/L4 safety flags, all forced to `False`

### `/team/front_auxiliary_evidence`

Type: `std_msgs/String`

Payload: JSON.

Contains:

- front LiDAR point count
- front RGB-D depth count
- LiDAR front-to-side density ratio
- LiDAR front weak flag
- RGB-D depth valid ratio
- RGB-D auxiliary usefulness flag
- `safe_for_navigation=False`

### `/team/traversability_status`

Type: `std_msgs/String`

Payload: JSON.

Contains:

- `diagnostic_only=True`
- `safe_for_navigation=False`
- `safe_for_frontier=False`
- `autonomous_l4_allowed=False`
- sensor input status
- odom traversed support
- local traversability status:
  - `FREE_SUPPORTED`
  - `OBSTACLE_SUPPORTED`
  - `UNKNOWN_INSUFFICIENT_EVIDENCE`
  - `CONFLICT_NEEDS_CAUTION`

## Sector Definition

Sectors are computed in the robot `base` frame:

- `front`: `abs(angle) <= 20 deg`
- `front_left`: `20 deg < angle <= 60 deg`
- `front_right`: `-60 deg <= angle < -20 deg`
- `left_side`: `60 deg < angle <= 120 deg`
- `right_side`: `-120 deg <= angle < -60 deg`

Each sector reports:

- LiDAR point count
- LiDAR front-to-side density ratio
- RGB-D depth valid count
- RGB-D depth valid ratio
- RGB-D front auxiliary useful
- odom traversed support
- conflict ratio
- final local traversability status

## Safety Boundary

This node is not a planner, not a map server, and not a navigation source. Any downstream use must keep the layer diagnostic until a separate manual navigation smoke-test review exists.
