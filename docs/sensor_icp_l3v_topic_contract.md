# Sensor–ICP–L3V topic contract

## Canonical chain

`liblivox_laser_simulation.so → /scan → l1s_scan_clean_cloud_node.py → /team/livox/scan_cloud_filtered → rtabmap_odom/icp_odometry → /team/livox/icp_odom_raw → l2_livox_odom_gate.py → /team/livox/icp_odom_gated → L3V`.

`/scan` is a direct `sensor_msgs/PointCloud` output configured in the Gazebo Livox plugin. It is not produced by `pointcloud2livox.py`. The latter is a parallel legacy converter: it consumes `/scan`, emits `/livox/lidar2` and `/livox/Pointcloud2`, and uses `/Odometry_gazebo`; it is not the canonical filtered-cloud input and must not be promoted into competition control.

| Producer | Topic | Type | Frame contract | Consumer | Observed startup result |
|---|---|---|---|---|---|
| Gazebo Livox plugin | `/scan` | `sensor_msgs/PointCloud` | `laser_livox` | L1S filter, legacy converter | 0 messages in each 3s-sim run |
| Legacy converter | `/livox/lidar2` | `unitree_guide/CustomMsg` | inherited | diagnostic only | 0 because `/scan` is 0 |
| Legacy converter | `/livox/Pointcloud2` | `sensor_msgs/PointCloud2` | declares `odom` | diagnostic only | 0 because `/scan` is 0 |
| L1S filter | `/team/livox/scan_cloud_filtered` | `sensor_msgs/PointCloud2` | preserves input header | ICP, L3V, V1 health | 0 because `/scan` is 0 |
| ICP | `/team/livox/icp_odom_raw` | `nav_msgs/Odometry` | `team_livox_odom → base` | odom gate, V1 health | 0 because filtered cloud is 0 |
| Odom gate | `/team/livox/icp_odom_gated` | `nav_msgs/Odometry` | `team_livox_odom → base` | L3V, V1 health | 0 because raw ICP is 0 |
| L3V | `/team/local_traversability_grid` | `nav_msgs/OccupancyGrid` | `base`; origin `(0, -1.5)` | V1 health | heartbeat exists; unknown/stale evidence |

## TF audit

- `base ← laser_livox`: observed static transform `[0.200, 0, 0.080]`, yaw about `45°`.
- `base ← real_sense`: observed static transform `[0.280, 0, 0.043]`.
- `team_livox_odom ← base`: absent when no ICP odometry exists. This is downstream of the first broken edge.
- L1S does not transform its cloud; ICP requests transform to `base`. The required sensor-to-base TF exists, so TF is not the first cause of zero `/scan`.

## L3V status semantics

L3V always publishes a heartbeat grid/status. A fresh header does not certify fresh sensor evidence: `input_freshness.all_required_inputs_fresh` is the authoritative freshness field. `safe_for_navigation` is deliberately false in this diagnostic node and was not changed by this audit.
