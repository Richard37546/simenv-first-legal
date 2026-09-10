# 关键接口契约

## 坐标与时间总则

- 机器人约定应为 `base` 中 x 向前、y 向左、z 向上；里程计目标在 `team_livox_odom`。
- 标准 `nav_msgs/OccupancyGrid` 应是 column 对应 x、row 对应 y，索引 `row * width + column`。当前 L3V 实现相反，是已证实的契约缺陷。
- ROS header stamp 表示仿真时间；watchdog 可以使用 wall time，但必须分别报告 ROS age、wall age 和 RTF，不得把 wall timeout 当作 ROS 数据时长。
- 预测、诊断、真实观测必须带 provenance。`diagnostic_only` 结果不得在没有显式升格与安全验收的情况下进入控制。

## 接口表

| 接口 | 类型 | 发布者 | 订阅者 | frame | 轴/坐标 | 频率 | 时间基准 | 最大新鲜度 | 参数来源 | free/occ/unknown | 错误码 | fallback | 掩盖风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /clock | rosgraph_msgs/Clock | Gazebo | all ROS-time consumers | n/a | simulation time | sim rate | ROS | consumer-specific | n/a | no unified code | no | implicit timeout | can hide clock-vs-upstream cause |
| /scan | sensor_msgs/PointCloud | Livox simulator | L1 preprocess | laser_livox expected | sensor xyz | sensor rate | ROS stamp | not archived | launch/default | n/a | node exception/log | none | no |
| /team/livox/scan_cloud_filtered | sensor_msgs/PointCloud2 | L1 preprocess | ICP + L3V + runner | laser_livox | x forward convention assumed | sensor rate | ROS+wall receipt | L3V 2 s | run_slam_bev_runtime.sh | points | timeout/log | RGB-D auxiliary in L3V | may mask weak LiDAR |
| /team/livox/icp_odom_raw | nav_msgs/Odometry | rtabmap icp_odometry | odom gate | team_livox_odom/base | SE(2/3) pose | ICP rate | ROS | gate callback only | run_slam_bev_runtime.sh | n/a | none persistent | none | no |
| /team/livox/icp_odom_gated | nav_msgs/Odometry | l2_livox_odom_gate | L3V/state machine/runner/detector | team_livox_odom; child base | x/y world, yaw | input-driven | ROS stamp | state/runner arg 20 s | gate defaults + launch remaps | n/a | status string only on callback | last accepted pose | stale pose may persist downstream |
| /team/local_traversability_grid | nav_msgs/OccupancyGrid | L3V | state machine/runner/detector/frontier | base | implementation: row=x, col=y (nonstandard) | 2 Hz heartbeat | new ROS stamp + wall input freshness | inputs 2 s; consumer 20 s | L3V ROS params | 0 free,100 occupied,-1 unknown | status separate | all-unknown heartbeat | fresh header can mask stale inputs |
| /team/traversability_status | std_msgs/String JSON | L3V | runner | JSON frame_id=base | health/semantic flags | 2 Hz | ROS publish + wall freshness fields | 2 s inputs | L3V params | explicit statuses | JSON fields, no typed error | UNKNOWN_INSUFFICIENT_EVIDENCE | grid remains fresh |
| /bev/occupancy_grid | nav_msgs/OccupancyGrid | bev_node | diagnostics | declared base but data/origin copied from /map | standard array intended | input-driven latch | input ROS stamp | none | bev_params.yaml/ROS params | 0/100/-1 | none | latched last grid | can mask missing updates |
| /team/doorway_candidate | std_msgs/String JSON | doorway detector | diagnostic/optional consumers | fields include base/odom context | opening profile | grid-driven | mixed ROS/wall/file ages | vision max-age parameter | source defaults; not resolved in manifest | ratios + segments | final_decision string | profile/vision fallback | fallback can dilute source provenance |
| /team/room_frontier_viewpoint | std_msgs/String JSON | room frontier selector | state machine via latest file | base target | x forward/y left | grid-driven | mixed topic/file | file age checks vary | source defaults | unknown policy internal | decision string | fallback x target | fallback may look like perception |
| debug/.../short_horizon_target_override.json | file JSON | state machine | runner | team_livox_odom target; base diagnostics | world target transformed at consume time | per iteration | wall file mtime + embedded stamps | no universal TTL | state machine args | n/a | decision fields | reuse locked target | stale target risk |
| /cmd_vel_raw | geometry_msgs/Twist | runner/state machine | imu_velocity_follower | base body | linear x forward; angular z yaw | slice loop | wall freshness in follower | follower configured age | state args/ROS params | n/a | follower status | zero on stale | safer than direct path |
| /cmd_vel | geometry_msgs/Twist | follower or direct turn/runner | State_RL | base body | linear x/y; angular z | command loop | callback ROS stamp but no controller age check | none in State_RL | hardcoded subscriber | n/a | no ack | last command retained | can hide publisher death |
| /trunk_imu | sensor_msgs/Imu | A1 | follower/runner | trunk | orientation/yaw rate | sensor rate | ROS+wall receipt | follower timeout | ROS params | n/a | status | follower stops | none |
| /imu_velocity_follower/status | std_msgs/String JSON | follower | state machine | n/a | command/IMU health | loop rate | mixed | state wait timeout | ROS params | n/a | JSON error fields | direct /cmd_vel bypass | bypass removes follower protections |
| /set_door_state | service (project-specific) | building control | start scripts | world entity | door command | on demand | wall wait | 120 s in non-tmux; unbounded tmux | launch/service | n/a | shell status | none | startup can hang |
| danger result output | missing formal contract | none found | competition evaluator | required world/local pose unspecified | detection/location | n/a | n/a | n/a | n/a | n/a | none | visible bool only | absence can be mistaken for no danger |

## grid/target 执行契约结论

1. L3V 发布 `width=cols(y)`、`height=rows(x)`，`origin=(0,y_min)`，同时 `data[row=x][col=y]`。这不是标准 OccupancyGrid 几何。
2. `navigation_state_machine.local_xy_to_cell()` 返回 `(x-index,y-index)` 后以二维数组 `[row,col]` 使用；runner 也复刻该特殊约定，形成内部耦合而非公开契约。
3. BEV 将 `/map` 的 data/info 原样复制并仅把 frame 改为 `base`，没有 TF 变换。
4. 目标以固定 JSON 文件交接，缺少统一 `run_id + seq + created_ros + created_wall + source_pose + ttl + consumed_ack`。
5. runner 的 sequential waits 和状态机的一次性 odom waits 无法区分 upstream stopped、gate stopped、clock stopped 和 consumer timeout。

## 建议的统一错误状态

每个关键输入至少应公开 `FRESH / STALE / LOST / INVALID_FRAME / INVALID_SCHEMA`，并附最后 ROS stamp、最后 wall receipt、message count、interval 和 source node alive。fallback 必须保留原错误，不得只输出 fallback 成功。
