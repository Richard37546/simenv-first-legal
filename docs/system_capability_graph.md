# 比赛系统能力图

## 审计边界

本图来自当前源码、启动脚本与 A/B 级历史归档。审计没有启动 ROS、没有发布 `/cmd_vel`，没有读取 `danger_truth.json`、`generated_building` 内容或 Gazebo 模型状态。Gazebo 真值只在测试矩阵中作为离线 oracle 设计。

![系统能力链](../debug/system_verification_baseline/system_capability_chain.png)

## 实际端到端链

`auto.sh → Gazebo/A1/传感器 → L1 filtered cloud → RTAB-Map ICP raw odom → odom gate → L3V/BEV → navigation_state_machine → target JSON → block_astar_dwa_mature_runner → /cmd_vel_raw → imu_velocity_follower → /cmd_vel → State_RL → A1`

危险源链目前止于 vision/room-viewpoint 的可见性布尔量；未发现在线定位并写出正式比赛结果的闭环。BEV 分支订阅 `/map`，但 `run_slam_bev_runtime.sh` 未启动 `/map` 发布者；实际导航主要消费 L3V。

## 能力清单

| 能力 | 代码入口 | 输入 | 输出 | 前置条件 | 成功条件 | 失败条件 | 证据状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 启动和控制模式 | auto.sh; start_runtime_stack*.sh; State_RL_test.cpp | launch args, keyboard userCmd | /cmd_vel subscriber and A1 controller | ROS/services ready; operator selects modes | controller mode ack and topic health | background input/mode unknown | partially proven |
| 仿真时间与topic健康 | Gazebo /clock; per-node timeouts | /clock and message stamps | implicit exceptions/status | use_sim_time and publishers alive | monotonic fresh messages | clock/upstream/gate/consumer not distinguished | failed contract |
| LiDAR、ICP和gated odom | run_slam_bev_runtime.sh; l2_livox_odom_gate.py | /scan, filtered cloud, raw odom | /team/livox/icp_odom_gated | sensor and ICP alive | bounded deltas and fresh stamps | raw/gate stop appears as consumer timeout | motion evidence exists |
| BEV/L3V可通行地图 | bev_node.py; l3v_local_traversability_node.py | cloud/odom/RGB-D or /map | OccupancyGrid/status | fresh required inputs and valid frame | standard contract plus fresh status | nonstandard axes, diagnostic/control mix, /map gap | proven inconsistent |
| ENTER_BUILDING | navigation_state_machine.py:5521 | odom/grid/anchor | runner target and FOLLOW transition | controller and runner ready | anchor progress threshold | stuck/input timeout | executed 63 frames |
| FOLLOW_CORRIDOR | navigation_state_machine.py:5544 | odom/grid/door/gap | corridor target or verify/turn | entry complete | continued progress or doorway transition | stuck/stale/behind target | executed 373 frames |
| room zone识别 | entry_anchor_metrics + room_zone_start_progress_m | anchor progress | room_zone_reached flag | stable anchor | progress >= resolved threshold | odom drift/threshold-only semantic | executed but no independent oracle |
| doorway/side-gap感知 | doorway_candidate_detector.py; profile functions in state machine | grid/odom/vision file | JSON candidate/latch/gap observation | fresh grid and active stage | geometry/profile/stability gates | file staleness/partial gaps | full-frame 1; partial-frame 285 |
| ROOM_SIDE_TURN | navigation_state_machine.py:6593 | committed side gap, odom/yaw | zero-linear turn then entry/follow | trigger_ready and alignment | minimum yaw and target available | candidate unavailable/yaw insufficient/watchdog | executed 8 frames |
| approach与commit目标 | door target/local_free_space/commit helpers | candidate + grid + pose | fixed JSON target | frame/grid validation | target ahead and valid | no component/stale/behind | approach covered=False; commit=True |
| ENTER_ROOM与inside确认 | ENTER_ROOM/CONFIRM_INSIDE_ROOM branches | entry target, odom, side-gap metrics | confirm or fallback | valid commit and bounded attempts | distance/lateral inside checks | progress insufficient/attempt limit | enter=0 confirm=0 |
| 房内扫描 | SCAN_ROOM_FOR_DANGER | vision/latest room viewpoint | DONE | inside confirmed | visible or scan cycle terminal | no localization/output contract | executed 0 frames |
| 危险源感知、定位和输出 | vision node + SCAN branch | RGB image/external API | visible bool only | fresh vision result | formal localized output written | writer/localization branch absent | not proven |
| 退出房间 | none | none | none | inside/scan complete | return to corridor | state absent | not implemented |
| 多房间调度和访问记忆 | none | none | none | exit available | all rooms visited once | scheduler/memory absent | not implemented |
| stuck/odom/传感器/规划失败恢复 | STUCK_RECOVERY + runner watchdogs | runner decision/exceptions | backoff/retry/FAILED | controller accepts zero/backoff | bounded recovery or fail closed | sensor loss conflated with timeout | recovery frames=9 |

## 模块依赖与关键断点

1. 控制器模式依赖人工键盘 `2 → 6`，没有 ROS 可读 mode acknowledgement。
2. L3V 的 `OccupancyGrid` 使用 x→row、y→column 的非标准布局；runner 等消费者通过复制同一假设工作，接口对通用消费者不成立。
3. L3V 明示 `diagnostic_only=true`、`safe_for_navigation=false`，但其 grid 已进入状态机和 runner 控制链。
4. 状态机有入口/扫描状态，但没有退出房间、多房间调度和访问记忆能力。
5. 危险源只有可见性终止条件，缺少定位、正式结果 schema 和 writer。
