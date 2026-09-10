# Livox 插件加载与发布契约

- SDF/xacro: `a1_description/xacro/gazebo.xacro`，ray sensor `laser_livox`，10 Hz，`liblivox_laser_simulation.so`。
- 输入: plugin 内 CSV `package://a1_description/scan_mode/mid360.csv` 和 `<ray>` 配置。
- 输出: `/scan`，`sensor_msgs/PointCloud`，frame `laser_livox`。
- 必要顺序: 类型转换成功 → CSV/shape 配置 → `RayPlugin::Load()` 连接 callback → `RaySensor::SetActive(true)` → callback → publish。
- 下游 canonical 链: `/scan` → `/team/livox/scan_cloud_filtered` → raw ICP → gated ICP → L3V。
- 就绪条件: 不得仅按 3 s sim 或短 wall timeout；必须观察 `load_complete` 后的实际 `/scan` 消息。
