# SimEnv First-Legal：四足机器人室内探索与房间搜索

基于 **ROS Noetic、Gazebo Classic 与 Unitree A1** 的室内探索项目，围绕未知室内环境中的局部导航、房间搜索和危险源感知组织感知、决策与控制模块。本仓库提供采用 **First-Legal（首个合法候选）** 选择机制的房间搜索实现及配套仿真代码。

## 项目内容

- **室内场景仿真**：生成包含房间、走廊、楼梯、门、电梯和危险源的多楼层建筑。
- **局部环境感知**：利用激光雷达、RGB-D 相机及里程计信息，为局部可通行性判断和观察目标生成提供输入。
- **分层导航决策**：通过状态机组织进入建筑、走廊通行、房间入口处理、房间搜索与返程。
- **房间搜索**：结合已观察区域、遮挡区域和危险源复查需求生成候选观察点，按任务优先级排序并逐个检查可执行性。
- **局部路径与运动控制**：衔接 A* 路径规划、DWA 局部控制和 A1 控制器。
- **执行反馈**：依据实际新增观察更新搜索记忆，结合收益递减与无进展保护决定继续搜索或退出。

## 系统结构

```text
Gazebo 场景与 A1 传感器
          │
          ▼
里程计 / 局部栅格 / 可通行性 / RGB-D 危险源感知
          │
          ▼
导航状态机 → 房间搜索候选生成与轻量排序
                          │
                          ▼
                  First-Legal 可执行性检查
                          │
                          ▼
                  A* / DWA → A1 控制器
                          │
                          ▼
                  实际观察反馈与搜索记忆
```

## First-Legal 搜索策略

### 1. 生成候选观察点

根据当前局部自由空间、房间坐标系和观察记忆生成候选点，以扇区代表点控制候选规模，并为候选关联覆盖、遮挡揭示和危险源复查信息。

### 2. 按任务需求排序

优先考虑满足条件的危险源复查候选；普通候选依次考虑新增可观察区域、遮挡揭示、同等收益下的重复访问偏好、几何距离和朝向变化等因素。轻量排序负责确定检查顺序，可执行性由后续正式检查判定。

### 3. 选择首个合法候选

按排序逐个执行正式可执行性检查，遇到第一个通过检查的候选即选中。常规路径先检查前 **5** 个候选；只有这些候选均未通过时，才继续检查剩余的有界候选集合。该机制能够在找到可执行目标后停止本轮后续候选检查。

### 4. 根据实际反馈继续搜索或返程

执行结束后，以实际观察结果更新记忆，记录未产生有效观察的局部机会，并结合边际收益、无进展保护和房间出口锚点处理搜索结束与返程。

## 技术栈与代码入口

| 层次 | 技术 / 主要入口 |
| --- | --- |
| 仿真与机器人 | Gazebo Classic、Unitree A1、`src/` |
| 通信与构建 | ROS Noetic、catkin、CMake |
| 开发语言 | Python、C++、Bash |
| 导航状态机 | `scripts/local_subgoal_runner_mvp/navigation_state_machine.py` |
| 房间搜索核心 | `scripts/local_subgoal_runner_mvp/room_search_v1.py` |
| 局部规划与控制 | `scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py` |
| 局部可通行性 | `scripts/l3v_local_traversability_diagnostic_node/` |
| 危险源感知 | `scripts/rgbd_danger_perception/` |
| 仿真启动 | `auto.sh` |

## 获取与运行

### 获取代码

```bash
git clone https://github.com/Richard37546/simenv-first-legal.git
cd simenv-first-legal
```

也可以通过仓库页面的 **Code → Download ZIP** 下载源码。

### 环境准备

使用 Ubuntu 20.04 / ROS Noetic 环境，准备 Gazebo Classic、catkin、Python 3.8 及相应 ROS 依赖。A1 控制器还需要 libtorch；依赖和构建配置见[快速启动文档](docs/quick-start.md)。

构建前请按本机安装位置核对 `src/unitree_guide/unitree_guide/unitree_guide/CMakeLists.txt` 中的 libtorch 与 CUDA 路径。部分历史辅助脚本使用 `/home/richard/simenv_official_clean` 作为固定工作目录，使用这些脚本前需将路径改为实际克隆目录。

### 编译工作空间

在仓库根目录执行：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -j2
source devel/setup.bash
```

### 启动仿真与机器人控制器

```bash
GUI=true PAUSED=false ./auto.sh
```

`auto.sh` 负责场景生成、Gazebo 启动、机器人加载、门和电梯服务以及控制器启动。该脚本会清理已有的相关仿真进程，请避免与其他 Gazebo 任务同时使用。控制器前台终端支持键盘 `2` 进入站立状态、`6` 切换至 RL 模式。无界面环境可设置 `GUI=false`。

### 启动导航链路

感知与局部导航的运行入口为 `scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh`。该入口会同时启动仿真和感知组件，可替代上面的独立仿真启动方式；使用前应核对脚本及 `scripts/slam_bev_runtime/run_slam_bev_runtime.sh` 的工作目录。

在机器人控制器、里程计、局部栅格和可通行性输入就绪后，在另一个已加载 ROS 与工作空间环境的终端执行：

```bash
bash scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh \
  --execute --enable-hierarchical-portal-local-autonomy --enable-room-search-v2
```

## 文档导航

- [快速启动与环境依赖](docs/quick-start.md)
- [算法接入接口](docs/algorithm-interfaces.md)
- [场景规则](docs/competition-rules.md)
- [传感器与 ROS 话题](docs/sensors-and-topics.md)
- [门与电梯控制](docs/doors-and-elevator.md)
- [结果格式与评估](docs/evaluation.md)
- [源码来源与校验清单](FIRST_LEGAL_SOURCE_MANIFEST.json)

## 版本与许可证

本仓库聚焦 First-Legal 版本，包含仿真环境、搜索实现及相关测试。源码来源与文件校验信息见清单。项目保留原有 [AFL-3.0 许可证](LICENSE)；第三方组件遵循各自随附的许可证。
