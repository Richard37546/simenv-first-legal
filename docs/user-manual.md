# MVP 使用手册

## 1. 固定环境基线

本项目不是无依赖的 Python 项目。乙方应优先使用以下基线环境：

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 20.04 x86_64 |
| ROS | ROS1 Noetic |
| 仿真器 | Gazebo Classic 11 |
| Python | Python 3.8 |
| 构建 | `catkin_make`、CMake、C++17、`build-essential` |
| 控制器依赖 | C++ LibTorch、NVIDIA CUDA、`liblcm-dev`、Boost |
| 建图运行时 | `ros-noetic-rtabmap-odom` |
| 工具 | `tmux`、`python3-numpy`、`zip`、`unzip` |

首次在新机器上运行前，先执行环境预检：

```bash
lsb_release -sc
rosversion -d
gazebo --version
python3 --version
nvcc --version
rospack find rtabmap_odom
test -x /usr/local/cuda/bin/nvcc
test -d /home/ros/Guoyulun/Download/libtorch
```

预期关键结果分别为 `focal`、`noetic`、Gazebo Classic 11，并且最后三个命令返回成功。

`unitree_guide/CMakeLists.txt` 当前硬编码 LibTorch 路径：

```text
/home/ros/Guoyulun/Download/libtorch
```

乙方必须使用与本机 CUDA 和 C++ ABI 兼容的 C++ LibTorch，并让它位于该路径；如果本机路径不同，先与项目负责人确认后再修改 CMake 路径。仓库没有锁定 CUDA 或 LibTorch 的具体版本，不能假设任意组合都可编译。

若目标机器尚未安装基础依赖，可在确认 Ubuntu 20.04 + Noetic 后安装：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake tmux zip unzip python3-numpy \
  libboost-all-dev liblcm-dev \
  ros-noetic-desktop-full ros-noetic-gazebo-ros \
  ros-noetic-controller-manager ros-noetic-joint-state-controller \
  ros-noetic-robot-state-publisher ros-noetic-tf \
  ros-noetic-geometry-msgs ros-noetic-nav-msgs \
  ros-noetic-sensor-msgs ros-noetic-std-msgs \
  ros-noetic-rtabmap-odom
```

若 `catkin_make` 首次失败，保留并报告第一处 CMake/Torch/CUDA 错误；不要为了绕过环境错误而删除 Torch、CUDA 或控制器依赖。

## 2. 解压后首次编译

当前 MVP 使用的项目根目录为：

```bash
cd /home/richard/simenv_official_clean
```

若不存在 `devel/setup.bash`，先编译一次工作区：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -j2
source devel/setup.bash
```

当 `simenv_stack` 已存在时，不要再次启动第二套运行栈。

如果状态机需要外部视觉语义 API，乙方还需要在项目根目录创建 `.env.local` 或 `config/vision_api.env` 并提供自己的配置。交付包刻意不包含这些本地配置或密钥；没有配置时，先查看 `debug/perception_pipeline/vision_scene_semantics.log`，不要将密钥写入仓库。

## 3. 启动 Gazebo、自动开门、建图和局部栅格

使用一条命令启动完整运行栈：

```bash
cd /home/richard/simenv_official_clean
bash scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh
```

该命令会打开名为 `simenv_stack` 的 tmux 会话：

- 窗格 0：启动 Gazebo、A1 和 `junior_ctrl`；
- 窗格 1：等待 `/set_door_state`，打开 `main_entrance`，然后启动 BEV/Livox/ICP 建图；
- 窗格 2：启动局部可通行栅格诊断节点。

Gazebo 窗口可能需要一段时间才会出现。在首次启动尚未完成前，不要重复运行启动命令。

## 4. 让机器人接受自主速度指令

在 tmux 的窗格 0 中，等待 `junior_ctrl` 打印键盘说明后，依次输入：

```text
2
6
```

`2` 用于站立，`6` 将控制器切换到 RL `/cmd_vel` 模式。必须完成该步骤后，状态机才可以驱动机器人。

常用 tmux 操作：

```text
Ctrl-b 然后方向键            切换窗格
Ctrl-b 然后 d                 脱离会话但保持运行
tmux attach -t simenv_stack   重新进入会话
```

## 5. 执行 MVP 状态机

另开一个终端执行。不要在控制器窗格内运行该命令。

```bash
cd /home/richard/simenv_official_clean
bash scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh --execute
```

脚本会启动感知辅助节点和状态机。运行结束时会在以下目录保存带编号的完整归档：

```text
debug/state_machine_navigation/run_archives/run_<id>_<timestamp>/
```

归档包含 `terminal_output.log`、运行 manifest，以及可用的状态机、目标选择、规划器、感知和房间入口 debug 输出。

### 可选：强制房间入口 MVP 路径

仅在任务明确要求测试 forced-room-entry MVP 时使用：

```bash
cd /home/richard/simenv_official_clean
bash scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh \
  --execute \
  --enable-forced-room-entry-mvp
```

不要同时运行两条状态机命令。

## 6. 检查服务和数据

在另一个已 source ROS 的终端中执行：

```bash
cd /home/richard/simenv_official_clean
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosservice list | grep -E '/set_door_state|/call_elevator'
rostopic list | grep -E '/scan|/livox|/trunk_imu|local_traversability'
```

需要人工打开主入口时：

```bash
rosservice call /set_door_state "{door_id: 'main_entrance', open: true}"
```

常用当前输出文件：

```bash
cat debug/state_machine_navigation/state_machine_navigation_summary.json
cat debug/block_astar_dwa_mature/block_astar_dwa_mature_summary.json
cat debug/short_horizon_target_selection/short_horizon_target_override.json
cat debug/navigation_stage/current_stage.json
```

这些是数据文件，不是可执行命令。请使用 `cat`、`less` 或 `jq` 查看。

## 7. 停止运行栈

状态机结束或被人工停止后，停止 tmux 运行栈：

```bash
tmux kill-session -t simenv_stack
```

等待 Gazebo 和 ROS 进程退出后再启动下一次运行。可以检查：

```bash
tmux ls
```

显示 `no server running` 表示 tmux 运行栈已经结束。

## 8. 常见问题

### `tmux session simenv_stack already exists`

进入已有会话：

```bash
tmux attach -t simenv_stack
```

或先停止它：

```bash
tmux kill-session -t simenv_stack
```

### 机器人不响应状态机运动指令

回到窗格 0，确认在 `junior_ctrl` 就绪后已先输入 `2`，再输入 `6`。同时确认状态机是在独立终端中启动的。

### 门或建图没有准备好

优先查看 tmux 窗格。窗格 1 必须先报告打开 `main_entrance`，随后才会启动建图。可检查服务：

```bash
rosservice list | grep /set_door_state
```

### 上一次运行遗留 ROS/Gazebo 进程

执行 `tmux kill-session -t simenv_stack` 后稍等，再检查 tmux 会话。未确认旧运行结束前，不要叠加启动新运行。若问题重复出现，保留日志和 run archive 用于排查，不要启动多个副本。

### `catkin_make` 找不到 Torch、CUDA 或 `lcm`

先重新执行第 1 节的环境预检。重点确认：

```bash
test -x /usr/local/cuda/bin/nvcc
test -d /home/ros/Guoyulun/Download/libtorch
dpkg -s liblcm-dev
```

若 LibTorch 不在硬编码路径，先确认其版本与 CUDA/C++ ABI 的兼容性，再决定是否调整 CMake 路径。不要用 Python 的 `torch` 包替代 C++ LibTorch。

## 9. 比赛数据边界

以下信息不能作为参赛算法输入：

- `results/danger_truth.json`；
- `generated_building` 下的布局或真值元数据；
- `/Odometry_gazebo`；
- Gazebo model/link state 相关话题或服务。

正式算法应使用已公开的传感器和控制接口。
