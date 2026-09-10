#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

SEED="${SEED:-}"
FLOOR_COUNT="${FLOOR_COUNT:-3}"
ROOMS_PER_FLOOR="${ROOMS_PER_FLOOR:-4}"
BUILDING_WIDTH="${BUILDING_WIDTH:-20.0}"
BUILDING_LENGTH="${BUILDING_LENGTH:-36.0}"
DANGER_COUNT="${DANGER_COUNT:-3:6}"
DISTRACTOR_COUNT="${DISTRACTOR_COUNT:-4:8}"
GUI="${GUI:-true}"
PAUSED="${PAUSED:-true}"
START_CONTROLLER="${START_CONTROLLER:-1}"
START_VIRTUAL_JOY="${START_VIRTUAL_JOY:-0}"
CONTROLLER_FOREGROUND="${CONTROLLER_FOREGROUND:-1}"
START_BUILDING_CONTROL="${START_BUILDING_CONTROL:-1}"
UNITREE_CTRL_DT="${UNITREE_CTRL_DT:-0.004}"
UNITREE_RL_POLICY="${UNITREE_RL_POLICY:-stair}"
P2KG15_ANGULAR_ACTUATION_TELEMETRY="${P2KG15_ANGULAR_ACTUATION_TELEMETRY:-0}"
P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ="${P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ:-10.0}"
ROBOT_X="${ROBOT_X:-0.0}"
ROBOT_Y="${ROBOT_Y:--2.2}"
ROBOT_Z="${ROBOT_Z:-0.6}"
ROBOT_YAW="${ROBOT_YAW:-1.5708}"
STARTUP_ANCHOR_AUDIT_ENABLED="${STARTUP_ANCHOR_AUDIT_ENABLED:-0}"
STARTUP_ANCHOR_RUN_ID="${STARTUP_ANCHOR_RUN_ID:-}"
STARTUP_ANCHOR_ARCHIVE_DIR="${STARTUP_ANCHOR_ARCHIVE_DIR:-}"
STARTUP_ANCHOR_READY_FILE="${STARTUP_ANCHOR_READY_FILE:-}"
STARTUP_ANCHOR_RECORDER_PID=""
STARTUP_ANCHOR_ROSCORE_PID=""
WORLD_RESULT_COORDINATE_PID=""

case "$UNITREE_RL_POLICY" in
  stair|plane) ;;
  *)
    echo "ERROR: UNITREE_RL_POLICY must be exactly 'stair' or 'plane'; got '$UNITREE_RL_POLICY'" >&2
    exit 2
    ;;
esac

case "$P2KG15_ANGULAR_ACTUATION_TELEMETRY" in
  0|1) ;;
  *)
    echo "ERROR: P2KG15_ANGULAR_ACTUATION_TELEMETRY must be exactly '0' or '1'; got '$P2KG15_ANGULAR_ACTUATION_TELEMETRY'" >&2
    exit 2
    ;;
esac

echo "Terminating previous Gazebo, launch, controller, and optional joystick processes..."
pkill -f "roslaunch unitree_guide multi_floor_gazeboSim.launch" 2>/dev/null || true
pkill -f "building_generator_classic_control" 2>/dev/null || true
pkill -f "gzserver|gzclient|gazebo" 2>/dev/null || true
pkill -f "junior_ctrl" 2>/dev/null || true
pkill -f "virtual_joy.py" 2>/dev/null || true

echo "Sourcing ROS environment..."
source /opt/ros/noetic/setup.bash
source "$WORKSPACE_DIR/devel/setup.bash"

BUILDING_OBSTACLES_DIR="$(rospack find building_obstacles)"
UNITREE_GAZEBO_MODELS="$(rospack find unitree_gazebo)/models"
SCENE_OUTPUT_DIR="$WORKSPACE_DIR/generated_building"
RESULTS_DIR="$WORKSPACE_DIR/results"
mkdir -p "$SCENE_OUTPUT_DIR" "$RESULTS_DIR" "$WORKSPACE_DIR/logs"

echo "Generating competition scene..."
GENERATOR_ARGS=(
  --output-dir "$SCENE_OUTPUT_DIR"
  --results-dir "$RESULTS_DIR"
  --floor-count "$FLOOR_COUNT"
  --rooms-per-floor "$ROOMS_PER_FLOOR"
  --width "$BUILDING_WIDTH"
  --length "$BUILDING_LENGTH"
  --danger-count "$DANGER_COUNT"
  --distractor-count "$DISTRACTOR_COUNT"
  --robot-x "$ROBOT_X"
  --robot-y "$ROBOT_Y"
  --robot-z "$ROBOT_Z"
  --robot-yaw "$ROBOT_YAW"
)
if [ -n "$SEED" ]; then
  GENERATOR_ARGS+=(--seed "$SEED")
fi
python3 "$BUILDING_OBSTACLES_DIR/scripts/generate_competition_scene.py" "${GENERATOR_ARGS[@]}" \
  > "$SCENE_OUTPUT_DIR/scene_manifest.stdout.json"

export BUILDING_WORLD_FILE="$SCENE_OUTPUT_DIR/competition_scene.world"
export COMPETITION_ROBOT_X="$ROBOT_X"
export COMPETITION_ROBOT_Y="$ROBOT_Y"
export COMPETITION_ROBOT_Z="$ROBOT_Z"
export COMPETITION_ROBOT_YAW="$ROBOT_YAW"
export ROBOT_X ROBOT_Y ROBOT_Z ROBOT_YAW
export STARTUP_ANCHOR_SPAWN_SOURCE STARTUP_ANCHOR_RUN_ID
export UNITREE_CTRL_DT
export UNITREE_RL_POLICY
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}:$SCENE_OUTPUT_DIR:$UNITREE_GAZEBO_MODELS"

stop_startup_anchor_audit() {
  if [ -n "$STARTUP_ANCHOR_RECORDER_PID" ] && kill -0 "$STARTUP_ANCHOR_RECORDER_PID" 2>/dev/null; then
    kill -INT "$STARTUP_ANCHOR_RECORDER_PID" 2>/dev/null || true
    wait "$STARTUP_ANCHOR_RECORDER_PID" 2>/dev/null || true
  fi
  if [ -n "$STARTUP_ANCHOR_ROSCORE_PID" ] && kill -0 "$STARTUP_ANCHOR_ROSCORE_PID" 2>/dev/null; then
    kill -INT "$STARTUP_ANCHOR_ROSCORE_PID" 2>/dev/null || true
    wait "$STARTUP_ANCHOR_ROSCORE_PID" 2>/dev/null || true
  fi
}

stop_world_result_coordinate() {
  if [ -n "$WORLD_RESULT_COORDINATE_PID" ] && kill -0 "$WORLD_RESULT_COORDINATE_PID" 2>/dev/null; then
    kill -INT "$WORLD_RESULT_COORDINATE_PID" 2>/dev/null || true
    wait "$WORLD_RESULT_COORDINATE_PID" 2>/dev/null || true
  fi
}

stop_runtime_sidecars() {
  stop_world_result_coordinate
  stop_startup_anchor_audit
}

start_startup_anchor_audit() {
  [ "$STARTUP_ANCHOR_AUDIT_ENABLED" = "1" ] || return 0
  [ -n "$STARTUP_ANCHOR_RUN_ID" ] && [ -n "$STARTUP_ANCHOR_ARCHIVE_DIR" ] && [ -n "$STARTUP_ANCHOR_READY_FILE" ] || {
    echo "STARTUP_ANCHOR_AUDIT_FAILED: missing run identity/archive/ready path" >&2
    return 65
  }
  trap stop_runtime_sidecars EXIT INT TERM HUP
  mkdir -p "$STARTUP_ANCHOR_ARCHIVE_DIR"
  rm -f "$STARTUP_ANCHOR_READY_FILE"
  echo "Starting audit-only ROS master before Gazebo spawn..."
  roscore > "$STARTUP_ANCHOR_ARCHIVE_DIR/roscore.log" 2>&1 &
  STARTUP_ANCHOR_ROSCORE_PID=$!
  for _ in $(seq 1 100); do
    rostopic list >/dev/null 2>&1 && break
    sleep 0.1
  done
  rostopic list >/dev/null 2>&1 || { echo "STARTUP_ANCHOR_AUDIT_FAILED: ROS master unavailable" >&2; return 66; }
  echo "Starting audit-only startup-anchor recorder before Gazebo spawn..."
  python3 "$WORKSPACE_DIR/scripts/startup_anchor_evidence/startup_anchor_audit_recorder.py" \
    --run-id "$STARTUP_ANCHOR_RUN_ID" \
    --archive-dir "$STARTUP_ANCHOR_ARCHIVE_DIR" \
    --ready-file "$STARTUP_ANCHOR_READY_FILE" \
    > "$STARTUP_ANCHOR_ARCHIVE_DIR/recorder.log" 2>&1 &
  STARTUP_ANCHOR_RECORDER_PID=$!
  for _ in $(seq 1 100); do
    [ -s "$STARTUP_ANCHOR_READY_FILE" ] && break
    kill -0 "$STARTUP_ANCHOR_RECORDER_PID" 2>/dev/null || break
    sleep 0.1
  done
  [ -s "$STARTUP_ANCHOR_READY_FILE" ] || { echo "STARTUP_ANCHOR_AUDIT_FAILED: recorder did not become ready before spawn" >&2; return 67; }
  grep -q '"status": "READY"' "$STARTUP_ANCHOR_READY_FILE" || { echo "STARTUP_ANCHOR_AUDIT_FAILED: invalid recorder ready file" >&2; return 68; }
}

start_startup_anchor_audit

# The audit wrapper has already made ROS available here, while Gazebo and the
# raw odometry producer have not started yet.  This ordering is required for
# the transform to bind to the run's actual first raw odometry sample.
if [ "$STARTUP_ANCHOR_AUDIT_ENABLED" = "1" ] && [ "$STARTUP_ANCHOR_SPAWN_SOURCE" = "EXPLICIT_AUDIT_WRAPPER_ENV" ] && [ -n "$STARTUP_ANCHOR_RUN_ID" ]; then
  echo "Starting result-coordinate-only world anchor sidecar before Gazebo spawn..."
  WORLD_RESULT_RUN_ID="$STARTUP_ANCHOR_RUN_ID" \
    "$WORKSPACE_DIR/scripts/rgbd_danger_perception/run_world_result_coordinate.sh" \
    > "$WORKSPACE_DIR/logs/world_result_coordinate.log" 2>&1 &
  WORLD_RESULT_COORDINATE_PID=$!
  trap stop_runtime_sidecars EXIT INT TERM HUP
else
  echo "WORLD_RESULT_COORDINATE_DISABLED: explicit effective spawn/run provenance required"
fi

echo "=========================================="
echo "Competition scene is ready"
echo "  World:   $BUILDING_WORLD_FILE"
echo "  Truth:   $RESULTS_DIR/danger_truth.json"
echo "  Manifest:$SCENE_OUTPUT_DIR/scene_manifest.json"
echo "  Result:  $RESULTS_DIR/detected_danger.json"
echo "=========================================="

if [ "$START_VIRTUAL_JOY" = "1" ]; then
  echo "Starting virtual joystick. This may require uinput permissions."
  rosrun unitree_guide virtual_joy.py > "$WORKSPACE_DIR/logs/virtual_joy.log" 2>&1 &
  echo $! > "$WORKSPACE_DIR/logs/virtual_joy.pid"
fi

echo "Launching Gazebo, Unitree A1 model, sensors, and ROS interfaces..."
if [ "$STARTUP_ANCHOR_AUDIT_ENABLED" = "1" ]; then
  python3 "$WORKSPACE_DIR/scripts/startup_anchor_evidence/startup_anchor_audit_recorder.py" \
    --run-id "$STARTUP_ANCHOR_RUN_ID" \
    --archive-dir "$STARTUP_ANCHOR_ARCHIVE_DIR" \
    --ready-file "$STARTUP_ANCHOR_READY_FILE" \
    --launcher-marker SPAWN_AND_PHYSICS_LAUNCH_REQUESTED \
    --paused "$PAUSED"
fi
roslaunch unitree_guide multi_floor_gazeboSim.launch \
  gui:="$GUI" \
  paused:="$PAUSED" \
  user_debug:=False \
  rname:=a1 \
  robot_x:="$ROBOT_X" \
  robot_y:="$ROBOT_Y" \
  robot_z:="$ROBOT_Z" \
  robot_yaw:="$ROBOT_YAW" \
  > "$WORKSPACE_DIR/logs/competition_gazebo.log" 2>&1 &
LAUNCH_PID=$!
echo "$LAUNCH_PID" > "$WORKSPACE_DIR/logs/competition_gazebo.pid"
sleep 6

if [ "$START_BUILDING_CONTROL" = "1" ]; then
  echo "Starting building door/elevator control service..."
  rosrun building_generator_classic building_generator_classic_control \
    --door-config "$SCENE_OUTPUT_DIR/door_config.yaml" \
    --elevator-config "$SCENE_OUTPUT_DIR/elevator_config.yaml" \
    > "$WORKSPACE_DIR/logs/building_control.log" 2>&1 &
  echo $! > "$WORKSPACE_DIR/logs/building_control.pid"
fi

if [ "$START_CONTROLLER" = "1" ]; then
  if [ "$P2KG15_ANGULAR_ACTUATION_TELEMETRY" = "1" ]; then
    echo "Enabling P2K-G15 angular-actuation telemetry before controller startup."
    rosparam set /unitree_gazebo_servo/p2kg15_angular_actuation_telemetry true
    rosparam set /unitree_gazebo_servo/p2kg15_angular_actuation_telemetry_rate_hz "$P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ"
  fi
  if [ "$CONTROLLER_FOREGROUND" = "1" ]; then
    echo "Starting junior_ctrl controller in the foreground."
    echo "UNITREE_CTRL_DT=$UNITREE_CTRL_DT seconds."
    echo "UNITREE_RL_POLICY=$UNITREE_RL_POLICY."
    echo "Use keyboard input in this terminal: 2 = stand, 6 = RL mode."
    "$WORKSPACE_DIR/devel/lib/unitree_guide/junior_ctrl"
  else
    echo "Starting junior_ctrl controller in the background. Keyboard state switching may not be available."
    echo "UNITREE_CTRL_DT=$UNITREE_CTRL_DT seconds."
    echo "UNITREE_RL_POLICY=$UNITREE_RL_POLICY."
    "$WORKSPACE_DIR/devel/lib/unitree_guide/junior_ctrl" \
      > "$WORKSPACE_DIR/logs/junior_ctrl.log" 2>&1 &
    echo $! > "$WORKSPACE_DIR/logs/junior_ctrl.pid"
  fi
fi

echo "Simulation startup command completed."
echo "Controller mode remains governed by unitree_guide keyboard/joy input; publish geometry_msgs/Twist to /cmd_vel after RL mode is enabled."
