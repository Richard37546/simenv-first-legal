# RL Mode And Zero Hold Contract

## State transition contract

This contract is derived from the current `junior_ctrl` source, not from terminal instructions.

| Current state | Input | `UserCommand` | Next state | Ready result |
|---|---|---|---|---|
| initial | none | `NONE` | `PASSIVE` | `false` from `State_RL` construction |
| `PASSIVE` | `2` | `L2_A` | `FIXEDSTAND` | false |
| `FIXEDSTAND` | `6` | `RL` | `RL` | true after `State_RL::enter()` starts the inference thread |
| `RL` | `2` | `L2_A` | `FIXEDSTAND` | false from `State_RL::exit()` |
| `RL` | `1` | `L2_B` | `PASSIVE` | false from `State_RL::exit()` |

The required test sequence is therefore `2`, wait for fixed stand to settle, then `6`. The initial FSM state is `PASSIVE` (`FSM::initialize()`), so a direct `6` is not a legal initial transition.

`FSM::_currentState`, `_nextStateName`, and `_mode` own the transition. `State_RL` owns the command subscriber, inference thread state, and ready publisher. It subscribes to `/cmd_vel` in its constructor, but the policy consumes `current_cmd_vel_` only through the RL observation path. `RL::exit()` is the normal exit path.

## Ready contract used by V1.1

The controller publishes a latched `std_msgs/Bool` to `/unitree/rl_mode_ready`. It sends false at construction and exit, and republishes true every 0.2 wall seconds while executing `State_RL::run()`.

`std_msgs/Bool` has no timestamp. The topic alone cannot make a prior latched true become false when the controller dies. The V1.1 *audit monitor* therefore defines an effective test-only ready state:

```
effective_ready = last_received_value == true
                  and receipt_wall_age <= 0.75 s
```

This is an observer rule, not a controller or navigation modification. Node liveness is not used as ready evidence. A future online health consumer must implement an equivalent persisted receipt-time watchdog before this signal is used as a safety gate.

## Zero command contract

The V1.1 audit owns the only command publisher it starts and publishes `geometry_msgs/Twist()` at 10 Hz. All emitted fields are exactly zero. It starts no navigation state machine, runner, follower, doorway process, or room-entry process.

The private controller field `FSMState::current_cmd_vel_` has no diagnostic topic. Receipt is therefore evidenced by the actual `/cmd_vel` subscription connection (`/unitree_gazebo_servo` in the V1.1 run), the source-proven RL policy use of that field, and the continuous all-zero input stream. It is not possible to observe its private command age directly without changing the controller; the audit records source-message age instead.

## Acceptance rule

Do not start an odometry stationary run unless all are true for at least 20 seconds of ROS simulation time: effective ready remains true, every test-published command is zero, required perception/odom topics remain fresh, truth position has no 10-second displacement above 0.05 m, yaw changes less than 2 degrees, and height/roll/pitch indicate an upright robot. A failed prerequisite is a robot/control/physics sample failure, not an ICP conclusion.

## Current V1.1 application

The current binary identity is
`devel/lib/unitree_guide/junior_ctrl` SHA-256
`b61417a271015b537f8a912d1f291ee1d63aaa492aecd055bc7cda181ae36e0c`.
The 2026-07-29 `stationary_02` run observed `false` before the user selected
RL and `true` after the user selected `6`, then maintained an all-zero command
source. It nevertheless failed the hold at 20.007 seconds: truth moved
0.07987 m and yaw changed 8.454 degrees. This exercises the contract's
fail-closed branch; it does not qualify odometry or authorize rotation.
