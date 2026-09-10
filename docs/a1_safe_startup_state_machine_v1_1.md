# A1 Safe Startup State Machine V1.1

`junior_ctrl` is launched as a supervised process immediately beside the Gazebo launch. Its process launch is not gated by controller-manager, Livox, ICP, L3V, or navigation freshness. The bootstrap waits only for ROS master availability, sets the existing global `/robot_name=a1` contract consumed by `IOROS`, then execs the unchanged controller binary.

```text
WAIT_GAZEBO_SPAWN
  -> START_JUNIOR_CTRL_EARLY
  -> WAIT_CONTROL_PREREQUISITES
  -> REQUEST_FIXEDSTAND
  -> VERIFY_FIXEDSTAND
  -> START_OR_WAIT_PERCEPTION
  -> WAIT_PERCEPTION_FRESH
  -> REQUEST_RL
  -> VERIFY_RL_ZERO_HOLD
  -> NAVIGATION_READY
```

Any failed health check transitions to `FAIL_SAFE_HOLD`, keeps `/unitree/navigation_ready=false`, and leaves the supervisor publishing zero `Twist` only.

## FixedStand Preconditions

The request requires: progressing `/clock`; live `junior_ctrl`; advertised mode service; three continuous finite state samples from each servo; a finite normalized IMU quaternion; live servo state updates; an observed zero `/cmd_vel`; and no non-expired nonzero command receipt. These conditions are tracked by ROS receipt time and their first simulation timestamp is archived.

The PASSIVE safety reference starts when continuous controller state becomes available. The limit is `1.0 s` simulation time. It is deliberately below the approximately `2.77 s` historical risk sample while allowing the prior stable `0.227 s` and `0.503 s` request/effective evidence. Both controller-ready-to-effective and clock/spawn-to-effective dwell are recorded.

`MotorCmd` is not a PASSIVE prerequisite: no FixedStand target exists before the request. It is instead observed during FixedStand verification together with joint state tracking and IMU attitude health. This avoids a circular request dependency.

## Truth Boundary

`a1_safe_startup_supervisor.py` imports no Gazebo model/link state message and makes no truth subscription. `collect_offline_truth.py` is a separate process that only writes post-run acceptance evidence. A truth-upright/supervisor disagreement is classified offline as `SUPERVISOR_ATTITUDE_PREDICATE_MISMATCH`; it is never fed to the reducer.

## Perception and RL

Perception starts only after FixedStand verification. Delayed Livox, ICP, or L3V cannot delay early `junior_ctrl` launch or the FixedStand request. RL is requested only after perception is fresh, then requires a fresh RL-ready heartbeat, zero command hold, and critical health before navigation-ready can be asserted. Any critical freshness loss revokes navigation-ready.
