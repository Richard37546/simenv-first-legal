# A1 deterministic safe-startup supervisor V1

The supervisor is a separate Python process.  It never subscribes to Gazebo
model/link state, `/Odometry_gazebo`, `generated_building` metadata, or any
truth file.  Gazebo truth remains allowed only in separate offline acceptance
capture and is not an input to this process.

```mermaid
stateDiagram-v2
  [*] --> WAIT_GAZEBO
  WAIT_GAZEBO --> WAIT_CONTROLLER_MANAGER: clock/Gazebo fresh
  WAIT_CONTROLLER_MANAGER --> START_JUNIOR_CTRL: 12 joint state streams fresh
  START_JUNIOR_CTRL --> REQUEST_FIXEDSTAND: controller mode interface alive
  REQUEST_FIXEDSTAND --> VERIFY_FIXEDSTAND: service accepted
  VERIFY_FIXEDSTAND --> START_PERCEPTION_STACK: 10 s sim stable
  START_PERCEPTION_STACK --> WAIT_PERCEPTION_FRESH
  WAIT_PERCEPTION_FRESH --> REQUEST_RL: scan/filtered/raw/gated/L3V fresh
  REQUEST_RL --> VERIFY_RL_ZERO_HOLD: service accepted
  VERIFY_RL_ZERO_HOLD --> NAVIGATION_READY: 10 s sim zero hold
  NAVIGATION_READY --> FAIL_SAFE_HOLD: any critical input stale
  WAIT_GAZEBO --> FAIL_SAFE_HOLD: watchdog/nonzero command
  WAIT_CONTROLLER_MANAGER --> FAIL_SAFE_HOLD: watchdog/nonzero command
  START_JUNIOR_CTRL --> FAIL_SAFE_HOLD: passive dwell > 0.75 s
  REQUEST_FIXEDSTAND --> FAIL_SAFE_HOLD: passive dwell > 0.75 s
  VERIFY_FIXEDSTAND --> FAIL_SAFE_HOLD: nonzero command
  WAIT_PERCEPTION_FRESH --> FAIL_SAFE_HOLD: FixedStand lost
  VERIFY_RL_ZERO_HOLD --> FAIL_SAFE_HOLD: nonzero command
```

The 0.75 s simulation-time PASSIVE limit is deliberately below the observed
failed delayed-start point (2.770 s) and above the previously observed
immediate controller startup latency.  It is a guard, not a performance goal.

The supervisor alone publishes `/cmd_vel`, at 10 Hz with all components zero.
It publishes `/unitree/navigation_ready` at 20 Hz.  A consumer must require a
fresh receipt within 0.75 wall seconds and treat any missing/stale heartbeat as
false.  `navigation_ready=true` requires only L3V input freshness, not
`safe_for_navigation=true`; no navigation runner is launched by this task.
