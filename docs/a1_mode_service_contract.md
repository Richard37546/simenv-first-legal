# A1 mode service contract

`junior_ctrl` provides three idempotent `std_srvs/Trigger` services.  They are
processed by the FSM control loop, not by the ROS callback thread.

| Service | Accepted precondition | Result | Idempotence |
| --- | --- | --- | --- |
| `/unitree/request_fixedstand` | current mode is `PASSIVE`, `FIXEDSTAND`, or `RL` | queues `FIXEDSTAND` | already `FIXEDSTAND` does not re-enter state |
| `/unitree/request_rl` | current mode is `FIXEDSTAND` or `RL` | queues `RL` | already `RL` does not re-enter state |
| `/unitree/request_safe_hold` | always accepts a fail-safe request | control loop selects `FIXEDSTAND` only when attitude and joints are healthy; otherwise `PASSIVE` | repeated requests do not reset an already-safe mode |

The response means that the request was accepted.  The supervisor must wait
for the latched `/unitree/controller_mode` confirmation before treating the
transition as complete.  It publishes one of `PASSIVE`, `FIXEDSTAND`, `RL`, or
`OTHER` on every actual FSM state transition.

Keyboard `2` and `6` are retained as manual fallback.  They go through the
same `FSM::queueModeRequest()` path as the services; the formal startup chain
does not use a PTY or keyboard injection.

`/unitree/rl_mode_ready` remains a `std_msgs/Bool` heartbeat from `State_RL`.
It is effective only when the value is true and its local receipt age is at
most 0.75 wall seconds.  A latched true value after controller death is thus
treated as false by the supervisor.
