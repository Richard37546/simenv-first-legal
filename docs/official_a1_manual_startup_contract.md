# Official A1 Manual Startup Contract

This contract describes only the verified official baseline at `a46d947b069f851493275e0b683cd796f570186c`.

1. Start the official stack from `/home/richard/simenv_official_baseline_repro` with `GUI=false ./auto.sh` (the reproduced command fixed `SEED=77`).
2. Keep `junior_ctrl` in the foreground. Input `2`; official source maps it to FixedStand.
3. Observe stable upright FixedStand. This audit used at least 10 seconds in the uninstrumented validation.
4. Input `6`; official source maps it to RL, after which the controller accepts `/cmd_vel`.
5. Start no navigation state machine and publish no `/cmd_vel`; this audit observed no `/cmd_vel` publisher/messages.

The contract does not authorize ROS mode services, ready topics, supervisors, parameter substitutions, or any current dirty wrapper. It also does not authorize a follow-on odom, movement, or navigation test without review.
