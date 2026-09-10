# SimEnv First-Legal

Historical complex room-search strategy for ROS Noetic, Gazebo Classic and Unitree A1.

## Strategy

The robot generates room-local observation candidates from free-space and observation memory, prioritizes danger re-observation, then ranks coverage, occlusion reveal, revisit preference, distance and heading. It checks candidates in order and selects the **first formally legal candidate**. The first five form the normal fast path; remaining bounded candidates are checked only if those fail. Actual observation feedback, diminishing-return completion and room-return behavior are preserved.

## Snapshot provenance

This is a reconstructed source release, not a copy of the original repository history.

- The two strategy files are byte-identical to the verified pre-change backup of 2026-09-04 07:09:09 UTC.
- Navigation support snapshots come from `controlled_yaw_shadow_20260903_091716` and their recorded SHA-256 checksums were verified.
- Remaining simulator/support files come from checkpoint `ba83ed62d8983ae8a29729b4146567acf89f9cfa`. Therefore this is not a claim that every file represents one historical instant.
- Only the first-legal selector is included. Later selector experiments, their tests, planning reports, runtime recordings, credentials and original Git history are excluded.
- The historical P_through default is **16**, preserved with its matching historical test.
- One legacy completion test fixture was given a positive path length to satisfy the already-existing termination contract; no production logic was changed for that adjustment.

`FIRST_LEGAL_SOURCE_MANIFEST.json` records per-file provenance and checksums.

## Environment and use

See [simulation setup](docs/quick-start.md), [simulation overview](docs/simulation-overview.md), and [algorithm interfaces](docs/algorithm-interfaces.md). Build in an existing ROS Noetic environment with the documented project dependencies; build products are not bundled.

```bash
source /opt/ros/noetic/setup.bash
catkin_make -j2
source devel/setup.bash
```

The historical runtime launcher is `scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh`. After simulator, controller and perception readiness, the historical navigation command is:

```bash
bash scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh \
  --execute --enable-hierarchical-portal-local-autonomy --enable-room-search-v2
```

Some historical helper scripts retain `/home/richard/simenv_official_clean` as their workspace path. Review these paths before running a clone elsewhere; do not overwrite an existing working directory. Machine-specific runtime configuration, including vision API environment files, must be supplied locally and must remain untracked.

## Verification

158 focused offline tests passed across 10 test modules. All 43 top-level navigation Python modules compiled successfully. No Gazebo mission, robot motion, full C++ rebuild or whole-repository test run was performed for this upload. See [verification record](VERIFICATION.md).
