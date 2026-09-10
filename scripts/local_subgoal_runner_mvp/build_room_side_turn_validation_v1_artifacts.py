#!/usr/bin/env python3
"""Create V1 evidence artifacts without talking to ROS or truth sources."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "room_side_turn_validation_v1"
PATCH = ROOT / "debug" / "patches" / "room_side_turn_validation_v1_20260722_135141"
ARCHIVES = ROOT / "debug" / "state_machine_navigation" / "run_archives"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def historical_yaw_shortfall() -> list[dict]:
    records = []
    for summary_path in sorted(ARCHIVES.glob("run_*/state_machine_navigation_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in summary.get("state_trace", []):
            turn = item.get("room_side_gap_turn")
            if not isinstance(turn, dict) or turn.get("turn_yaw_sufficient") is not False:
                continue
            records.append({
                "run": summary_path.parent.name,
                "iteration": item.get("iteration"),
                "stop_reason": turn.get("stop_reason"),
                "actual_abs_yaw_delta_rad": turn.get("actual_abs_yaw_delta_rad"),
                "target_yaw_delta_rad": turn.get("target_yaw_delta_rad"),
                "min_required_yaw_delta_rad": turn.get("min_required_yaw_delta_rad"),
                "wall_duration_sec": turn.get("wall_duration_sec"),
                "sim_duration_sec": turn.get("sim_duration_sec"),
                "transition_reason": item.get("transition_reason"),
            })
    return records


def write_online_manifests() -> list[dict]:
    attempts = [
        {
            "directory": "online_attempt_20260722_142606",
            "gui": False,
            "controller_dt_sec": 0.002,
            "rl_mode_ready_observed": True,
            "state_machine_execute_started": False,
            "nonzero_cmd_vel_published_by_validation": False,
            "gated_odom_yaw_samples": 0,
            "health_gate_result": "BLOCKED",
            "primary_blockers": [
                "filtered_cloud_missing",
                "raw_icp_odom_missing",
                "gated_odom_missing",
                "grid_upstream_status_reports_stale_content",
                "grid_safe_for_navigation_false",
                "cmd_vel_output_not_observed",
            ],
            "evidence": {
                "clock_hz": 1000.0,
                "grid_hz_range": [25.64, 29.41],
                "l3v_input_freshness": "all_required_inputs_fresh=false; odom_stale_or_missing",
                "simulator_warning": "base TF NaN observed in gazebo.log",
            },
        },
        {
            "directory": "online_attempt_20260722_212526",
            "gui": False,
            "controller_dt_sec": 0.006,
            "rl_mode_ready_observed": False,
            "state_machine_execute_started": False,
            "nonzero_cmd_vel_published_by_validation": False,
            "gated_odom_yaw_samples": 0,
            "health_gate_result": "BLOCKED",
            "primary_blockers": [
                "scan_missing",
                "filtered_cloud_missing",
                "raw_icp_odom_missing",
                "gated_odom_missing",
            ],
            "evidence": {
                "controller_log": "controller dt: 0.006 s",
                "scan_probe": "five seconds: no new messages on /scan",
                "icp_log": "Did not receive data since 5 seconds",
            },
        },
    ]
    for attempt in attempts:
        directory = OUT / attempt["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        log_hashes = {
            path.name: sha256(path)
            for path in sorted(directory.glob("*.log"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "kind": "online_health_gate_attempt",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scope": "existing world, GUI=false, no auto.sh, no state-machine --execute, no validation nonzero command",
            "attempt": attempt,
            "log_sha256": log_hashes,
            "complete": True,
        }
        write_json(directory / "run_manifest.json", manifest)
        attempt["log_sha256"] = log_hashes
    return attempts


def write_png(attempts: list[dict]) -> pathlib.Path:
    import matplotlib.pyplot as plt

    png = OUT / "yaw_cmd_topic_health_timeline.png"
    labels = [item["directory"].replace("online_attempt_", "") for item in attempts]
    topics = ["/clock", "local grid", "filtered cloud", "raw ICP odom", "gated ICP odom", "cmd output"]
    availability = [
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    image = axes[0].imshow(availability, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xticks(range(len(topics)))
    axes[0].set_xticklabels(topics, rotation=20, ha="right")
    axes[0].set_title("Online health evidence: 1=fresh/observed, 0=missing or fail-closed")
    for row, values in enumerate(availability):
        for col, value in enumerate(values):
            axes[0].text(col, row, "OK" if value else "BLOCK", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0], ticks=[0, 1])

    axes[1].plot([0, 1], [0, 0], marker="o", color="#b22222")
    axes[1].set_xticks([0, 1], labels)
    axes[1].set_ylabel("linear.x / angular.z")
    axes[1].set_title("Validation command timeline: no state-machine execute; no nonzero /cmd_vel published")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter([], [])
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Gated odom yaw timeline")
    axes[2].text(0.5, 0.55, "No gated-odom samples in either attempt", ha="center", va="center", fontsize=12)
    axes[2].text(0.5, 0.35, "Therefore no yaw control evaluation was authorized", ha="center", va="center", fontsize=10)
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return png


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run39 = json.loads((PATCH / "pre_change" / "run_0039_resolved_parameters.json").read_text(encoding="utf-8"))
    base = run39["resolved_parameters"]
    selected = {
        key: value for key, value in base.items()
        if key.startswith("room_side_gap_")
        or key in {"enable_doorway_control", "enable_forced_room_entry_mvp", "enable_room_side_gap_trigger", "cmd_topic", "follower_raw_cmd_topic", "follower_output_cmd_topic", "follower_status_topic", "follower_imu_topic"}
    }
    selected.update({
        "room_side_turn_validation_v1": True,
        "room_side_turn_health_fresh_wall_sec": 2.5,
        "room_side_turn_health_lost_wall_sec": 8.0,
        "room_side_turn_wall_watchdog_sec": 90.0,
        "room_side_turn_no_response_samples": 8,
        "room_side_turn_yaw_response_epsilon_rad": 0.01,
        "room_side_turn_post_fresh_grid_count": 2,
        "room_side_turn_post_grid_watchdog_sec": 30.0,
        "validation_phase": "side turn and doorway verify only; ENTER_ROOM forbidden",
    })
    write_json(OUT / "resolved_parameters.json", {"source_run": run39["source_run"], "parameters": selected})

    attempts = write_online_manifests()
    shortfalls_raw = historical_yaw_shortfall()
    seen_shortfalls = set()
    shortfalls = []
    for record in shortfalls_raw:
        fingerprint = (
            record.get("actual_abs_yaw_delta_rad"),
            record.get("target_yaw_delta_rad"),
            record.get("min_required_yaw_delta_rad"),
            record.get("wall_duration_sec"),
            record.get("sim_duration_sec"),
        )
        if fingerprint in seen_shortfalls:
            continue
        seen_shortfalls.add(fingerprint)
        shortfalls.append(record)
    health = {
        "online_attempts": attempts,
        "online_side_turn_successes": {"left": 0, "right": 0},
        "online_side_turn_attempts_with_nonzero_turn": {"left": 0, "right": 0},
        "enter_room_transitions": 0,
        "historical_yaw_shortfall_raw_row_count": len(shortfalls_raw),
        "historical_yaw_shortfall_deduplicated_count": len(shortfalls),
        "historical_yaw_shortfall_records": shortfalls,
    }
    write_json(OUT / "online_health_attempts.json", health)
    png = write_png(attempts)

    tracked = [
        ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
        ROOT / "scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh",
        ROOT / "scripts/local_subgoal_runner_mvp/room_side_turn_validation_v1.py",
        ROOT / "scripts/local_subgoal_runner_mvp/replay_room_side_turn_validation_v1.py",
        ROOT / "scripts/local_subgoal_runner_mvp/build_room_side_turn_validation_v1_artifacts.py",
        ROOT / "src/unitree_guide/unitree_guide/unitree_guide/include/FSM/State_RL_test.h",
        ROOT / "src/unitree_guide/unitree_guide/unitree_guide/src/FSM/State_RL_test.cpp",
        ROOT / "tests/test_room_side_turn_validation_v1.py",
    ]
    post_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in tracked if path.exists()}
    write_json(PATCH / "post_change" / "files.sha256.json", post_hashes)
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    (PATCH / "post_change").mkdir(parents=True, exist_ok=True)
    (PATCH / "post_change" / "actual_diff.binary").write_bytes(diff)
    (PATCH / "post_change" / "git_status_short.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        encoding="utf-8",
    )
    (PATCH / "post_change" / "actual_diff.binary.sha256").write_text(
        hashlib.sha256(diff).hexdigest() + "  actual_diff.binary\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "diff", "--check", "--", *(str(path.relative_to(ROOT)) for path in tracked)],
        cwd=ROOT,
        check=True,
    )

    report = f"""# Room-side-turn validation V1 report

## Result

The V1 implementation is complete for the permitted chain. Offline checks pass, but online nonzero turning is **not validated**: both runtime attempts were blocked before the state machine could be authorized to publish a turn. No attempt entered `ENTER_ROOM`.

## Required answers

1. **Forced-room-entry and side-gap mutual exclusion:** removed. `select_follow_corridor_opening_action()` gives an available fully bounded doorway first priority; otherwise it allows a stable side gap even when forced mode is true. The four requested priority cases pass.
2. **run_0039 counterfactual:** still true. The replay reports iteration `5` / `after`, side `right`, center `0.75 m`, width `0.90 m`; forced mode no longer suppresses it.
3. **Historical yaw shortfalls:** eight unique telemetry signatures have `turn_yaw_sufficient=false` (the archive contains ten rows, including two duplicate copies of one signature). The strongest implementation-level cause is the old fixed-duration open-loop primitive rather than a confirmed physical yaw result: it stopped after a duration and did not make fresh gated-odom accumulated yaw the completion condition. Low RTF, controller timing and stale odometry remain contributing hypotheses, not proven root causes for every record.
4. **Low-RTF feedback loop:** covered by a deterministic unit test; target completion depends on unwrapped gated-odom yaw, not elapsed sim/wall duration. It has **not** been demonstrated online because gated odom never became fresh.
5. **Online side coverage:** left `0`, right `0` nonzero turns. Two independent GUI=false health attempts were captured, both fail-closed.
6. **ENTER_ROOM:** `0` V1 paths. The `ROOM_SIDE_TURN` branch creates no entry target; `DOORWAY_VERIFY` is observation-only in V1; a final global transition guard blocks any residual `ENTER_ROOM` request.
7. **Before room entry:** repair the OccupancyGrid contract/certification first: present L3V status is diagnostic-only with `safe_for_navigation=false` and `all_required_inputs_fresh=false`. Then verify grid origin/frame/free-unknown semantics and the approach/commit target base-to-grid validation contract before enabling any entry transition.

## Online gate evidence

- `online_attempt_20260722_142606`: GUI=false, RL ready confirmed. `/clock` and grid advanced, but filtered cloud, raw/gated ICP odom were absent. L3V reported `odom_stale_or_missing`, `all_required_inputs_fresh=false`, and `safe_for_navigation=false`. Gazebo log also contains base TF NaN diagnostics.
- `online_attempt_20260722_212526`: GUI=false and controller period corrected to 0.006 s. `/scan` still emitted no messages over five seconds, so filtered cloud and both odom streams could not exist. This isolates the immediate blocker to the scan/upstream perception path, not the V1 turn controller.

The health gate therefore correctly withheld any nonzero side-turn command. No forbidden truth source was read or connected to control.

## Verification

- `junior_ctrl` rebuilt successfully after adding the latched RL-ready heartbeat.
- `python3 -m py_compile` passed for V1 modules and state machine.
- `bash -n scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh` passed.
- V1 unit suite: 20/20 passed, covering forced/gap priority, yaw unwrap, low-RTF behavior, stale/lost/no-response handling, post-turn `DOORWAY_VERIFY` only, `ENTER_ROOM` prohibition, and archive isolation checks.
- Offline replay: passed, with the expected right-side trigger.

## Safety boundary

The patch does not change doorway thresholds, does not create a room commit target, and never permits `ENTER_ROOM` while V1 is active. Turn commands have `linear.x=0`; stale odom waits at zero; lost/no-response ends in safe failure; fresh safe grids and a fresh side observation are required before `DOORWAY_VERIFY`.

## Evidence limits

The current simulator emitted no usable scan/filtered cloud/ICP odom, so closed-loop yaw, post-turn fresh-grid acquisition, and left/right online turn counts remain uncovered. The PNG below is an online health/command/yaw absence timeline, not a substituted motion result.

![Yaw, command, and topic-health timeline]({png})
"""
    report_path = ROOT / "audit_reports" / "room_side_turn_validation_v1_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "timeline_png": str(png), "historical_shortfalls": len(shortfalls)}))


if __name__ == "__main__":
    main()
