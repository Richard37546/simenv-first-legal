#!/usr/bin/env python3
"""Capture the exact navigation-control source set into a run archive."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE_PATHS = (
    "auto.sh",
    "scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh",
    "scripts/slam_bev_runtime/run_slam_bev_runtime.sh",
    "scripts/l2_livox_icp_rtabmap_readiness/l2_livox_odom_gate.py",
    "scripts/l2_livox_icp_rtabmap_readiness/continuous_yaw_consistency.py",
    "scripts/l2_livox_icp_rtabmap_readiness/continuous_odom_imu_yaw_shadow.py",
    "scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh",
    "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
    "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
    "scripts/local_subgoal_runner_mvp/room_local_online_validation_observer.py",
    "scripts/local_subgoal_runner_mvp/tests/test_room_local_online_enablement.py",
    "scripts/local_subgoal_runner_mvp/tests/test_room_local_online_enablement_closure.py",
    "scripts/local_subgoal_runner_mvp/finalize_room_local_online_validation_archive.py",
    "scripts/local_subgoal_runner_mvp/room_search_v1.py",
    "scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_capture.py",
    "scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_sidecar.py",
    "scripts/room_search_stage_b_shadow/start_room_search_stage_b_shadow.sh",
    "scripts/room_search_stage_b_shadow/verify_room_search_stage_b_navigation_contract.sh",
    "scripts/p2kg9_portal_shadow/prepare_p2kg9u_audit_bundle.sh",
    "scripts/p2kg9_portal_shadow/start_p2kg9u_navigation_runner.sh",
    "scripts/controlled_online_yaw_shadow/prepare_manual_run.sh",
)
POLICY_PATH = "src/unitree_guide/logs/policy_act_inference_stair.pt"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-command", required=True)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=False)
    files = []
    for relative in SOURCE_PATHS:
        source = root / relative
        if not source.is_file():
            files.append({"path": relative, "status": "MISSING", "sha256": None})
            continue
        destination = sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(destination))
        files.append({"path": relative, "status": "COPIED", "sha256": digest(source)})
    policy = root / POLICY_PATH
    policy_record = {
        "path": POLICY_PATH,
        "status": "PRESENT" if policy.is_file() else "MISSING",
        "sha256": digest(policy) if policy.is_file() else None,
    }
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS],
        cwd=str(root), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    diff_path = output / "control_source_worktree.diff"
    diff_path.write_bytes(diff)
    manifest = {
        "schema_version": "navigation_control_source_provenance_v1",
        "run_command": args.run_command,
        "source_snapshot_dir": "sources",
        "source_files": files,
        "policy": policy_record,
        "control_source_worktree_diff": "control_source_worktree.diff",
        "control_source_worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
