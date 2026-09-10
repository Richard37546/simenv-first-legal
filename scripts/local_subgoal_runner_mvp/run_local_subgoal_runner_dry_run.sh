#!/usr/bin/env bash
set -eo pipefail

cd /home/richard/simenv_official_clean
source /opt/ros/noetic/setup.bash
source devel/setup.bash 2>/dev/null || true
set -u

TARGET_REGEN_CMD="python3 scripts/local_subgoal_runner_mvp/regenerate_corrected_target_from_frame_contract.py"
if ! ${TARGET_REGEN_CMD}; then
  echo '{"final_decision":"DRY_RUN_BLOCKED_BY_TARGET_REGENERATION_FAILURE","target_regeneration_pass":false}'
  exit 1
fi

FRAME_CONTRACT_CMD="python3 scripts/local_subgoal_runner_mvp/frame_contract_validation.py"
if ! ${FRAME_CONTRACT_CMD}; then
  echo '{"final_decision":"DRY_RUN_BLOCKED_BY_TARGET_REGENERATION_FAILURE","target_regeneration_pass":true,"frame_contract_validation_pass":false}'
  exit 1
fi

export LSR_TARGET_REGENERATED_BY_WRAPPER=1
export LSR_TARGET_REGENERATION_PASS=1
export LSR_TARGET_REGENERATION_COMMAND="${TARGET_REGEN_CMD}"
export LSR_FRAME_CONTRACT_VALIDATION_RERUN_BY_WRAPPER=1
export LSR_FRAME_CONTRACT_VALIDATION_PASS=1

python3 scripts/local_subgoal_runner_mvp/local_subgoal_runner.py --dry-run "$@"
