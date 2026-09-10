#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/n5_target_selection_subgoal_audit.py"

cd "${REPO_ROOT}"

python3 - <<PY
from pathlib import Path
path = Path("${PY_SCRIPT}")
compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY
python3 "${PY_SCRIPT}"
python3 - <<'PY'
import json
from pathlib import Path

root = Path("debug/short_horizon_target_selection")
required = [
    root / "n5a_candidate_selection_report.json",
    root / "n5b_subgoal_audit_report.json",
    root / "n5_target_selection_subgoal_summary.json",
    Path("audit_reports/n5_target_selection_subgoal_audit_report.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing expected outputs: {missing}")
for path in required:
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
summary = json.loads((root / "n5_target_selection_subgoal_summary.json").read_text(encoding="utf-8"))
boundary = summary["execution_boundary"]
assert boundary["execution_allowed"] is False
assert boundary["send_to_navigation"] is False
assert boundary["safe_for_navigation"] is False
assert boundary["planner_ready"] is False
assert boundary["diagnostic_only"] is True
assert boundary["would_publish_cmd_vel"] is False
assert boundary["published_cmd_vel"] is False
assert boundary["called_move_base"] is False
assert boundary["sent_navigation_goal"] is False
print("N5_TARGET_SELECTION_OUTPUTS_VALID")
print("final_decision=" + summary["final_decision"])
PY
