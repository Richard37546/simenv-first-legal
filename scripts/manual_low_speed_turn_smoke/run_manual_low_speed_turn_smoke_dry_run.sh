#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/manual_low_speed_turn_smoke.py"

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

summary_path = Path("debug/manual_low_speed_turn_smoke/manual_low_speed_turn_smoke_summary.json")
report_path = Path("audit_reports/manual_low_speed_turn_smoke_report.md")
missing = [str(path) for path in [summary_path, report_path] if not path.exists()]
if missing:
    raise SystemExit(f"missing expected outputs: {missing}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["mode"] == "dry_run_preaudit"
assert summary["nonzero_cmd_vel_published"] is False
assert summary["called_move_base"] is False
assert summary["sent_navigation_goal"] is False
boundary = summary["execution_boundary"]
assert boundary["execution_allowed"] is False
assert boundary["send_to_navigation"] is False
assert boundary["safe_for_navigation"] is False
assert boundary["planner_ready"] is False
assert boundary["diagnostic_only"] is True
assert boundary["would_publish_cmd_vel"] is False
assert boundary["published_cmd_vel"] is False
print("MANUAL_LOW_SPEED_TURN_SMOKE_DRY_RUN_OUTPUTS_VALID")
print("final_decision=" + summary["final_decision"])
PY
