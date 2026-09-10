#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/online_local_safety_source_audit.py"

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

root = Path("debug/online_local_safety_source")
required = [
    root / "local_traversability_online_audit.json",
    root / "bev_front_sector_fallback_report.json",
    root / "online_local_safety_source_summary.json",
    Path("audit_reports/online_local_safety_source_audit_report.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing expected outputs: {missing}")
for path in required:
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
summary = json.loads((root / "online_local_safety_source_summary.json").read_text(encoding="utf-8"))
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
print("ONLINE_LOCAL_SAFETY_SOURCE_OUTPUTS_VALID")
print("final_decision=" + summary["final_decision"])
PY
