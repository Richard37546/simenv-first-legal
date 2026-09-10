#!/usr/bin/env python3
"""Read one frozen Stage-B artifact and emit an authority-free V1 replay."""

import argparse
import json
from pathlib import Path

from formal_mission_comparison_v1 import replay_stage_b_shadow_result


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-result", required=True, type=Path)
    parser.add_argument("--frozen-candidates", type=Path,
                        help="defaults to candidates.json next to the frozen bundle named in the Stage-B result")
    parser.add_argument("--summary", action="store_true", help="emit only replay comparison fields")
    args = parser.parse_args()
    result = _load_json(args.stage_b_result)
    candidates_path = args.frozen_candidates
    if candidates_path is None:
        bundle = result.get("bundle") or {}
        bundle_dir = Path(bundle if isinstance(bundle, str) else bundle.get("bundle_dir") or "")
        candidates_path = bundle_dir / "candidates.json"
    if not candidates_path.is_file():
        parser.error("frozen candidates file is required and was not found: {}".format(candidates_path))
    frozen = _load_json(candidates_path)
    candidates = frozen.get("ranked_candidates") if isinstance(frozen, dict) else frozen
    if not isinstance(candidates, list):
        parser.error("frozen candidates do not contain a ranked_candidates list: {}".format(candidates_path))
    replay = replay_stage_b_shadow_result(result, candidates)
    if args.summary:
        comparison = replay["comparison"]
        selection = replay["shadow_selection"]
        replay = {
            "epoch_id": comparison["epoch_id"],
            "candidate_count": len(comparison["all_candidates"]),
            "legal_executable_count": len(comparison["legal_executable_candidate_ids"]),
            "production_winner_candidate_id": replay["production_winner_candidate_id"],
            "shadow_status": selection["status"],
            "shadow_winner_candidate_id": selection["winner_candidate_id"],
            "shadow_reason": selection["reason"],
            "win_reason": selection["win_reason"],
            "same_as_production": replay["same_as_production"],
            "context_mismatch": replay["replay_context_mismatch"],
            "context_mismatch_reason": replay["replay_context_mismatch_reason"],
        }
    print(json.dumps(replay, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
