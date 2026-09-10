#!/usr/bin/env python3
"""Regression guard for zero-iteration state-machine summary construction."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py"


class NavigationStartupSummaryContractTests(unittest.TestCase):
    def test_visual_audit_counters_are_defined_before_the_main_iteration_loop(self):
        module = ast.parse(SOURCE.read_text(encoding="utf-8"))
        main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        loop_index = next(index for index, node in enumerate(main.body) if isinstance(node, ast.For))
        assigned = set()
        for node in main.body[:loop_index]:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
        self.assertTrue({
            "side_gap_visual_audit_event_count",
            "side_gap_visual_audit_broad_event_count",
        }.issubset(assigned))

    def test_n5_semantic_readiness_is_checked_before_target_regeneration(self):
        source = SOURCE.read_text(encoding="utf-8")
        start = source.index("def build_initial_anchor")
        end = source.index("\ndef write_anchor_target", start)
        body = source[start:end]
        self.assertIn("N5_SUMMARY_PATH", body)
        self.assertIn("n5_target_selection_not_ready", body)
        self.assertLess(
            body.index("n5_target_selection_not_ready"),
            body.index("regenerate_corrected_target_from_frame_contract.py"),
        )

    def test_target_preparation_failure_is_not_normalized_to_generic_incomplete(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('not final_decision.startswith("STATE_MACHINE_TARGET_PREP_FAILED")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
