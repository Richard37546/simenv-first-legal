#!/usr/bin/env python3
"""No-node regression test for L3V's repository-relative shared-contract import."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
L3V_PATH = ROOT / "scripts" / "l3v_local_traversability_diagnostic_node" / "l3v_local_traversability_node.py"
CONTRACT_PATH = ROOT / "scripts" / "local_subgoal_runner_mvp" / "local_grid_contract.py"


class L3VImportPathTests(unittest.TestCase):
    def test_import_is_repository_relative_without_pythonpath(self):
        self.assertTrue(CONTRACT_PATH.is_file())
        prior_module = sys.modules.pop("local_grid_contract", None)
        prior_paths = list(sys.path)
        try:
            sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != CONTRACT_PATH.parent]
            spec = importlib.util.spec_from_file_location("p2kg15_l3v_import_path_072", L3V_PATH)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            imported = sys.modules["local_grid_contract"]
            self.assertEqual(module.CONTRACT_DIR.resolve(), CONTRACT_PATH.parent.resolve())
            self.assertEqual(Path(imported.__file__).resolve(), CONTRACT_PATH.resolve())
        finally:
            sys.path[:] = prior_paths
            sys.modules.pop("local_grid_contract", None)
            if prior_module is not None:
                sys.modules["local_grid_contract"] = prior_module


if __name__ == "__main__":
    unittest.main()
