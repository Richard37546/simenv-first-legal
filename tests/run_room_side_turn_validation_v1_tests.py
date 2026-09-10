#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
import time
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.started = {}

    def startTest(self, test):
        self.started[test.id()] = time.monotonic()
        super().startTest(test)

    def _record(self, test, status, detail=None):
        self.records.append({
            "test": test.id(),
            "status": status,
            "duration_sec": time.monotonic() - self.started.get(test.id(), time.monotonic()),
            "detail": detail,
        })

    def addSuccess(self, test):
        self._record(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test, err):
        self._record(test, "failed", self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):
        self._record(test, "error", self._exc_info_to_string(err, test))
        super().addError(test, err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_room_side_turn_validation_v1")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    payload = {
        "schema_version": 1,
        "offline_only": True,
        "ros_commands_published": False,
        "tests_run": result.testsRun,
        "passed": sum(1 for item in result.records if item["status"] == "passed"),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "records": result.records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
