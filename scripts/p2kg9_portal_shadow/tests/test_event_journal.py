#!/usr/bin/env python3
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from event_journal import GlobalEventJournal  # noqa: E402
from portal_audit_writer import PortalAuditWriter  # noqa: E402


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class EventJournalTests(unittest.TestCase):
    def test_concurrent_events_are_unique_and_file_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.jsonl"
            journal = GlobalEventJournal(path)
            threads = [threading.Thread(target=lambda: [journal.emit("cloud_received") for _ in range(40)]) for _ in range(8)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            journal.close()
            ids = [row["event_id"] for row in rows(path)]
            self.assertEqual(ids, list(range(1, 321)))

    def test_failure_and_writer_overflow_paths_use_the_same_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = GlobalEventJournal(root / "events.jsonl")
            writer = PortalAuditWriter(root / "writer", max_queue=1, event_callback=journal.emit)
            # Hold the writer queue full before it can drain by replacing its queue.
            writer.q.put_nowait((0, {"source_stamp": 0.0, "portal_candidates": []}))
            self.assertFalse(writer.enqueue(1, {"source_stamp": 1.0, "portal_candidates": []}))
            journal.emit("tf_failure", failure_reason="synthetic")
            journal.emit("processing_failure", failure_reason="synthetic")
            writer.close()
            journal.emit("shutdown_complete")
            journal.close()
            recorded = rows(root / "events.jsonl")
            ids = [row["event_id"] for row in recorded]
            self.assertEqual(ids, list(range(1, len(ids) + 1)))
            self.assertIn("writer_queue_overflow", [row["event_type"] for row in recorded])
            self.assertIn("writer_worker_stopped", [row["event_type"] for row in recorded])

    def test_independent_instances_restart_event_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "second"):
                journal = GlobalEventJournal(root / f"{name}.jsonl")
                journal.emit("startup")
                journal.emit("shutdown_complete")
                journal.close()
                self.assertEqual([row["event_id"] for row in rows(root / f"{name}.jsonl")], [1, 2])


if __name__ == "__main__":
    unittest.main()
