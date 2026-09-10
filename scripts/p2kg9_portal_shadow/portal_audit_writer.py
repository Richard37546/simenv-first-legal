#!/usr/bin/env python3
"""Bounded, atomic candidate persistence for the P2K-G9 audit node."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from pathlib import Path


class PortalAuditWriter:
    def __init__(self, out: Path, max_queue: int = 64, event_callback=None) -> None:
        self.out = out
        self.frames = out / "portal_frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        # Each run owns its output directory. Truncate stale offline-replay
        # indexes rather than silently appending a second temporal sequence.
        self.index = (out / "portal_outputs.jsonl").open("w", encoding="utf-8")
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stop_requested = False
        self.queue_overflow = 0
        self.errors: list[str] = []
        self.persisted = 0
        self.event_callback = event_callback
        self.worker = threading.Thread(target=self._loop, name="p2kg9-portal-writer", daemon=True)
        self.worker.start()

    def enqueue(self, sequence: int, payload: dict) -> bool:
        try:
            self.q.put_nowait((sequence, payload))
            return True
        except queue.Full:
            self.queue_overflow += 1
            if self.event_callback is not None:
                self.event_callback("writer_queue_overflow", source_stamp=payload.get("source_stamp"), queue_size=self.q.qsize(), failure_reason="WRITER_QUEUE_FULL")
            return False

    def _loop(self) -> None:
        while not self.stop_requested or not self.q.empty():
            try:
                sequence, payload = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                target = self.frames / f"frame_{sequence:06d}.json"
                temporary = target.with_suffix(".json.tmp")
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                self.index.write(json.dumps({"frame_sequence": sequence, "file": str(target.relative_to(self.out)), "sha256": digest, "source_stamp": payload["source_stamp"]}, sort_keys=True) + "\n")
                self.index.flush()
                self.persisted += 1
                if self.event_callback is not None:
                    self.event_callback("audit_persisted", source_stamp=payload["source_stamp"], frame_sequence=sequence, queue_size=self.q.qsize())
            except Exception as exc:
                self.errors.append(repr(exc))
                if self.event_callback is not None:
                    self.event_callback("writer_failure", source_stamp=payload.get("source_stamp"), frame_sequence=sequence, failure_reason=repr(exc))
            finally:
                self.q.task_done()

    def close(self) -> dict:
        self.stop_requested = True
        self.worker.join(timeout=30.0)
        self.index.close()
        if self.event_callback is not None:
            self.event_callback("writer_worker_stopped", source_stamp=None, queue_size=self.q.qsize(), worker_alive=self.worker.is_alive())
        return {"persisted_frames": self.persisted, "queue_overflow": self.queue_overflow, "writer_errors": self.errors, "queue_remaining": self.q.qsize(), "worker_alive": self.worker.is_alive()}
