#!/usr/bin/env python3
"""Bounded, atomic sparse-ray persistence for the audit-only shadow."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


class SparseFrameWriter:
    """Write complete source-frame evidence without inventing grid cells.

    A JSONL index row is committed only after the compressed NPZ has been
    atomically renamed and hashed.  Queue overflow is observable and never
    overwrites an earlier frame.
    """

    def __init__(self, root: Path, max_queue: int = 64) -> None:
        self.root = root
        self.frames = root / "side_evidence_frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue[Tuple[int, np.ndarray, Dict[str, Any], float]] = queue.Queue(maxsize=max_queue)
        self.index = (root / "side_evidence_index.jsonl").open("a", encoding="utf-8")
        self.max_queue = max_queue
        self.queue_peak = 0
        self.dropped = 0
        self.persisted = 0
        self.errors: list[str] = []
        self.writer_ms: list[float] = []
        self.end_to_end_ms: list[float] = []
        self.stop = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="p2kg7r-sparse-writer", daemon=True)
        self.thread.start()

    def enqueue(self, seq: int, rays: np.ndarray, meta: Dict[str, Any], enqueued: float) -> bool:
        try:
            # The compute worker reuses its array on later callbacks.
            self.q.put_nowait((seq, rays.copy(), dict(meta), enqueued))
            with self.lock:
                self.queue_peak = max(self.queue_peak, self.q.qsize())
            return True
        except queue.Full:
            with self.lock:
                self.dropped += 1
            return False

    def _run(self) -> None:
        while not self.stop or not self.q.empty():
            try:
                seq, rays, meta, enqueued = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            name = f"frame_{seq:06d}.npz"
            final = self.frames / name
            tmp = self.frames / f"{name}.tmp.npz"
            try:
                np.savez_compressed(tmp, rays=rays, metadata=np.asarray(json.dumps(meta, sort_keys=True)))
                os.replace(tmp, final)
                digest = hashlib.sha256(final.read_bytes()).hexdigest()
                ended = time.monotonic()
                writer_ms = (ended - started) * 1000.0
                end_to_end_ms = (ended - enqueued) * 1000.0
                row = {
                    "frame_sequence": seq,
                    "file": str(final.relative_to(self.root)),
                    "sha256": digest,
                    "ray_count": int(len(rays)),
                    "source_stamp": meta["source_stamp"],
                    "writer_ms": writer_ms,
                    "end_to_end_ms": end_to_end_ms,
                }
                self.index.write(json.dumps(row, sort_keys=True) + "\n")
                self.index.flush()
                with self.lock:
                    self.persisted += 1
                    self.writer_ms.append(writer_ms)
                    self.end_to_end_ms.append(end_to_end_ms)
            except Exception as exc:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                with self.lock:
                    self.errors.append(str(exc))
            finally:
                self.q.task_done()

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(values, percentile)) if values else None

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "persisted_frames": self.persisted,
                "queue_overflow": self.dropped,
                "writer_errors": list(self.errors),
                "queue_remaining": self.q.qsize(),
                "queue_peak": self.queue_peak,
                "writer_ms_median": self._percentile(self.writer_ms, 50),
                "writer_ms_p90": self._percentile(self.writer_ms, 90),
                "end_to_end_ms_median": self._percentile(self.end_to_end_ms, 50),
                "end_to_end_ms_p90": self._percentile(self.end_to_end_ms, 90),
            }

    def close(self, timeout_sec: float = 30.0) -> Dict[str, Any]:
        self.stop = True
        self.thread.join(timeout=timeout_sec)
        summary = self.snapshot()
        summary["writer_thread_alive"] = self.thread.is_alive()
        self.index.close()
        return summary
