#!/usr/bin/env python3
"""Thread-safe, process-lifetime event sequence for P2K-G9 audit paths."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable


class GlobalEventJournal:
    """Allocates and persists one strictly increasing ID under one lock."""

    def __init__(self, path: Path, publish: Callable[[str], None] | None = None) -> None:
        self.path = path
        self._file = path.open("a", encoding="utf-8")
        self._publish = publish
        self._lock = threading.Lock()
        self._next_id = 0
        self.event_types: dict[str, int] = {}

    def emit(self, event_type: str, **fields) -> int:
        """Assign and write atomically so on-disk IDs cannot reorder."""
        with self._lock:
            event_id = self._next_id + 1
            row = {
                "event_id": event_id,
                "event_type": event_type,
                "source_stamp": fields.pop("source_stamp", None),
                "receipt_monotonic_time": fields.pop("receipt_monotonic_time", time.monotonic()),
                "processing_start": fields.pop("processing_start", None),
                "processing_end": fields.pop("processing_end", time.monotonic()),
                "input_state_version": fields.pop("input_state_version", None),
                "portal_track_version": fields.pop("portal_track_version", None),
                "thread_name": threading.current_thread().name,
                **fields,
            }
            encoded = json.dumps(row, sort_keys=True)
            self._file.write(encoded + "\n")
            self._file.flush()
            # Do not commit the sequence number until its complete JSONL row
            # is durable in the process buffer. A write error cannot advance
            # the allocator and leave a normal-path gap for the next event.
            self._next_id = event_id
            self.event_types[event_type] = self.event_types.get(event_type, 0) + 1
            if self._publish is not None:
                self._publish(encoded)
            return event_id

    @property
    def last_event_id(self) -> int:
        with self._lock:
            return self._next_id

    def close(self) -> None:
        with self._lock:
            self._file.close()
