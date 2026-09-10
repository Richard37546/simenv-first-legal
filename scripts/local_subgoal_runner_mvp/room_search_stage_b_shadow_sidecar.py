#!/usr/bin/env python3
"""Authority-free ROOM_SEARCH Stage-B shadow sidecar.

The sidecar consumes only atomically renamed ``*.ready`` bundles.  Evaluation
runs in a separately killable subprocess with ``execute=False`` inherited from
the validated bundle.  Results are advisory files outside the immutable bundle.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import resource
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "room_search_stage_b_shadow_result_v1"
AUTHORITY_FALSE = {
    "production_authority": False,
    "selection_authority": False,
    "command_authority": False,
    "completion_authority": False,
    "recoverability_authority": False,
    "fallback_authority": False,
}
REQUIRED_PRODUCTION_EVENTS = {
    "PRODUCTION_ADMISSION", "PRODUCTION_SINGLETON_NBV", "PRODUCTION_COMPLETION",
    "PRODUCTION_EXECUTION_HANDOFF", "PRODUCTION_EXECUTION_OUTCOME",
}


class ReadOnlyGridStatusObserver:
    """Lazy ROS subscriber used only for start/finish churn identity."""

    def __init__(
        self,
        grid_topic: str = "/team/local_traversability_grid",
        status_topic: str = "/team/traversability_status",
    ) -> None:
        import rospy
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String
        from local_grid_contract import ExactGridStatusPairCache

        self._json = json
        self._cache = ExactGridStatusPairCache(maxlen=8)
        self._subscribers = [
            rospy.Subscriber(grid_topic, OccupancyGrid, self._cache.add_grid, queue_size=8),
            rospy.Subscriber(status_topic, String, self._status_callback, queue_size=8),
        ]

    def _status_callback(self, message: Any) -> None:
        try:
            payload = self._json.loads(message.data)
        except Exception:
            payload = {}
        self._cache.add_status(payload if isinstance(payload, dict) else {})

    def identity(self) -> Dict[str, Any]:
        pair = self._cache.matching_pair()
        if pair is None:
            return {"matched_pair_found": False}
        _grid, status = pair
        return {
            "matched_pair_found": True,
            "producer_instance_id": status.get("producer_instance_id"),
            "content_generation_id": status.get("content_generation_id"),
            "grid_content_stamp": status.get("grid_content_stamp"),
            "grid_content_hash": status.get("grid_content_hash"),
            "local_traversability_status": status.get("local_traversability_status"),
        }

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _read_events(bundle: Path) -> List[Dict[str, Any]]:
    path = bundle / "production_events.jsonl"
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _rss_bytes(pid: int) -> Optional[int]:
    """Linux-only best-effort RSS observer for the evaluator process."""
    try:
        for row in Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8").splitlines():
            if row.startswith("VmRSS:"):
                return int(row.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _event_lifecycle(rows: Sequence[Mapping[str, Any]], decision_id: Any) -> Dict[str, Any]:
    relevant = [dict(row) for row in rows if row.get("decision_id") == decision_id]
    names = {row.get("event") for row in relevant}
    advanced = [row for row in relevant if row.get("event") == "MISSION_DECISION_ADVANCED"]
    finish = [row for row in relevant if row.get("event") == "PRODUCTION_FINISH"]
    state: Dict[str, str] = {}
    for required in sorted(REQUIRED_PRODUCTION_EVENTS):
        if required in names:
            state[required] = "SEEN"
        elif required == "PRODUCTION_COMPLETION" and "PRODUCTION_EXECUTION_OUTCOME" in names:
            state[required] = "NOT_APPLICABLE"
        elif required == "PRODUCTION_EXECUTION_OUTCOME" and "PRODUCTION_COMPLETION" in names:
            state[required] = "NOT_APPLICABLE"
        elif advanced:
            state[required] = "DECISION_ADVANCED_BEFORE_EVENT"
        elif finish:
            state[required] = "RUN_FINISHED_BEFORE_EVENT"
        else:
            state[required] = "NOT_YET_OBSERVED"
    observed = [row.get("observed_decision_id") for row in advanced]
    return {
        "required_event_lifecycle": state,
        "production_baseline_complete": all(value in {"SEEN", "NOT_APPLICABLE"} for value in state.values()),
        "mission_decision_advanced": bool(advanced),
        "decision_id_at_start": decision_id,
        "decision_id_at_finish": observed[-1] if observed else decision_id,
    }


def _staleness(
    start_identity: Mapping[str, Any], finish_identity: Mapping[str, Any], lifecycle: Mapping[str, Any],
) -> Dict[str, Any]:
    start_generation = start_identity.get("content_generation_id")
    finish_generation = finish_identity.get("content_generation_id")
    grid_changed = bool(start_identity and finish_identity and dict(start_identity) != dict(finish_identity))
    generation_delta = (
        int(finish_generation) - int(start_generation)
        if isinstance(start_generation, int) and not isinstance(start_generation, bool)
        and isinstance(finish_generation, int) and not isinstance(finish_generation, bool) else None
    )
    mission_advanced = bool(lifecycle.get("mission_decision_advanced"))
    if grid_changed and mission_advanced:
        state = "GRID_STALE_AND_MISSION_ADVANCED"
    elif grid_changed:
        state = "GRID_STALE_ONLY"
    elif mission_advanced:
        state = "MISSION_DECISION_ADVANCED"
    else:
        state = "CURRENT_SAME_DECISION"
    return {
        "grid_identity_changed": grid_changed,
        "grid_generation_at_start": start_generation,
        "grid_generation_at_finish": finish_generation,
        "grid_generation_delta": generation_delta,
        "mission_decision_advanced": mission_advanced,
        "staleness_state": state,
    }


def _default_evaluator_command(bundle: Path, output: Path) -> List[str]:
    adapter = Path(__file__).resolve().with_name("room_search_stage_a_contract.py")
    return [sys.executable, str(adapter), str(bundle), "--stage-b", "--output", str(output)]


def _terminate_evaluator(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=2.0)


def _close_process_pipes(process: subprocess.Popen) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def run_sidecar_once(
    bundle_path: Path,
    *,
    result_path: Optional[Path] = None,
    evaluator_timeout_sec: float = 30.0,
    evaluator_command_factory: Callable[[Path, Path], Sequence[str]] = _default_evaluator_command,
    identity_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    production_event_wait_sec: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate one ready bundle; all errors remain isolated to this process."""
    bundle = Path(bundle_path).resolve()
    destination = Path(result_path).resolve() if result_path else bundle.with_suffix(".shadow-result.json")
    base: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle": str(bundle),
        "result_path": str(destination),
        "started_wall_sec": time.time(),
        **AUTHORITY_FALSE,
    }
    if not bundle.is_dir() or not bundle.name.endswith(".ready"):
        result = {**base, "status": "SHADOW_FAILED_ISOLATED", "reason": "READY_BUNDLE_REQUIRED"}
        _atomic_write_json(destination, result)
        return result
    try:
        manifest = _read_json(bundle / "manifest.json")
        decision = _read_json(bundle / "decision.json")
        grid = _read_json(bundle / "grid.json")["planning_identity"]
        start_identity = dict(identity_provider()) if identity_provider is not None else {}
        events_before = _read_events(bundle)
    except Exception as exc:
        result = {
            **base, "status": "SHADOW_FAILED_ISOLATED", "reason": "BUNDLE_READ_FAILED",
            "exception": f"{type(exc).__name__}:{exc}",
        }
        _atomic_write_json(destination, result)
        return result

    fd, evaluator_output_name = tempfile.mkstemp(prefix="stage-b-evaluator-", suffix=".json")
    os.close(fd)
    evaluator_output = Path(evaluator_output_name)
    process: Optional[subprocess.Popen] = None
    try:
        command = list(evaluator_command_factory(bundle, evaluator_output))
        if "--execute" in command:
            raise ValueError("evaluator_execute_flag_forbidden")
        evaluator_started_ns = time.perf_counter_ns()
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        peak_rss_bytes: List[int] = []
        rss_stop = threading.Event()
        def _sample_rss() -> None:
            while not rss_stop.is_set():
                sample = _rss_bytes(process.pid)
                if sample is not None:
                    peak_rss_bytes.append(sample)
                rss_stop.wait(0.01)
        rss_thread = threading.Thread(target=_sample_rss, name="stage-b-rss-observer", daemon=True)
        rss_thread.start()
        try:
            stdout, stderr = process.communicate(timeout=max(0.01, float(evaluator_timeout_sec)))
        except subprocess.TimeoutExpired:
            rss_stop.set()
            rss_thread.join(0.1)
            _terminate_evaluator(process)
            _close_process_pipes(process)
            result = {
                **base, "status": "SHADOW_FAILED_ISOLATED", "reason": "EVALUATOR_TIMEOUT",
                "evaluator_timeout_sec": float(evaluator_timeout_sec),
                "evaluator_process_only_terminated": True,
            }
            _atomic_write_json(destination, result)
            return result
        finally:
            rss_stop.set()
            rss_thread.join(0.1)
        if process.returncode != 0:
            result = {
                **base, "status": "SHADOW_FAILED_ISOLATED", "reason": "EVALUATOR_FAILED",
                "evaluator_returncode": process.returncode,
                "evaluator_stdout_tail": stdout[-2000:], "evaluator_stderr_tail": stderr[-2000:],
            }
            _atomic_write_json(destination, result)
            return result
        evaluation = _read_json(evaluator_output)
        evaluator_elapsed_ns = time.perf_counter_ns() - evaluator_started_ns
        events_after = _read_events(bundle)
        finish_identity = dict(identity_provider()) if identity_provider is not None else {}
        lifecycle = _event_lifecycle(events_after, decision.get("decision_id"))
        stale = _staleness(start_identity, finish_identity, lifecycle)
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        expected_epoch = manifest.get("epoch_id")
        evaluated_epoch = (evaluation.get("room_search_decision_epoch") or {}).get("epoch_id")
        grid_match = (
            (evaluation.get("same_epoch_constraints") or {}).get("grid_hash") == grid.get("grid_content_hash")
            and (evaluation.get("same_epoch_constraints") or {}).get("grid_generation") == grid.get("content_generation_id")
        )
        if expected_epoch != evaluated_epoch or not grid_match:
            final_status, reason = "SHADOW_FAILED_ISOLATED", "EVALUATED_IDENTITY_MISMATCH"
        elif stale["staleness_state"] != "CURRENT_SAME_DECISION":
            final_status, reason = "SHADOW_STALE", stale["staleness_state"]
        elif evaluation.get("status") == "DECISION_EPOCH_UNQUALIFIED":
            final_status, reason = "SHADOW_EPOCH_UNQUALIFIED", str(
                evaluation.get("reason") or "DECISION_EPOCH_UNQUALIFIED"
            )
        elif evaluation.get("status") != "STAGE_B_SHADOW_COMPLETE":
            final_status, reason = "SHADOW_INCOMPLETE", "OFFLINE_COMPARISON_INCOMPLETE"
        else:
            final_status, reason = "SHADOW_COMPLETE", None
        result = {
            **base,
            "status": final_status,
            "reason": reason,
            "finished_wall_sec": time.time(),
            "ready_bundle_consumed": True,
            "execute": False,
            "epoch_id": expected_epoch,
            "grid_identity": grid,
            "identity_at_start": start_identity,
            "identity_at_finish": finish_identity,
            **stale,
            "production_events_before_count": len(events_before),
            "production_events_after_count": len(events_after),
            "production_events": events_after,
            **lifecycle,
            "timing": {
                "bundle_observed_monotonic_ns": evaluator_started_ns,
                "capture_to_evaluator_start_ns": (
                    evaluator_started_ns - int(decision["capture_monotonic_ns"])
                    if isinstance(decision.get("capture_monotonic_ns"), int) else None
                ),
                "evaluator_elapsed_ns": evaluator_elapsed_ns,
                "evaluator_cpu_user_sec": max(0.0, cpu_after.ru_utime - cpu_before.ru_utime),
                "evaluator_cpu_system_sec": max(0.0, cpu_after.ru_stime - cpu_before.ru_stime),
                "evaluator_peak_rss_bytes": max(peak_rss_bytes) if peak_rss_bytes else None,
                "production_event_wait_sec_ignored": float(production_event_wait_sec),
            },
            "evaluation": evaluation,
        }
        _atomic_write_json(destination, result)
        return result
    except Exception as exc:
        if process is not None and process.poll() is None:
            _terminate_evaluator(process)
        result = {
            **base, "status": "SHADOW_FAILED_ISOLATED", "reason": "SIDECAR_EXCEPTION",
            "exception": f"{type(exc).__name__}:{exc}",
        }
        _atomic_write_json(destination, result)
        return result
    finally:
        try:
            evaluator_output.unlink()
        except OSError:
            pass


def refresh_result_lifecycle(
    bundle: Path, result_path: Path, identity_provider: Optional[Callable[[], Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Refresh only asynchronous evidence; never invokes the evaluator again."""
    try:
        result = _read_json(result_path)
        decision = _read_json(bundle / "decision.json")
        rows = _read_events(bundle)
        lifecycle = _event_lifecycle(rows, decision.get("decision_id"))
        finish_identity = dict(identity_provider()) if identity_provider is not None else {}
        stale = _staleness(result.get("identity_at_start") or {}, finish_identity, lifecycle)
        result.update({
            "identity_at_finish": finish_identity,
            "production_events_after_count": len(rows),
            "production_events": rows,
            **lifecycle,
            **stale,
        })
        if result.get("status") in {"SHADOW_COMPLETE", "SHADOW_STALE"}:
            result["status"] = "SHADOW_STALE" if stale["staleness_state"] != "CURRENT_SAME_DECISION" else "SHADOW_COMPLETE"
            result["reason"] = stale["staleness_state"] if result["status"] == "SHADOW_STALE" else None
        _atomic_write_json(result_path, result)
        return result
    except Exception:
        return None


def _write_run_counters(result_paths: Iterable[Path], result_dir: Path) -> None:
    """A bounded, advisory summary of already-written independent epoch results."""
    rows = []
    for path in result_paths:
        try:
            rows.append(_read_json(path))
        except Exception:
            continue
    counters = {
        "total_epochs_seen": len(rows),
        "evaluations_started": len(rows),
        "evaluations_completed": sum(row.get("status") in {"SHADOW_COMPLETE", "SHADOW_STALE"} for row in rows),
        "evaluations_failed": sum(row.get("status") == "SHADOW_FAILED_ISOLATED" for row in rows),
        "evaluations_epoch_unqualified": sum(row.get("status") == "SHADOW_EPOCH_UNQUALIFIED" for row in rows),
        "evaluations_stale_by_grid": sum(bool(row.get("grid_identity_changed")) for row in rows),
        "evaluations_stale_by_mission": sum(bool(row.get("mission_decision_advanced")) for row in rows),
        "evaluator_timeout_count": sum(row.get("reason") == "EVALUATOR_TIMEOUT" for row in rows),
        "evaluator_concurrency": 1,
    }
    _atomic_write_json(Path(result_dir) / "stage_b2_run_counters.json", {
        "schema_version": SCHEMA_VERSION, "authority": "ADVISORY_ONLY", "counters": counters,
    })


def watch_ready_bundles(
    watch_dir: Path,
    *,
    result_dir: Path,
    timeout_sec: float,
    poll_sec: float,
    one_shot: bool,
    identity_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> int:
    processed: Dict[str, Path] = {}
    while True:
        ready = sorted(path for path in Path(watch_dir).rglob("*.ready") if path.is_dir())
        for bundle in ready:
            key = str(bundle.resolve())
            if key in processed:
                continue
            result_path = Path(result_dir) / f"{bundle.parent.name}_{bundle.stem}.json"
            processed[key] = result_path
            result = run_sidecar_once(
                bundle,
                result_path=result_path,
                evaluator_timeout_sec=timeout_sec,
                identity_provider=identity_provider,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if one_shot:
                return 0 if result.get("status") in {"SHADOW_COMPLETE", "SHADOW_STALE"} else 2
        for key, stored_result in tuple(processed.items()):
            refresh_result_lifecycle(Path(key), stored_result, identity_provider)
        _write_run_counters(processed.values(), Path(result_dir))
        time.sleep(max(0.05, float(poll_sec)))


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--poll-sec", type=float, default=0.2)
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--mode", choices=["ONE_SHOT", "MULTI_DECISION"], default="ONE_SHOT")
    parser.add_argument("--ros-grid-observer", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    observer = None
    try:
        if args.ros_grid_observer:
            import rospy
            rospy.init_node("room_search_stage_b_shadow_sidecar", anonymous=False, disable_signals=True)
            observer = ReadOnlyGridStatusObserver()
        one_shot = bool(args.one_shot or args.mode == "ONE_SHOT")
        mode = "ONE_SHOT" if one_shot else "MULTI_DECISION"
        if args.ready_file is not None:
            _atomic_write_json(args.ready_file, {
                "schema_version": SCHEMA_VERSION,
                "status": "SIDECAR_READY",
                "pid": os.getpid(),
                "watch_dir": str(args.watch_dir.resolve()),
                "result_dir": str(args.result_dir.resolve()),
                "mode": mode,
                "one_shot": one_shot,
                "evaluator_concurrency": 1,
                "read_only_grid_status_observer": observer is not None,
                "latest_grid_identity": observer.identity() if observer is not None else {},
                **AUTHORITY_FALSE,
            })
        return watch_ready_bundles(
            args.watch_dir, result_dir=args.result_dir, timeout_sec=args.timeout_sec,
            poll_sec=args.poll_sec, one_shot=one_shot,
            identity_provider=(observer.identity if observer is not None else None),
        )
    finally:
        if observer is not None:
            observer.close()


if __name__ == "__main__":
    raise SystemExit(_main())
