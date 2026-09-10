#!/usr/bin/env python3
"""Read-only RGB scene semantic observer.

This is a sanitized adapter inspired by the uploaded vision package. It never
publishes motion commands and never reads Gazebo truth. If no API key is
provided, it still publishes a structured "no_api_key" status so downstream
diagnostics can run without blocking the robot chain.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "vision_scene_semantics"
LATEST_PATH = OUT / "latest_project_scene_analysis.json"
JSONL_PATH = OUT / "project_scene_analysis.jsonl"


DEFAULT_PROMPT = """You are a robot scene semantic observer.
Return strict JSON only. Classify the current robot RGB view for navigation.
Schema:
{
  "primary_scene": "wall|corridor|room|room_opening|main_gate|normal_door|elevator|elevator_door|stairs|obstacle_area|danger_source_view|unknown",
  "passability": "passable|blocked|need_open_door|narrow|unknown",
  "door": {"visible": true/false, "type": "normal|main_gate|elevator|unknown", "state": "open|closed|partial|unknown"},
  "room_opening": {"visible": true/false, "looks_enterable": true/false},
  "elevator": {"visible": true/false, "state": "open|closed|unknown"},
  "stairs": {"visible": true/false, "direction": "up|down|unknown"},
  "danger_source": {"visible": true/false, "class_name": "danger_red_sphere|unknown", "confidence": 0.0},
  "navigation_hint": "go_forward|turn_left|turn_right|enter_left|enter_right|stop|unknown",
  "confidence": 0.0,
  "description": "short evidence"
}
Do not invent metric coordinates."""


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def ros_image_to_rgb_bytes(msg: Image) -> bytes:
    try:
        from PIL import Image as PILImage
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(f"PIL_unavailable:{exc}") from exc

    width = int(msg.width)
    height = int(msg.height)
    encoding = (msg.encoding or "").lower()
    raw = bytes(msg.data)
    step = int(msg.step) if msg.step else 0

    if encoding in ("rgb8", "bgr8"):
        channels = 3
        rows = []
        for r in range(height):
            start = r * step if step else r * width * channels
            rows.append(raw[start : start + width * channels])
        packed = b"".join(rows)
        image = PILImage.frombytes("RGB", (width, height), packed)
        if encoding == "bgr8":
            r, g, b = image.split()
            image = PILImage.merge("RGB", (b, g, r))
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
        rows = []
        for r in range(height):
            start = r * step if step else r * width * channels
            rows.append(raw[start : start + width * channels])
        image = PILImage.frombytes("RGBA", (width, height), b"".join(rows)).convert("RGB")
        if encoding == "bgra8":
            r, g, b = image.split()
            image = PILImage.merge("RGB", (b, g, r))
    elif encoding == "mono8":
        rows = []
        for r in range(height):
            start = r * step if step else r * width
            rows.append(raw[start : start + width])
        image = PILImage.frombytes("L", (width, height), b"".join(rows)).convert("RGB")
    else:
        raise RuntimeError(f"unsupported_image_encoding:{msg.encoding}")

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("llm_response_json_not_found")
    obj = json.loads(stripped[start : end + 1])
    if not isinstance(obj, dict):
        raise RuntimeError("llm_response_root_not_object")
    return obj


def normalize_semantics(raw: Dict[str, Any]) -> Dict[str, Any]:
    def nested(name: str) -> Dict[str, Any]:
        value = raw.get(name)
        return value if isinstance(value, dict) else {}

    confidence = raw.get("confidence")
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence_f = 0.0
    return {
        "primary_scene": str(raw.get("primary_scene") or "unknown"),
        "passability": str(raw.get("passability") or "unknown"),
        "door": nested("door"),
        "room_opening": nested("room_opening"),
        "elevator": nested("elevator"),
        "stairs": nested("stairs"),
        "danger_source": nested("danger_source"),
        "navigation_hint": str(raw.get("navigation_hint") or "unknown"),
        "confidence": confidence_f,
        "description": str(raw.get("description") or ""),
    }


class VisionSceneSemanticsNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.latest_image: Optional[Image] = None
        self.latest_image_wall_time = 0.0
        self.busy = False
        self.last_payload: Optional[Dict[str, Any]] = None
        self.pub = rospy.Publisher(args.output_topic, String, queue_size=2)
        rospy.Subscriber(args.image_topic, Image, self.image_cb, queue_size=1)
        threading.Thread(target=self.wall_loop, daemon=True).start()

    def image_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_image = msg
            self.latest_image_wall_time = time.time()

    def wall_loop(self) -> None:
        while not rospy.is_shutdown():
            self.timer_cb(None)
            time.sleep(max(0.05, float(self.args.interval_sec)))

    def no_api_payload(self, msg: Optional[Image]) -> Dict[str, Any]:
        return {
            "final_decision": "VISION_SCENE_SEMANTICS_NO_API_KEY",
            "semantic_source": "vision_scene_semantics_node",
            "api_call_enabled": False,
            "api_key_present": False,
            "image_topic": self.args.image_topic,
            "image_stamp_sec": float(msg.header.stamp.to_sec()) if msg and msg.header.stamp else None,
            "analysis_wall_time_sec": time.time(),
            "semantics": normalize_semantics({}),
            "forbidden_sources_used": [],
            "called_move_base": False,
            "sent_navigation_goal": False,
            "cmd_vel_published": False,
        }

    def publish_payload(self, payload: Dict[str, Any]) -> None:
        self.last_payload = payload
        write_json(LATEST_PATH, payload)
        append_jsonl(JSONL_PATH, payload)
        self.pub.publish(String(data=json.dumps(payload, sort_keys=True, ensure_ascii=False)))

    def call_llm(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "model": self.args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.args.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            self.args.api_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.args.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.args.api_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vision_api_request_failed:{exc}") from exc
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        return extract_json_object(content)

    def api_worker(self, msg: Image, request_started_wall_time: float) -> None:
        try:
            jpeg = ros_image_to_rgb_bytes(msg)
            raw = self.call_llm(jpeg)
            payload = {
                "final_decision": "VISION_SCENE_SEMANTICS_READY",
                "semantic_source": "vision_scene_semantics_node",
                "api_call_enabled": True,
                "api_key_present": True,
                "api_busy": False,
                "model": self.args.model,
                "api_timeout_sec": self.args.api_timeout_sec,
                "api_request_started_wall_time_sec": request_started_wall_time,
                "image_topic": self.args.image_topic,
                "image_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
                "analysis_wall_time_sec": time.time(),
                "semantics": normalize_semantics(raw),
                "raw_semantics": raw,
                "forbidden_sources_used": [],
                "called_move_base": False,
                "sent_navigation_goal": False,
                "cmd_vel_published": False,
            }
            self.publish_payload(payload)
        except Exception as exc:
            payload = {
                "final_decision": "VISION_SCENE_SEMANTICS_ERROR",
                "error": str(exc),
                "semantic_source": "vision_scene_semantics_node",
                "api_call_enabled": True,
                "api_key_present": True,
                "api_busy": False,
                "api_timeout_sec": self.args.api_timeout_sec,
                "api_request_started_wall_time_sec": request_started_wall_time,
                "analysis_wall_time_sec": time.time(),
                "forbidden_sources_used": [],
                "called_move_base": False,
                "sent_navigation_goal": False,
                "cmd_vel_published": False,
            }
            self.publish_payload(payload)
        finally:
            with self.lock:
                self.busy = False

    def timer_cb(self, _event: Any) -> None:
        with self.lock:
            msg = self.latest_image
            busy = self.busy
        try:
            if msg is None:
                payload = self.no_api_payload(None)
                payload["final_decision"] = "VISION_SCENE_SEMANTICS_WAITING_FOR_IMAGE"
            elif not self.args.api_key:
                payload = self.no_api_payload(msg)
            elif busy:
                last = self.last_payload or {}
                last_time = last.get("analysis_wall_time_sec")
                payload = {
                    "final_decision": "VISION_SCENE_SEMANTICS_API_IN_FLIGHT",
                    "semantic_source": "vision_scene_semantics_node",
                    "api_call_enabled": True,
                    "api_key_present": True,
                    "api_busy": True,
                    "model": self.args.model,
                    "api_timeout_sec": self.args.api_timeout_sec,
                    "image_topic": self.args.image_topic,
                    "image_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
                    "analysis_wall_time_sec": time.time(),
                    "last_result_final_decision": last.get("final_decision"),
                    "last_result_age_sec": time.time() - float(last_time) if isinstance(last_time, (int, float)) else None,
                    "semantics": last.get("semantics") if isinstance(last.get("semantics"), dict) else normalize_semantics({}),
                    "forbidden_sources_used": [],
                    "called_move_base": False,
                    "sent_navigation_goal": False,
                    "cmd_vel_published": False,
                }
            else:
                request_started = time.time()
                with self.lock:
                    self.busy = True
                threading.Thread(target=self.api_worker, args=(msg, request_started), daemon=True).start()
                payload = {
                    "final_decision": "VISION_SCENE_SEMANTICS_API_REQUEST_STARTED",
                    "semantic_source": "vision_scene_semantics_node",
                    "api_call_enabled": True,
                    "api_key_present": True,
                    "api_busy": True,
                    "model": self.args.model,
                    "api_timeout_sec": self.args.api_timeout_sec,
                    "api_request_started_wall_time_sec": request_started,
                    "image_topic": self.args.image_topic,
                    "image_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
                    "analysis_wall_time_sec": time.time(),
                    "semantics": normalize_semantics({}),
                    "forbidden_sources_used": [],
                    "called_move_base": False,
                    "sent_navigation_goal": False,
                    "cmd_vel_published": False,
                }
            self.publish_payload(payload)
        except Exception as exc:
            payload = {
                "final_decision": "VISION_SCENE_SEMANTICS_ERROR",
                "error": str(exc),
                "api_timeout_sec": self.args.api_timeout_sec,
                "analysis_wall_time_sec": time.time(),
                "forbidden_sources_used": [],
                "called_move_base": False,
                "sent_navigation_goal": False,
                "cmd_vel_published": False,
            }
            self.publish_payload(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default="/real_sense/rgb/image_raw")
    parser.add_argument("--output-topic", default="/team/vision_scene_semantics")
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--api-base-url", default=os.getenv("VISION_LLM_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key", default=os.getenv("VISION_LLM_API_KEY") or os.getenv("FIRE_AGENT_API_KEY") or "")
    parser.add_argument("--model", default=os.getenv("VISION_LLM_MODEL", "qwen3.6-flash"))
    parser.add_argument("--api-timeout-sec", type=float, default=6.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("vision_scene_semantics_node", anonymous=False)
    VisionSceneSemanticsNode(args)
    rospy.spin()


if __name__ == "__main__":
    main()
