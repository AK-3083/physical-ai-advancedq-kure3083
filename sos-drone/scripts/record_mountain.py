"""山道探索をGazebo固定視点と機体カメラの左右映像で録画する。"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from scripts.evaluate import color_comparison, stop_simulator
from scripts.mountain_scene import generate_mountain
from scripts.record_gazebo import GazeboCapture, find_viewport, fit_viewport
from scripts.render_mountain import main as render_mountain
from scripts.render_mountain import VIDEO_SIZE, panel_areas, read_events, recording_paths


class Recorder(Node):
    """Gazebo固定視点と機体カメラを合成するROSノード。"""

    def __init__(self, gazebo):
        super().__init__("sos_demo_recorder")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.bridge = CvBridge()
        self.downward = None
        self.gazebo = gazebo
        self.create_subscription(Image, "/simple_drone/bottom/image_raw",
                                 self.on_downward, qos_profile_sensor_data)

    def on_downward(self, message):
        self.downward = message

    def wait_fresh(self):
        """機体カメラの遅れを0.5秒以内に確認し、Gazebo画像を取得する。"""
        deadline = time.monotonic() + 5
        while True:
            message = self.downward
            age = (self.get_clock().now().nanoseconds / 1e9
                   - message.header.stamp.sec - message.header.stamp.nanosec / 1e9)
            if age <= 0.5:
                self.gazebo.frame()
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"Camera timestamps remain stale (maximum age {age:.2f}s)")
            rclpy.spin_once(self, timeout_sec=0.03)

    def frame(self):
        """1秒以内の実カメラ画像を文字なしの1080pフレームに合成する。"""
        canvas = np.full((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), 22, dtype=np.uint8)
        left, right = panel_areas(VIDEO_SIZE)
        x, y, width, height = left
        canvas[y:y+height, x:x+width] = fit_viewport(self.gazebo.frame(), width, height)
        message = self.downward
        timestamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if self.get_clock().now().nanoseconds / 1e9 - timestamp > 1.0:
            raise RuntimeError(f"Camera {message.header.frame_id} stale at {timestamp:.2f}s")
        image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        x, y, width, height = right
        canvas[y:y+height, x:x+width] = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_CUBIC)
        return canvas


def prepare_scene(directory):
    """固定視点録画用の山道シーンを生成する。"""
    if directory.exists():
        raise FileExistsError("Mountain demonstration scene already exists")
    generate_mountain(directory, 503, 2)


def main():
    """一括実行、シーン生成のみ、起動済みシミュレータの録画のみを実行する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/color-demo"))
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--destination", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                      help="Create the recording scene without launching or recording")
    mode.add_argument("--record-only", action="store_true",
                      help="Record an externally launched simulator and mission; do not render")
    args = parser.parse_args()
    output, directory, destination = recording_paths(args.output, args.scene, args.destination)
    if output.exists():
        raise FileExistsError("Mountain demonstration output already exists")
    if destination.exists() and any(destination.glob("color-*")):
        raise FileExistsError("Color demonstration deliverables already exist")
    if args.record_only:
        for name in ("search.world", "terrain.json", "truth.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Missing {directory / name}; run --prepare-only first")
    else:
        if not args.prepare_only:
            for name in ("gzserver", "gzclient"):
                result = subprocess.run(["pgrep", "-x", name], capture_output=True)
                if result.returncode == 0:
                    raise RuntimeError("Stop existing Gazebo and GUI before automatic recording")
                if result.returncode != 1:
                    raise subprocess.CalledProcessError(result.returncode, result.args)
        prepare_scene(directory)
    if args.prepare_only:
        print(f"Recording scene ready: {directory / 'search.world'}", flush=True)
        return
    raw = directory / "full-patrol.avi"
    # OpenCV hides filesystem errors behind generic encoder warnings.
    raw.write_bytes(b"")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"FFV1"), 5, VIDEO_SIZE)
    if not writer.isOpened():
        raise OSError(f"Cannot open video encoder for {raw}")
    rclpy.init()
    recorder = None
    mission, simulator, gui, screen = None, None, None, None
    environment = dict(os.environ, SOS_WORLD=str(directory / "search.world"))
    records, phases = [], []
    try:
        with ExitStack() as stack:
            if not args.record_only:
                log = stack.enter_context((directory / "sim.log").open("w"))
                mission_log = stack.enter_context((directory / "mission.log").open("w"))
                simulator = subprocess.Popen(
                    ["ros2", "launch", "/workspace/sos/scripts/sim.launch.py"],
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                stack.callback(stop_simulator, simulator)
            try:
                if not args.record_only:
                    gui = subprocess.Popen(
                        ["gzclient", "--verbose"], env=environment,
                        stdout=stack.enter_context((directory / "gui.log").open("w")),
                        stderr=subprocess.STDOUT, start_new_session=True)
                    stack.callback(stop_simulator, gui)
                screen = GazeboCapture(find_viewport(timeout=45, stable_seconds=2),
                                       stack.enter_context((directory / "capture.log").open("w")))
                stack.callback(screen.close)
                recorder = Recorder(screen)
                deadline = time.monotonic() + 45
                while recorder.downward is None or not screen.ready:
                    rclpy.spin_once(recorder, timeout_sec=0.03)
                    if time.monotonic() > deadline:
                        raise TimeoutError("Mountain cameras or state service are missing")
                if args.record_only:
                    print(f"Recording ready. Start the mission with --output {output}", flush=True)
                    deadline = time.monotonic() + 300
                    while not output.exists():
                        rclpy.spin_once(recorder, timeout_sec=0.03)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Mission did not start within 5 minutes")
                else:
                    mission = subprocess.Popen(
                        ["/opt/sos/train/bin/python", "-m", "sos.mission", "--terrain",
                         str(directory / "terrain.json"), "--output", str(output)],
                        stdout=mission_log, stderr=subprocess.STDOUT,
                    )
                start = time.monotonic()
                frames, last_capture = 0, -1
                last_geometry_check = 0.
                while (mission.poll() is None if mission is not None
                       else not (output / "report.json").exists()):
                    rclpy.spin_once(recorder, timeout_sec=0.03)
                    elapsed = time.monotonic() - start
                    if elapsed - last_geometry_check >= 1:
                        if find_viewport() != screen.viewport:
                            raise RuntimeError("Gazebo viewport changed size or identity during recording")
                        last_geometry_check = elapsed
                    if elapsed > 12000:
                        raise TimeoutError("Mountain recording exceeded deadline")
                    expected = int(elapsed * 5)
                    if frames < expected:
                        recorder.wait_fresh()
                        image = recorder.frame()
                        while frames < expected:
                            writer.write(image)
                            frames += 1
                        scan = output / "scans.jsonl"
                        if scan.exists():
                            items = read_events(scan)
                            if items and items[-1]["index"] > last_capture:
                                last_capture = items[-1]["index"]
                                records.append({"wall_seconds": frames / 5, **items[-1]})
                        events = output / "mission-events.jsonl"
                        if events.exists():
                            phases.extend({"wall_seconds": frames / 5, **event}
                                          for event in read_events(events)[len(phases):])
                if mission is not None and mission.returncode != 0:
                    raise RuntimeError("Mountain demonstration mission failed; inspect logs")
                duration = time.monotonic() - start
                report = json.loads((output / "report.json").read_text())
                if report["status"] != "complete":
                    raise RuntimeError("Mountain mission not complete")
                recorder.wait_fresh()
                final = recorder.frame()
                for _ in range(40):
                    writer.write(final)
            finally:
                if mission is not None and mission.poll() is None:
                    mission.terminate()
                    mission.wait(timeout=10)
    finally:
        writer.release()
        if recorder is not None:
            recorder.destroy_node()
        rclpy.try_shutdown()
    metrics = json.loads((output / "flight-metrics.json").read_text())
    if len(records) != metrics["viewpoint_count"]:
        raise RuntimeError(f"Expected {metrics['viewpoint_count']} real capture events, recorded {len(records)}")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"flight_wall_seconds": duration, "capture_times": records,
                "scheme": "color",
                "external_camera": "Fixed Gazebo GUI view",
                "recording_source": "gazebo_composite",
                "source_video_size": list(VIDEO_SIZE), "source_codec": "FFV1 lossless",
                "phase_events": phases,
                "status": report["status"], "sites": len(report["sites"])}
    metadata.update({key: metrics[key] for key in ("return_method", "exploration_method")})
    if screen is not None:
        metadata["viewport"] = {"window_id": screen.viewport[0],
                                "width": screen.viewport[1], "height": screen.viewport[2]}
    truth = json.loads((directory / "truth.json").read_text())
    metadata["color_comparison"] = color_comparison(truth, output)
    (destination / "color-demo-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    if args.record_only:
        print(f"Recording saved: {raw}\nMetadata saved: {destination / 'color-demo-metadata.json'}",
              flush=True)
    else:
        render_mountain(output, directory, destination)


if __name__ == "__main__":
    main()
