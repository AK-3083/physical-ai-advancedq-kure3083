"""ROSの観測・位置指令・CNN推論を接続し、SOS探索と証拠保存を実行する。"""

import argparse
from collections import deque
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose as PoseMessage, Twist
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from gazebo_msgs.msg import ContactsState
from std_msgs.msg import Bool, Empty, Int8

from sos.geometry import Pose, terrain_point
from sos.reporting import Detection, write_report
from sos.terrain import Terrain
from sos.vision import MarkerClassifier, annotate_candidates


class Mission(Node):
    """姿勢・カメラを観測し、静止撮影を伴う自動探索を管理するROSノード。"""

    def __init__(self, directory, model, threshold, terrain, timeout=900, publish_preview=False):
        """モデル・出力先・ROS入出力を準備する。"""

        if terrain is None or terrain.data.get("scheme") != "color":
            raise ValueError("Color mission requires the matching color terrain")
        classifier = MarkerClassifier(model, threshold)
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        super().__init__("sos_search")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.output = output
        self.classifier = classifier
        self.deadline = time.monotonic() + timeout
        self.terrain = terrain
        self.patrol_targets = terrain.search_route()
        from sos.search import SearchTrigger
        self.search_trigger = SearchTrigger()
        self.search_image_time = None
        self.search_wall = 0.
        self.capture_index = 0
        self.survey_index = 0
        self.minimum_clearance = float("inf")
        self.maximum_altitude = 0.
        self.collision = False
        self.enroute = False
        self.safety_error = None
        self.bridge = CvBridge()
        self.poses = deque(maxlen=1000)
        self.velocity = None
        self.info = None
        self.image = None
        self.state = None
        self.detections = []
        self.video_writer = None
        self.frame_count = 0
        self.frame_wall = None
        self.preview = self.create_publisher(Image, "/sos/preview/image", 1) if publish_preview else None
        self.cmd = self.create_publisher(Twist, "/simple_drone/cmd_vel", 10)
        self.mode = self.create_publisher(Bool, "/simple_drone/posctrl", 10)
        self.takeoff = self.create_publisher(Empty, "/simple_drone/takeoff", 10)
        self.land = self.create_publisher(Empty, "/simple_drone/land", 10)
        self.create_subscription(PoseMessage, "/simple_drone/gt_pose", self.on_pose, qos_profile_sensor_data)
        self.create_subscription(Twist, "/simple_drone/gt_vel", self.on_velocity, qos_profile_sensor_data)
        self.create_subscription(Image, "/simple_drone/bottom/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/simple_drone/bottom/camera_info", self.on_info, qos_profile_sensor_data)
        self.create_subscription(Int8, "/simple_drone/state", self.on_state, qos_profile_sensor_data)
        self.create_subscription(ContactsState, "/simple_drone/bumper_states",
                                 self.on_contact, qos_profile_sensor_data)

    def now(self):
        """現在のROSシミュレーション時刻を秒で返す。"""

        return self.get_clock().now().nanoseconds / 1e9

    def on_pose(self, message):
        """ヘッダーのない姿勢メッセージに受信時刻を付けて履歴へ追加する。"""

        p, q = message.position, message.orientation
        if not np.isfinite([p.x, p.y, p.z, q.x, q.y, q.z, q.w]).all():
            self.safety_error = "Simulator returned non-finite aircraft pose"
            return
        self.poses.append(Pose(np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]), self.now()))
        self.maximum_altitude = max(self.maximum_altitude, p.z)
        if self.enroute:
            gap = p.z - self.terrain.height(p.x, p.y)
            self.minimum_clearance = min(self.minimum_clearance, gap)
            if gap < 1.0:
                self.safety_error = f"Insufficient actual terrain clearance: {gap:.2f}m"

    def on_contact(self, message):
        """飛行中の地形・樹木への接触を失敗として保持する。"""
        if (message.states and self.poses and self.state == 1
                and self.poses[-1].position[2] - self.terrain.height(
                    *self.poses[-1].position[:2]) > 1.0):
            self.collision = True

    def on_velocity(self, message):
        """静止判定に使う最新の線速度を保持する。"""

        self.velocity = np.array([message.linear.x, message.linear.y, message.linear.z])
        if not np.isfinite(self.velocity).all():
            self.safety_error = "Simulator returned non-finite aircraft velocity"

    def on_info(self, message):
        """画素の視線復元に使う最新のカメラ内部情報を保持する。"""

        self.info = message

    def on_state(self, message):
        """離陸・着陸の完了判定に使う飛行状態を保持する。"""

        self.state = message.data

    def on_image(self, message):
        """最新の機体画像を保持し、約5fpsでカメラ動画へ追記する。"""

        self.image = message
        wall = time.monotonic()
        if self.frame_wall is None or wall - self.frame_wall >= 0.2:
            frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
            if self.video_writer is None:
                height, width = frame.shape[:2]
                self.video_writer = cv2.VideoWriter(
                    str(self.output / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    5, (width, height),
                )
                if not self.video_writer.isOpened():
                    raise OSError("Could not create camera video")
            self.video_writer.write(frame)
            self.frame_count += 1
            self.frame_wall = wall

    def wait(self, predicate, timeout, description, target=None):
        """ROSコールバックを処理しながら、条件成立または期限まで待つ。"""

        deadline = min(time.monotonic() + timeout, self.deadline)
        while time.monotonic() < deadline:
            if target is not None:
                message = Twist()
                message.linear.x, message.linear.y, message.linear.z = map(float, target)
                self.cmd.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.collision:
                raise RuntimeError("Collision detected during mountain flight")
            if self.safety_error:
                raise RuntimeError(self.safety_error)
            if predicate():
                return
        if time.monotonic() >= self.deadline:
            raise TimeoutError(f"Global mission timeout: {description}")
        raise TimeoutError(description)

    def move(self, target, timeout=45):
        """目標近傍での静止が1秒継続するまで位置指令を送る。"""

        stable_since = None

        def stable():
            """静止条件の継続時間を更新し、1秒以上の成立を判定する。"""

            nonlocal stable_since
            good = (self.poses and self.velocity is not None
                    and self.now() - self.poses[-1].time < 0.2
                    and np.linalg.norm(self.poses[-1].position - target) < 0.2
                    and np.linalg.norm(self.velocity) < 0.1)
            if not good:
                stable_since = None
                return False
            if stable_since is None:
                stable_since = self.now()
            return self.now() - stable_since >= 1.0

        try:
            self.wait(stable, timeout, f"Did not settle at {target}", target)
        except TimeoutError as error:
            position = self.poses[-1].position.tolist() if self.poses else None
            velocity = self.velocity.tolist() if self.velocity is not None else None
            raise TimeoutError(f"{error}; actual position={position}, velocity={velocity}") from error

    def inspect(self, index, target):
        """静止後の新しい画像を認識し、座標・証拠・撮影ログを保存する。"""

        timestamp = self.now()
        self.wait(
            lambda: self.image is not None and (
                self.image.header.stamp.sec + self.image.header.stamp.nanosec / 1e9
            ) > timestamp,
            5, "No new image after settling", target,
        )
        image, pose, intrinsic, capture_time = self.camera_observation()
        observations = []
        from sos.vision import candidates
        candidate_records = []
        for x, y, w, h in candidates(image):
            point = terrain_point((x+w/2, y+h/2), intrinsic, pose, self.terrain)
            candidate_records.append({"pixel": [x, y, w, h],
                                      "score": self.classifier.score(image[y:y+h, x:x+w]),
                                      "x": float(point[0]), "y": float(point[1]), "z": float(point[2])})
        annotated = annotate_candidates(image, candidate_records, self.classifier.threshold)
        for x, y, w, h, score in self.classifier.detect(image):
            point = terrain_point((x + w / 2, y + h / 2), intrinsic, pose, self.terrain)
            filename = f"evidence_{index:02d}_{len(observations)}.png"
            if not cv2.imwrite(str(self.output / filename), image):
                raise OSError(f"Could not save {filename}")
            self.detections.append(Detection(
                float(point[0]), float(point[1]), capture_time, score, filename,
                float(point[2]),
            ))
            observations.append({"pixel": [x, y, w, h], "x": float(point[0]), "y": float(point[1]), "score": score})
            observations[-1]["z"] = float(point[2])
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 80), 2)
        if not cv2.imwrite(str(self.output / f"scan_{index:02d}.png"), annotated):
            raise OSError("Could not save scan image")
        self.publish_preview(image, self.image.header, candidate_records)
        record = {"index": index, "target": target, "pose": pose.position.tolist(),
                  "time": capture_time, "observations": observations}
        filename = f"raw_{index:02d}.png"
        if not cv2.imwrite(str(self.output / filename), image):
            raise OSError("Could not save raw color comparison image")
        record.update({"candidates": candidate_records, "raw_image": filename,
                       "survey": self.survey_index})
        with (self.output / "scans.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.get_logger().info(f"scan={index} SOS detections={len(observations)}")

    def publish_preview(self, image, header, records):
        """Publish the actual inference frame, not boxes over a newer camera frame."""
        if self.preview is None:
            return
        annotated = annotate_candidates(image, records, self.classifier.threshold)
        for record in records:
            x, y, _, _ = record["pixel"]
            cv2.putText(annotated, f'{record["score"]:.2f}', (x, max(16, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        annotated = cv2.copyMakeBorder(annotated, 0, 44, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        height = annotated.shape[0]
        timestamp = header.stamp.sec + header.stamp.nanosec / 1e9
        cv2.putText(annotated, f"Latest inference | sim time {timestamp:.2f}s",
                    (8, height-27), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Yellow: color | Green: CNN >= {self.classifier.threshold:.2f}",
                    (8, height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        message = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        message.header.stamp.sec = header.stamp.sec
        message.header.stamp.nanosec = header.stamp.nanosec
        message.header.frame_id = header.frame_id
        self.preview.publish(message)

    def camera_observation(self):
        """画像・姿勢・内部行列の整合性を検証して返す。"""
        message = self.image
        capture_time = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        pose = min(self.poses, key=lambda value: abs(value.time - capture_time))
        if abs(pose.time - capture_time) > 0.1:
            raise RuntimeError("Image and pose timestamps differ by more than 0.1s")
        image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        if (image.shape[:2] != (self.info.height, self.info.width)
                or message.header.frame_id != self.info.header.frame_id):
            raise RuntimeError("Image and CameraInfo dimensions or frames disagree")
        intrinsic = np.asarray(self.info.k).reshape(3, 3)
        if any(abs(value) > 1e-8 for value in self.info.d):
            raise RuntimeError("Nonzero distortion requires explicit calibration support")
        return image, pose, intrinsic, capture_time

    def mission_event(self, phase, **details):
        """飛行状態の遷移をROS時刻とともに保存する。"""
        with (self.output / "mission-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": self.now(), "phase": phase, **details},
                                    allow_nan=False) + "\n")

    def search_frame(self):
        """移動中の独立画像を判定し、継続検出した未調査地点を返す。"""
        if self.image is None or time.monotonic() - self.search_wall < 0.4:
            return None
        stamp = self.image.header.stamp
        timestamp = stamp.sec + stamp.nanosec / 1e9
        if timestamp == self.search_image_time:
            return None
        image, pose, intrinsic, timestamp = self.camera_observation()
        self.search_wall = time.monotonic()
        self.search_image_time = timestamp
        from sos.vision import candidates
        records, points = [], []
        for x, y, w, h in candidates(image):
            score = self.classifier.score(image[y:y+h, x:x+w])
            point = terrain_point((x+w/2, y+h/2), intrinsic, pose, self.terrain)
            records.append({"pixel": [x, y, w, h], "score": score,
                            "x": float(point[0]), "y": float(point[1]), "z": float(point[2])})
            if score >= self.classifier.threshold:
                points.append(point)
        filename = None
        if records:
            filename = f"search_{stamp.sec}_{stamp.nanosec}.jpg"
            if not cv2.imwrite(str(self.output / filename), image):
                raise OSError("Could not save moving-camera evidence")
        with (self.output / "search-frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": timestamp, "pose": pose.position.tolist(),
                                    "candidates": records, "image": filename},
                                    allow_nan=False) + "\n")
        self.publish_preview(image, self.image.header, records)
        return self.search_trigger.update(points)

    def explore(self, target, finish=False):
        """山道を移動し、CNNによる継続検出時だけ停止して周囲を撮影する。"""
        point = None
        if finish:
            self.move(target)
        finish_deadline = time.monotonic() + 2.

        def arrived_or_detected():
            """到達か継続検出のいずれかが成立したかを返す。"""
            nonlocal point
            point = self.search_frame()
            return point is not None or (
                time.monotonic() >= finish_deadline if finish
                else np.linalg.norm(self.poses[-1].position - target) < 0.6)

        while True:
            self.wait(arrived_or_detected, 45, f"Search transit failed: {target}", target)
            if point is None:
                return
            hold = self.poses[-1].position.tolist()
            self.mission_event("detected", point=point.tolist(), survey=self.survey_index)
            self.move(hold)
            survey = self.terrain.survey_targets(self.poses[-1].position.tolist(), point)
            for location in survey:
                self.move(location)
                self.inspect(self.capture_index, location)
                self.capture_index += 1
            self.search_trigger.mark_visited(point)
            self.survey_index += 1
            self.terrain.validate_segment(self.poses[-1].position, target, clearance=2.)
            self.mission_event("searching")

    def run(self):
        """入力準備、離陸、山道での継続検出と撮影、直接帰還、着陸を実行する。"""

        self.wait(lambda: self.poses and self.info is not None and self.image is not None
                  and self.state is not None and self.cmd.get_subscription_count() > 0,
                  30, "Missing drone/camera/control input")
        self.run_mountain()

    def run_mountain(self):
        """道中の継続検出で周囲を撮影し、麓へ直行して着陸する。"""
        targets = self.patrol_targets
        self.mode.publish(Bool(data=True))
        self.takeoff.publish(Empty())
        self.wait(lambda: self.state == 1, 10, "Mountain takeoff failed", targets[0]["target"])
        self.move(targets[0]["target"])
        self.enroute = True
        self.mission_event("searching")
        for item in targets:
            self.explore(item["target"])
        self.explore(targets[-1]["target"], finish=True)
        self.mission_event("returning")
        home = targets[0]["target"]
        return_targets = self.terrain.direct_return(self.poses[-1].position, home)
        for target in return_targets[:-1]:
            distance = np.linalg.norm(self.poses[-1].position - target)
            # 水平速度制限と描画負荷に余裕を持たせる。
            self.move(target, timeout=45 + 2 * distance)
        self.move(home)
        self.enroute = False
        self.move([home[0], home[1], self.terrain.height(*home[:2]) + 1.])
        self.mode.publish(Bool(data=False))
        self.land.publish(Empty())
        self.wait(lambda: self.state == 0 and self.poses[-1].position[2]
                  < self.terrain.height(*home[:2]) + 0.25, 30, "Mountain landing failed")
        metrics = {"maximum_altitude_m": self.maximum_altitude,
                   "minimum_airborne_clearance_m": self.minimum_clearance,
                   "collision": self.collision, "viewpoint_count": self.capture_index,
                   "registered_site_count": 0,
                   "scope": "Camera-visible known trail corridor; no fixed capture sites",
                   "gps": False,
                   "return_method": "direct",
                   "return_targets": return_targets,
                   "exploration_method": "cnn_triggered",
                   "survey_count": self.survey_index, "fixed_capture_count": 0}
        (self.output / "flight-metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        if self.maximum_altitude < self.terrain.data["summit_height_m"]:
            raise RuntimeError("Mountain summit altitude was not reached")

    def save_video(self):
        """カメラ動画ライターを解放し、少なくとも1フレームの保存を確認する。"""

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.frame_count == 0:
            raise RuntimeError("No camera frames recorded; video deliverable is missing")

    def hold(self):
        """最新位置を0.5秒繰り返し指令し、失敗時の位置保持を要求する。"""

        self.enroute = False
        if not self.poses:
            self.get_logger().error("No pose available for emergency hold; stop the simulator")
            return
        message = Twist()
        message.linear.x, message.linear.y, message.linear.z = map(float, self.poses[-1].position)
        self.mode.publish(Bool(data=True))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            self.cmd.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warning("Position hold requested; stop/reset simulator before another run")


def main():
    """CLIから探索を開始し、成功・失敗をレポートと終了コードに反映する。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/color/marker.onnx")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--terrain", default="assets/color/terrain.json")
    parser.add_argument("--publish-preview", action="store_true",
                        help="Publish latest annotated inference frames on /sos/preview/image")
    args = parser.parse_args()
    model = args.model
    threshold = json.loads(Path(model).with_name("metrics.json").read_text())["threshold"]
    terrain = Terrain.load(args.terrain)
    if terrain.data.get("scheme") != "color":
        parser.error("Color mission requires the matching color terrain")
    timeout = args.timeout if args.timeout is not None else (
        900 + 10 * sum(np.linalg.norm(np.subtract(b["target"], a["target"]))
                       for a, b in zip(terrain.search_route(), terrain.search_route()[1:]))
    )
    if not np.isfinite(timeout) or timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    rclpy.init()
    node = None
    status, detail = "failed", "Mission did not complete"
    try:
        node = Mission(args.output, model, threshold, terrain, timeout,
                       publish_preview=args.publish_preview)
        node.run()
        node.save_video()
        status, detail = "complete", "Route, evidence recording and landing completed"
    except (TimeoutError, RuntimeError, ValueError, OSError, cv2.error, CvBridgeError,
            KeyboardInterrupt, ExternalShutdownException) as error:
        detail = str(error) or type(error).__name__
        if node is None:
            logging.error("Mission initialization failed: %s", detail)
        else:
            if rclpy.ok():
                node.get_logger().error(detail)
                node.hold()
            else:
                logging.error("%s: ROS context closed; stop/reset simulator before another run", detail)
    finally:
        try:
            if node is not None:
                try:
                    node.save_video()
                except (RuntimeError, OSError) as error:
                    status, detail = "failed", str(error)
                    if rclpy.ok():
                        node.get_logger().error(detail)
                    else:
                        logging.error(detail)
                write_report(node.output, node.detections, status, detail)
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.try_shutdown()
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
