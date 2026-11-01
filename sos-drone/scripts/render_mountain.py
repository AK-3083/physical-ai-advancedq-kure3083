"""全飛行の左右映像を、字幕なし・右カメラの候補/CNN枠付き動画へ変換する。"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sos.vision import MarkerClassifier, candidates


VIDEO_SIZE = (1920, 1080)


def panel_areas(size):
    scale = size[0] / 1280
    return [tuple(round(value*scale) for value in area)
            for area in ((0, 110, 640, 480), (640, 170, 640, 360))]


def annotate_camera(frame, classifier):
    """画像判定結果をもとに、右カメラ映像に枠を描く。"""
    size = (frame.shape[1], frame.shape[0])
    scale = size[0] / 1280
    x, y, width, height = panel_areas(size)[1]
    camera = cv2.resize(frame[y:y+height, x:x+width], (640, 360),
                        interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for color, thickness, boxes in (
        ("#ffd000", 2, candidates(camera)),
        ("#00ff70", 3, classifier.detect(camera)),
    ):
        for box in boxes:
            bx, by, bw, bh = box[:4]
            draw.rectangle(tuple(round(value*scale) for value in
                                 (640+bx, 170+by, 640+bx+bw, 170+by+bh)),
                           outline=color, width=round(thickness*scale))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def read_events(path):
    """追記中のJSONLから、改行で完結したイベントだけを読む。"""
    content = path.read_text()
    return [json.loads(line) for line in content[:content.rfind("\n")+1].splitlines()]


def video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {"codec": stream["codec_name"], "width": int(stream["width"]),
            "height": int(stream["height"]), "fps": float(Fraction(stream["r_frame_rate"])),
            "frames": int(stream["nb_frames"]), "duration": float(stream["duration"])}


def recording_paths(output, scene=None, destination=None):
    """飛行結果、シーン、完成動画の保存先を分離する。"""
    output = Path(output).resolve()
    scene = Path(scene or output.with_name(output.name + "-scene")).resolve()
    destination = Path(destination or (
        Path("deliverables") if output == Path("runs/color-demo").resolve()
        else output / "rendered")).resolve()
    if scene == output or output in scene.parents:
        raise ValueError("Scene directory must not create the mission output directory")
    if destination == scene or scene in destination.parents:
        raise ValueError("Video destination must be separate from the scene directory")
    return output, scene, destination


def speed_segments(duration, return_start):
    """帰還前は1倍速、帰還・着陸は2倍速、最後の静止表示は8秒間とする。"""
    landing_end = duration - 8
    if not np.isfinite([duration, return_start]).all() or not 0 <= return_start < landing_end:
        raise ValueError("Invalid continuous video timing")
    segments = [(0., return_start, 1.)] if return_start > 0 else []
    segments += [(return_start, landing_end, 2.), (landing_end, duration, 1.)]
    if not 60 <= sum((end-start)/speed for start, end, speed in segments) <= 180:
        raise ValueError("Continuous observation video outside 1-3 minutes")
    return segments


def main(output=None, scene=None, destination=None):
    if output is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, default=Path("runs/color-demo"))
        parser.add_argument("--scene", type=Path)
        parser.add_argument("--destination", type=Path)
        args = parser.parse_args()
        output, scene, destination = args.output, args.scene, args.destination
    output, scene, destination = recording_paths(output, scene, destination)
    source = scene / "full-patrol.avi"
    if not source.exists():
        source = scene / "full-patrol.mp4"
    metadata_path = destination / "color-demo-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("recording_source") != "gazebo_composite":
        raise ValueError("Expected a fixed Gazebo view and downward-camera recording")
    return_times = [event["wall_seconds"] for event in metadata.get("phase_events", [])
                    if event["phase"] == "returning"]
    if len(return_times) != 1:
        raise ValueError("Exactly one recorded returning event is required for playback timing")
    model = Path("models/color/marker.onnx")
    threshold = json.loads(model.with_name("metrics.json").read_text(encoding="utf-8"))["threshold"]
    classifier = MarkerClassifier(model, threshold)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Cannot read {source}")
    temporary = destination / "color-no-subtitles.tmp.avi"
    writer = None
    frames = 0
    output_segments = []
    try:
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if size not in ((1280, 720), VIDEO_SIZE):
            raise ValueError("Expected a 720p or 1080p composite with left view and right downward camera")
        areas = panel_areas(size)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(source_fps) or source_fps <= 0 or count <= 8*source_fps:
            raise ValueError("Source video has invalid frame count or sampling rate")
        duration = count / source_fps
        segments = speed_segments(duration, return_times[0])
        fps = 20
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
        if not writer.isOpened():
            raise OSError(f"Cannot open video output: {temporary}")
        source_index = -1
        cached = None
        annotated_index, annotated = -1, None
        annotation_end = metadata["flight_wall_seconds"] - 15.
        for start, end, speed in segments:
            segment_frames = round((end-start)/speed*fps)
            output_start = frames/fps
            for index in range(segment_frames):
                timestamp = min(end, start + index/fps*speed)
                wanted = min(int(timestamp*source_fps), count-int(8*source_fps)-1)
                while source_index < wanted:
                    success, cached = capture.read()
                    if not success:
                        raise OSError(f"Cannot decode source frame {source_index+1}")
                    source_index += 1
                clean = np.full((size[1], size[0], 3), 22, dtype=np.uint8)
                for x, y, width, height in areas:
                    clean[y:y+height, x:x+width] = cached[y:y+height, x:x+width]
                if timestamp < annotation_end:
                    if annotated_index != source_index:
                        annotated = annotate_camera(clean, classifier)
                        annotated_index = source_index
                    writer.write(annotated)
                else:
                    writer.write(clean)
                frames += 1
            output_segments.append({"source_start": start, "source_end": end, "speed": speed,
                                    "output_start": output_start, "output_end": frames/fps})
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    video = destination / "color-marker-demo-no-subtitles.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(temporary),
                    "-map_metadata", "-1", "-an", "-sn", "-c:v", "libx264", "-threads", "2",
                    "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(video)], check=True)
    info = video_info(video)
    if not 60 <= info["duration"] <= 180 or abs(info["duration"]-frames/fps) > 0.1:
        raise ValueError(f"Invalid rendered duration: {info['duration']}")
    if (info["codec"] != "h264" or (info["width"], info["height"]) != size
            or info["fps"] != fps or info["frames"] != frames):
        raise ValueError(f"Invalid rendered video properties: {info}")
    temporary.unlink()
    for key in ("previous_excerpt_intervals_seconds", "cut_intervals_seconds",
                "normal_speed_return_reason", "annotation_source", "subtitles", "final_card"):
        metadata.pop(key, None)
    metadata.update({"video_seconds": frames/fps, "source_video_seconds": duration,
                     "processing": "Continuous full flight; variable speed; no travel cuts",
                     "speed_segments": output_segments, "output_fps": fps,
                     "video_size": list(size), "output_crf": 16,
                     "intermediate_codec": "FFV1 lossless",
                     "playback_speed_policy": "Before returning: 1x; returning through landing: 2x; final hold: 1x",
                     "return_speed_start_seconds": return_times[0],
                     "final_hold_seconds": 8, "annotations": True, "subtitles": False,
                     "annotation_source": "Candidate and CNN inference on the displayed 640x360 camera frame",
                     "annotation_end_source_seconds": annotation_end,
                     "annotation_style": {"candidate": "#ffd000", "cnn": "#00ff70",
                                          "candidate_width_720p": 2, "cnn_width_720p": 3},
                     "video_file": video.name})
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created text-free continuous video: {frames/fps:.2f}s")


if __name__ == "__main__":
    main()
