"""Read the native Gazebo viewport for the left panel of the recording."""

import logging
import os
import re
import signal
import subprocess
from threading import Thread
import time

import cv2
import numpy as np


def find_viewport(timeout=0, stable_seconds=0):
    """Identify the native render surface, not the surrounding Qt controls."""
    deadline = time.monotonic() + timeout
    previous, stable_since = None, time.monotonic()
    while True:
        viewport = None
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True,
                              text=True, check=True).stdout
        windows = re.findall(r'^\s*(0x[0-9a-f]+) "Gazebo":', tree, re.MULTILINE)
        if len(windows) > 1:
            raise ValueError("Exactly one Gazebo window is required; close duplicate windows")
        if windows:
            tree = subprocess.run(["xwininfo", "-id", windows[0], "-tree"],
                                  capture_output=True, text=True, check=True).stdout
            surfaces = [(identifier, int(width), int(height)) for identifier, width, height in re.findall(
                r"^\s*(0x[0-9a-f]+) \(has no name\):.*?\s(\d+)x(\d+)[+-]\d+[+-]\d+",
                tree, re.MULTILINE,
            ) if int(width) >= 300 and int(height) >= 200]
            if len(surfaces) > 1:
                raise ValueError("Cannot identify a unique Gazebo 3D render surface; close extra views")
            if surfaces:
                details = subprocess.run(["xwininfo", "-id", surfaces[0][0]],
                                         capture_output=True, text=True, check=True).stdout
                if "Map State: IsViewable" in details:
                    viewport = surfaces[0]
        if viewport != previous:
            previous, stable_since = viewport, time.monotonic()
        if viewport is not None and time.monotonic()-stable_since >= stable_seconds:
            return viewport
        if time.monotonic() >= deadline:
            raise TimeoutError("Gazebo viewport is not visible or ready; start gzclient")
        time.sleep(0.1)


def capture_command(viewport):
    identifier, _, _ = viewport
    return ["ffmpeg", "-nostdin", "-loglevel", "warning",
            "-f", "x11grab", "-window_id", str(int(identifier, 16)), "-draw_mouse", "0",
            "-framerate", "5", "-i", os.environ.get("DISPLAY", ":1"),
            "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]


def fit_viewport(image, width=640, height=480):
    """Keep the complete viewport aspect ratio inside the existing left panel."""
    source_height, source_width = image.shape[:2]
    scale = min(width/source_width, height/source_height)
    size = (max(1, round(source_width*scale)), max(1, round(source_height*scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    result = np.full((height, width, 3), 22, dtype=np.uint8)
    x, y = (width-size[0])//2, (height-size[1])//2
    result[y:y+size[1], x:x+size[0]] = resized
    return result


class GazeboCapture:
    """Read a bounded latest-frame slot without blocking ROS callbacks."""

    def __init__(self, viewport, log):
        self.viewport = viewport
        self.latest = None
        self.error = None
        self.closed = False
        self.process = subprocess.Popen(capture_command(viewport), stdout=subprocess.PIPE, stderr=log)
        self.thread = Thread(target=self.read_frames, name="gazebo-viewport")
        self.thread.start()

    def read_frames(self):
        _, width, height = self.viewport
        try:
            while True:
                data = self.process.stdout.read(width*height*3)
                if len(data) != width*height*3:
                    if self.closed:
                        return
                    self.error = RuntimeError("Gazebo capture ended or returned an incomplete frame")
                    return
                if self.closed:
                    continue
                self.latest = (time.monotonic(), np.frombuffer(data, np.uint8).reshape(height, width, 3))
        except OSError as error:
            self.error = error

    @property
    def ready(self):
        if self.error is not None:
            raise self.error
        if self.process.poll() is not None:
            raise RuntimeError("Gazebo capture process exited; inspect capture.log")
        return self.latest is not None

    def frame(self):
        if not self.ready:
            raise RuntimeError("Gazebo has not produced a frame")
        timestamp, image = self.latest
        if time.monotonic() - timestamp > 0.5:
            raise RuntimeError("Gazebo viewport frame is stale")
        return image

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logging.warning("Gazebo capture did not stop; killing the owned reader process")
                self.process.kill()
                self.process.wait(timeout=10)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Gazebo frame reader did not stop")
        self.process.stdout.close()
