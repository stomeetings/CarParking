"""Low-latency RTSP reader: a background thread drains the stream and keeps only the newest frame.

Why a thread: OpenCV/FFmpeg buffers decoded frames. When inference (0.1–3 s) is slower than the camera (25 fps), a plain
`cap.read()` loop returns ever older frames and the feed lags by minutes. CAP_PROP_BUFFERSIZE is ignored by the FFmpeg
backend, so the only reliable fix is to keep reading at stream speed and drop everything but the latest frame.
Drops (camera reboot, Wi-Fi, NVR restart) are retried with backoff inside the thread; the caller never blocks or crashes.
"""
from __future__ import annotations

import os
import random
import re
import threading
import time

import cv2

# TCP instead of UDP: no smeared grey frames from lost packets on a busy LAN. Read when the capture is opened.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def env_setting(name: str, env_file=None) -> str:
    """$name, else its NAME=value line in env_file (the project's .env: never in the image, never in the source), else
    "". Lets the bind-mounted dev container pick up the camera URL without being recreated with -e; under
    docker compose the same .env arrives as real environment variables (env_file)."""
    value = os.environ.get(name, "")
    if value or env_file is None or not os.path.isfile(env_file):
        return value
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            key, sep, val = line.strip().partition("=")
            if sep and key.strip() == name:
                return val.strip().strip("'\"")
    return ""


def mask_url(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://***@host/... (for UI messages and logs)."""
    return re.sub(r"//[^/@]*@", "//***@", url)


class LatestFrameReader:
    """reader.latest() -> (frame BGR or None, seq, capture time). seq grows by 1 per new frame, so the caller can skip
    frames it has already processed. reader.status is a short human-readable connection state."""

    def __init__(self, url: str, open_timeout_ms: int = 8000, read_timeout_ms: int = 5000, max_backoff: float = 30,
                 rng: random.Random | None = None):
        self.url, self.open_timeout_ms, self.read_timeout_ms, self.max_backoff = url, open_timeout_ms, read_timeout_ms, max_backoff
        self._rng = rng or random.Random()
        self._frame, self._seq, self._t = None, 0, 0.0
        self._lock, self._stop = threading.Lock(), threading.Event()
        self.status, self.reconnects = "connecting", 0
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._frame, self._seq, self._t

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _open(self):
        if isinstance(self.url, int) or self.url.isdigit():  # local webcam index, handy for testing without the camera
            return cv2.VideoCapture(int(self.url))
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
                                                           cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms])

    def delay(self, backoff: float) -> float:
        """The wait before a retry: the backoff plus up to half of it at random, so cameras that dropped together (a switch
        reboot) do not all reconnect in the same second."""
        return backoff + self._rng.uniform(0, backoff / 2)

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                wait = self.delay(backoff)
                self.status = f"cannot open stream, retrying in {wait:.0f} s"
                self._stop.wait(wait)
                backoff = min(backoff * 2, self.max_backoff)
                continue
            self.status, backoff = "connected", 1.0
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()  # blocks at most read_timeout_ms
                    if not ok:
                        break
                    with self._lock:
                        self._frame, self._seq, self._t = frame, self._seq + 1, time.time()
            finally:
                cap.release()
            if not self._stop.is_set():
                self.reconnects += 1
                self.status = f"stream dropped, reconnecting (#{self.reconnects})"
                self._stop.wait(self.delay(1.0))
