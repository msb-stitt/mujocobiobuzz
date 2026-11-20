"""A run drawn to an H.264 video, from a camera on the field, with the run's time on each frame.

The video's first frame is the run's time 0, so a replay's video starts at its log's first
record, and the time drawn on it lines it up with the log in AdvantageScope's video tab.
`ffmpeg` encodes it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

# The field seen from its long side, raised, as a match video sees it.
LOOKAT = (1.83, 1.83, 0.0)
DISTANCE_M = 5.2
AZIMUTH_DEG = 90.0
ELEVATION_DEG = -38.0


class Video:
    """Frames of a run at `fps`, each taken at the first state handed to it at or after its
    time, written to `path`."""

    def __init__(
        self, m: mujoco.MjModel, path: Path, fps: int = 30, width: int = 1280, height: int = 720
    ) -> None:
        self.m, self.fps, self.frames = m, fps, 0
        self.renderer = mujoco.Renderer(m, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = LOOKAT
        self.camera.distance = DISTANCE_M
        self.camera.azimuth, self.camera.elevation = AZIMUTH_DEG, ELEVATION_DEG
        self.encoder = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def __call__(self, d: mujoco.MjData) -> None:
        while self.frames / self.fps <= d.time:
            self.renderer.update_scene(d, camera=self.camera)
            frame = Image.fromarray(self.renderer.render())
            ImageDraw.Draw(frame).text((16, 12), f"{self.frames / self.fps:8.2f} s", fill="white")
            assert self.encoder.stdin is not None
            self.encoder.stdin.write(np.asarray(frame).tobytes())
            self.frames += 1

    def close(self) -> None:
        """Ends the video; raises if `ffmpeg` failed."""
        assert self.encoder.stdin is not None
        self.encoder.stdin.close()
        if self.encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg exited {self.encoder.returncode}")
        self.renderer.close()


def of_replay(log_path: Path, out: Path, prefix: str = "robot_") -> int:
    """A replay of the log at `log_path`, on the field from its localizer's first pose, drawn to
    `out`; how many frames."""
    from mujocobiobuzz import field, replay

    log = replay.read(log_path)
    if log.start is None:
        raise ValueError(f"{log_path} logs no localizer pose to start the robot at")
    m = field.with_robots({prefix: log.start})
    video = Video(m, out)
    try:
        replay.replay(m, log, prefix, every=5, sample=video)
    finally:
        video.close()
    return video.frames


def main() -> None:
    """`python -m mujocobiobuzz.video LOG OUT.mp4`: a replay of LOG drawn to OUT.mp4."""
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    logging.getLogger(__name__).info("%d frames to %s", of_replay(a.log, a.out), a.out)


if __name__ == "__main__":
    main()
