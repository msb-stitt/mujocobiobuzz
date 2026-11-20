import math
import subprocess
from pathlib import Path

import mujoco
import pytest

from mujocobiobuzz import field, video
from test_replay import write_log

FIELD = Path(__file__).parent.parent / "models" / "biobuzz" / "biobuzz.xml"
needs_models = pytest.mark.skipif(
    not FIELD.exists(), reason="the models are built by `ninja model`"
)


@pytest.mark.slow
@needs_models
def test_a_robot_starts_at_its_pose_on_the_field() -> None:
    m = field.with_robots({"a_": (1.0, 2.0, math.pi / 2)})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_starterbot")
    assert d.xpos[b][:2] == pytest.approx([1.0, 2.0], abs=1e-9)
    assert d.xmat[b].reshape(3, 3)[:, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


@pytest.mark.slow
@needs_models
def test_a_replay_is_drawn_at_30_frames_a_second_from_its_logs_start(tmp_path: Path) -> None:
    log = tmp_path / "run.wpilog"
    rows: list[tuple[str, str, int, object]] = [
        ("/Localizer/pinPoint/x_in", "double", 0, 72.0),
        ("/Localizer/pinPoint/y_in", "double", 0, 24.0),
        ("/Localizer/pinPoint/heading_deg", "double", 0, 90.0),
        ("/loop/count", "double", 10**6, 1.0),
    ]
    write_log(log, rows)
    out = tmp_path / "run.mp4"
    assert video.of_replay(log, out) == 31
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert probe.stdout.split() == ["30/1,31"]
