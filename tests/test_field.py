import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujocobiobuzz import balls as B

SCENE = Path(__file__).parent.parent / "models" / "biobuzz" / "scene.xml"


@pytest.mark.slow
@pytest.mark.skipif(not SCENE.exists(), reason="the model is built by `ninja model`")
@pytest.mark.parametrize("hive", ["blue", "red"])
def test_a_hive_falls_to_the_stable_state_nearer_it(hive: str) -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"biobuzz_{hive}_hive")
    lo, hi = m.jnt_range[j]
    for start, end in ((lo + 0.3 * (hi - lo), lo), (lo + 0.7 * (hi - lo), hi)):
        d = mujoco.MjData(m)
        d.qpos[m.jnt_qposadr[j]] = start
        for _ in range(int(3.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        assert d.qpos[m.jnt_qposadr[j]] == pytest.approx(end, abs=math.radians(1))


# Blue's raised opening in its CAD state, from waggle's `data/hive/openings.json`: its centre,
# Pedro in, and its outward normal.
OPENING_IN = np.array([84.79, 88.38, 57.67])
NORMAL = np.array([-0.001, 0.87, 0.493])


def load_blue(kinds: list[B.Kind], seed: int, hold_s: float = 5.0) -> tuple[bool, float | None]:
    """The kinds placed one by one 0.12 m inside blue's raised cell, each 0.6 s after the last, as
    the Event Field Guide's § 12.3 places a calibration load; whether the hive tips within `hold_s`
    after the last, and its time from level to 2° short of its other stop."""
    rng = np.random.default_rng(seed)
    parked = [(k, f"b{i}", (0.2 + 0.2 * i, 0.1, 0.2)) for i, k in enumerate(kinds)]
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=SCENE.parent) as f:
        f.write(B.scene("biobuzz", parked))
        f.flush()
        m = mujoco.MjModel.from_xml_path(f.name)
    d = mujoco.MjData(m)
    hinge = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "biobuzz_blue_hive")]
    centre, across = OPENING_IN * 0.0254, np.cross(NORMAL, [1, 0, 0])
    level_t = down_s = None
    steps = [int(0.6 / m.opt.timestep)] * len(kinds)
    steps[-1] += int(hold_s / m.opt.timestep)
    for i, n in enumerate(steps):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"b{i}")
        q = m.jnt_qposadr[m.body_jntadr[b]]
        jitter = np.array([rng.uniform(-0.12, 0.12), 0, 0]) + across * rng.uniform(-0.04, 0.04)
        d.qpos[q : q + 7] = [*(centre - NORMAL * 0.12 + jitter), 1, 0, 0, 0]
        for _ in range(n):
            mujoco.mj_step(m, d)
            a = d.qpos[hinge]
            if level_t is None and a < math.radians(-30):
                level_t = d.time
            if level_t is not None and down_s is None and a < math.radians(-58):
                down_s = d.time - level_t
    return level_t is not None, down_s


@pytest.mark.slow
@pytest.mark.skipif(not SCENE.exists(), reason="the model is built by `ninja model`")
@pytest.mark.parametrize(
    "load, tips",
    [
        ([B.POLLEN] * 7, False),
        ([B.POLLEN] * 8, True),
        ([B.BLUE_NECTAR] * 3 + [B.POLLEN] * 2, False),
        ([B.BLUE_NECTAR] * 3 + [B.POLLEN] * 3, True),
    ],
)
def test_a_hive_tips_on_the_field_guides_calibration_loads(load: list[B.Kind], tips: bool) -> None:
    tipped, down_s = load_blue(load, seed=1)
    assert tipped == tips
    if tips:
        # The video's swing from level to down is 8 to 9 frames at 30 fps, by the floor plan's
        # `floor.physics.down`.
        assert down_s is not None and 0.24 < down_s < 0.32
