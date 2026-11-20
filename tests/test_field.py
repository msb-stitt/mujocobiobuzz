import math
from pathlib import Path

import mujoco
import pytest

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
