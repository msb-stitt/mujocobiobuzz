from pathlib import Path

import mujoco
import pytest

SCENE = Path(__file__).parent.parent / "models" / "starterbot" / "scene.xml"


@pytest.mark.slow
@pytest.mark.skipif(not SCENE.exists(), reason="the model is built by `ninja model`")
def test_starterbot_settles_on_its_four_wheels() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.02
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    touching = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        for c in d.contact[: d.ncon]
        for g in (c.geom1, c.geom2)
        if m.geom_bodyid[g]
    }
    assert touching == {f"starterbot_wheel_{w}" for w in ("fl", "fr", "bl", "br")}
    assert abs(d.qpos[2]) < 0.001
