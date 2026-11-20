import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

SCENE = Path(__file__).parent.parent / "models" / "starterbot" / "scene.xml"
WHEELS = ("fl", "fr", "bl", "br")
MOTORS = ("Front Left", "Front Right", "Back Left", "Back Right")
IN = 0.0254

needs_model = pytest.mark.skipif(not SCENE.exists(), reason="the model is built by `ninja model`")


def _load() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _wheel_of(m: mujoco.MjModel, body: int) -> str:
    """The wheel a roller belongs to, or the body's own name."""
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return name.split("_ROLLER")[0]


def _drive(m: mujoco.MjModel, powers: tuple[float, ...], seconds: float) -> mujoco.MjData:
    d = mujoco.MjData(m)
    for motor, u in zip(MOTORS, powers, strict=True):
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, motor)] = u
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


@pytest.mark.slow
@needs_model
def test_starterbot_settles_on_its_four_wheels() -> None:
    m = _load()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.02
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    touching = {
        _wheel_of(m, int(m.geom_bodyid[g]))
        for c in d.contact[: d.ncon]
        for g in (c.geom1, c.geom2)
        if m.geom_bodyid[g]
    }
    assert touching == {f"starterbot_wheel_{w}" for w in WHEELS}
    assert abs(d.qpos[2]) < 0.001


@pytest.mark.slow
@needs_model
def test_starterbot_drives_as_a_mecanum_robot() -> None:
    """At full power it drives forward near its wheels' free speed, strafes left at most of
    that, as the robots' own tuning measured, and turns counterclockwise, each without drift."""
    m = _load()
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "Front Left")
    rim = -m.actuator_gainprm[a, 0] / m.actuator_biasprm[a, 2] * 0.052
    fwd, left, turn = (_drive(m, p, 2.0) for p in ((1, 1, 1, 1), (-1, 1, 1, -1), (-1, 1, -1, 1)))
    v = {k: d.xmat[1].reshape(3, 3).T @ d.qvel[:3] for k, d in (("f", fwd), ("l", left))}
    assert 0.9 * rim < v["f"][0] < rim
    assert abs(v["f"][1]) < 2 * IN
    assert 0.8 * v["f"][0] < v["l"][1] < v["f"][0]
    assert abs(v["l"][0]) < 2 * IN
    assert turn.qvel[5] > 4.0
    assert np.linalg.norm(turn.qvel[:2]) < 3 * IN


@pytest.mark.slow
@needs_model
def test_starterbot_mechanisms_turn() -> None:
    """Each mechanism's motor at full power turns it forward, to most of its free speed."""
    m = _load()
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name in MOTORS:
            continue
        d = mujoco.MjData(m)
        d.ctrl[a] = 1.0
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        free = -m.actuator_gainprm[a, 0] / m.actuator_biasprm[a, 2]
        speed = d.qvel[m.jnt_dofadr[m.actuator_trnid[a, 0]]]
        assert 0.6 * free < speed <= free * (1 + 1e-6), (name, speed * 60 / 2 / math.pi)


@pytest.mark.slow
@needs_model
def test_starterbot_intake_pulls_in() -> None:
    """At the code's in, +1, the side of the intake roller and of each Gecko wheel that meets a
    ball, the roller's underside and each wheel's inner side, moves back, into the robot."""
    m = _load()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for motor in ("intake", "left_intake_servo", "right_intake_servo"):
        j = m.actuator_trnid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, motor), 0]
        b = m.jnt_bodyid[j]
        omega = d.xmat[b].reshape(3, 3) @ m.jnt_axis[j]
        meets = [0.0, -np.sign(d.xpos[b][1]), 0.0] if "servo" in motor else [0.0, 0.0, -1.0]
        assert np.cross(omega, meets)[0] < 0, motor
