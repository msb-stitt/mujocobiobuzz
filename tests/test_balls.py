import math
from pathlib import Path

import mujoco
import pytest

from mujocobiobuzz import balls as B


def model(
    tmp_path: Path, kind: B.Kind, pos: tuple[float, float, float], gravity: bool = True
) -> mujoco.MjModel:
    (tmp_path / "floor.xml").write_text(
        '<mujoco><worldbody><geom type="plane" size="2 2 0.01"/></worldbody></mujoco>'
    )
    xml = B.scene("floor", [(kind, "ball", pos)])
    if not gravity:
        xml = xml.replace("<option ", '<option gravity="0 0 0" ')
    (tmp_path / "scene.xml").write_text(xml)
    return mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))


@pytest.mark.parametrize("kind", B.KINDS)
def test_a_ball_weighs_and_measures_as_its_drawing(tmp_path: Path, kind: B.Kind) -> None:
    m = model(tmp_path, kind, (0, 0, 1))
    assert m.body_mass[1] == pytest.approx(kind.mass_kg)
    # A hollow shell: 2/3 m r², not a solid ball's 2/5.
    assert m.body_inertia[1][0] == pytest.approx(2 / 3 * kind.mass_kg * kind.radius_m**2)


@pytest.mark.parametrize("kind", [B.POLLEN, B.BLUE_NECTAR])
def test_drag_is_quadratic_at_the_floor_plans_coefficient(tmp_path: Path, kind: B.Kind) -> None:
    m = model(tmp_path, kind, (0, 0, 1), gravity=False)
    d = mujoco.MjData(m)
    d.qvel[0] = 10.0
    for _ in range(int(0.1 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    k = B.AIR_KG_M3 * B.DRAG_CD * math.pi * kind.radius_m**2 / (2 * kind.mass_kg)
    assert k * 0.0254 == pytest.approx(0.0007, abs=0.00005)  # the floor plan's 0.0007 /in
    assert d.qvel[0] == pytest.approx(10.0 / (1 + k * 10.0 * 0.1), rel=1e-3)


@pytest.mark.parametrize("height", [0.1, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("kind", [B.POLLEN, B.BLUE_NECTAR])
def test_a_floor_bounce_keeps_about_the_restitution(
    tmp_path: Path, kind: B.Kind, height: float
) -> None:
    m = model(tmp_path, kind, (0, 0, height + kind.radius_m))
    d = mujoco.MjData(m)
    v_in = None
    for _ in range(int(3 / m.opt.timestep)):
        vz = d.qvel[2]
        mujoco.mj_step(m, d)
        if v_in is None and d.ncon:
            v_in = -vz
        if v_in is not None and not d.ncon and d.qvel[2] > 0:
            break
    assert v_in is not None
    assert 0.55 < d.qvel[2] / v_in < 0.7
