from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from mujocobiobuzz.cad.model import CAD_TO_ROBOT, assign, to_robot, weld, write_model


@dataclass
class Box:
    """A solid as the converter sees one: a box of the given corners, in CAD mm."""

    path: list[str]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    volume_mm3: float = 0.0
    centroid: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    vertices: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: NDArray[np.int64] = field(default_factory=lambda: np.zeros((0, 3), np.int64))

    def __post_init__(self) -> None:
        lo, hi = self.lo, self.hi
        self.volume_mm3 = float(np.prod(hi - lo))
        self.centroid = (lo + hi) / 2
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
            float,
        )
        self.vertices = corners
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        self.triangles = np.array([t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))])


def box(name: str, lo: list[float], hi: list[float]) -> Box:
    return Box(["robot", name, "body"], np.array(lo, float), np.array(hi, float))


SPEC: dict[str, Any] = {
    "name": "toy",
    "origin_cad_mm": [0, 0, -50],
    "density_g_cm3": 1.0,
    "min_box_mm3": 1,
    "look_triangles": {"default": 1000},
    "hull_triangles": 100,
    "bodies": [
        {
            "name": "wheel",
            "parts": ["wheel:1"],
            "joint": {"through": "wheel:1", "axis_cad": [1, 0, 0]},
            "collide": {"radius_mm": 50, "half_mm": 10},
        },
    ],
}


def solids() -> list[Box]:
    return [
        box("frame:1", [-100, -100, 0], [100, 100, 20]),
        box("wheel:1", [110, -50, -50], [130, 50, 50]),
    ]


def test_cad_axes_become_the_robots() -> None:
    # The CAD's x is the robot's left, its −y forward and its z up.
    assert np.allclose(CAD_TO_ROBOT @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(CAD_TO_ROBOT @ [0, -1, 0], [1, 0, 0])
    assert np.allclose(to_robot(np.array([[0, 0, -50]]), np.array([0, 0, -50])), 0)
    assert np.allclose(to_robot(np.array([[10, -20, 0]]), np.zeros(3)), [[0.02, 0.01, 0]])


def test_assign_puts_unnamed_parts_on_the_chassis() -> None:
    groups = assign(solids(), SPEC)  # type: ignore[arg-type]
    assert [s.path[1] for s in groups["chassis"]] == ["frame:1"]
    assert [s.path[1] for s in groups["wheel"]] == ["wheel:1"]


def test_assign_refuses_a_part_the_cad_lacks() -> None:
    spec = {**SPEC, "bodies": [{**SPEC["bodies"][0], "parts": ["axle:1"]}]}
    with pytest.raises(ValueError, match="axle:1"):
        assign(solids(), spec)  # type: ignore[arg-type]


def test_weld_joins_coincident_corners() -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    t = np.array([[0, 1, 2], [3, 5, 4]])
    wv, wt = weld(v, t, 1e-6)
    assert len(wv) == 4 and len(wt) == 2


def test_model_loads_and_rests_on_its_wheel_and_frame(tmp_path: Path) -> None:
    write_model(solids(), SPEC, tmp_path)  # type: ignore[arg-type]
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    # 1 g/cm³ over a 200 mm × 200 mm × 20 mm frame and a 20 mm × 100 mm × 100 mm wheel.
    assert m.body_mass[1:].sum() == pytest.approx(0.8 + 0.2)
    wheel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "toy_wheel")
    assert np.allclose(m.body_pos[wheel], [0, 0.12, 0.05])
    hinge = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "toy_wheel")
    assert np.allclose(m.jnt_axis[hinge], [0, 1, 0])
