import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from mujocobiobuzz.cad.model import (
    CAD_TO_ROBOT,
    Body,
    Frame,
    _roller_geom,
    assign,
    patches,
    thicken,
    to_robot,
    weld,
    write_model,
)


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


def test_a_field_frame_puts_cad_y_up() -> None:
    # The field's CAD has y up and its origin at the field's centre; the model's origin is a corner.
    f = Frame(np.array([-1828.8, 0, 1828.8]), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    assert np.allclose(f.points(np.array([[0, 1000, 0]])), [[1.8288, 1.8288, 1.0]])


def test_patches_cut_a_bent_sheet_at_its_bend() -> None:
    # Two flat squares meeting at a right angle, two triangles each.
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], float)
    t = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    got = sorted(sorted(p.tolist()) for p in patches(v, t, 2))
    assert got == [[0, 1], [2, 3]]


def test_thicken_gives_a_flat_patch_depth_and_leaves_a_solid_one() -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    tv, tt = thicken(v, np.array([[0, 1, 2]]), 0.001)
    assert len(tv) == 6 and len(tt) == 2
    assert np.ptp(tv[:, 2]) == pytest.approx(0.001)
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    tri = np.array([[0, 1, 2], [4, 5, 6], [0, 4, 1], [3, 7, 2]])
    cv, ct = thicken(cube, tri, 0.001)
    assert len(cv) == 8 and len(ct) == 4


FIXED: dict[str, Any] = {
    **SPEC,
    "name": "gate",
    "base": "fixed",
    "hull_by": "solid",
    "bodies": [
        {
            "name": "flap",
            "parts": ["wheel:1"],
            "joint": {
                "through": "body",
                "axis_cad": [1, 0, 0],
                "range_deg": [-60, 0],
                "stop_solref": [0.05, 1.2],
            },
            "collide": {"hulls": True},
        }
    ],
}


def test_a_fixed_base_holds_still_and_its_hinge_stops(tmp_path: Path) -> None:
    # The flap's bearing reaches 20 mm into the frame, as a hive's does into its A-frame.
    overlapping = [
        box("frame:1", [-100, -100, 0], [100, 100, 20]),
        box("wheel:1", [80, -50, 10], [100, 50, 110]),
    ]
    write_model(overlapping, FIXED, tmp_path)  # type: ignore[arg-type]
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    hinge = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gate_flap")
    assert m.njnt == 1 and m.jnt_limited[hinge]
    assert np.allclose(np.degrees(m.jnt_range[hinge]), [-60, 0])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pairs = {tuple(sorted(int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))) for c in d.contact}
    base, flap = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gate", "gate_flap"))
    assert tuple(sorted((base, flap))) not in pairs


def test_roller_ellipsoid_keeps_the_wheel_round() -> None:
    """A roller 43.4 mm from the hub, 8.6 mm in radius and 23.45 mm in half-length, leaning 45°:
    its ellipsoid is as wide as it, and at the roller's ends still reaches the wheel's radius."""
    c, r0, half = 0.0434, 0.0086, 0.02345
    axis = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
    anchor = np.array([0.0, 0.0, -c])
    s = np.linspace(-half, half, 9)
    ring = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 16)])
    u, v = np.array([1.0, 1.0, 0.0]) / math.sqrt(2), np.array([0.0, 0.0, 1.0])
    pts = np.array([anchor + a * axis + r0 * (x * u + y * v) for a in s for x, y in ring])
    wheel = Body("wheel", axis=np.array([0.0, 1.0, 0.0]))
    g = _roller_geom(pts, anchor, axis, wheel)
    ra, _, a = g["size"]
    assert ra == pytest.approx(r0)
    end = ra * math.sqrt(1 - (half / a) ** 2)
    assert math.hypot(c, half / math.sqrt(2)) + end == pytest.approx(c + r0)
