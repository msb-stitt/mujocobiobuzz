"""A robot's CAD solids grouped into MuJoCo bodies, and written as MJCF and meshes.

Pure but for `write_stl` and `write_model`, which write files. The spec, a JSON file under
`robots/`, names which of the assembly's top-level parts move with each body and how each body
turns; every other part is fixed to the chassis.

Frames: the CAD's x is the robot's left, its y the robot's back and its z up, in mm. The robot's
frame is MuJoCo's: x forward, y left, z up, in m, from the spec's origin on the floor.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

# CAD axes to robot axes: robot x = −CAD y, robot y = CAD x, robot z = CAD z.
CAD_TO_ROBOT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MM = 0.001


class SolidLike(Protocol):
    path: list[str]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    volume_mm3: float
    centroid: NDArray[np.float64]
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int64]


@dataclass
class Body:
    """A body MuJoCo moves: its solids, its joint and its collision shapes, in the robot frame."""

    name: str
    solids: list[SolidLike] = field(default_factory=list)
    anchor: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    axis: NDArray[np.float64] | None = None
    geoms: list[dict[str, Any]] = field(default_factory=list)


def to_robot(p_mm: NDArray[np.float64], origin_mm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Points in the CAD frame, mm, as points in the robot frame, m; rows are points."""
    return np.asarray((np.asarray(p_mm) - origin_mm) @ CAD_TO_ROBOT.T * MM)


def top_part(solid: SolidLike) -> str:
    """The top-level instance a solid belongs to: the second name in its path."""
    return solid.path[1] if len(solid.path) > 1 else solid.path[0]


def assign(solids: list[SolidLike], spec: dict[str, Any]) -> dict[str, list[SolidLike]]:
    """Each solid to the body whose `parts` names its top-level instance, else to the chassis."""
    owner = {p: b["name"] for b in spec["bodies"] for p in b["parts"]}
    groups: dict[str, list[SolidLike]] = {"chassis": []}
    for b in spec["bodies"]:
        groups[b["name"]] = []
    for s in solids:
        groups[owner.get(top_part(s), "chassis")].append(s)
    missing = [p for p in owner if not any(top_part(s) == p for s in solids)]
    if missing:
        raise ValueError(f"spec names parts the CAD does not have: {missing}")
    return groups


def mass_and_inertia(
    solids: list[SolidLike], density_g_cm3: float, origin_mm: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Mass in kg, centre of mass and inertia about it in the robot frame, each solid taken as a
    uniform box of its volume's mass filling its bounding box."""
    m = np.array([s.volume_mm3 * 1e-3 * density_g_cm3 * 1e-3 for s in solids])
    c = to_robot(np.array([s.centroid for s in solids]), origin_mm)
    total = float(m.sum())
    com = (m[:, None] * c).sum(0) / total
    inertia = np.zeros((3, 3))
    for mi, ci, s in zip(m, c, solids, strict=True):
        d = np.abs((s.hi - s.lo) @ CAD_TO_ROBOT.T) * MM
        sq = d**2
        own = mi / 12 * np.diag([sq[1] + sq[2], sq[0] + sq[2], sq[0] + sq[1]])
        r = ci - com
        inertia += own + mi * (r @ r * np.eye(3) - np.outer(r, r))
    return total, com, inertia


def box_of(solids: list[SolidLike], origin_mm: NDArray[np.float64]) -> tuple[Any, Any]:
    """The robot-frame centre and half sizes, m, of the solids' joint bounding box."""
    lo = np.min([s.lo for s in solids], 0)
    hi = np.max([s.hi for s in solids], 0)
    corners = to_robot(np.array([lo, hi]), origin_mm)
    return corners.mean(0), np.abs(corners[1] - corners[0]) / 2


def build(solids: list[SolidLike], spec: dict[str, Any]) -> list[Body]:
    """The chassis and the spec's bodies, each with its anchor, axis and collision geoms."""
    origin = np.array(spec["origin_cad_mm"], float)
    groups = assign(solids, spec)
    drop = set(spec.get("no_collision", []))
    chassis = Body("chassis", groups["chassis"])
    by_part: dict[str, list[SolidLike]] = {}
    for s in groups["chassis"]:
        by_part.setdefault(top_part(s), []).append(s)
    for part, ss in sorted(by_part.items()):
        if part in drop or sum(s.volume_mm3 for s in ss) < spec["min_box_mm3"]:
            continue
        chassis.geoms.append(dict(type="hull", part=part, solids=ss))
    bodies = [chassis]
    for b in spec["bodies"]:
        body = Body(b["name"], groups[b["name"]])
        through = [s for s in body.solids if top_part(s) == b["joint"]["through"]]
        vol = np.array([t.volume_mm3 for t in through])
        body.anchor = to_robot(
            (vol[:, None] * np.array([t.centroid for t in through])).sum(0) / vol.sum(), origin
        )
        body.axis = CAD_TO_ROBOT @ np.array(b["joint"]["axis_cad"], float)
        g = b["collide"]
        body.geoms.append(
            dict(
                type="cylinder",
                pos=body.anchor,
                axis=body.axis,
                size=np.array([g["radius_mm"] * MM, g["half_mm"] * MM]),
            )
        )
        bodies.append(body)
    return bodies


def write_stl(path: Path, verts: NDArray[np.float64], tris: NDArray[np.int64]) -> None:
    """A binary STL of the triangles."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            a, b, c = verts[t]
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


# The robot alone on a floor, lit, for a look or a drop: MuJoCo reads meshes from the folder of
# the file it is given, so the scene sits beside the robot's file.
SCENE = """<mujoco model="{name} on a floor">
  <include file="{name}.xml"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" directional="true"/>
    <light pos="-1 -1 2" dir="0.3 0.3 -1" directional="true" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="2 2 0.01" rgba="0.3 0.3 0.32 1"/>
  </worldbody>
</mujoco>
"""

Reduce = Callable[
    [NDArray[np.float64], NDArray[np.int64], int], tuple[NDArray[np.float64], NDArray[np.int64]]
]


def weld(
    verts: NDArray[np.float64], tris: NDArray[np.int64], tol: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The mesh with vertices within `tol` of each other made one, and triangles left with two
    corners the same dropped."""
    keys = np.round(verts / tol).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    t = inverse.reshape(-1)[tris]
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return verts[first], t[ok]


def _v(x: Any) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.ravel(x))


def _hull_name(spec: dict[str, Any], part: str) -> str:
    return spec["name"] + "_hull_" + "".join(c if c.isalnum() else "_" for c in part)


def _geom(g: dict[str, Any], rel: NDArray[np.float64]) -> str:
    if g["type"] == "hull":
        return f'<geom type="mesh" mesh="{g["mesh"]}" class="collision"/>'
    pos = np.asarray(g["pos"]) - rel
    extra = ""
    if g["type"] == "cylinder":
        a = np.asarray(g["axis"])
        h = g["size"][1]
        extra = f' fromto="{_v(pos - a * h)} {_v(pos + a * h)}"'
        return f'<geom type="cylinder" size="{g["size"][0]:.6g}"{extra} class="collision"/>'
    return f'<geom type="box" pos="{_v(pos)}" size="{_v(g["size"])}" class="collision"/>'


def mjcf(bodies: list[Body], spec: dict[str, Any], masses: dict[str, Any]) -> str:
    """The robot's MJCF: the chassis on a free joint, each other body hinged to it."""
    name = spec["name"]
    out = [
        f'<mujoco model="{name}">',
        '  <compiler angle="radian" meshdir="meshes"/>',
        "  <default>",
        '    <default class="visual">',
        '      <geom contype="0" conaffinity="0" group="1" rgba="0.75 0.75 0.78 1"/>',
        "    </default>",
        '    <default class="collision">',
        '      <geom group="3" rgba="0.9 0.4 0.1 0.4" friction="1 0.005 0.0001"/>',
        "    </default>",
        "  </default>",
        "  <asset>",
    ]
    out += [f'    <mesh name="{name}_{b.name}" file="{name}_{b.name}.stl"/>' for b in bodies]
    out += [
        f'    <mesh name="{g["mesh"]}" file="{g["mesh"]}.stl"/>'
        for g in bodies[0].geoms
        if g["type"] == "hull"
    ]
    out += [
        "  </asset>",
        "  <worldbody>",
        f'    <body name="{name}" pos="0 0 0">',
        "      <freejoint/>",
    ]
    m, com, inertia = masses["chassis"]
    out.append(
        f'      <inertial pos="{_v(com)}" mass="{m:.6g}" '
        f'fullinertia="{_v(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])}"/>'
    )
    out.append(f'      <geom type="mesh" mesh="{name}_chassis" class="visual"/>')
    out += ["      " + _geom(g, np.zeros(3)) for g in bodies[0].geoms]
    for b in bodies[1:]:
        m, com, inertia = masses[b.name]
        out += [
            f'      <body name="{name}_{b.name}" pos="{_v(b.anchor)}">',
            f'        <joint name="{name}_{b.name}" type="hinge" axis="{_v(b.axis)}" '
            'damping="0.001"/>',
            f'        <inertial pos="{_v(com - b.anchor)}" mass="{m:.6g}" fullinertia='
            f'"{_v(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])}"/>',
            f'        <geom type="mesh" mesh="{name}_{b.name}" class="visual"/>',
        ]
        out += ["        " + _geom(g, b.anchor) for g in b.geoms]
        out.append("      </body>")
    out += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    return "\n".join(out)


def write_model(
    solids: list[SolidLike], spec: dict[str, Any], out: Path, reduce: Reduce | None = None
) -> dict[str, Any]:
    """The robot's MJCF, its meshes and a summary of each body, written under `out`. Each body's
    look is welded and, given `reduce`, cut to the spec's `look_triangles` for it."""
    origin = np.array(spec["origin_cad_mm"], float)
    bodies = build(solids, spec)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    look_drop = set(spec.get("no_look", []))
    masses: dict[str, Any] = {}
    summary: dict[str, Any] = {"name": spec["name"], "bodies": {}}
    for g in bodies[0].geoms:
        if g["type"] != "hull":
            continue
        g["mesh"] = _hull_name(spec, g["part"])
        verts = [to_robot(s.vertices, origin) for s in g["solids"]]
        tris, base = [], 0
        for s, v in zip(g["solids"], verts, strict=True):
            tris.append(s.triangles + base)
            base += len(v)
        v, t = weld(np.vstack(verts), np.vstack(tris), 1e-5)
        if reduce is not None and len(t) > spec["hull_triangles"]:
            v, t = reduce(v, t, spec["hull_triangles"])
        write_stl(out / "meshes" / f"{g['mesh']}.stl", v, t)
    for b in bodies:
        masses[b.name] = mass_and_inertia(b.solids, spec["density_g_cm3"], origin)
        shown = [s for s in b.solids if top_part(s) not in look_drop and len(s.triangles)]
        verts, tris, base = [], [], 0
        for s in shown:
            verts.append(to_robot(s.vertices, origin) - b.anchor)
            tris.append(s.triangles + base)
            base += len(s.vertices)
        v, t = weld(np.vstack(verts), np.vstack(tris), 1e-5)
        budget = spec["look_triangles"].get(b.name, spec["look_triangles"]["default"])
        if reduce is not None and len(t) > budget:
            v, t = reduce(v, t, budget)
        write_stl(out / "meshes" / f"{spec['name']}_{b.name}.stl", v, t)
        summary["bodies"][b.name] = dict(
            solids=len(b.solids),
            mass_kg=round(masses[b.name][0], 4),
            triangles=len(t),
            collision_geoms=len(b.geoms),
        )
    (out / f"{spec['name']}.xml").write_text(mjcf(bodies, spec, masses))
    (out / "scene.xml").write_text(SCENE.format(name=spec["name"]))
    return summary
