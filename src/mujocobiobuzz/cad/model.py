"""An assembly's CAD solids grouped into MuJoCo bodies, and written as MJCF and meshes.

Pure but for `write_stl` and `write_model`, which write files. The spec, a JSON file under
`robots/` or `fields/`, names which of the assembly's parts move with each body and how each body
turns; every other part is fixed to the base. A part is a top-level instance of the assembly; a
joint turns through an instance at any depth. A robot's base is free; a field's is fixed to the
world.

Frames: the spec's `axes` turns the CAD's axes into the model's and `origin_cad_mm` is the model's
origin in the CAD, in mm; the model is in m. Left out, `axes` is a goBILDA robot's: the CAD's x
is the robot's left, its y the robot's back and its z up, and the model's x forward, y left, z up.
"""

from __future__ import annotations

import math
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
class Frame:
    """The CAD frame, mm, to the model's, m: `axes` turns, from `origin_mm` in the CAD."""

    origin_mm: NDArray[np.float64]
    axes: NDArray[np.float64]

    @staticmethod
    def of(spec: dict[str, Any]) -> Frame:
        return Frame(
            np.array(spec["origin_cad_mm"], float),
            np.array(spec.get("axes", CAD_TO_ROBOT), float),
        )

    def points(self, p_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Points in the CAD frame, mm, in the model's, m; rows are points."""
        return np.asarray((np.asarray(p_mm) - self.origin_mm) @ self.axes.T * MM)


@dataclass
class Body:
    """A body MuJoCo moves: its solids, its joint and its collision shapes, in the model frame."""

    name: str
    solids: list[SolidLike] = field(default_factory=list)
    anchor: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    axis: NDArray[np.float64] | None = None
    joint: dict[str, Any] = field(default_factory=dict)
    geoms: list[dict[str, Any]] = field(default_factory=list)
    looks: list[str] = field(default_factory=list)
    children: list[Body] = field(default_factory=list)
    look_triangles: int | None = None


def to_robot(p_mm: NDArray[np.float64], origin_mm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Points in a goBILDA robot's CAD frame, mm, in the robot frame, m; rows are points."""
    return Frame(origin_mm, CAD_TO_ROBOT).points(p_mm)


def top_part(solid: SolidLike) -> str:
    """The top-level instance a solid belongs to: the second name in its path."""
    return solid.path[1] if len(solid.path) > 1 else solid.path[0]


def assign(solids: list[SolidLike], spec: dict[str, Any]) -> dict[str, list[SolidLike]]:
    """Each solid to the body whose `parts` names its top-level instance, else to the base;
    solids under `leave_out` to none."""
    owner = {p: b["name"] for b in spec["bodies"] for p in b["parts"]}
    out = set(spec.get("leave_out", []))
    groups: dict[str, list[SolidLike]] = {"chassis": []}
    for b in spec["bodies"]:
        groups[b["name"]] = []
    for s in solids:
        if top_part(s) not in out:
            groups[owner.get(top_part(s), "chassis")].append(s)
    missing = [p for p in [*owner, *out] if not any(top_part(s) == p for s in solids)]
    if missing:
        raise ValueError(f"spec names parts the CAD does not have: {missing}")
    return groups


def mass_and_inertia(
    solids: list[SolidLike], density_g_cm3: float, frame: Frame
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Mass in kg, centre of mass and inertia about it in the model frame, each solid taken as a
    uniform box of its volume's mass filling its bounding box."""
    m = np.array([s.volume_mm3 * 1e-3 * density_g_cm3 * 1e-3 for s in solids])
    c = frame.points(np.array([s.centroid for s in solids]))
    total = float(m.sum())
    com = (m[:, None] * c).sum(0) / total
    inertia = np.zeros((3, 3))
    for mi, ci, s in zip(m, c, solids, strict=True):
        d = np.abs((s.hi - s.lo) @ frame.axes.T) * MM
        sq = d**2
        own = mi / 12 * np.diag([sq[1] + sq[2], sq[0] + sq[2], sq[0] + sq[1]])
        r = ci - com
        inertia += own + mi * (r @ r * np.eye(3) - np.outer(r, r))
    return total, com, inertia


def hull_groups(solids: list[SolidLike], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The solids that collide, grouped for one convex hull each: by top-level part, or, with
    `hull_by` "solid", each solid alone. Parts under `no_collision` and groups under
    `min_box_mm3` are left out."""
    drop = set(spec.get("no_collision", []))
    by: dict[str, list[SolidLike]] = {}
    for s in solids:
        if top_part(s) not in drop:
            key = "/".join(s.path) if spec.get("hull_by") == "solid" else top_part(s)
            by.setdefault(key, []).append(s)
    return [
        dict(type="hull", part=k, solids=ss)
        for k, ss in sorted(by.items())
        if sum(s.volume_mm3 for s in ss) >= spec["min_box_mm3"]
    ]


def pieces(group: dict[str, Any], spec: dict[str, Any]) -> int:
    """How many convex pieces a hull group is cut into: one, unless its box holds more than
    `split_ratio` times its volume, as a bent sheet's does; then one for each `split_ratio`,
    to `max_pieces`."""
    ratio = spec.get("split_ratio")
    if ratio is None:
        return 1
    ss = group["solids"]
    box = float(np.prod(np.max([s.hi for s in ss], 0) - np.min([s.lo for s in ss], 0)))
    vol = sum(s.volume_mm3 for s in ss)
    return int(min(spec.get("max_pieces", 16), max(1, math.ceil(box / vol / ratio))))


def patches(verts: NDArray[np.float64], tris: NDArray[np.int64], k: int) -> list[NDArray[np.int64]]:
    """The triangles cut into up to `k` patches by their centres, nearest first: seeds taken
    farthest first, then moved to their patches' means. Each patch is a list of triangle rows."""
    c = verts[tris].mean(1)
    seeds = [c[0]]
    for _ in range(1, min(k, len(c))):
        d = np.min([np.sum((c - s) ** 2, 1) for s in seeds], 0)
        seeds.append(c[int(np.argmax(d))])
    sv = np.array(seeds)
    for _ in range(10):
        lab = np.argmin(((c[:, None, :] - sv[None]) ** 2).sum(2), 1)
        sv = np.array([c[lab == i].mean(0) if np.any(lab == i) else sv[i] for i in range(len(sv))])
    lab = np.argmin(((c[:, None, :] - sv[None]) ** 2).sum(2), 1)
    return [np.flatnonzero(lab == i) for i in range(len(sv)) if np.any(lab == i)]


# A patch thinner than this, m, is flat to MuJoCo, which cannot hull it; it is given this depth.
FLAT_M = 0.001


def thicken(
    verts: NDArray[np.float64], tris: NDArray[np.int64], depth: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The patch's own vertices and triangles; where they lie flat, within `depth`, a copy of
    them `depth` behind, so the patch has a hull."""
    used, inv = np.unique(tris, return_inverse=True)
    v, t = verts[used], inv.reshape(tris.shape)
    c = v - v.mean(0)
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    if len(sv) == 3 and sv[2] / math.sqrt(len(v)) >= depth:
        return v, t
    return np.vstack([v, v - vt[-1] * depth]), np.vstack([t, t[:, ::-1] + len(v)])


def build(solids: list[SolidLike], spec: dict[str, Any]) -> list[Body]:
    """The base and the spec's bodies, each with its anchor, axis, joint and collision geoms."""
    frame = Frame.of(spec)
    groups = assign(solids, spec)
    base = Body("chassis", groups["chassis"])
    base.geoms = hull_groups(base.solids, spec)
    bodies = [base]
    for b in spec["bodies"]:
        body = Body(b["name"], groups[b["name"]], joint=b["joint"])
        through = [s for s in body.solids if b["joint"]["through"] in s.path]
        vol = np.array([t.volume_mm3 for t in through])
        body.anchor = frame.points(
            (vol[:, None] * np.array([t.centroid for t in through])).sum(0) / vol.sum()
        )
        body.axis = frame.axes @ np.array(b["joint"]["axis_cad"], float)
        g = b["collide"]
        if "rollers" in g:
            body.solids, body.children = _rollers(body, g, spec, frame)
        elif g.get("hulls"):
            body.geoms = hull_groups(body.solids, spec)
        else:
            body.geoms.append(
                dict(
                    type="cylinder",
                    pos=body.anchor,
                    axis=body.axis,
                    size=np.array([g["radius_mm"] * MM, g["half_mm"] * MM]),
                    **({"condim": g["condim"]} if "condim" in g else {}),
                )
            )
        bodies.append(body)
    return bodies


def _rollers(
    body: Body, g: dict[str, Any], spec: dict[str, Any], frame: Frame
) -> tuple[list[SolidLike], list[Body]]:
    """A mecanum wheel's rollers, the instances inside the assembly `g["rollers"]`, each a body
    of its own on a free hinge along its length, its principal axis, through its centre; and the
    wheel's other solids. A roller collides as an ellipsoid as wide as it, long enough that the
    wheel's outline is a circle at its middle and at its ends, so the wheel rolls without a
    bump; and with the floor and the balls, not with its neighbours."""
    under = g["rollers"]
    by: dict[str, list[SolidLike]] = {}
    rest = []
    for s in body.solids:
        i = s.path.index(under) if under in s.path else -1
        if 0 <= i < len(s.path) - 1:
            by.setdefault(s.path[i + 1], []).append(s)
        else:
            rest.append(s)
    children = []
    for k, ss in sorted(by.items()):
        name = f"{body.name}_" + "".join(c if c.isalnum() else "_" for c in k)
        pts = frame.points(np.vstack([s.vertices for s in ss]))
        vol = np.array([s.volume_mm3 for s in ss])
        anchor = frame.points(
            (vol[:, None] * np.array([s.centroid for s in ss])).sum(0) / vol.sum()
        )
        axis = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0]
        child = Body(name, ss, anchor, axis, {"damping": g["roller_damping"]})
        child.geoms = [_roller_geom(pts, anchor, axis, body)]
        child.look_triangles = g["roller_look_triangles"]
        children.append(child)
    return rest, children


def _roller_geom(
    pts: NDArray[np.float64], anchor: NDArray[np.float64], axis: NDArray[np.float64], wheel: Body
) -> dict[str, Any]:
    assert wheel.axis is not None
    rel = pts - anchor
    along = rel @ axis
    half = float(np.abs(along).max())
    r0 = float(np.linalg.norm(rel - np.outer(along, axis), axis=1).max())
    out = anchor - wheel.anchor
    out -= (out @ wheel.axis) * wheel.axis
    c = float(np.linalg.norm(out))
    # The roller's axis leans across the wheel by `t`; its end sits sqrt(c² + (half t)²) from
    # the hub, and the ellipsoid's radius there makes up the wheel's radius, c + r0.
    t = abs(float(axis @ np.cross(wheel.axis, out / c)))
    r_end = c + r0 - math.hypot(c, half * t)
    a = half / math.sqrt(1 - (r_end / r0) ** 2)
    return dict(type="ellipsoid", pos=anchor, axis=axis, size=np.array([r0, r0, a]))


def write_stl(path: Path, verts: NDArray[np.float64], tris: NDArray[np.int64]) -> None:
    """A binary STL of the triangles."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            a, b, c = verts[t]
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


# The model lit, on a floor if it has none, for a look or a drop: MuJoCo reads meshes from the
# folder of the file it is given, so the scene sits beside the model's file.
SCENE = """<mujoco model="{name} scene">
  <include file="{name}.xml"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" directional="true"/>
    <light pos="-1 -1 2" dir="0.3 0.3 -1" directional="true" diffuse="0.4 0.4 0.4"/>
{floor}  </worldbody>
</mujoco>
"""
FLOOR = '    <geom name="floor" type="plane" size="2 2 0.01" rgba="0.3 0.3 0.32 1"/>\n'

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


def _geom(g: dict[str, Any], rel: NDArray[np.float64]) -> list[str]:
    """The geom's MJCF; a hull cut into pieces gives a line a piece."""
    if g["type"] == "hull":
        return [f'<geom type="mesh" mesh="{m}" class="collision"/>' for m in g["meshes"]]
    pos = np.asarray(g["pos"]) - rel
    extra = ""
    if g["type"] == "cylinder":
        a = np.asarray(g["axis"])
        h = g["size"][1]
        extra = f' fromto="{_v(pos - a * h)} {_v(pos + a * h)}"'
        if "condim" in g:
            extra += f' condim="{g["condim"]}"'
        return [f'<geom type="cylinder" size="{g["size"][0]:.6g}"{extra} class="collision"/>']
    if g["type"] == "ellipsoid":
        return [
            f'<geom type="ellipsoid" pos="{_v(pos)}" zaxis="{_v(g["axis"])}" '
            f'size="{_v(g["size"])}" contype="2" conaffinity="1" class="collision"/>'
        ]
    if g["type"] == "plane":
        return [f'<geom type="plane" pos="{_v(pos)}" size="{_v(g["size"])}" class="collision"/>']
    return [f'<geom type="box" pos="{_v(pos)}" size="{_v(g["size"])}" class="collision"/>']


def _lines(indent: str, geoms: list[dict[str, Any]], rel: NDArray[np.float64]) -> list[str]:
    return [indent + line for g in geoms for line in _geom(g, rel)]


def _joint(name: str, b: Body) -> str:
    j = b.joint
    extra = ""
    if "range_deg" in j:
        lo, hi = sorted(math.radians(a) for a in j["range_deg"])
        extra += f' limited="true" range="{lo:.6g} {hi:.6g}"'
    if "stop_solref" in j:
        extra += f' solreflimit="{_v(j["stop_solref"])}"'
    return (
        f'<joint name="{name}" type="hinge" axis="{_v(b.axis)}" '
        f'damping="{j.get("damping", 0.001):.6g}"{extra}/>'
    )


def _motor(joint: str, mo: dict[str, Any]) -> str:
    """A DC motor or a continuous servo on the joint, its control the power, −1 to 1: its torque
    is the stall torque at full power, falling to none at its free speed."""
    stall = mo["stall_kg_cm"] * 9.80665 / 100
    free = mo["free_rpm"] * 2 * math.pi / 60
    return (
        f'    <general name="{joint}" joint="{joint}" ctrlrange="-1 1" ctrllimited="true" '
        f'gainprm="{stall:.6g}" biastype="affine" biasprm="0 0 {-stall / free:.6g}"/>'
    )


def _meshes(b: Body) -> list[str]:
    return [m for g in b.geoms if g["type"] == "hull" for m in g["meshes"]]


def _body(
    name: str, b: Body, parent: NDArray[np.float64], indent: str, masses: dict[str, Any]
) -> list[str]:
    """A hinged body's MJCF, placed relative to its parent's anchor, with its own children."""
    m, com, inertia = masses[b.name]
    i = indent + "  "
    out = [
        f'{indent}<body name="{name}_{b.name}" pos="{_v(b.anchor - parent)}">',
        i + _joint(f"{name}_{b.name}", b),
        f'{i}<inertial pos="{_v(com - b.anchor)}" mass="{m:.6g}" fullinertia='
        f'"{_v(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])}"/>',
    ]
    out += [f'{i}<geom type="mesh" mesh="{lk}" class="visual"/>' for lk in b.looks]
    out += _lines(i, b.geoms, b.anchor)
    for c in b.children:
        out += _body(name, c, b.anchor, i, masses)
    out.append(f"{indent}</body>")
    return out


def _all(bodies: list[Body]) -> list[Body]:
    """The bodies and, after each, its children's, depth first."""
    return [x for b in bodies for x in [b, *_all(b.children)]]


def mjcf(bodies: list[Body], spec: dict[str, Any], masses: dict[str, Any]) -> str:
    """The model's MJCF: the base on a free joint, or fixed, each other body hinged to it."""
    name = spec["name"]
    out = [
        f'<mujoco model="{name}">',
        '  <compiler angle="radian" meshdir="meshes"/>',
        # A motor's back-EMF is a damping large for a light body's inertia; implicitfast takes it
        # implicitly, where the default integrator goes unstable.
        '  <option integrator="implicitfast"/>',
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
    out += [f'    <mesh name="{m}" file="{m}.stl"/>' for b in _all(bodies) for m in b.looks]
    out += [f'    <mesh name="{m}" file="{m}.stl"/>' for b in _all(bodies) for m in _meshes(b)]
    out += [
        "  </asset>",
        "  <worldbody>",
        f'    <body name="{name}" pos="0 0 0">',
    ]
    if spec.get("base", "free") == "free":
        out.append("      <freejoint/>")
    m, com, inertia = masses["chassis"]
    out.append(
        f'      <inertial pos="{_v(com)}" mass="{m:.6g}" '
        f'fullinertia="{_v(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])}"/>'
    )
    out += [f'      <geom type="mesh" mesh="{m}" class="visual"/>' for m in bodies[0].looks]
    out += _lines("      ", bodies[0].geoms, np.zeros(3))
    for b in bodies[1:]:
        out += _body(name, b, np.zeros(3), "      ", masses)
    out += ["    </body>", "  </worldbody>"]
    motors = [(b, j["motor"]) for b in bodies[1:] if "motor" in (j := b.joint)]
    if motors:
        out.append("  <actuator>")
        out += [_motor(f"{name}_{b.name}", mo) for b, mo in motors]
        out.append("  </actuator>")
    if spec.get("base", "free") != "free":
        # MuJoCo leaves out a body's contacts with its parent, but not where the parent is welded
        # to the world, as a fixed base is; a hinged body's bearings sit inside the base's hulls.
        out.append("  <contact>")
        out += [f'    <exclude body1="{name}" body2="{name}_{b.name}"/>' for b in bodies[1:]]
        out.append("  </contact>")
    out += ["</mujoco>", ""]
    return "\n".join(out)


def _mesh_of(
    solids: list[SolidLike], frame: Frame, rel: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    verts, tris, base = [], [], 0
    for s in solids:
        verts.append(frame.points(s.vertices) - rel)
        tris.append(s.triangles + base)
        base += len(s.vertices)
    return weld(np.vstack(verts), np.vstack(tris), 1e-5)


def _write_hulls(
    b: Body, spec: dict[str, Any], frame: Frame, out: Path, reduce: Reduce | None
) -> None:
    """Each hull group's mesh, cut into its pieces, one STL a piece, relative to the body."""
    for g in b.geoms:
        if g["type"] != "hull":
            continue
        name = _hull_name(spec, g["part"])
        v, t = _mesh_of(g["solids"], frame, b.anchor)
        k = pieces(g, spec)
        if reduce is not None and len(t) > spec["hull_triangles"] * k:
            v, t = reduce(v, t, spec["hull_triangles"] * k)
        g["meshes"] = []
        for i, rows in enumerate(patches(v, t, k) if k > 1 else [np.arange(len(t))]):
            if len(np.unique(t[rows])) < 3:
                continue
            m = name if k == 1 else f"{name}_{i}"
            write_stl(out / "meshes" / f"{m}.stl", *thicken(v, t[rows], FLAT_M))
            g["meshes"].append(m)


def _looks(shown: list[SolidLike], b: Body, spec: dict[str, Any]) -> dict[str, list[SolidLike]]:
    """The body's look meshes, each named for its `look_triangles` budget: one for the body, or,
    for the base with `look_by` "part", one for each top-level part."""
    if b.name != "chassis" or spec.get("look_by") != "part":
        return {b.name: shown}
    by: dict[str, list[SolidLike]] = {}
    for s in shown:
        key = "".join(c if c.isalnum() else "_" for c in top_part(s))
        by.setdefault(key, []).append(s)
    return by


def _floor(spec: dict[str, Any]) -> dict[str, Any] | None:
    f = spec.get("floor")
    if f is None:
        return None
    return dict(type="plane", pos=np.array(f["centre_m"], float), size=[*f["half_m"], 0.01])


def write_model(
    solids: list[SolidLike], spec: dict[str, Any], out: Path, reduce: Reduce | None = None
) -> dict[str, Any]:
    """The model's MJCF, its meshes and a summary of each body, written under `out`. Each body's
    look is welded and, given `reduce`, cut to the spec's `look_triangles` for it."""
    frame = Frame.of(spec)
    bodies = build(solids, spec)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    look_drop = set(spec.get("no_look", []))
    masses: dict[str, Any] = {}
    summary: dict[str, Any] = {"name": spec["name"], "bodies": {}}
    for b in _all(bodies):
        _write_hulls(b, spec, frame, out, reduce)
        own: dict[str, Any] = next((x for x in spec["bodies"] if x["name"] == b.name), {})
        density = own.get("density_g_cm3", spec["density_g_cm3"])
        masses[b.name] = mass_and_inertia(b.solids, density, frame)
        shown = [s for s in b.solids if top_part(s) not in look_drop and len(s.triangles)]
        triangles = 0
        for key, ss in _looks(shown, b, spec).items():
            v, t = _mesh_of(ss, frame, b.anchor)
            budget = b.look_triangles or spec["look_triangles"].get(
                key, spec["look_triangles"]["default"]
            )
            if reduce is not None and len(t) > budget:
                v, t = reduce(v, t, budget)
            m = f"{spec['name']}_{key}"
            write_stl(out / "meshes" / f"{m}.stl", v, t)
            b.looks.append(m)
            triangles += len(t)
        summary["bodies"][b.name] = dict(
            solids=len(b.solids),
            mass_kg=round(masses[b.name][0], 4),
            triangles=triangles,
            collision_geoms=sum(len(g.get("meshes", [0])) for g in b.geoms),
        )
    floor = _floor(spec)
    if floor is not None:
        bodies[0].geoms.append(floor)
    (out / f"{spec['name']}.xml").write_text(mjcf(bodies, spec, masses))
    (out / "scene.xml").write_text(
        SCENE.format(name=spec["name"], floor="" if floor is not None else FLOOR)
    )
    return summary
