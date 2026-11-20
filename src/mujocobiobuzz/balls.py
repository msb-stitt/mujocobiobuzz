"""POLLEN and NECTAR as MuJoCo bodies, and a scene that adds them to a model.

The sizes and masses are AndyMark's drawings am-5851 and am-5852, as `ftc-claude`'s floor plan,
`.docs/2026-11-18-waggle-floor-count.md`, gives them under `floor.physics.balls`; each ball is a
hollow shell. The drag is quadratic, C_d 0.30 on the ball's cross-section in still air, through
MuJoCo's ellipsoid fluid model with its lift left out. A bounce off the floor keeps `RESTITUTION`
of the speed into it, the floor plan's `floor.physics.bounce`, by the contact's damping ratio,
`DAMPRATIO`, fitted to it by a drop.
"""

from __future__ import annotations

from dataclasses import dataclass

IN = 0.0254
LB = 0.45359237
AIR_KG_M3 = 1.204
DRAG_CD = 0.30
RESTITUTION = 0.6
# The step, s, and the contact's time constant, s, ten steps, as a shorter one gains energy at a
# bounce; and the damping ratio a drop from 0.5 m onto a plane found to keep `RESTITUTION` of the
# speed. From 0.1 m to 1.5 m it keeps 0.60 to 0.68, as `tests/test_balls.py` checks.
TIMESTEP = 0.001
TIMECONST = 0.01
DAMPRATIO = 0.162
FRICTION = 0.5


@dataclass(frozen=True)
class Kind:
    """A ball type: its class name, radius, mass and colour."""

    name: str
    radius_m: float
    mass_kg: float
    rgba: str


POLLEN = Kind("pollen", 1.4 * IN, 0.055 * LB, "0.95 0.85 0.1 1")
BLUE_NECTAR = Kind("blue_nectar", 1.8 * IN, 0.091 * LB, "0.1 0.3 0.9 1")
RED_NECTAR = Kind("red_nectar", 1.8 * IN, 0.091 * LB, "0.85 0.1 0.1 1")
KINDS = (POLLEN, BLUE_NECTAR, RED_NECTAR)


def defaults() -> str:
    """The balls' default classes. MuJoCo's blunt drag is its coefficient times the air's density,
    the projected area and the speed squared, so it is half of C_d."""
    out = ["  <default>"]
    for k in KINDS:
        out += [
            f'    <default class="{k.name}">',
            f'      <geom type="sphere" size="{k.radius_m:.6g}" mass="{k.mass_kg:.6g}" '
            f'shellinertia="true" rgba="{k.rgba}" priority="1" friction="{FRICTION} 0.005 0.0001" '
            f'solref="{TIMECONST} {DAMPRATIO}" fluidshape="ellipsoid" '
            f'fluidcoef="{DRAG_CD / 2:.6g} 0.25 1.5 0 0"/>',
            "    </default>",
        ]
    out.append("  </default>")
    return "\n".join(out)


def body(kind: Kind, name: str, pos: tuple[float, float, float]) -> str:
    """A ball on a free joint at `pos`, m."""
    p = " ".join(f"{v:.6g}" for v in pos)
    return f'    <body name="{name}" pos="{p}"><freejoint/><geom class="{kind.name}"/></body>'


def scene(model: str, balls: list[tuple[Kind, str, tuple[float, float, float]]]) -> str:
    """A scene of the model, by its MJCF file's name beside the scene, lit, in still air, with
    the balls, stepped at `TIMESTEP`. The scene is written in the model's folder, as MuJoCo reads
    meshes from there."""
    return "\n".join(
        [
            f'<mujoco model="{model} with balls">',
            f'  <include file="{model}.xml"/>',
            f'  <option timestep="{TIMESTEP}" density="{AIR_KG_M3}"/>',
            '  <visual><global offwidth="1280" offheight="960"/></visual>',
            defaults(),
            "  <worldbody>",
            '    <light pos="1 1 3" dir="-0.3 -0.3 -1" directional="true"/>',
            *[body(k, n, p) for k, n, p in balls],
            "  </worldbody>",
            "</mujoco>",
            "",
        ]
    )
