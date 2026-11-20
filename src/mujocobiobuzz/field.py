"""The BIOBUZZ field with robots on it, each at a pose in the field's frame.

The field's frame is Pedro Pathing's, in m, from a corner; a robot's pose is its x and y there and
its heading, counterclockwise from x. Each robot is the StarterBot model under a prefix, its
motors named as biobuzz names them after it.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

MODELS = Path(__file__).parent.parent.parent / "models"
ROBOT = "starterbot"
# A robot's look, so it stands out on the grey field.
ROBOT_RGBA = [0.95, 0.55, 0.1, 1.0]


def with_robots(
    robots: dict[str, tuple[float, float, float]], models: Path = MODELS
) -> mujoco.MjModel:
    """The field, lit, with each robot named by its prefix at its pose, (x m, y m, heading
    rad)."""
    field = mujoco.MjSpec.from_file(str(models / "biobuzz" / "biobuzz.xml"))
    field.visual.global_.offwidth, field.visual.global_.offheight = 1920, 1080
    light = field.worldbody.add_light(pos=[1.83, 1.83, 4], dir=[0.2, 0.3, -1])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for prefix, (x, y, h) in robots.items():
        robot = mujoco.MjSpec.from_file(str(models / ROBOT / f"{ROBOT}.xml"))
        for g in robot.geoms:
            if g.classname.name == "visual":
                g.rgba = ROBOT_RGBA
        frame = field.worldbody.add_frame(
            pos=[x, y, 0.0], quat=[math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)]
        )
        frame.attach_body(robot.body(ROBOT), prefix, "")
    return field.compile()
