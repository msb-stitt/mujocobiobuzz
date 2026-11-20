"""The robot in MuJoCo, driven from its own flight log, a WPILOG file `corbelsflightlog` wrote.

The log's powers set the model's motors as `mujocobiobuzz.nt` sets them from `SimRun`: each
held from its record until the next. The robot's own velocity, from its Pinpoint, is read beside
them, so a replay can be put beside what the robot did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from wpiutil.log import DataLogReader

from mujocobiobuzz import nt

IN = 0.0254
# Each log topic that carries a power, and the `SimRun` topic `nt.controls` reads it as. The
# drivetrain logs the powers it wrote, before a motor's direction, as `SimRun` publishes them.
POWERS = {f"/wheel/{w}/speed": f"sim/wheels/{w}" for w in nt.WHEELS} | {
    t: t for t in ("/intake/speed", nt.FEEDING, nt.TARGET_TPS)
}
# The robot's velocity along itself and its turn, as its localizer logged them.
VELOCITY = ("/Robot/vel/forward_ips", "/Robot/vel/strafe_ips", "/Robot/vel/omega_radps")


@dataclass
class Log:
    """A log's powers, as (time, `SimRun` topic, value) in time order, its time from the log's
    first record, s; its velocity topics' records, as rows of time and value; and the time of its
    last record."""

    end_s: float = 0.0
    powers: list[tuple[float, str, float | bool]] = field(default_factory=list)
    velocity: dict[str, NDArray[np.float64]] = field(default_factory=dict)


def read(path: Path) -> Log:
    """The powers and velocity out of the WPILOG at `path`."""
    names: dict[int, tuple[str, str]] = {}
    rows: list[tuple[float, str, float | bool]] = []
    vel: dict[str, list[tuple[float, float]]] = {k: [] for k in VELOCITY}
    t0: float | None = None
    end = 0.0
    for rec in DataLogReader(str(path)):
        if rec.isStart():
            s = rec.getStartData()
            names[s.entry] = (s.name, s.type)
            continue
        if rec.isControl() or rec.getEntry() not in names:
            continue
        t = rec.getTimestamp() / 1e6
        t0 = t if t0 is None else t0
        name, kind = names[rec.getEntry()]
        if name in POWERS:
            v = rec.getBoolean() if kind == "boolean" else rec.getDouble()
            rows.append((t - t0, POWERS[name], v))
        elif name in vel:
            vel[name].append((t - t0, rec.getDouble()))
        end = t - t0
    rows.sort(key=lambda r: r[0])
    return Log(end, rows, {k: np.array(v, float).reshape(-1, 2) for k, v in vel.items()})


def replay(scene: Path, log: Log, every: int = 20) -> NDArray[np.float64]:
    """The scene's robot driven by the log's powers to the log's end; rows of time, s, its velocity
    along itself and to its left, in/s, and its turn, rad/s, each `every` steps."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    held: dict[str, float | bool] = {}
    out = []
    i = step = 0
    while d.time < log.end_s:
        while i < len(log.powers) and log.powers[i][0] <= d.time:
            _, topic, v = log.powers[i]
            held[topic] = v
            i += 1
        nt.apply(m, d, nt.controls(held))
        mujoco.mj_step(m, d)
        step += 1
        if step % every == 0:
            v_body = d.xmat[1].reshape(3, 3).T @ d.qvel[:3] / IN
            out.append((d.time, v_body[0], v_body[1], d.qvel[5]))
    return np.array(out, float).reshape(-1, 4)
