"""A Monte Carlo of a hive's spill: balls loaded at random into blue's raised cell until it tips,
and each ball followed out of the cell to its first landing.

A run places balls one by one inside the cell, each 0.6 s after the last, as the Event Field
Guide's § 12.3 places a calibration load, each POLLEN or NECTAR at random, at a place clear of the
balls already there, until the hive passes level; a cell with no clear place stops the load. Each
ball's leaving is when its centre crosses the plane of the cell's opening, outward; its landing,
its first touch of anything but the hive, a ball still in the cell or a ball in the air. Between
the two it is in the air, and its touches there are counted, each kind with its largest normal
force: of a ball also in the air, and of the cell, the hive or a ball still in it.

Places and velocities are in Pedro Pathing's field frame, in inches, as `waggle`'s are; times
from when the hive passes level, s.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from mujocobiobuzz import balls as B

log = logging.getLogger(__name__)

IN = 0.0254
SCENE_DIR = Path(__file__).parent.parent.parent / "models" / "biobuzz"
HIVE = "biobuzz_blue_hive"
# Blue's raised cell's opening at the model's 0°, from waggle's `data/hive/openings.json`: its
# centre, Pedro in, and its outward normal.
OPENING_IN = np.array([84.791, 88.376, 57.667])
NORMAL = np.array([-0.001, 0.87, 0.493])
INSIDE_M = 0.12
LEVEL_RAD = math.radians(-30)
PLACE_S = 0.6
MAX_BALLS = 12
AFTER_S = 3.0
TRIES = 50


@dataclass
class Ball:
    """One ball of a run: its kind, when it left the cell and with what place, velocity, in/s,
    and spin, rad/s; when it landed; and its touches in the air, of balls in the air and of the
    cell, each with the largest normal force, N."""

    kind: str
    left_s: float | None = None
    place_in: list[float] | None = None
    velocity_ips: list[float] | None = None
    spin_radps: list[float] | None = None
    landed_s: float | None = None
    air_touches: int = 0
    air_force_n: float = 0.0
    cell_touches: int = 0
    cell_force_n: float = 0.0


@dataclass
class Run:
    seed: int
    nectar_share: float
    tipped: bool = False
    full: bool = False
    down_s: float | None = None
    balls: list[Ball] = field(default_factory=list)


def _model(n: int) -> mujoco.MjModel:
    """The field with `n` balls of each kind parked off it, to be placed; MuJoCo reads the
    meshes from the model's folder, so the scene is written there."""
    parked = [
        (k, f"{k.name}{i}", (-0.5 - 0.15 * i, -0.5 - 0.15 * j, 0.1))
        for j, k in enumerate((B.POLLEN, B.BLUE_NECTAR))
        for i in range(n)
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=SCENE_DIR) as f:
        f.write(B.scene("biobuzz", parked))
        f.flush()
        return mujoco.MjModel.from_xml_path(f.name)


def run(m: mujoco.MjModel, seed: int, nectar_share: float = 0.3) -> Run:
    """One spill: a random load placed until the hive passes level, then followed `AFTER_S`."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    hinge = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, HIVE)]
    hive = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, HIVE)]
    # The opening in the hive's own frame, so it turns with the hive.
    r0, p0 = d.xmat[hive].reshape(3, 3).copy(), d.xpos[hive].copy()
    c_local, n_local = r0.T @ (OPENING_IN * IN - p0), r0.T @ NORMAL
    across = np.cross(NORMAL, [1.0, 0.0, 0.0])
    out = Run(seed, nectar_share)
    used = {"pollen": 0, "blue_nectar": 0}
    placed: list[tuple[int, Ball]] = []
    level_t = None
    while level_t is None and len(placed) < MAX_BALLS:
        kind = "blue_nectar" if rng.random() < nectar_share else "pollen"
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{kind}{used[kind]}")
        used[kind] += 1
        q = m.jnt_qposadr[m.body_jntadr[b]]
        # The cell as it lies now, as the hive starts to turn under its load.
        turn = d.xmat[hive].reshape(3, 3) @ r0.T
        cell = (d.xpos[hive] - turn @ p0, turn)
        at = _clear_place(m, d, rng, across, m.geom_size[m.body_geomadr[b]][0], placed, cell)
        if at is None:
            out.full = True
            break
        d.qpos[q : q + 7] = [*at, 1, 0, 0, 0]
        d.qvel[m.jnt_dofadr[m.body_jntadr[b]] : m.jnt_dofadr[m.body_jntadr[b]] + 6] = 0
        placed.append((b, Ball(kind)))
        for _ in range(int(PLACE_S / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if d.qpos[hinge] < LEVEL_RAD:
                level_t = d.time
                break
    out.balls = [ball for _, ball in placed]
    if level_t is None:
        return out
    out.tipped = True
    ids = {b: ball for b, ball in placed}
    end = d.time + AFTER_S
    while d.time < end:
        mujoco.mj_step(m, d)
        t = d.time - level_t
        if out.down_s is None and d.qpos[hinge] < math.radians(-58):
            out.down_s = t
        r, p = d.xmat[hive].reshape(3, 3), d.xpos[hive]
        c, n = p + r @ c_local, r @ n_local
        for b, ball in ids.items():
            if ball.left_s is None and (d.xpos[b] - c) @ n > 0:
                dof = m.jnt_dofadr[m.body_jntadr[b]]
                ball.left_s = t
                ball.place_in = (d.xpos[b] / IN).round(2).tolist()
                ball.velocity_ips = (d.qvel[dof : dof + 3] / IN).round(2).tolist()
                spin = d.xmat[b].reshape(3, 3) @ d.qvel[dof + 3 : dof + 6]
                ball.spin_radps = spin.round(2).tolist()
        force = np.zeros(6)
        for i, con in enumerate(d.contact[: d.ncon]):
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            for me, other in ((b1, b2), (b2, b1)):
                touched = ids.get(me)
                if touched is None or touched.left_s is None or touched.landed_s is not None:
                    continue
                mujoco.mj_contactForce(m, d, i, force)
                f = round(float(force[0]), 3)
                mate = ids.get(other)
                if mate is not None and mate.left_s is not None and mate.landed_s is None:
                    touched.air_touches += 1
                    touched.air_force_n = max(touched.air_force_n, f)
                elif other == hive or (mate is not None and mate.left_s is None):
                    touched.cell_touches += 1
                    touched.cell_force_n = max(touched.cell_force_n, f)
                else:
                    touched.landed_s = t
    return out


def _clear_place(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    rng: np.random.Generator,
    across: np.ndarray,
    radius: float,
    placed: list[tuple[int, Ball]],
    cell: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    """A place inside the cell, at random, clear of each ball already in it; or None after
    `TRIES` places that are not. `cell` moves a place in the cell at the model's 0° to where the
    cell lies now."""
    shift, turn = cell
    for _ in range(TRIES):
        jitter = np.array([rng.uniform(-0.12, 0.12), 0, 0]) + across * rng.uniform(-0.04, 0.04)
        at = shift + turn @ (OPENING_IN * IN - NORMAL * INSIDE_M + jitter)
        if all(
            np.linalg.norm(d.xpos[b] - at) > radius + m.geom_size[m.body_geomadr[b]][0]
            for b, _ in placed
        ):
            return at
    return None


def monte_carlo(runs: int, out: Path, nectar_share: float = 0.3, first_seed: int = 0) -> None:
    """`runs` spills, seeds from `first_seed`, each a line of JSON in `out`."""
    m = _model(MAX_BALLS)
    with out.open("w") as f:
        for seed in range(first_seed, first_seed + runs):
            r = run(m, seed, nectar_share)
            f.write(json.dumps(asdict(r)) + "\n")
            log.info("seed %d: %d balls, tipped %s", seed, len(r.balls), r.tipped)


def main() -> None:
    """`python -m mujocobiobuzz.spill RUNS OUT.jsonl`: a Monte Carlo of blue's spill."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("runs", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--nectar-share", type=float, default=0.3)
    p.add_argument("--first-seed", type=int, default=0)
    a = p.parse_args()
    monte_carlo(a.runs, a.out, a.nectar_share, a.first_seed)


if __name__ == "__main__":
    main()
