"""The robot in MuJoCo, driven by what biobuzz's simulator, `SimRun`, publishes on NetworkTables.

`SimRun` is the NT4 server; this is a client, as AdvantageScope is. It reads the powers biobuzz's
code set, by the topics `SimPublisher` and `Tracker` already publish, and sets each motor of the
model, named as biobuzz names it, to them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path

import mujoco
import ntcore

log = logging.getLogger(__name__)

# Each power topic, and the model's motors it sets. `SimPublisher` writes the wheels' powers as
# the code set them, before a motor's direction, so +1 drives each wheel forward; `Intake` sets
# its motor and both Gecko wheels to one speed, +1 in.
POWERS: dict[str, tuple[str, ...]] = {
    "sim/wheels/frontLeft": ("Front Left",),
    "sim/wheels/frontRight": ("Front Right",),
    "sim/wheels/backLeft": ("Back Left",),
    "sim/wheels/backRight": ("Back Right",),
    "/intake/speed": ("intake", "left_intake_servo", "right_intake_servo"),
}
# `Launcher` feeds by turning the windmill at +1 while this is true.
FEEDING = "/launcher/feeding"
# `Launcher` asks its motor for a speed, in encoder ticks a second, which the Control Hub holds
# with its own PIDF. The model is given that speed's share of the motor's free speed, its
# feedforward only.
TARGET_TPS = "/launcher/target_tps"
LAUNCHER_TICKS_PER_REV = 28
LAUNCHER_FREE_RPM = 6000.0


def controls(values: Mapping[str, float | bool]) -> dict[str, float]:
    """Each motor's power, −1 to 1, from the topics' last values; a topic not yet seen leaves
    its motors out."""
    out: dict[str, float] = {}
    for topic, motors in POWERS.items():
        if topic in values:
            for m in motors:
                out[m] = float(values[topic])
    if FEEDING in values:
        out["windmillServo"] = 1.0 if values[FEEDING] else 0.0
    if TARGET_TPS in values:
        free_tps = LAUNCHER_FREE_RPM / 60 * LAUNCHER_TICKS_PER_REV
        out["launcher"] = float(values[TARGET_TPS]) / free_tps
    return {k: max(-1.0, min(1.0, v)) for k, v in out.items()}


def apply(m: mujoco.MjModel, d: mujoco.MjData, powers: Mapping[str, float]) -> None:
    """Sets each named motor's control."""
    for name, u in powers.items():
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = u


class Client:
    """A NetworkTables client of `SimRun`, subscribed to the power topics."""

    def __init__(self, host: str, name: str = "mujocobiobuzz") -> None:
        self.inst = ntcore.NetworkTableInstance.create()
        self.inst.startClient4(name)
        self.inst.setServer(host)
        topics = [*POWERS, FEEDING, TARGET_TPS]
        self.subs = {t: self.inst.getTopic(t).genericSubscribe() for t in topics}

    def values(self) -> dict[str, float | bool]:
        """Each topic's last value, for those the server has sent."""
        out: dict[str, float | bool] = {}
        for t, s in self.subs.items():
            v = s.get()
            if v.isValid():
                out[t] = v.value()
        return out

    def close(self) -> None:
        ntcore.NetworkTableInstance.destroy(self.inst)


def run(scene: Path, host: str, seconds: float | None = None) -> mujoco.MjData:
    """Steps the scene in real time, setting its motors from `SimRun` each step, for `seconds`
    or until stopped."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    client = Client(host)
    start = time.monotonic()
    try:
        while seconds is None or d.time < seconds:
            apply(m, d, controls(client.values()))
            mujoco.mj_step(m, d)
            ahead = d.time - (time.monotonic() - start)
            if ahead > 0:
                time.sleep(ahead)
    finally:
        client.close()
    return d
