import struct
from pathlib import Path

import numpy as np
import pytest

from mujocobiobuzz.replay import read, replay

SCENE = Path(__file__).parent.parent / "models" / "starterbot" / "scene.xml"


def _record(entry: int, t_us: int, payload: bytes) -> bytes:
    # 4-byte id, 4-byte size and 8-byte time, as the header byte's bits say.
    return bytes([3 | 3 << 2 | 7 << 4]) + struct.pack("<IIQ", entry, len(payload), t_us) + payload


def _start(entry: int, name: str, kind: str) -> bytes:
    n, k = name.encode(), kind.encode()
    body = struct.pack("<BII", 0, entry, len(n)) + n + struct.pack("<I", len(k)) + k
    return _record(0, 0, body + struct.pack("<I", 0))


def write_log(path: Path, rows: list[tuple[str, str, int, object]]) -> None:
    """A WPILOG 1.0 file of (topic, type, time in µs, value) rows."""
    ids: dict[str, int] = {}
    out = b"WPILOG" + b"\x00\x01" + struct.pack("<I", 0)
    for topic, kind, t_us, v in rows:
        if topic not in ids:
            ids[topic] = len(ids) + 1
            out += _start(ids[topic], topic, kind)
        payload = struct.pack("<d", v) if kind == "double" else bytes([bool(v)])
        out += _record(ids[topic], t_us, payload)
    path.write_bytes(out)


def test_read_takes_the_powers_by_simruns_names_from_the_first_record(tmp_path: Path) -> None:
    log = tmp_path / "run.wpilog"
    write_log(
        log,
        [
            ("/loop/count", "double", 1_000_000, 1.0),
            ("/wheel/frontLeft/speed", "double", 1_500_000, 0.5),
            ("/launcher/feeding", "boolean", 2_000_000, True),
            ("/Robot/vel/forward_ips", "double", 2_500_000, 12.0),
        ],
    )
    got = read(log)
    assert got.powers == [(0.5, "sim/wheels/frontLeft", 0.5), (1.0, "/launcher/feeding", True)]
    assert got.velocity["/Robot/vel/forward_ips"].tolist() == [[1.5, 12.0]]


@pytest.mark.slow
@pytest.mark.skipif(not SCENE.exists(), reason="the model is built by `ninja model`")
def test_replay_drives_the_model_as_the_log_did(tmp_path: Path) -> None:
    """Each wheel at full power for 1 s, then stopped: the model drives forward, then stops."""
    log = tmp_path / "run.wpilog"
    rows: list[tuple[str, str, int, object]] = []
    for w in ("frontLeft", "frontRight", "backLeft", "backRight"):
        topic = f"/wheel/{w}/speed"
        rows += [(topic, "double", 0, 1.0), (topic, "double", 10**6, 0.0)]
    rows.append(("/loop/count", "double", 2 * 10**6, 2.0))
    write_log(log, sorted(rows, key=lambda r: r[2]))
    trace = replay(SCENE, read(log))
    forward, left = (lambda t, c=c: np.interp(t, trace[:, 0], trace[:, c]) for c in (1, 2))
    assert forward(0.9) > 55
    assert abs(left(0.9)) < 2
    assert abs(forward(1.9)) < 2
