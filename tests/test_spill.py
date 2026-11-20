from pathlib import Path

import numpy as np
import pytest

from mujocobiobuzz import spill

FIELD = Path(__file__).parent.parent / "models" / "biobuzz" / "scene.xml"
needs_model = pytest.mark.skipif(not FIELD.exists(), reason="the model is built by `ninja model`")


def check_spills(runs: list[spill.Run]) -> None:
    """Each run tips and swings down as the video's does; most balls leave and land, and none
    leaves at more than 150 in/s. A fall from the raised opening, 57.7 in up, to the lowered lip,
    about 34 in, is 133 in/s; a ball put down inside another or inside the cell's wall leaves at
    178 in/s."""
    for r in runs:
        assert r.tipped and not r.full
        assert r.down_s is not None and 0.2 < r.down_s < 0.32
    left = [b for r in runs for b in r.balls if b.left_s is not None]
    assert len(left) >= 0.8 * sum(len(r.balls) for r in runs)
    assert all(b.landed_s is not None for b in left[: int(0.9 * len(left))])
    assert max(np.linalg.norm(b.velocity_ips) for b in left if b.velocity_ips) < 150


@pytest.mark.slow
@needs_model
def test_spills_tip_and_empty_the_cell() -> None:
    m = spill._model(spill.MAX_BALLS)
    check_spills([spill.run(m, seed) for seed in range(5)])
