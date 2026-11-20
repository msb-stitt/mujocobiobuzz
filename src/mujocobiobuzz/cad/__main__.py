"""Convert a robot's STEP to MJCF: `uv run --group cad python -m mujocobiobuzz.cad SPEC STEP OUT`.

SPEC is a robot's JSON under `robots/`, STEP the CAD it names, and OUT the folder the model goes
in. The summary of each body is printed as JSON on stdout.
"""

import json
import logging
import sys
from pathlib import Path

import fast_simplification
import numpy as np
from numpy.typing import NDArray

from mujocobiobuzz.cad.model import write_model
from mujocobiobuzz.cad.step import read_solids


def reduce(
    verts: NDArray[np.float64], tris: NDArray[np.int64], target: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The mesh cut to about `target` triangles by quadric decimation."""
    v, t = fast_simplification.simplify(verts, tris.astype(np.int32), target_count=target)[:2]
    return np.asarray(v, float), np.asarray(t, np.int64)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if len(argv) != 3:
        logging.critical("usage: python -m mujocobiobuzz.cad SPEC STEP OUT")
        return 2
    spec = json.loads(Path(argv[0]).read_text())
    summary = write_model(list(read_solids(Path(argv[1]))), spec, Path(argv[2]), reduce)
    print(json.dumps(summary, indent=1))  # noqa: T201 — the summary is the command's data
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
