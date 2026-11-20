# mujocobiobuzz

**Status: controlling.**

The BIOBUZZ field and FTC robots 31459 and 37523 in [MuJoCo](https://mujoco.org): the hives
tipping, the balls, and the robots driven by a gamepad over NetworkTables, through
`ftc31459-26-27-biobuzz`'s simulation, or by their own logs. `waggle` uses it as a library.

The plan it follows is `.docs/2026-11-20-mujocobiobuzz.md` in `ftc-claude`.

## Working on it

```sh
uv sync
uv run prek install
ninja check
```

`ninja check` runs ruff, ruff format, mypy and the fast tests, and runs in GitHub Actions on Linux.

## The models

`ninja model` downloads goBILDA's StarterBot STEP and FIRST's BIOBUZZ field STEP and converts
them to MuJoCo models in `models/starterbot/` and `models/biobuzz/`, with the internal parts left
out. Each folder's `scene.xml` lights the model, the robot on a floor. The conversion reads the
CAD with `cadquery-ocp`, in the `cad` dependency group; the simulator needs only the models it
writes. `robots/starterbot.json` and `fields/biobuzz.json` name which CAD parts make each moving
body and how each collides. The field is in Pedro Pathing's field frame, in m, from a corner, and
its hives turn on their pivots between two stable states. `uv run pytest -m slow` checks the
built models.

Licensed under Apache-2.0; see `LICENSE`.
