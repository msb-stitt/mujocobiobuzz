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
its hives turn on their pivots between two stable states. The robot's mecanum wheels roll on
free rollers, and a motor drives each wheel and mechanism, named as biobuzz's code names it, its
control the power, −1 to 1, as that code sets it. `uv run pytest -m slow` checks the built models.

## Driving the model from biobuzz's simulator

`mujocobiobuzz.nt.run` steps the robot's scene in real time as a NetworkTables 4 client of
biobuzz's `simRun`, as AdvantageScope is one, setting each motor to the power biobuzz's code set.
`mujocobiobuzz.nt` names the topics it reads.

## Replaying a robot's log

`mujocobiobuzz.replay` drives the model from a robot's flight log, a WPILOG file
`corbelsflightlog` wrote, by the same powers, and reads the robot's own velocity beside them.

## A replay's video

`uv run python -m mujocobiobuzz.video LOG OUT.mp4` replays a log on the field, from its
localizer's first pose, and draws it to a video whose first frame is the log's first record, its
time on each frame. In AdvantageScope's Video tab, with the log open, load the video, lock it
at 0 s, and it plays beside the log. It needs `ffmpeg` on the path; 9.0.2 from Homebrew made the
first.

## A spill's Monte Carlo

`uv run python -m mujocobiobuzz.spill RUNS OUT.jsonl` loads blue's raised cell at random until
the hive tips, RUNS times, and writes each run as a line of JSON: each ball's kind, when it left
the cell, its place, velocity and spin then, when it landed, and its touches in the air.
`mujocobiobuzz.spill` says how each is read.

Licensed under Apache-2.0; see `LICENSE`.
