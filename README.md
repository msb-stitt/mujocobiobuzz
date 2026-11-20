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

Licensed under Apache-2.0; see `LICENSE`.
