# engineering-ai

GPU-accelerated engineering design agent. CLI/TUI only, no GUI.

## Setup

The venv is already created at `.venv`. **Always unset `PYTHONPATH`** when using
it — ROS 2 Humble exports a `PYTHONPATH` that would otherwise shadow the venv's
packages with `/opt/ros/humble` ones:

```bash
env -u PYTHONPATH .venv/bin/python ...
env -u PYTHONPATH .venv/bin/pip install -r requirements.txt
```

## Check the GPU stack

```bash
env -u PYTHONPATH .venv/bin/python scripts/gpu_sanity.py
```

## GPU profiles

Size-dependent settings live in `configs/profiles/`. Selection order:
`--profile` → `ENG_PROFILE` env var → VRAM auto-detection → `laptop_4gb`.

```bash
env -u PYTHONPATH .venv/bin/python -m interfaces.cli.main info
ENG_PROFILE=cloud_a100 env -u PYTHONPATH .venv/bin/python -m interfaces.cli.main info
```
