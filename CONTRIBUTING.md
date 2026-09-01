# Contributing to Daedalus Engineering AI

Contributions are welcome. This document is a stub and will grow as the project
opens up: issues and discussion are the best place to start right now.

## Before you start

Open an issue describing what you intend to change. For anything touching the
physics, the optimizer or the Brain's evidence rules, that conversation will
save you time.

## Development setup

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install -U pip wheel
env -u PYTHONPATH .venv/bin/pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/python scripts/gpu_sanity.py
env -u PYTHONPATH .venv/bin/python -m pytest tests/ -q -n 8 --dist loadfile
```

Run the venv with a clean `PYTHONPATH`: a sourced shell environment can export
one and shadow the venv's packages.

## What this project asks of a change

The project's distinguishing property is that **every layer states what it does
not know**. Please keep that intact:

- **Verify critical calculations independently.** A new closed-form result
  should be checked against a separately derived reference (numerical
  integration, an analytical limit, or a second method), not against the same
  algebra that produced it. See `tests/reference_beam.py` and
  `tests/test_section_properties.py` for the pattern.
- **Do not overclaim.** If something is an approximation, an assumption, or a
  heuristic, say so in the code and in the docs. `[ASSUMED]` marks values that
  are not derived.
- **Nothing reaches `EXPERIMENTALLY_VALIDATED` without physical test
  evidence.** No amount of simulation may promote a claim past that gate.
- **Add tests that can fail.** A test that passes against a broken
  implementation is worse than no test. Where practical, confirm a new test
  actually fails before the fix.

## Pull requests

- Keep the change focused, and make sure the full suite passes.
- Use `-n 8 --dist loadfile` for the WHOLE suite only. It is about
  2.1x faster there and slower on a single file, because each worker
  pays its own CUDA and Warp startup. pytest.ini carries the numbers.
- Explain *why*, not just *what*, in the description.
- Report measured numbers honestly, including ones that are worse than hoped.
