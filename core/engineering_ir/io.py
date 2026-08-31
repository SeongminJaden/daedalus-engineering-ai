"""core.engineering_ir.io - YAML round-trip for the Engineering IR."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import EngineeringProblem


def problem_to_dict(problem: EngineeringProblem) -> dict:
    """Plain-python dict with enums flattened to their string values."""
    return problem.model_dump(mode="json")


def problem_from_dict(data: dict) -> EngineeringProblem:
    return EngineeringProblem.model_validate(data)


def save_problem(problem: EngineeringProblem, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(problem_to_dict(problem), fh, sort_keys=False,
                       default_flow_style=False)
    return path


def load_problem(path: str | Path) -> EngineeringProblem:
    with Path(path).open() as fh:
        return problem_from_dict(yaml.safe_load(fh))
