"""projects.robotic_link.problem - the MVP design problem.

A cantilever robotic arm link: fixed at the root, carrying a vertical point
load at the tip. Minimize mass subject to a stress ceiling, a tip-deflection
limit and a safety factor.

Everything is SI. The 196.2 N tip load is a 20 kg payload under standard
gravity (20 * 9.81); it is written out here rather than hard-coded as a bare
number so the intent survives.
"""

from __future__ import annotations

from pathlib import Path

from core.engineering_ir import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryLocation,
    Constraints,
    EngineeringProblem,
    Geometry,
    Load,
    LoadApplication,
    LoadType,
    Objective,
    ObjectiveQuantity,
    ObjectiveSense,
    SectionType,
    Vec3,
    load_problem,
)
from core.units import MPA, MM

MVP_YAML = Path(__file__).resolve().parent / "mvp_problem.yaml"

PAYLOAD_KG = 20.0
TIP_LOAD_N = 196.2          # 20 kg * 9.81 m/s^2, as specified for the MVP


def build_mvp_problem() -> EngineeringProblem:
    """The MVP problem, constructed in Python.

    Kept equivalent to mvp_problem.yaml - tests assert the two agree, so the
    YAML cannot silently drift from the code.
    """
    return EngineeringProblem(
        name="mvp_cantilever_link",
        geometry=Geometry(
            length_m=0.5,
            max_width_m=0.1,
            max_height_m=0.1,
            section_type=SectionType.HOLLOW_RECTANGLE,
        ),
        material_id="al_7075_t6",
        loads=[
            Load(
                type=LoadType.POINT_FORCE,
                magnitude_n=TIP_LOAD_N,
                direction=Vec3(x=0.0, y=-1.0, z=0.0),   # straight down
                application=LoadApplication.TIP,
            )
        ],
        boundary_conditions=[
            BoundaryCondition(
                type=BoundaryConditionType.FIXED,
                location=BoundaryLocation.ROOT,
            )
        ],
        constraints=Constraints(
            max_stress_pa=120.0 * MPA,
            max_deflection_m=1.0 * MM,
            min_safety_factor=2.0,
        ),
        objectives=[
            Objective(
                sense=ObjectiveSense.MINIMIZE,
                quantity=ObjectiveQuantity.MASS,
            )
        ],
    )


def load_mvp_problem() -> EngineeringProblem:
    """The same problem, read from mvp_problem.yaml."""
    return load_problem(MVP_YAML)
