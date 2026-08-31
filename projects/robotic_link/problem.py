"""projects.robotic_link.problem - MVP design problem.

Optimize a robotic arm link: minimize mass subject to a stress ceiling and a
stiffness floor, under a tip load. Placeholder - the real definition lands with
DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoboticLinkProblem:
    """Design-space and load-case definition for the MVP link."""

    length_m: float = 0.30
    max_tip_load_n: float = 200.0
    material: str = "aluminium_6061"
    yield_stress_pa: float = 276e6
    safety_factor: float = 2.0
    min_stiffness_n_per_m: float = 1.0e5

    def objective(self):
        """Minimize mass. Not implemented yet."""
        raise NotImplementedError

    def constraints(self):
        """Stress ceiling + stiffness floor. Not implemented yet."""
        raise NotImplementedError
