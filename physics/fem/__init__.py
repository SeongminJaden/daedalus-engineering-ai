"""physics.fem - 3D linear-elastic finite element verification (Phase 7)."""

from .element import (
    elasticity_matrix, element_stiffness, element_stiffness_from_c,
    element_stress_operator_from_c, von_mises,
)
from .mesh import Mesh, hollow_rect_mesh, solid_box_mesh
from .solver import FemSolution, SolveReport, solve_linear_elasticity
from .verify import (
    FIDELITY, HighFidelityResult, high_fidelity_verify,
    size_mesh_for_budget,
)

__all__ = [
    "FIDELITY", "FemSolution", "HighFidelityResult", "Mesh", "SolveReport",
    "elasticity_matrix", "element_stiffness_from_c",
    "element_stress_operator_from_c", "high_fidelity_verify",
    "size_mesh_for_budget",
    "element_stiffness", "hollow_rect_mesh", "solid_box_mesh",
    "solve_linear_elasticity", "von_mises",
]
