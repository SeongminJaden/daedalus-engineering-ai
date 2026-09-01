"""General shapes route to CalculiX, because this project cannot mesh them.

The Warp solver is an eight node hexahedron on a structured uniform grid. A
shape that grid cannot cover has, until now, had no route at all. Gmsh meshes
it with tetrahedra and CalculiX solves it, and this file covers what that
route can and cannot claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.registry import Category, ProblemContext
from nodes import calculix as ccx
from nodes import gmsh_node as gm
from nodes.gmsh_node import (GMSH_TET10_TO_CALCULIX, TetMesh,
                             tetrahedral_box_mesh)
from nodes.roster import build_roster
from physics.fem.mesh import solid_box_mesh
from physics.fem.solver import solve_linear_elasticity

requires_both = pytest.mark.skipif(
    not (gm.is_available() and ccx.is_available()),
    reason="needs both gmsh and CalculiX")

BOX = (0.2, 0.04, 0.03)
E_PA, NU, LOAD = 71.7e9, 0.33, -100.0


@pytest.fixture(scope="module")
def hex_reference():
    mesh = solid_box_mesh(*BOX, 20, 4, 3)
    solution = solve_linear_elasticity(
        mesh, E_PA, NU, mesh.nodes_at_x(0.0), mesh.nodes_at_x(BOX[0]),
        total_load_n=LOAD, load_direction=1)
    return solution.tip_deflection()


def tip_deflection(mesh, element_type) -> float:
    result = ccx.solve(mesh, E_PA, NU, mesh.nodes_at_x(0.0),
                       mesh.nodes_at_x(BOX[0]), total_load_n=LOAD,
                       load_direction=1, element_type=element_type)
    return float(result.displacements[mesh.nodes_at_x(BOX[0]), 1].mean())


# ------------------------------------------------------------- the mesh type

@pytest.mark.skipif(not gm.is_available(), reason="gmsh is not installed")
@pytest.mark.parametrize("order,per", [(1, 4), (2, 10)])
def test_the_tet_mesh_has_the_right_shape_and_volume(order, per):
    mesh = tetrahedral_box_mesh(*BOX, 0.012, order=order)
    assert mesh.nodes_per_element == per
    assert mesh.is_quadratic == (order == 2)
    assert mesh.volume_m3() == pytest.approx(BOX[0] * BOX[1] * BOX[2],
                                             rel=1e-9)


@pytest.mark.skipif(not gm.is_available(), reason="gmsh is not installed")
def test_an_invalid_order_is_refused():
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        tetrahedral_box_mesh(*BOX, 0.012, order=3)


def test_the_element_types_declare_their_node_counts():
    assert ccx.ElementType.C3D4.nodes_per_element == 4
    assert ccx.ElementType.C3D10.nodes_per_element == 10
    assert ccx.ElementType.C3D8I.nodes_per_element == 8


@pytest.mark.skipif(not gm.is_available(), reason="gmsh is not installed")
def test_a_mesh_that_does_not_match_the_element_type_is_refused(tmp_path):
    """A mismatched deck either fails to read or describes another solid."""
    mesh = tetrahedral_box_mesh(*BOX, 0.02, order=1)
    with pytest.raises(ValueError, match="nodes per element"):
        ccx.write_deck(tmp_path / "bad.inp", mesh, E_PA, NU,
                       mesh.nodes_at_x(0.0), mesh.nodes_at_x(BOX[0]),
                       total_load_n=LOAD,
                       element_type=ccx.ElementType.C3D10)


# ----------------------------------------- the tets agree with the hex answer

@requires_both
def test_quadratic_tets_agree_with_the_structured_hex_solution(hex_reference):
    """A box is the one domain both meshers cover, so the answer is checkable.

    That is the whole reason the first general shape is a box: everywhere else
    this route has nothing to be compared against.
    """
    mesh = tetrahedral_box_mesh(*BOX, 0.012, order=2)
    deflection = tip_deflection(mesh, ccx.ElementType.C3D10)
    assert deflection == pytest.approx(hex_reference, rel=0.02)


@requires_both
def test_linear_tets_are_far_too_stiff_which_is_why_they_are_not_the_default(
        hex_reference):
    """The control. Measured at 11 to 18 percent low, and refining barely helps.

    Linear tetrahedra lock in bending. Offering them as the default would give
    a confidently wrong deflection on exactly the problems this route exists
    for.
    """
    coarse = tip_deflection(tetrahedral_box_mesh(*BOX, 0.012, order=1),
                            ccx.ElementType.C3D4)
    fine = tip_deflection(tetrahedral_box_mesh(*BOX, 0.008, order=1),
                          ccx.ElementType.C3D4)
    for value in (coarse, fine):
        assert abs(value) < abs(hex_reference) * 0.95      # too stiff
    # refining moves it the right way, and nowhere near far enough
    assert abs(fine) > abs(coarse)
    assert abs(fine) < abs(hex_reference) * 0.95


# ------------------------------- the node ordering, and how it fails when wrong

@requires_both
def test_a_wrong_mid_edge_ordering_is_raised_not_returned_as_zeros():
    """The failure mode is silent, so the adapter has to break the silence.

    A permutation CalculiX rejects makes it write an EMPTY result file. The
    arrays would come back full of zeros, which reads as a perfectly rigid
    structure rather than as a failure. That is now an exception.
    """
    mesh = tetrahedral_box_mesh(*BOX, 0.015, order=2)
    inverse = np.argsort(np.array(GMSH_TET10_TO_CALCULIX))
    scrambled = TetMesh(node_coords=mesh.node_coords,
                        connectivity=mesh.connectivity[:, inverse])
    with pytest.raises(RuntimeError, match="no displacements"):
        tip_deflection(scrambled, ccx.ElementType.C3D10)


@requires_both
def test_the_shipped_permutation_is_the_one_that_works(hex_reference):
    """Stated as data and checked, rather than trusted to a comment."""
    assert GMSH_TET10_TO_CALCULIX == (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)
    mesh = tetrahedral_box_mesh(*BOX, 0.015, order=2)
    assert tip_deflection(mesh, ccx.ElementType.C3D10) == pytest.approx(
        hex_reference, rel=0.05)


# ------------------------------------------------------------------ routing

def test_a_general_shape_routes_here_and_a_structured_one_does_not():
    """The two CalculiX capabilities answer different questions."""
    registry = build_roster()
    assert ccx.CALCULIX_GENERAL_CAPABILITY in registry

    structured = ProblemContext(geometry="prismatic_beam",
                                representations=("prismatic_beam",),
                                needs_stress_field=True)
    general = ProblemContext(geometry="organic",
                             representations=("organic",),
                             needs_stress_field=True)
    structured_names = registry.query(structured, Category.ANALYSIS).names()
    general_names = registry.query(general, Category.ANALYSIS).names()

    assert ccx.CALCULIX_GENERAL_CAPABILITY not in structured_names
    assert ccx.CALCULIX_GENERAL_CAPABILITY in general_names
    # And the hex cross-check does not claim a shape it cannot mesh.
    assert ccx.CALCULIX_CAPABILITY not in general_names


def test_the_general_route_admits_it_has_no_second_opinion():
    """The hex route is cross-validated. This one cannot be, and says so."""
    method = ccx.calculix_general_capability_method()
    assert "no second opinion" in method.notes
    assert method.evidence == "SIMULATED"
