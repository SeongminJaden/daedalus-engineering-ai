"""Cross-validating the Warp FEM against CalculiX.

Every FEM result so far was checked against beam theory, a patch test, or
itself. Those do not catch a shared misunderstanding of the element
formulation. CalculiX is independently written, in Fortran and C, by people who
made different choices, so agreeing with it is evidence of a different kind.

The test that carries this is `test_the_two_solvers_agree_across_the_field`.
`test_the_fully_integrated_element_shear_locks` is the control: it shows the
comparison is capable of detecting a real difference, so the agreement is not
an artefact of comparing something with itself.
"""

import numpy as np
import pytest

from core.registry import Category, ProblemContext
from nodes import (CALCULIX_CAPABILITY, CROSS_VALIDATED, SELF_FEM_ONLY,
                   STATUS_ORDER, CrossValidation, ElementType, VerificationStatus,
                   build_roster, calculix_descriptor, calculix_solve,
                   calculix_version, cross_validated_status, is_available,
                   write_deck)
from physics.fem.mesh import l_bracket_mesh, solid_box_mesh
from physics.fem.solver import solve_linear_elasticity

E, NU, LOAD = 71.7e9, 0.33, 200.0

requires_ccx = pytest.mark.skipif(
    not is_available(),
    reason="the ccx binary is not on PATH; CalculiX cross-validation skipped")


def cantilever(length=0.40, height=0.020, width=0.020, nx=40, ny=4, nz=4):
    mesh = solid_box_mesh(length, height, width, nx, ny, nz)
    return mesh, mesh.nodes_at_x(0.0), mesh.nodes_at_x(length)


# --- availability ------------------------------------------------------------

@requires_ccx
def test_calculix_reports_a_version():
    assert "Version" in (calculix_version() or "")


def test_the_node_reports_availability_from_the_filesystem():
    """Whether the binary is there is a fact, not a policy flag."""
    descriptor = calculix_descriptor()
    assert descriptor.available is is_available()
    if not descriptor.available:
        assert "not on PATH" in descriptor.unavailable_reason


def test_an_absent_binary_is_reported_not_assumed():
    forced = calculix_descriptor(available=False)
    assert not forced.available
    assert forced.unavailable_reason


# --- the deck ----------------------------------------------------------------

def test_the_deck_uses_one_based_numbering_and_matching_corner_order(tmp_path):
    """CalculiX expects the corner order this project's NODE_OFFSETS already
    produce, so the connectivity maps straight across with a one-based shift."""
    mesh, fixed, loaded = cantilever(nx=2, ny=1, nz=1)
    path = write_deck(tmp_path / "job.inp", mesh, E, NU, fixed, loaded, -LOAD)
    text = path.read_text()

    assert "*ELEMENT, TYPE=C3D8I" in text
    assert "*STATIC" in text
    # Node ids run from 1, never 0.
    node_lines = text.split("*NODE, NSET=Nall")[1].split("*ELEMENT")[0]
    first_id = int(node_lines.strip().splitlines()[0].split(",")[0])
    assert first_id == 1
    # Every element references the same eight nodes the mesh does, shifted.
    # After the *ELEMENT header line, the first stripped line IS element one.
    element_block = text.split("*ELEMENT, TYPE=C3D8I, ELSET=Eall")[1]
    for offset, connectivity in enumerate(mesh.connectivity):
        line = element_block.strip().splitlines()[offset]
        ids = [int(v) for v in line.split(",")[1:]]
        assert ids == [int(n) + 1 for n in connectivity]
        assert int(line.split(",")[0]) == offset + 1


def test_the_load_is_divided_the_same_way_both_solvers_divide_it(tmp_path):
    """So a difference in the answer is not a difference in the loading."""
    mesh, fixed, loaded = cantilever(nx=2, ny=1, nz=1)
    text = write_deck(tmp_path / "job.inp", mesh, E, NU, fixed, loaded,
                      -LOAD).read_text()
    cload = text.split("*CLOAD")[1].split("*NODE PRINT")[0].strip().splitlines()
    assert len(cload) == len(loaded)
    per_node = float(cload[0].split(",")[2])
    assert per_node == pytest.approx(-LOAD / len(loaded))


def test_a_deck_with_no_restraint_is_refused(tmp_path):
    """Singular for both solvers; refusing early says why."""
    mesh, _, loaded = cantilever(nx=2, ny=1, nz=1)
    with pytest.raises(ValueError, match="singular"):
        write_deck(tmp_path / "job.inp", mesh, E, NU, np.array([], dtype=int),
                   loaded, -LOAD)


def test_a_zero_load_is_refused(tmp_path):
    mesh, fixed, loaded = cantilever(nx=2, ny=1, nz=1)
    with pytest.raises(ValueError, match="checks nothing"):
        write_deck(tmp_path / "job.inp", mesh, E, NU, fixed, loaded, 0.0)


# --- the cross-validation ----------------------------------------------------

@requires_ccx
@pytest.mark.parametrize("label,builder", [
    ("slender cantilever", lambda: cantilever(0.40, 0.020, 0.020, 40, 4, 4)),
    ("stocky block", lambda: cantilever(0.16, 0.050, 0.020, 16, 6, 3)),
])
def test_the_two_solvers_agree_across_the_field(label, builder):
    """Not just at the tip: every node and every degree of freedom.

    Agreeing on one scalar could happen by luck or by cancelling errors.
    Agreeing on the whole displacement field, and on the stresses, is what
    makes this evidence about the element formulation.
    """
    mesh, fixed, loaded = builder()
    ours = solve_linear_elasticity(mesh, E, NU, fixed_nodes=fixed,
                                   load_nodes=loaded, total_load_n=-LOAD,
                                   load_direction=1)
    theirs = calculix_solve(mesh, E, NU, fixed, loaded, -LOAD, 1,
                            ElementType.C3D8I)

    scale = np.abs(ours.displacements).max()
    assert scale > 0
    field_error = np.abs(ours.displacements - theirs.displacements).max() / scale
    assert field_error < 1e-5, f"{label}: field error {field_error:.2e}"

    stress_error = abs(ours.element_von_mises.max()
                       / theirs.max_von_mises_pa() - 1.0)
    assert stress_error < 1e-4, f"{label}: stress error {stress_error:.2e}"


@requires_ccx
def test_the_agreement_holds_on_a_re_entrant_corner():
    """The L-bracket, where the stress field is genuinely awkward."""
    mesh = l_bracket_mesh(0.10, 0.4, 0.01, 16, nz=2)
    top = mesh.nodes_where(np.abs(mesh.node_coords[:, 1] - 0.10) < 1e-9)
    tip = mesh.nodes_where(np.abs(mesh.node_coords[:, 0] - 0.10) < 1e-9)

    ours = solve_linear_elasticity(mesh, E, NU, fixed_nodes=top,
                                   load_nodes=tip, total_load_n=-300.0,
                                   load_direction=1)
    theirs = calculix_solve(mesh, E, NU, top, tip, -300.0, 1,
                            ElementType.C3D8I)
    scale = np.abs(ours.displacements).max()
    assert np.abs(ours.displacements - theirs.displacements).max() / scale < 1e-5
    assert abs(ours.element_von_mises.max()
               / theirs.max_von_mises_pa() - 1.0) < 1e-4


@requires_ccx
def test_the_fully_integrated_element_shear_locks():
    """The control, and it is what makes the agreement meaningful.

    If every element type agreed, the comparison would be incapable of
    detecting a difference and the agreement would prove nothing. The fully
    integrated C3D8 locks in bending, coming out roughly ten percent stiff on a
    slender beam, which is exactly why this project uses incompatible modes.
    The locking eases as the beam gets stubbier, which is the signature of
    shear locking rather than of an unrelated error.
    """
    def stiffness_error(length, nx):
        mesh, fixed, loaded = cantilever(length, 0.020, 0.020, nx, 4, 4)
        with_modes = calculix_solve(mesh, E, NU, fixed, loaded, -LOAD, 1,
                                    ElementType.C3D8I)
        without = calculix_solve(mesh, E, NU, fixed, loaded, -LOAD, 1,
                                 ElementType.C3D8)
        a = abs(with_modes.displacements[loaded, 1].mean())
        b = abs(without.displacements[loaded, 1].mean())
        return a / b - 1.0

    slender = stiffness_error(0.40, 40)
    stubby = stiffness_error(0.08, 24)
    assert slender > 0.05, "C3D8 was expected to lock on a slender beam"
    assert slender > 2.0 * stubby, "locking should ease as the beam thickens"


@requires_ccx
def test_both_solvers_approach_beam_theory_as_the_beam_slims():
    """Two independent implementations converging on a third, analytical answer."""
    mesh, fixed, loaded = cantilever(0.40, 0.020, 0.020, 40, 4, 4)
    ours = solve_linear_elasticity(mesh, E, NU, fixed_nodes=fixed,
                                   load_nodes=loaded, total_load_n=-LOAD,
                                   load_direction=1)
    theirs = calculix_solve(mesh, E, NU, fixed, loaded, -LOAD, 1,
                            ElementType.C3D8I)
    second_moment = 0.020 * 0.020 ** 3 / 12.0
    euler = LOAD * 0.40 ** 3 / (3.0 * E * second_moment)

    for tip in (abs(ours.displacements[loaded, 1].mean()),
                abs(theirs.displacements[loaded, 1].mean())):
        assert abs(tip / euler - 1.0) < 0.02


# --- verification status -----------------------------------------------------

def test_agreement_promotes_a_design_and_carries_the_measurement():
    agreement = CrossValidation("CalculiX", "2.17", 1.1e-7, 0.0, 1e-3, "C3D8I")
    status = cross_validated_status("d1", agreement)
    assert status.status == CROSS_VALIDATED
    assert status.is_cross_validated
    assert status.cross_validation is agreement
    assert status.as_dict()["cross_validated_against"] == "CalculiX"


def test_disagreement_does_not_promote_and_says_by_how_much():
    """A silent downgrade would lose the only useful part of the result."""
    disagreement = CrossValidation("CalculiX", "2.17", 0.08, 0.05, 1e-3,
                                   "C3D8")
    status = cross_validated_status("d2", disagreement)
    assert status.status == SELF_FEM_ONLY
    assert not status.is_cross_validated
    assert "8.000e-02" in status.reason


def test_a_cross_validation_claim_must_carry_its_measurement():
    with pytest.raises(ValueError, match="measured agreement"):
        VerificationStatus(design_id="d", status=CROSS_VALIDATED)


def test_a_measurement_cannot_travel_without_the_claim_it_supports():
    agreement = CrossValidation("CalculiX", "2.17", 1e-7, 0.0, 1e-3, "C3D8I")
    with pytest.raises(ValueError, match="must travel with the claim"):
        VerificationStatus(design_id="d", status=SELF_FEM_ONLY,
                           reason="none", cross_validation=agreement)


def test_no_amount_of_solver_agreement_reaches_physical_validation():
    """The invariant that matters most here.

    Two simulations agreeing is stronger evidence than one simulation alone and
    it is still not a measurement. Nothing this module produces means anything
    was built.
    """
    agreement = CrossValidation("CalculiX", "2.17", 0.0, 0.0, 1e-9, "C3D8I")
    status = cross_validated_status("d", agreement)
    assert not status.is_physically_validated
    assert status.as_dict()["physically_validated"] is False


def test_the_status_ladder_is_ordered_and_starts_at_self_only():
    assert STATUS_ORDER[0] == SELF_FEM_ONLY
    assert CROSS_VALIDATED in STATUS_ORDER
    assert STATUS_ORDER.index(CROSS_VALIDATED) > STATUS_ORDER.index(
        SELF_FEM_ONLY)


# --- registry ----------------------------------------------------------------

def test_calculix_is_registered_and_fusion_is_still_not_available():
    """One real external node and one blocked one, side by side."""
    registry = build_roster()
    assert CALCULIX_CAPABILITY in registry
    context = ProblemContext(geometry="prismatic_beam",
                             representations=("prismatic_beam",),
                             slenderness=20.0, needs_stress_field=True)
    candidates = registry.query(context, Category.ANALYSIS)
    if is_available():
        assert CALCULIX_CAPABILITY in candidates.names()
    assert "analysis.fea.fusion" not in candidates.names()
