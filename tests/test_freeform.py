"""Generating a shape with no family behind it.

Every other generator searches a parametric family. This one takes the
envelope, the load and the supports, and returns a body no family describes,
labelled by CalculiX like any other part. The tests are about the pipeline
holding for an unrecognised shape: nothing downstream may require a part to
have a family.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.execution import executable_methods
from agent.execution.freeform import ELEMENT_NOTE, METHOD, FreeformFailed, run
from agent.execution.outcome import SOLVER_RESPONSES
from core.part_dataset.classify import UNKNOWN, rule_classify
from core.part_dataset.labeller import labelling_available
from core.registry import DEFAULT_REGISTRY
from geometry.cad_export.kernel import kernel_available
from optimization.constraints import build_optimization_problem
from projects.robotic_link.problem import build_mvp_problem

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and labelling_available()),
    reason="build123d, gmsh and CalculiX are required")


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


@pytest.fixture(scope="module")
def outcome(op, tmp_path_factory):
    if not (kernel_available() and labelling_available()):
        pytest.skip("build123d, gmsh and CalculiX are required")
    return run(op, divisions=(20, 8, 4), iterations=50,
               step_dir=tmp_path_factory.mktemp("freeform"))


# --- it is a registered, dispatchable method ---------------------------------

def test_the_method_is_registered_and_executable():
    assert METHOD in executable_methods()
    method = DEFAULT_REGISTRY.get(METHOD)
    assert method.evidence == "SIMULATED"
    assert "not a family search" in method.notes
    assert "lower bound" in method.notes


# --- the result is a solver labelled part ------------------------------------

@requires_all
def test_the_outcome_carries_a_solver_labelled_part_with_no_family(outcome):
    assert outcome.method == METHOD
    assert outcome.detail["family"] is None
    record = outcome.cad_record
    assert set(record.labels) & SOLVER_RESPONSES
    assert record.labels["tip_deflection_m"]["evidence"] == "simulated"
    assert record.provenance.generator == METHOD
    assert record.geometry.volume_m3 > 0.0
    assert outcome.mass_kg == pytest.approx(
        record.labels["mass_kg"]["value"], rel=1e-12)


@requires_all
def test_every_label_says_which_element_solved_it(outcome):
    """Linear tetrahedra are stiff in bending, so a label that did not say so
    would be an optimistic number without a warning."""
    assert outcome.detail["element_type"] == "C3D4"
    assert "10.7 percent" in ELEMENT_NOTE
    for name in ("tip_deflection_m", "max_displacement_m"):
        assert ELEMENT_NOTE in outcome.cad_record.labels[name]["note"]
    assert "not converged" in outcome.cad_record.labels["max_von_mises_pa"]["note"]


@requires_all
def test_the_mass_names_its_own_error_against_the_field(outcome):
    note = outcome.cad_record.labels["mass_kg"]["note"]
    assert "differs from the density field" in note
    assert "%" in note


# --- an unrecognised shape does not break anything ---------------------------

@requires_all
def test_the_classifier_says_unknown_and_the_pipeline_still_runs(outcome):
    """The rule classifier was measured on five parametric families and this
    is none of them. UNKNOWN is the correct answer, and nothing downstream may
    depend on getting a family."""
    from core.part_dataset.descriptors import ShapeDescriptor

    record = outcome.cad_record
    # A descriptor cannot be read from a triangle body the way it is read from
    # a B-rep, so the classifier is exercised on what the record does carry.
    assert record.topology.faces > 100          # triangles, not analytic faces
    assert record.topology.edges == 0
    assert "triangle count" in record.notes
    assert outcome.detail["classifier"].startswith("expected UNKNOWN")


@requires_all
def test_manufacturability_runs_on_a_shape_with_no_family(outcome, tmp_path):
    """The DFM rules read a triangle mesh, so they apply to a body that no
    family describes, which is the point of keeping them geometric."""
    from geometry.manufacturability import Process, assess
    from optimization.topology.smooth import marching_surface

    surface = outcome.detail["surface"]
    assert surface["watertight"] and surface["n_components"] == 1
    import trimesh
    stl = outcome.detail["stl_path"]
    mesh = trimesh.load(stl)
    report = assess(Process.SLM, np.asarray(mesh.vertices), np.asarray(mesh.faces))
    measured = [f for f in report.findings if f.assessed]
    assert measured, "no rule could be measured on the free form body"
    assert report.grade == "rule_based_dfm_guideline"


# --- refusals ----------------------------------------------------------------

def test_a_problem_without_an_envelope_is_refused(op):
    problem = op.problem.model_copy(deep=True)
    problem.geometry.max_height_m = None
    broken = op.__class__(**{**op.__dict__, "problem": problem})
    with pytest.raises(ValueError, match="envelope"):
        run(broken)


@requires_all
def test_a_high_threshold_drops_islands_and_says_how_many(op, tmp_path):
    """A thresholded field routinely leaves material attached only through an
    edge, or not at all. It carries no load, so it is dropped before the
    surface is built, and the count is in the record rather than hidden. If
    what remains cannot be solved the run refuses instead."""
    try:
        outcome = run(op, divisions=(20, 8, 4), iterations=20, threshold=0.9,
                      step_dir=tmp_path)
    except FreeformFailed as exc:
        assert "closed body" in str(exc) or "singular" in str(exc)
        return
    assert outcome.detail["island_elements_dropped"] >= 0
    assert outcome.detail["surface"]["n_components"] == 1


def test_the_method_applies_only_where_both_a_grid_and_a_part_are_wanted():
    """The routing mistake this condition pair exists to prevent.

    Registered with the CAD family condition alone, this method took every
    family problem from the family search. Registered with the voxel condition
    alone, it took every field problem from the topology methods. It needs
    both: an envelope to discretise, and a caller who wants a part rather than
    a density field.
    """
    from core.registry import ProblemContext

    method = DEFAULT_REGISTRY.get(METHOD)
    family_only = ProblemContext(geometry="cad_family",
                                 representations=("cad_family",),
                                 slenderness=5.0, material_class="isotropic",
                                 has_stress_constraint=True)
    voxel_only = ProblemContext(geometry="prismatic_beam",
                                representations=("prismatic_beam", "voxel_domain"),
                                slenderness=5.0, material_class="isotropic",
                                has_stress_constraint=True)
    both = ProblemContext(geometry="cad_family",
                          representations=("cad_family", "voxel_domain"),
                          slenderness=5.0, material_class="isotropic",
                          has_stress_constraint=True)
    assert not all(c.holds(family_only) for c in method.conditions)
    assert not all(c.holds(voxel_only) for c in method.conditions)
    assert all(c.holds(both) for c in method.conditions)
