"""The generative CAD strategy: the loop emits a STEP file the solver verified.

Each verified candidate is a Gmsh mesh and two CalculiX solves, so the runs
here build few parts and label fewer. What the tests pin is provenance: the
outcome's design is a B-rep the analyzer checked and the solver labelled, the
ranker never decides, and the episode names the method that made it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent.execution import DesignOutcome, execute, executable_methods
from agent.execution.cad import (DEFAULT_FAMILIES, METHOD, proxy_ranker, run)
from agent.loop.engine import DesignLoop, LoopConfig
from agent.reasoner import RegistryRoutingReasoner
from agent.strategy import StrategySelector
from core.materials import get_material
from core.part_dataset.labeller import labelling_available
from core.registry import DEFAULT_REGISTRY, Category, ProblemContext
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa
from optimization.constraints import build_optimization_problem
from projects.robotic_link.problem import build_mvp_problem

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="needs build123d, OCP, gmsh and CalculiX")

FAST = {"candidates": 6, "top_k": 2}


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


# ------------------------------------------------------------ the registry

def test_generative_cad_is_registered_and_applies_only_to_cad_problems():
    method = DEFAULT_REGISTRY.get(METHOD)
    assert method.category is Category.DESIGN_GENERATION
    assert "Not free-form" in method.notes
    beam_only = ProblemContext(geometry="prismatic_beam")
    with_cad = ProblemContext(geometry="prismatic_beam",
                              representations=("prismatic_beam", "cad_family"))
    assert not method.applies_to(beam_only)
    assert method.applies_to(with_cad)
    assert METHOD in executable_methods()


def test_a_cad_outcome_must_carry_a_labelled_record():
    class Unlabelled:
        labels = {}

    with pytest.raises(ValueError, match="solver has labelled"):
        DesignOutcome(method=METHOD, mass_kg=1.0, feasible=True,
                      cad_record=Unlabelled())
    with pytest.raises(ValueError, match="exactly one representation"):
        DesignOutcome(method=METHOD, mass_kg=1.0, feasible=True,
                      design_vector=np.zeros(3), cad_record=Unlabelled())


# ------------------------------------------------------------ the executor

@requires_all
def test_the_executor_returns_a_solver_verified_step_part(op, tmp_path):
    outcome = execute(METHOD, op, step_dir=tmp_path, seed=0, **FAST)
    assert outcome.method == METHOD
    assert outcome.representation == "cad_record"
    record = outcome.cad_record
    labels = record.labels
    assert labels["tip_deflection_m"]["evidence"] == "simulated"
    assert outcome.detail["evidence"] == "simulated"
    assert outcome.mass_kg == pytest.approx(labels["mass_kg"]["value"])
    material = get_material(op.problem.material_id)
    assert outcome.mass_kg == pytest.approx(
        record.geometry.volume_m3 * material.density_kg_m3, rel=1e-9)
    assert Path(outcome.detail["step_path"]).exists()
    assert record.provenance.generator == outcome.detail["family"]
    assert outcome.detail["family"] in DEFAULT_FAMILIES
    assert outcome.detail["parameters"]["length_m"] == pytest.approx(
        op.problem.geometry.length_m)
    assert outcome.detail["n_verified"] <= FAST["top_k"]
    assert outcome.detail["n_screened"] <= FAST["candidates"]
    assert "not assessed" in outcome.detail["stress_note"]
    if outcome.feasible:
        assert outcome.constraints["deflection"] >= 0.0
        assert outcome.detail["tip_deflection_m"] <= op.max_deflection_m


@requires_all
def test_the_ranker_orders_and_the_solver_decides(op, tmp_path):
    """A ranker that lies about every candidate still cannot put an unverified
    part in the outcome: the winner is whatever the solver labelled feasible
    among the parts the lie sent it. It may be a worse part. It is never an
    unchecked one."""
    def lying_ranker(candidates, material, case):
        honest = proxy_ranker(candidates, material, case)
        return honest[::-1] * 1e-3            # reversed and absurdly optimistic

    lied = run(op, ranker=lying_ranker, step_dir=tmp_path / "lie", seed=3, **FAST)
    assert lied.representation == "cad_record"
    assert lied.detail["ranker"] == "shape_surrogate"   # a supplied ranker
    assert lied.cad_record.labels["tip_deflection_m"]["evidence"] == "simulated"
    solver = lied.detail["tip_deflection_m"]
    assert abs(lied.cad_record.labels["tip_deflection_m"]["value"]) == solver
    # the prediction it acted on is kept beside the solver number, and here
    # it is wrong by construction
    assert lied.detail["predicted_deflection_m"] < solver


@requires_all
def test_candidates_respect_the_problem_envelope(op, tmp_path):
    outcome = run(op, step_dir=tmp_path, seed=1, **FAST)
    geometry = op.problem.geometry
    params = outcome.detail["parameters"]
    height = params.get("height_m", params.get("thickness_m"))
    if geometry.max_height_m is not None:
        assert height <= geometry.max_height_m
    if geometry.max_width_m is not None:
        assert params["width_m"] <= geometry.max_width_m


# ---------------------------------------------------------------- the loop

@requires_all
def test_the_loop_runs_the_cad_strategy_and_logs_a_step_path(op, tmp_path):
    """Routed through the registry with only the CAD representation
    available, the loop has one applicable method and has to run it."""
    context = ProblemContext(geometry="cad_family",
                             representations=("cad_family",),
                             slenderness=op.problem.geometry.length_m
                             / op.problem.geometry.max_height_m,
                             material_class="isotropic",
                             has_stress_constraint=True)
    selector = StrategySelector(DEFAULT_REGISTRY, context)
    reasoner = RegistryRoutingReasoner(selector)
    config = LoopConfig(max_iterations=1, seed=5,
                        cad_options={**FAST, "step_dir": str(tmp_path)})
    loop = DesignLoop(op, config=config, reasoner=reasoner)
    result = loop.run()
    assert result.iterations == 1
    episode = result.episodes[0]
    assert episode.strategy_used.startswith(METHOD)
    genome = episode.design_genome
    assert genome["method"] == METHOD
    assert genome["representation"] == "cad_record"
    assert genome["family"] in DEFAULT_FAMILIES
    assert Path(genome["step_path"]).exists()
    assert genome["evidence"] == "simulated"
    assert reasoner.unmet_recommendations == []
