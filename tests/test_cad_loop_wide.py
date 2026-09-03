"""The generative CAD search widened: eleven families, five load cases,
nineteen materials, and manufacturability as a preference.

Every check here is about what the widening must not break. A disc family
must be refused by name rather than stretched into a cantilever; a load case
must reach the solver as itself; a material chosen by scaling must be solved
again before it is reported; and a rule based preference must not be able to
pass or fail a part on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.execution.cad import (LENGTH_PARAMETER, NO_LENGTH_REASON, METHOD,
                                 SEARCHABLE_FAMILIES, UnsearchableFamily,
                                 impose_length, run)
from agent.execution.outcome import SOLVER_RESPONSES
from core.part_dataset import FAMILIES, sample_parameters
from core.part_dataset.labeller import PRIMARY_LABEL, LoadKind, labelling_available
from geometry.cad_export.kernel import kernel_available
from geometry.manufacturability import Process
from nodes import step_analyzer as sa
from optimization.constraints import build_optimization_problem
from projects.robotic_link.problem import build_mvp_problem

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="needs build123d, OCP, gmsh and CalculiX")


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


# --- which families can be searched at all -----------------------------------

def test_eleven_families_take_a_length_and_the_two_discs_are_named():
    """Thirteen families exist; eleven can be a cantilever of a stated length.
    The flange and the gear blank are discs whose axial extent is a thickness,
    and they are refused by name with that reason rather than stretched."""
    assert len(FAMILIES) == 13
    assert len(SEARCHABLE_FAMILIES) == 11
    assert set(NO_LENGTH_REASON) == set(FAMILIES) - set(SEARCHABLE_FAMILIES)
    assert set(NO_LENGTH_REASON) == {"flange", "gear_blank"}
    for name, reason in NO_LENGTH_REASON.items():
        assert "thickness" in reason or "hub" in reason


def test_the_imposed_length_is_the_span_for_every_searchable_family():
    """Whatever parameter carries the span, the part ends up that long."""
    rng = np.random.default_rng(0)
    for name in SEARCHABLE_FAMILIES:
        fam = FAMILIES[name]
        params = impose_length(fam, sample_parameters(fam, rng), 0.5)
        if LENGTH_PARAMETER[name] == "split":
            assert params["length_1_m"] + params["length_2_m"] == pytest.approx(0.5)
        else:
            assert params[LENGTH_PARAMETER[name]] == pytest.approx(0.5)


def test_the_split_keeps_the_ratio_the_sampler_drew():
    """The stepped shaft is scaled, not reshaped: the step stays where the
    sampler put it, proportionally."""
    fam = FAMILIES["stepped_shaft"]
    params = sample_parameters(fam, np.random.default_rng(3))
    before = params["length_1_m"] / (params["length_1_m"] + params["length_2_m"])
    after = impose_length(fam, dict(params), 0.5)
    assert after["length_1_m"] / 0.5 == pytest.approx(before)


def test_a_disc_family_is_refused_by_name():
    fam = FAMILIES["flange"]
    with pytest.raises(UnsearchableFamily, match="thickness"):
        impose_length(fam, sample_parameters(fam, np.random.default_rng(0)), 0.5)


@requires_all
def test_a_search_of_only_discs_refuses_instead_of_returning_nothing(op, tmp_path):
    with pytest.raises(UnsearchableFamily, match="no family"):
        run(op, families=("flange", "gear_blank"), step_dir=tmp_path / "discs")


# --- the load cases ----------------------------------------------------------

@requires_all
@pytest.mark.parametrize("kind", [LoadKind.AXIAL, LoadKind.TORSION,
                                  LoadKind.THERMAL_GRADIENT])
def test_each_load_case_reaches_the_solver_and_names_its_own_response(
        op, tmp_path, kind):
    """The record has to carry the label that load case produces, which is
    only true if the kind survived every hand-off down to CalculiX."""
    out = run(op, candidates=2, top_k=1, seed=2, families=("box",),
              load_kind=kind, step_dir=tmp_path / kind.value)
    name = PRIMARY_LABEL[kind][0]
    assert out.detail["load_kind"] == kind.value
    assert out.detail["primary_response_name"] == name
    assert name in out.cad_record.labels
    assert name in SOLVER_RESPONSES
    assert out.detail["primary_response"] > 0.0
    assert out.cad_record.labels[name]["evidence"] == "simulated"


@requires_all
def test_without_a_stated_limit_there_is_no_margin(op, tmp_path):
    """The engineering problem states a deflection limit and nothing else, so
    a torsion search has no limit to report. Zero would read as a part sitting
    exactly on a limit nobody set."""
    out = run(op, candidates=2, top_k=1, seed=2, families=("box",),
              load_kind=LoadKind.TORSION, step_dir=tmp_path / "no_limit")
    assert out.constraints == {}
    assert out.detail["response_limit"] is None
    assert out.feasible


@requires_all
def test_a_stated_limit_is_applied_to_the_case_it_belongs_to(op, tmp_path):
    """An impossible twist limit must fail the search, which is the check
    that the limit is compared against the twist and not against something
    else that happens to be small."""
    out = run(op, candidates=2, top_k=1, seed=2, families=("box",),
              load_kind=LoadKind.TORSION, response_limit=1e-12,
              step_dir=tmp_path / "tight")
    assert not out.feasible
    assert out.constraints["twist_rad"] < 0.0


# --- materials ---------------------------------------------------------------

@requires_all
def test_a_material_chosen_by_scaling_is_solved_again_before_it_is_reported(
        op, tmp_path):
    """Scaling picks among materials for free; the answer is a solve. The
    reported response must be the solved one, and it must sit within the
    Poisson residual bound of the scaled value that chose it."""
    out = run(op, candidates=3, top_k=2, seed=5, families=("box", "hollow_rect"),
              materials=["al_7075_t6", "ss_304", "ti_6al_4v"],
              step_dir=tmp_path / "materials")
    assert set(out.detail["materials_searched"]) == {"al_7075_t6", "ss_304", "ti_6al_4v"}
    assert out.detail["n_options"] == 3 * out.detail["n_verified"]
    record = out.cad_record
    assert record.material_id == out.detail["material_id"]
    for name, item in record.labels.items():
        if isinstance(item, dict) and "value" in item:
            assert not item.get("derived"), f"{name} is a scaled label"
    if out.detail["chosen_by_scaling_then_solved"]:
        scaled = out.detail["scaled_response_before_solve"]
        solved = out.detail["primary_response"]
        assert abs(scaled - solved) / solved < 0.05, (scaled, solved)


# --- manufacturability as a preference ---------------------------------------

@requires_all
def test_manufacturability_orders_but_cannot_pass_or_fail(op, tmp_path):
    """The DFM report is attached and used to order, and the part is still
    the one the solver verified. The grade is a rule set, not evidence."""
    out = run(op, candidates=4, top_k=2, seed=7, families=("box", "hollow_rect"),
              process=Process.CNC_MILLING, step_dir=tmp_path / "dfm")
    assert out.detail["process"] == "cnc_milling"
    assert out.detail["dfm_rules_measured"] >= 1
    assert out.detail["dfm_rules_failed"] <= out.detail["dfm_rules_measured"]
    assert "not an evidence level" in out.detail["dfm_note"]
    assert out.method == METHOD
    # Feasibility is the solver's, and the DFM count cannot change it.
    limit = op.max_deflection_m
    assert out.feasible == (out.detail["primary_response"] <= limit)


@requires_all
def test_the_widened_search_spans_families_and_returns_one_solved_part(op, tmp_path):
    """The whole thing at once: eleven families, several materials, a process,
    and a disc family in the request that is refused by name."""
    out = run(op, candidates=11, top_k=3, seed=1,
              families=SEARCHABLE_FAMILIES + ("flange",),
              materials=["al_7075_t6", "al_6061_t6", "ss_304"],
              process=Process.CNC_MILLING, step_dir=tmp_path / "wide")
    assert out.detail["family"] in SEARCHABLE_FAMILIES
    assert "flange" in dict(out.detail["refused"])
    assert len(out.detail["families_searched"]) == 11
    assert out.cad_record.labels["tip_deflection_m"]["evidence"] == "simulated"
    assert out.mass_kg > 0.0
