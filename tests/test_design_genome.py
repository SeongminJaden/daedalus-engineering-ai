"""Design genome: validity, mass, bounds/mutation, realize()."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.design_genome import (  # noqa: E402
    DesignBounds, DesignGenome, HollowRectangleSection, Interval, realize,
)
from core.materials import get_material  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402


def make_section(b=0.05, h=0.08, t=0.005):
    return HollowRectangleSection(
        outer_width_m=b, outer_height_m=h, wall_thickness_m=t
    )


def make_genome(**kw):
    return DesignGenome(section=make_section(**kw), material_id="al_7075_t6")


# --- validity --------------------------------------------------------------
def test_reasonable_section_is_valid():
    assert make_section().is_valid()
    assert make_section().validity_reason() is None


def test_wall_too_thick_is_invalid():
    """t >= min(b,h)/2 leaves no cavity."""
    s = make_section(b=0.05, h=0.08, t=0.025)     # min(b,h)/2 == 0.025
    assert not s.is_valid()
    assert "no cavity" in s.validity_reason()


def test_wall_beyond_half_is_invalid():
    s = make_section(b=0.02, h=0.02, t=0.015)
    assert not s.is_valid()


def test_inner_dimensions_negative_is_invalid():
    s = make_section(b=0.01, h=0.01, t=0.009)
    assert s.inner_width_m() < 0
    assert not s.is_valid()


def test_invalid_section_properties_raises():
    with pytest.raises(ValueError, match="invalid section"):
        make_section(b=0.02, h=0.02, t=0.02).section_properties()


def test_inner_dimensions():
    s = make_section(b=0.05, h=0.08, t=0.005)
    assert s.inner_width_m() == pytest.approx(0.04)
    assert s.inner_height_m() == pytest.approx(0.07)


# --- mass: hand-checked ----------------------------------------------------
def test_mass_hand_check():
    """b=0.05 h=0.08 t=0.005 -> A = 0.05*0.08 - 0.04*0.07 = 0.0012 m^2.
    mass = A * L * rho = 0.0012 * 0.5 * 2810 = 1.686 kg."""
    s = make_section()
    assert s.section_properties().area_m2 == pytest.approx(0.0012, rel=1e-12)
    assert s.mass(0.5, 2810.0) == pytest.approx(1.686, rel=1e-12)


def test_mass_scales_linearly():
    s = make_section()
    assert s.mass(1.0, 2810.0) == pytest.approx(2.0 * s.mass(0.5, 2810.0))
    assert s.mass(0.5, 5620.0) == pytest.approx(2.0 * s.mass(0.5, 2810.0))


def test_mass_rejects_bad_inputs():
    s = make_section()
    with pytest.raises(ValueError):
        s.mass(0.0, 2810.0)
    with pytest.raises(ValueError):
        s.mass(0.5, -1.0)


# --- serialization ---------------------------------------------------------
def test_genome_round_trip():
    g = make_genome()
    assert DesignGenome.from_dict(g.to_dict()) == g


def test_genome_dict_is_plain_json_types():
    d = make_genome().to_dict()
    assert d["material_id"] == "al_7075_t6"
    assert isinstance(d["section"]["outer_width_m"], float)


# --- bounds and mutation ---------------------------------------------------
def test_interval_rejects_inverted_range():
    with pytest.raises(Exception):
        Interval(min=1.0, max=0.0)


def test_mutation_stays_in_bounds_and_valid():
    """Many draws with a large scale must never escape the bounds."""
    bounds, rng, g = DesignBounds(), random.Random(1234), make_genome()
    for _ in range(500):
        g = bounds.mutate(g, rng, scale=0.5)
        s = g.section
        assert bounds.outer_width_m.min <= s.outer_width_m <= bounds.outer_width_m.max
        assert bounds.outer_height_m.min <= s.outer_height_m <= bounds.outer_height_m.max
        assert (bounds.wall_thickness_m.min <= s.wall_thickness_m
                <= bounds.wall_thickness_m.max)
        assert g.is_valid(), g.validity_reason()


def test_mutation_actually_changes_something():
    bounds, rng = DesignBounds(), random.Random(7)
    g = make_genome()
    assert bounds.mutate(g, rng, scale=0.2).section != g.section


def test_mutation_is_deterministic_for_a_seed():
    bounds = DesignBounds()
    a = bounds.mutate(make_genome(), random.Random(42), scale=0.2)
    b = bounds.mutate(make_genome(), random.Random(42), scale=0.2)
    assert a == b


def test_mutation_rejects_bad_scale():
    with pytest.raises(ValueError):
        DesignBounds().mutate(make_genome(), random.Random(0), scale=0.0)


def test_clamp_repairs_impossible_wall():
    """Clamping each variable alone can still leave t >= min(b,h)/2."""
    bounds = DesignBounds()
    repaired = bounds.clamp_section(
        HollowRectangleSection(
            outer_width_m=0.010, outer_height_m=0.010, wall_thickness_m=0.010
        )
    )
    assert repaired.is_valid()


def test_sample_is_always_valid():
    bounds, rng = DesignBounds(), random.Random(99)
    for _ in range(200):
        assert bounds.sample(rng, "al_7075_t6").is_valid()


# --- realize ---------------------------------------------------------------
def test_realize_is_fully_specified():
    problem, genome = build_mvp_problem(), make_genome()
    spec = realize(genome, problem)

    assert spec["problem_name"] == "mvp_cantilever_link"
    assert spec["length_m"] == problem.geometry.length_m
    assert spec["material"]["id"] == "al_7075_t6"
    for key in ("section", "section_properties", "loads",
                "boundary_conditions", "constraints", "objectives", "mass_kg"):
        assert key in spec, f"realized spec missing {key}"


def test_realize_mass_is_consistent():
    problem, genome = build_mvp_problem(), make_genome()
    spec = realize(genome, problem)
    rho = get_material("al_7075_t6").density_kg_m3
    expected = genome.section.mass(problem.geometry.length_m, rho)
    assert spec["mass_kg"] == pytest.approx(expected, rel=1e-12)
    # and against the hand figure: 0.0012 * 0.5 * 2810
    assert spec["mass_kg"] == pytest.approx(1.686, rel=1e-12)


def test_realize_rejects_material_mismatch():
    problem = build_mvp_problem()
    genome = DesignGenome(section=make_section(), material_id="al_6061_t6")
    with pytest.raises(ValueError, match="material mismatch"):
        realize(genome, problem)


def test_realize_rejects_invalid_genome():
    problem = build_mvp_problem()
    genome = DesignGenome(
        section=make_section(b=0.02, h=0.02, t=0.02), material_id="al_7075_t6"
    )
    with pytest.raises(ValueError, match="invalid genome"):
        realize(genome, problem)
