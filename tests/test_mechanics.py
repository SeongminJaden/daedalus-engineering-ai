"""Stress states, torsion of non-circular sections, vessels and contact.

The test that carries the phase is
`test_the_in_plane_shear_is_not_the_absolute_shear_under_biaxial_tension`. An
equibiaxial state has zero in-plane shear and a real absolute shear of half the
principal stress, so a design reading the in-plane number concludes there is no
shear at all.
"""

import math

import numpy as np
import pytest

from core.registry import ProblemContext, build_default_registry
from physics.mechanics import (SUBSURFACE_SHEAR_DEPTH_RATIO, effective_modulus_pa,
                               effective_radius_m, kt_plate_with_hole,
                               kt_shoulder_fillet_bending,
                               line_contact_half_width_m,
                               line_contact_pressure_pa, principal_stress_2d,
                               principal_stress_3d, rectangle_coefficients,
                               solid_rectangle, sphere_contact, thick_wall,
                               thin_closed_section, thin_open_section,
                               thin_wall, thin_wall_error,
                               transform_stress_2d, von_mises_3d)

STEEL_E, STEEL_NU, STEEL_G = 207e9, 0.29, 79e9


# --- stress state ------------------------------------------------------------

def test_principal_stresses_of_a_uniaxial_state():
    state = principal_stress_2d(100e6, 0.0, 0.0)
    assert state.sigma_1_pa == pytest.approx(100e6)
    assert state.sigma_2_pa == pytest.approx(0.0)
    assert state.in_plane_max_shear_pa == pytest.approx(50e6)
    assert state.absolute_max_shear_pa == pytest.approx(50e6)
    assert not state.absolute_shear_is_out_of_plane


def test_pure_shear_gives_equal_and_opposite_principals_at_45_degrees():
    state = principal_stress_2d(0.0, 0.0, 60e6)
    assert state.sigma_1_pa == pytest.approx(60e6)
    assert state.sigma_2_pa == pytest.approx(-60e6)
    assert state.principal_angle_deg == pytest.approx(45.0)


def test_the_in_plane_shear_is_not_the_absolute_shear_under_biaxial_tension():
    """The trap, and it is silent.

    Equibiaxial tension has ZERO in-plane shear, because the Mohr circle
    collapses to a point. The real maximum shear is half the principal stress,
    on a plane through the zero third principal that plane stress always
    carries. A design reading the in-plane number concludes there is no shear
    at all.
    """
    state = principal_stress_2d(100e6, 100e6, 0.0)
    assert state.in_plane_max_shear_pa == pytest.approx(0.0)
    assert state.absolute_max_shear_pa == pytest.approx(50e6)
    assert state.absolute_shear_is_out_of_plane

    # It happens whenever the two principals share a sign, not only when equal.
    same_sign = principal_stress_2d(120e6, 40e6, 0.0)
    assert same_sign.absolute_max_shear_pa > same_sign.in_plane_max_shear_pa
    # And not when they straddle zero, where the in-plane value already spans
    # the full range.
    opposite = principal_stress_2d(120e6, -40e6, 0.0)
    assert opposite.absolute_max_shear_pa == pytest.approx(
        opposite.in_plane_max_shear_pa)


def test_transformation_round_trips():
    original = (120e6, -40e6, 50e6)
    rotated = transform_stress_2d(*original, 37.0)
    assert transform_stress_2d(*rotated, -37.0) == pytest.approx(original)


def test_the_principal_directions_carry_no_shear():
    """Which is what makes them principal."""
    state = principal_stress_2d(120e6, -40e6, 50e6)
    _, _, shear = transform_stress_2d(120e6, -40e6, 50e6,
                                      state.principal_angle_deg)
    assert abs(shear) < 1e-6 * abs(state.sigma_1_pa)


def test_the_two_and_three_dimensional_paths_agree():
    """An independent cross-check: eigenvalues against the closed form."""
    sx, sy, txy = 120e6, -40e6, 50e6
    tensor = np.array([[sx, txy, 0.0], [txy, sy, 0.0], [0.0, 0.0, 0.0]])
    state = principal_stress_2d(sx, sy, txy)
    assert principal_stress_3d(tensor) == pytest.approx(
        sorted([state.sigma_1_pa, state.sigma_2_pa, 0.0], reverse=True))
    assert von_mises_3d(tensor) == pytest.approx(state.von_mises_pa)


def test_an_unsymmetric_stress_tensor_is_refused():
    """No equilibrium state has one."""
    with pytest.raises(ValueError, match="symmetric"):
        principal_stress_3d(np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]]))


# --- torsion -----------------------------------------------------------------

def test_the_rectangle_coefficients_match_the_published_table():
    assert rectangle_coefficients(1.0) == pytest.approx((0.208, 0.1406))
    assert rectangle_coefficients(2.0) == pytest.approx((0.246, 0.229))
    # Both tend to one third as the section becomes a thin strip.
    alpha, beta = rectangle_coefficients(50.0)
    assert alpha == pytest.approx(1.0 / 3.0)
    assert beta == pytest.approx(1.0 / 3.0)


def test_solid_rectangle_torsion_matches_the_hand_calculation():
    torque, long_side, short_side = 500.0, 0.050, 0.025
    alpha, beta = rectangle_coefficients(2.0)
    result = solid_rectangle(torque, long_side, short_side, STEEL_G)
    assert result.max_shear_stress_pa == pytest.approx(
        torque / (alpha * long_side * short_side ** 2))
    assert result.torsion_constant_m4 == pytest.approx(
        beta * long_side * short_side ** 3)


def test_slitting_a_tube_destroys_its_torsional_stiffness():
    """The largest single error available in this module.

    A closed tube carries torque as shear flow around its enclosed area.
    Slitting it lengthwise removes that path entirely. Measured on a 50 mm
    square tube with a 2 mm wall, the loss is a factor of several hundred in
    stiffness and tens in stress.
    """
    torque, side, wall = 500.0, 0.050, 0.002
    midline = side - wall
    closed = thin_closed_section(torque, midline ** 2, wall, 4.0 * midline,
                                 STEEL_G)
    opened = thin_open_section(torque, [(4.0 * midline, wall)], STEEL_G)

    assert closed.max_shear_stress_pa == pytest.approx(
        torque / (2.0 * midline ** 2 * wall))
    assert opened.torsion_constant_m4 == pytest.approx(
        4.0 * midline * wall ** 3 / 3.0)
    assert closed.torsion_constant_m4 / opened.torsion_constant_m4 > 100.0
    assert opened.max_shear_stress_pa > 20.0 * closed.max_shear_stress_pa


def test_an_open_section_is_governed_by_its_thickest_strip():
    thin_strips = thin_open_section(500.0, [(0.100, 0.002), (0.100, 0.002)],
                                    STEEL_G)
    mixed = thin_open_section(500.0, [(0.100, 0.002), (0.100, 0.004)],
                              STEEL_G)
    assert mixed.torsion_constant_m4 > thin_strips.torsion_constant_m4


# --- vessels -----------------------------------------------------------------

def test_hoop_is_twice_longitudinal_in_a_thin_cylinder():
    """Which is why a cylinder splits along its length."""
    stress = thin_wall(10e6, 0.050, 0.005)
    assert stress.hoop_pa == pytest.approx(2.0 * stress.longitudinal_pa)
    assert stress.hoop_pa == pytest.approx(10e6 * 0.050 / 0.005)


def test_the_radial_stress_at_the_bore_equals_minus_the_pressure():
    """A check on the Lame algebra that needs no reference value."""
    for outer in (0.055, 0.060, 0.100):
        stress = thick_wall(10e6, 0.050, outer)
        assert stress.radial_pa == pytest.approx(-10e6, rel=1e-12)


def test_the_hoop_stress_peaks_at_the_bore():
    inner, outer = 0.050, 0.070
    at_bore = thick_wall(10e6, inner, outer, inner).hoop_pa
    at_mid = thick_wall(10e6, inner, outer, 0.060).hoop_pa
    at_outer = thick_wall(10e6, inner, outer, outer).hoop_pa
    assert at_bore > at_mid > at_outer


def test_the_thin_wall_error_depends_on_which_radius_is_used():
    """The familiar r/t rule is about the INNER-radius form, not thin-wall
    theory in general.

    Measured: with the inner radius the error is 4.98% at r/t of 10, which is
    where the rule comes from. With the mean radius, which this module uses, it
    is 0.23% there and still under 4% at r/t of 2.
    """
    thickness = 0.005
    for ratio, mean_error in ((2, 0.0385), (10, 0.0023)):
        inner = ratio * thickness
        assert thin_wall_error(inner, thickness) == pytest.approx(
            mean_error, abs=5e-4)

    inner = 10 * thickness
    exact = thick_wall(1.0, inner, inner + thickness).hoop_pa
    inner_form = 1.0 * inner / thickness
    assert (exact - inner_form) / exact == pytest.approx(0.0498, abs=5e-4)


def test_an_impossible_vessel_is_refused():
    with pytest.raises(ValueError, match="outer radius"):
        thick_wall(10e6, 0.060, 0.050)
    with pytest.raises(ValueError, match="inside the wall"):
        thick_wall(10e6, 0.050, 0.060, at_radius_m=0.070)


# --- contact -----------------------------------------------------------------

def test_hertz_sphere_contact_matches_the_hand_calculation():
    force, radius = 1000.0, 0.010
    contact = sphere_contact(force, radius, None, STEEL_E, STEEL_NU, STEEL_E,
                             STEEL_NU)
    modulus = effective_modulus_pa(STEEL_E, STEEL_NU, STEEL_E, STEEL_NU)
    expected_a = (3.0 * force * radius / (4.0 * modulus)) ** (1.0 / 3.0)
    assert contact.contact_radius_m == pytest.approx(expected_a, rel=1e-12)
    assert contact.max_pressure_pa == pytest.approx(
        3.0 * force / (2.0 * math.pi * expected_a ** 2), rel=1e-12)


def test_contact_pressure_grows_only_as_the_cube_root_of_load():
    """Which is why contact stress is governed by geometry, not by load."""
    single = sphere_contact(1000.0, 0.010, None, STEEL_E, STEEL_NU, STEEL_E,
                            STEEL_NU)
    double = sphere_contact(2000.0, 0.010, None, STEEL_E, STEEL_NU, STEEL_E,
                            STEEL_NU)
    assert double.max_pressure_pa / single.max_pressure_pa == pytest.approx(
        2.0 ** (1.0 / 3.0), rel=1e-9)


def test_the_peak_shear_is_below_the_surface():
    """Which is why rolling contact fatigue starts subsurface."""
    contact = sphere_contact(1000.0, 0.010, None, STEEL_E, STEEL_NU, STEEL_E,
                             STEEL_NU)
    assert contact.max_shear_depth_m == pytest.approx(
        SUBSURFACE_SHEAR_DEPTH_RATIO * contact.contact_radius_m)
    assert contact.max_shear_depth_m > 0.0
    assert contact.max_shear_pa < contact.max_pressure_pa


def test_a_flat_partner_is_expressed_as_an_infinite_radius():
    assert effective_radius_m(0.010, None) == pytest.approx(0.010)
    assert effective_radius_m(0.010, 0.010) == pytest.approx(0.005)
    with pytest.raises(ValueError, match="None for a flat"):
        effective_radius_m(0.010, 0.0)


def test_line_contact_pressure_matches_its_definition():
    load = 5.0e5
    half_width = line_contact_half_width_m(load, 0.010, None, STEEL_E,
                                           STEEL_NU, STEEL_E, STEEL_NU)
    assert line_contact_pressure_pa(load, half_width) == pytest.approx(
        2.0 * load / (math.pi * half_width))


def test_kt_for_a_small_hole_approaches_the_classical_three():
    """The known result for a circular hole in an infinite plate."""
    assert kt_plate_with_hole(1e-9, 1.0) == pytest.approx(3.0, abs=1e-6)
    # And it falls as the hole takes up more of the width.
    assert (kt_plate_with_hole(0.1, 1.0) > kt_plate_with_hole(0.3, 1.0)
            > kt_plate_with_hole(0.5, 1.0))


def test_kt_rises_steeply_as_a_fillet_sharpens():
    """Which is why a sharp shoulder is where a shaft breaks."""
    sharp = kt_shoulder_fillet_bending(0.0015, 0.030, 0.045)
    generous = kt_shoulder_fillet_bending(0.006, 0.030, 0.045)
    assert sharp > generous > 1.0
    assert sharp / generous > 1.2


# --- registry ----------------------------------------------------------------

def test_the_new_methods_are_gated():
    registry = build_default_registry()
    none = ProblemContext(geometry="assembly", representations=("assembly",),
                          has_multiaxial_stress=False,
                          has_noncircular_torsion=False,
                          has_internal_pressure=False,
                          has_concentrated_contact=False)
    candidates = registry.query(none)
    for name in ("stress_transformation", "noncircular_torsion",
                 "pressure_vessel", "hertz_contact"):
        assert name not in candidates.names()
        assert candidates.reason(name)

    present = ProblemContext(geometry="assembly",
                             representations=("assembly",),
                             has_multiaxial_stress=True,
                             has_noncircular_torsion=True,
                             has_internal_pressure=True,
                             has_concentrated_contact=True)
    names = registry.query(present).names()
    for name in ("stress_transformation", "noncircular_torsion",
                 "pressure_vessel", "hertz_contact"):
        assert name in names
