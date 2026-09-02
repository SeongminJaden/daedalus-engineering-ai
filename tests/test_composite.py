"""Classical laminate theory and ply failure.

Three tests carry the phase. `test_isotropic_plies_recover_isotropic_stiffness`
checks the whole ABD path against a closed form nobody can argue with.
`test_tsai_wu_reproduces_every_uniaxial_strength` checks the criterion against
the data it was built from. `test_layup_changes_stiffness_strength_and_the_failing_ply`
is the reason CLT exists: the same plies stacked differently are a different
material.
"""

import math

import numpy as np
import pytest

from core.materials import get_material
from core.registry import Category, ProblemContext, build_default_registry
from physics.composite import (FailureMode, Lamina, LaminaStrength, Laminate,
                               abd_matrices, first_ply_failure,
                               max_stress_ratio, ply_states, reduced_stiffness,
                               stress_transformation, transformed_stiffness,
                               tsai_wu_coefficients, tsai_wu_index,
                               tsai_wu_strength_ratio)

PLY_THICKNESS_M = 0.125e-3


@pytest.fixture(scope="module")
def cfrp():
    return get_material("cfrp_ud")


@pytest.fixture(scope="module")
def strength(cfrp):
    return LaminaStrength.from_material(cfrp)


def cfrp_lamina(cfrp) -> Lamina:
    return Lamina(e1_pa=cfrp.e1_pa, e2_pa=cfrp.e2_pa, g12_pa=cfrp.g12_pa,
                  nu12=cfrp.nu12, thickness_m=PLY_THICKNESS_M)


# --- lamina stiffness --------------------------------------------------------

def test_reduced_stiffness_matches_the_hand_calculation(cfrp):
    """Q11 = E1/(1 - nu12 nu21), and nu21 follows from reciprocity."""
    lamina = cfrp_lamina(cfrp)
    nu21 = cfrp.nu12 * cfrp.e2_pa / cfrp.e1_pa
    assert lamina.nu21 == pytest.approx(nu21)
    denominator = 1.0 - cfrp.nu12 * nu21
    q = reduced_stiffness(lamina)
    assert q[0, 0] == pytest.approx(cfrp.e1_pa / denominator, rel=1e-12)
    assert q[1, 1] == pytest.approx(cfrp.e2_pa / denominator, rel=1e-12)
    assert q[0, 1] == pytest.approx(cfrp.nu12 * cfrp.e2_pa / denominator,
                                    rel=1e-12)
    assert q[2, 2] == pytest.approx(cfrp.g12_pa)
    assert q[0, 1] == q[1, 0]


def test_the_transformation_is_the_identity_at_zero_degrees(cfrp):
    q = reduced_stiffness(cfrp_lamina(cfrp))
    assert transformed_stiffness(q, 0.0) == pytest.approx(q, rel=1e-12)


def test_ninety_degrees_swaps_the_two_axes(cfrp):
    q = reduced_stiffness(cfrp_lamina(cfrp))
    rotated = transformed_stiffness(q, 90.0)
    assert rotated[0, 0] == pytest.approx(q[1, 1], rel=1e-12)
    assert rotated[1, 1] == pytest.approx(q[0, 0], rel=1e-12)
    assert rotated[2, 2] == pytest.approx(q[2, 2], rel=1e-12)
    # The off-axis terms are zero to floating point. cos(90 degrees) is 6.1e-17
    # rather than zero, so this has to be judged relative to Q11 and not
    # against an absolute tolerance meaningless at these magnitudes.
    assert abs(rotated[0, 2]) / q[0, 0] < 1e-15
    assert abs(rotated[1, 2]) / q[0, 0] < 1e-15


def test_an_off_axis_ply_couples_extension_to_shear(cfrp):
    """Q16 and Q26 are zero only at 0 and 90 degrees.

    Nonzero, they are why an unbalanced laminate shears when pulled.
    """
    q = reduced_stiffness(cfrp_lamina(cfrp))
    rotated = transformed_stiffness(q, 45.0)
    assert abs(rotated[0, 2]) > 0.01 * q[0, 0]
    # And the sign flips with the angle, which is what lets a balanced pair
    # cancel.
    assert (transformed_stiffness(q, -45.0)[0, 2]
            == pytest.approx(-rotated[0, 2], rel=1e-12))


def test_a_ply_with_an_inadmissible_poisson_ratio_is_refused():
    with pytest.raises(ValueError, match="nu12"):
        Lamina(e1_pa=100e9, e2_pa=10e9, g12_pa=5e9, nu12=1.5,
               thickness_m=1e-4)


# --- ABD ---------------------------------------------------------------------

def test_isotropic_plies_recover_isotropic_stiffness():
    """The check with a closed form: A11 must be E t / (1 - nu^2).

    Stacked at arbitrary angles, isotropic plies must give an isotropic A with
    A11 = A22, A16 = 0 and A66 = (A11 - A12)/2, whatever the angles were. If
    the transformation or the integration were wrong, an arbitrary stack would
    not come out isotropic.
    """
    e, nu = 70e9, 0.33
    ply = Lamina(e1_pa=e, e2_pa=e, g12_pa=e / (2.0 * (1.0 + nu)), nu12=nu,
                 thickness_m=PLY_THICKNESS_M)
    laminate = Laminate(plies=[ply] * 8,
                        angles_deg=[0, 30, 60, 90, 90, 60, 30, 0])
    a = abd_matrices(laminate).a
    thickness = laminate.thickness_m

    assert a[0, 0] == pytest.approx(e * thickness / (1.0 - nu ** 2), rel=1e-12)
    assert a[1, 1] == pytest.approx(a[0, 0], rel=1e-12)
    assert abs(a[0, 2]) / a[0, 0] < 1e-15
    assert a[2, 2] == pytest.approx(0.5 * (a[0, 0] - a[0, 1]), rel=1e-12)


def test_a_symmetric_stack_has_no_extension_bending_coupling(cfrp):
    """B = 0 for symmetry, which is why symmetric layups are the default.

    It depends on measuring the ply heights from the MIDPLANE. From any other
    datum B would not vanish and the coupling would look real.
    """
    symmetric = Laminate.from_material(cfrp, [0, 90, 90, 0], PLY_THICKNESS_M)
    abd = abd_matrices(symmetric)
    assert symmetric.is_symmetric()
    assert np.abs(abd.b).max() / np.abs(abd.a).max() < 1e-15
    assert not abd.couples_extension_to_bending


def test_an_unsymmetric_stack_does_couple(cfrp):
    unsymmetric = Laminate.from_material(cfrp, [0, 90], PLY_THICKNESS_M)
    abd = abd_matrices(unsymmetric)
    assert not unsymmetric.is_symmetric()
    assert abd.couples_extension_to_bending
    assert np.abs(abd.b).max() > 0.0


def test_abd_matches_a_hand_summed_two_ply_stack(cfrp):
    """A_11 = sum Qbar_11 t, summed by hand for [0/90]s."""
    laminate = Laminate.from_material(cfrp, [0, 90, 90, 0], PLY_THICKNESS_M)
    q = reduced_stiffness(cfrp_lamina(cfrp))
    q0 = transformed_stiffness(q, 0.0)[0, 0]
    q90 = transformed_stiffness(q, 90.0)[0, 0]
    expected = (q0 + q90) * 2.0 * PLY_THICKNESS_M
    assert abd_matrices(laminate).a[0, 0] == pytest.approx(expected, rel=1e-12)


def test_a_quasi_isotropic_layup_is_isotropic_in_plane(cfrp):
    """[0/45/-45/90]s built from a material with a 13:1 stiffness ratio."""
    quasi = Laminate.from_material(cfrp, [0, 45, -45, 90, 90, -45, 45, 0],
                                   PLY_THICKNESS_M)
    a = abd_matrices(quasi).a
    assert a[0, 0] == pytest.approx(a[1, 1], rel=1e-12)
    assert abs(a[0, 2]) / a[0, 0] < 1e-12
    assert a[2, 2] == pytest.approx(0.5 * (a[0, 0] - a[0, 1]), rel=1e-12)

    # The plies themselves are strongly anisotropic; the stack is not.
    unidirectional = Laminate.from_material(cfrp, [0] * 8, PLY_THICKNESS_M)
    au = abd_matrices(unidirectional).a
    assert au[0, 0] / au[1, 1] > 10.0


def test_balance_and_symmetry_are_different_properties(cfrp):
    balanced_unsymmetric = Laminate.from_material(cfrp, [45, -45],
                                                  PLY_THICKNESS_M)
    assert balanced_unsymmetric.is_balanced()
    assert not balanced_unsymmetric.is_symmetric()

    symmetric_unbalanced = Laminate.from_material(cfrp, [45, 0, 45],
                                                  PLY_THICKNESS_M)
    assert symmetric_unbalanced.is_symmetric()
    assert not symmetric_unbalanced.is_balanced()


def test_a_laminate_needs_an_angle_for_every_ply(cfrp):
    with pytest.raises(ValueError, match="angles"):
        Laminate(plies=[cfrp_lamina(cfrp)] * 3, angles_deg=[0.0, 90.0])


# --- ply stresses ------------------------------------------------------------

def test_a_unidirectional_laminate_carries_its_load_along_the_fibres(cfrp):
    """N_x over the thickness, with nothing transverse and no shear."""
    laminate = Laminate.from_material(cfrp, [0] * 8, PLY_THICKNESS_M)
    load = 2.0e5
    states = ply_states(laminate, np.array([load, 0.0, 0.0]))
    expected = load / laminate.thickness_m
    for state in states:
        assert state.stress_material[0] == pytest.approx(expected, rel=1e-9)
        assert abs(state.stress_material[1]) < 1e-6 * expected
        assert abs(state.stress_material[2]) < 1e-6 * expected


def test_the_stress_transformation_is_orthogonal_in_the_right_sense():
    """Rotating by an angle and back returns the original stress."""
    stress = np.array([100e6, 20e6, 30e6])
    forward = stress_transformation(37.0)
    backward = stress_transformation(-37.0)
    assert backward @ (forward @ stress) == pytest.approx(stress, rel=1e-12)


def test_a_singular_abd_is_reported_clearly(cfrp, monkeypatch):
    """A stack with no unique response says so rather than propagating LinAlgError.

    The condition is hard to reach with physically admissible plies, since the
    Lamina validator already refuses zero moduli, so the solve is forced to
    fail here. That tests the error path that exists rather than asserting a
    singularity that a real laminate does not have: an earlier version of this
    test built a single ply with tiny moduli and asserted it was singular, and
    it was not. A single centred ply gives B = 0 and a perfectly invertible ABD.
    """
    import physics.composite.clt as clt

    def refuse(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(clt.np.linalg, "solve", refuse)
    laminate = Laminate.from_material(cfrp, [0, 90, 90, 0], PLY_THICKNESS_M)
    with pytest.raises(ValueError, match="singular"):
        ply_states(laminate, np.array([1.0, 0.0, 0.0]))


def test_a_single_centred_ply_has_no_coupling(cfrp):
    """Because its own midplane is the laminate midplane, B vanishes."""
    laminate = Laminate.from_material(cfrp, [0.0], PLY_THICKNESS_M)
    abd = abd_matrices(laminate)
    assert np.abs(abd.b).max() / np.abs(abd.a).max() < 1e-15
    assert not abd.couples_extension_to_bending


# --- failure -----------------------------------------------------------------

def test_the_five_strengths_are_required_and_not_assumed(cfrp):
    """Transverse compressive is about 2.5 times transverse tensile here.

    It was four times when the transverse tensile strength was a 50 MPa
    typical; the Hexcel 8552/AS4 sheet gives 81 MPa, and the compressive
    value is still the unsourced typical 200 MPa. Defaulting one to the other
    would still overstate transverse tensile capacity severalfold, so an
    incomplete material is refused.
    """
    complete = LaminaStrength.from_material(cfrp)
    assert (complete.transverse_compression_pa
            > 2.0 * complete.transverse_tension_pa)
    assert (complete.longitudinal_compression_pa
            < complete.longitudinal_tension_pa)

    stripped = cfrp.model_copy(update={"strength_trans_compressive_pa": None})
    with pytest.raises(ValueError, match="transverse compressive"):
        LaminaStrength.from_material(stripped)


def test_tsai_wu_coefficients_match_their_definitions(strength):
    coefficients = tsai_wu_coefficients(strength)
    xt, xc = strength.longitudinal_tension_pa, strength.longitudinal_compression_pa
    yt, yc = strength.transverse_tension_pa, strength.transverse_compression_pa
    assert coefficients["F1"] == pytest.approx(1 / xt - 1 / xc, rel=1e-12)
    assert coefficients["F2"] == pytest.approx(1 / yt - 1 / yc, rel=1e-12)
    assert coefficients["F11"] == pytest.approx(1 / (xt * xc), rel=1e-12)
    assert coefficients["F22"] == pytest.approx(1 / (yt * yc), rel=1e-12)
    assert coefficients["F66"] == pytest.approx(
        1 / strength.shear_pa ** 2, rel=1e-12)
    assert coefficients["F12"] == pytest.approx(
        -0.5 * math.sqrt(coefficients["F11"] * coefficients["F22"]), rel=1e-12)


def test_tsai_wu_reproduces_every_uniaxial_strength(strength):
    """The defining property: at each measured strength the index is exactly 1.

    A criterion that does not reproduce the data it was fitted to is wrong
    before any combined state is considered.
    """
    cases = (
        np.array([strength.longitudinal_tension_pa, 0.0, 0.0]),
        np.array([-strength.longitudinal_compression_pa, 0.0, 0.0]),
        np.array([0.0, strength.transverse_tension_pa, 0.0]),
        np.array([0.0, -strength.transverse_compression_pa, 0.0]),
        np.array([0.0, 0.0, strength.shear_pa]),
        np.array([0.0, 0.0, -strength.shear_pa]),
    )
    for stress in cases:
        assert tsai_wu_index(stress, strength) == pytest.approx(1.0, rel=1e-12)
        assert tsai_wu_strength_ratio(stress, strength) == pytest.approx(
            1.0, rel=1e-9)


def test_the_failure_index_is_not_a_reciprocal_safety_factor(strength):
    """The error the strength ratio exists to prevent.

    The criterion has linear terms, so halving the index does not double the
    allowable load. Where the material is asymmetric the discrepancy is large
    and in the unconservative direction.
    """
    stress = np.array([0.0, 0.6 * strength.transverse_tension_pa, 0.0])
    index = tsai_wu_index(stress, strength)
    ratio = tsai_wu_strength_ratio(stress, strength)
    assert ratio != pytest.approx(1.0 / index, rel=0.05)
    # And the ratio is the one that actually predicts failure.
    scaled = stress * ratio
    assert tsai_wu_index(scaled, strength) == pytest.approx(1.0, rel=1e-9)


def test_the_strength_ratio_scales_a_proportional_load_to_failure(strength):
    stress = np.array([300e6, 12e6, 20e6])
    ratio = tsai_wu_strength_ratio(stress, strength)
    assert tsai_wu_index(stress * ratio, strength) == pytest.approx(1.0,
                                                                    rel=1e-9)


def test_max_stress_names_the_governing_component(strength):
    """Which Tsai-Wu cannot: it returns one number for the whole state."""
    ratio, mode = max_stress_ratio(
        np.array([0.0, strength.transverse_tension_pa * 0.5, 0.0]), strength)
    assert mode is FailureMode.MATRIX_TENSION
    assert ratio == pytest.approx(2.0)

    _, compressive = max_stress_ratio(
        np.array([-strength.longitudinal_compression_pa * 0.5, 0.0, 0.0]),
        strength)
    assert compressive is FailureMode.FIBRE_COMPRESSION

    _, shear = max_stress_ratio(np.array([0.0, 0.0, strength.shear_pa * 0.9]),
                                strength)
    assert shear is FailureMode.SHEAR


# --- the point of the phase --------------------------------------------------

def test_layup_changes_stiffness_strength_and_the_failing_ply(cfrp, strength):
    """The same plies stacked differently are a different material.

    One load, four layups, and the stiffness spans a factor of three, the
    strength a factor of eleven, and the ply that fails first changes both its
    angle and its failure mode.
    """
    # 2.0e5 N/m failed the angle ply when the in-plane shear strength was a
    # 70 MPa typical; with the Hexcel 8552/AS4 value of 114 MPa the same load
    # gives a strength ratio of 1.08, so the load is raised to keep the case
    # the docstring describes: one layup that fails and one that does not.
    load = np.array([2.6e5, 0.0, 0.0])
    layups = {
        "unidirectional": [0] * 8,
        "cross_ply": [0, 90, 0, 90, 90, 0, 90, 0],
        "angle_ply": [45, -45, 45, -45, -45, 45, -45, 45],
        "quasi_isotropic": [0, 45, -45, 90, 90, -45, 45, 0],
    }
    results = {}
    for name, angles in layups.items():
        laminate = Laminate.from_material(cfrp, angles, PLY_THICKNESS_M)
        modulus = (abd_matrices(laminate).a[0, 0] / laminate.thickness_m)
        results[name] = (modulus, first_ply_failure(laminate, strength, load))

    stiffness = {k: v[0] for k, v in results.items()}
    assert stiffness["unidirectional"] > stiffness["cross_ply"]
    assert stiffness["cross_ply"] > stiffness["quasi_isotropic"]
    assert stiffness["quasi_isotropic"] > stiffness["angle_ply"]
    assert stiffness["unidirectional"] / stiffness["angle_ply"] > 3.0

    ratios = {k: v[1].strength_ratio for k, v in results.items()}
    assert ratios["unidirectional"] > 5.0
    assert ratios["angle_ply"] < 1.0          # this layup fails under this load
    assert max(ratios.values()) / min(ratios.values()) > 10.0

    # And the mode changes with the layup, which a single number could not say.
    assert results["unidirectional"][1].governing_mode is FailureMode.FIBRE_TENSION
    assert results["cross_ply"][1].governing_mode is FailureMode.MATRIX_TENSION
    assert results["angle_ply"][1].governing_mode is FailureMode.SHEAR
    assert results["cross_ply"][1].angle_deg == pytest.approx(90.0)


def test_unidirectional_first_ply_failure_matches_the_hand_calculation(
        cfrp, strength):
    """N over t against Xt, with no transformation to get wrong."""
    laminate = Laminate.from_material(cfrp, [0] * 8, PLY_THICKNESS_M)
    load = 2.0e5
    stress = load / laminate.thickness_m
    result = first_ply_failure(laminate, strength, np.array([load, 0.0, 0.0]))
    assert result.strength_ratio == pytest.approx(
        strength.longitudinal_tension_pa / stress, rel=1e-6)
    assert result.governing_mode is FailureMode.FIBRE_TENSION
    assert result.passes


def test_the_off_axis_plies_fail_first_in_a_cross_ply(cfrp, strength):
    """The classic composite result: the matrix cracks before the fibres break.

    Transverse tensile strength is 50 MPa against 1500 along the fibres, so the
    90 degree plies of a cross-ply laminate go long before the 0 degree ones.
    """
    laminate = Laminate.from_material(cfrp, [0, 90, 90, 0], PLY_THICKNESS_M)
    result = first_ply_failure(laminate, strength, np.array([1.0e5, 0.0, 0.0]))
    assert result.angle_deg == pytest.approx(90.0)
    assert result.governing_mode is FailureMode.MATRIX_TENSION


# --- registry ----------------------------------------------------------------

def test_clt_is_gated_on_being_a_laminate():
    registry = build_default_registry()
    isotropic = ProblemContext(geometry="prismatic_beam",
                               representations=("prismatic_beam",),
                               material_class="isotropic", has_layup=False)
    candidates = registry.query(isotropic, Category.ANALYSIS)
    assert "laminate_clt" not in candidates.names()
    assert "layup" in candidates.reason("laminate_clt")[0]

    laminate = ProblemContext(geometry="prismatic_beam",
                              representations=("prismatic_beam",),
                              material_class="orthotropic", has_layup=True)
    assert "laminate_clt" in registry.query(laminate, Category.ANALYSIS).names()


def test_unimplemented_composite_methods_are_not_registered():
    registry = build_default_registry()
    for absent in ("progressive_damage", "free_edge_stress", "hashin",
                   "puck", "hygrothermal"):
        assert absent not in registry
