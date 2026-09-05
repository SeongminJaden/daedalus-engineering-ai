"""The cycloidal reducer of the joint module, as geometry that can be checked.

The arm's deflection budget asks for a torsional stiffness at each loaded
joint, and it asks the REDUCER for it, not the bearing. At the shoulder the
gravity moment lies exactly along the joint axis: the tool hangs out along x
at full reach, gravity pulls along minus y, and the cross product points
along minus z, which is the axis the joint turns on. A joint bearing resists
moments about the two axes across the joint and cannot resist that one. So
the tool sags because the drive train twists.

This module computes that twist from the drive's own geometry rather than
from a rule of thumb, because the rules of thumb were wrong twice.

WHAT THE GEOMETRY TURNED OUT TO SAY, both of which cost a factor:

A cycloidal contact normal passes through the instantaneous pitch point, and
in the disc's own frame that point sits at e * N from the disc centre, where
e is the eccentricity and N the lobe count. So NO RING PIN CAN HAVE A MOMENT
ARM LARGER THAN e * N about the disc centre, whatever radius the pin circle
is drawn at. With 2.5 mm and ten lobes that is 25 mm, against a 45 mm pin
circle. The first estimate written here used the pin circle radius as the
lever and was optimistic by a factor of more than three in the sum of
squares. `ring_pin_moment_arms` computes the arms from the envelope and the
maximum comes out 24.98 mm, which is that bound to two decimal places.

The disc's smallest outer radius is its ROOT radius, R - r_pin - e, not its
tip radius R - r_pin + e. With a 90 mm pin circle and 10 mm pins that is
37.5 mm against 42.5. An output pin hole is r_pin + e across its own radius,
so a 5 mm web puts the output pin circle at 25 mm and no further. A Ø60
output circle, which would have bought 44 percent on that term, breaks
through the root exactly: the web comes out 0.000 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Palmgren's empirical approach for a steel line contact, delta in mm from
#: F in newtons and L in millimetres. It is a ROLLER BEARING formula and it
#: is used here on a cycloidal flank and on a pin in a hole, which is the
#: least defensible assumption in this file.
APPROACH_COEFFICIENT = 3.84e-5
APPROACH_FORCE_EXPONENT = 0.9
APPROACH_LENGTH_EXPONENT = 0.8


def line_contact_approach_m(force_n, length_m):
    force = np.asarray(force_n, dtype=float)
    return 1e-3 * APPROACH_COEFFICIENT * np.abs(force) ** APPROACH_FORCE_EXPONENT / (
        1e3 * length_m) ** APPROACH_LENGTH_EXPONENT


def line_contact_force_n(approach_m, length_m):
    """The inverse, so a rotation can be turned into a load share."""
    approach = np.asarray(approach_m, dtype=float)
    scale = 1e-3 * APPROACH_COEFFICIENT / (1e3 * length_m) ** APPROACH_LENGTH_EXPONENT
    return (np.maximum(approach, 0.0) / scale) ** (1.0 / APPROACH_FORCE_EXPONENT)


@dataclass(frozen=True)
class CycloidalGeometry:
    """Every number a stiffness estimate here needs, and where each came from.

    All of them are CHOSEN. Nothing in this file is read off a drawing,
    because no cycloidal unit in this project's catalogue publishes its
    internal geometry, and the joint module is being designed rather than
    bought. The two that are constrained rather than free are said so in
    `constraints`.
    """

    ring_pin_circle_radius_m: float = 0.045
    ring_pin_radius_m: float = 0.005
    lobes: int = 10
    eccentricity_m: float = 0.003
    output_pin_circle_radius_m: float = 0.024
    output_pin_radius_m: float = 0.005
    output_pin_count: int = 6
    disc_thickness_m: float = 0.008
    disc_count: int = 2
    shear_modulus_pa: float = 79.3e9

    @property
    def ring_pin_count(self) -> int:
        return self.lobes + 1

    @property
    def ratio(self) -> int:
        return self.lobes

    @property
    def k1_factor(self) -> float:
        """The eccentricity, expressed the way a cycloidal design states it.

        THE ECCENTRICITY IS NOT A FREE VARIABLE. It is K1 times the pin
        circle radius over the pin count, and K1 is what the usual design
        band is written in. Raising e alone at a fixed pin circle raises K1,
        and past the band the lobe tips sharpen, the pressure angle grows and
        the profile eventually undercuts. `undercut_margin_m` measures the
        last of those directly rather than trusting the band.
        """
        return (self.eccentricity_m * self.ring_pin_count
                / self.ring_pin_circle_radius_m)

    @property
    def pitch_radius_m(self) -> float:
        """Where every contact normal passes, and so the largest moment arm
        any ring pin can have about the disc centre."""
        return self.eccentricity_m * self.lobes

    @property
    def disc_root_radius_m(self) -> float:
        return (self.ring_pin_circle_radius_m - self.ring_pin_radius_m
                - self.eccentricity_m)

    @property
    def disc_tip_radius_m(self) -> float:
        return (self.ring_pin_circle_radius_m - self.ring_pin_radius_m
                + self.eccentricity_m)

    @property
    def output_hole_radius_m(self) -> float:
        return self.output_pin_radius_m + self.eccentricity_m

    @property
    def output_web_m(self) -> float:
        """Material between an output pin hole and the disc's outer profile.

        Measured to the ROOT, which is what binds. Negative means the hole
        breaks out of the disc.
        """
        return (self.disc_root_radius_m - self.output_pin_circle_radius_m
                - self.output_hole_radius_m)

    @property
    def output_ligament_m(self) -> float:
        """Material between two adjacent output pin holes."""
        pitch = 2.0 * self.output_pin_circle_radius_m * math.sin(
            math.pi / self.output_pin_count)
        return pitch - 2.0 * self.output_hole_radius_m


def pin_centre_locus(geometry: CycloidalGeometry, samples: int = 2001):
    """The path a ring pin's centre traces in the DISC's frame.

    The disc profile is this curve offset inward by the pin radius, so every
    clearance question about the disc can be asked of this curve directly.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    points = []
    for index in range(geometry.ring_pin_count):
        points.append(_locus(geometry, index, angles))
    return np.concatenate(points)


def _locus(geometry: CycloidalGeometry, index: int, angles):
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    phi = 2.0 * np.pi * index / geometry.ring_pin_count
    pin = np.array([geometry.ring_pin_circle_radius_m * math.cos(phi),
                    geometry.ring_pin_circle_radius_m * math.sin(phi)])
    centre = np.stack([geometry.eccentricity_m * np.cos(angles),
                       geometry.eccentricity_m * np.sin(angles)], axis=1)
    offset = pin - centre
    turn = angles / geometry.lobes
    return np.stack([np.cos(turn) * offset[:, 0] - np.sin(turn) * offset[:, 1],
                     np.sin(turn) * offset[:, 0] + np.cos(turn) * offset[:, 1]],
                    axis=1)


def ring_pin_contacts(geometry: CycloidalGeometry, input_angle: float,
                      step: float = 1e-6):
    """Contact point, outward normal and signed moment arm for every ring pin.

    The normal is the locus normal at the pin's own point. A positive arm
    drives, a negative one is on the trailing side and carries nothing under
    a single direction of torque.
    """
    points = np.empty((geometry.ring_pin_count, 2))
    normals = np.empty((geometry.ring_pin_count, 2))
    for index in range(geometry.ring_pin_count):
        here = _locus(geometry, index, input_angle)[0]
        tangent = (_locus(geometry, index, input_angle + step)[0]
                   - _locus(geometry, index, input_angle - step)[0])
        tangent /= np.linalg.norm(tangent)
        normal = np.array([tangent[1], -tangent[0]])
        if float(np.dot(normal, here)) < 0.0:
            normal = -normal
        points[index] = here
        normals[index] = normal
    arms = points[:, 0] * normals[:, 1] - points[:, 1] * normals[:, 0]
    return points, normals, arms


def ring_pin_moment_arms(geometry: CycloidalGeometry, input_angle: float,
                         step: float = 1e-6):
    return ring_pin_contacts(geometry, input_angle, step)[2]


def undercut_margin_m(geometry: CycloidalGeometry, samples: int = 20001
                      ) -> float:
    """How much curvature is left before the profile eats itself.

    The disc profile is the pin centre locus offset INWARD by the pin radius,
    and an inward offset is singular wherever the centre of curvature already
    lies inward at less than that radius. That happens at the lobe tips, and
    it is what a K1 band is a proxy for. Returns the tightest such radius
    less the pin radius: positive is clear, negative undercuts.

    Computed, because the band is a convention this project has no source
    for. The margin falls smoothly with K1 and reaches zero near 1.0, so the
    usual 0.5 to 0.75 band is conservative rather than a cliff edge.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    curve = _locus(geometry, 0, angles)
    first = np.gradient(curve, angles, axis=0)
    second = np.gradient(first, angles, axis=0)
    speed = np.linalg.norm(first, axis=1)
    curvature = (first[:, 0] * second[:, 1]
                 - first[:, 1] * second[:, 0]) / speed ** 3
    left = np.stack([-first[:, 1], first[:, 0]], axis=1) / speed[:, None]
    centre = curve + left / curvature[:, None]
    inward = np.linalg.norm(centre, axis=1) < np.linalg.norm(curve, axis=1)
    if not inward.any():
        return float("inf")
    return float(np.abs(1.0 / curvature)[inward].min()
                 - geometry.ring_pin_radius_m)


def eccentric_bearing_load_n(geometry: CycloidalGeometry, torque_nm: float,
                             input_angle: float) -> dict:
    """What the eccentric bearing carries, from the disc's own equilibrium.

    Worth computing rather than estimating, because raising the eccentricity
    pulls the two components in OPPOSITE directions and only the arithmetic
    says which wins. The tangential component follows from power, T / (N e),
    and so falls as e rises. The radial component comes from the pressure
    angle, which grows with K1, and so rises. The magnitudes turn out to be
    close enough that the resultant barely moves.

    It matters for bearing SELECTION and life. It does not enter the
    stiffness relation: only the tangential deflection turns the output, and
    an isotropic radial stiffness has no cross term, so
    `eccentric_bearing_stiffness_nm_rad` is unaffected by the radial share.
    """
    points, normals, arms = ring_pin_contacts(geometry, input_angle)
    rotation = pin_set_rotation(torque_nm, arms, geometry.disc_thickness_m,
                                geometry.disc_count)
    forces = np.where(arms > 0.0,
                      line_contact_force_n(rotation * np.maximum(arms, 0.0),
                                           geometry.disc_thickness_m), 0.0)
    from_ring = -(forces[:, None] * normals).sum(axis=0)

    output_arms = output_pin_moment_arms(geometry, input_angle)
    output_rotation = pin_set_rotation(torque_nm, output_arms,
                                       geometry.disc_thickness_m,
                                       geometry.disc_count)
    output_forces = np.where(
        output_arms > 0.0,
        line_contact_force_n(output_rotation * np.maximum(output_arms, 0.0),
                             geometry.disc_thickness_m), 0.0)
    offset = np.array([math.cos(input_angle), math.sin(input_angle)])
    from_output = -float(output_forces.sum()) * offset

    load = -(from_ring + from_output)
    tangential = np.array([-offset[1], offset[0]])
    return {
        "magnitude_n": float(np.linalg.norm(load)),
        "tangential_n": float(np.dot(load, tangential)),
        "radial_n": float(np.dot(load, offset)),
        "power_estimate_n": torque_nm / (geometry.ratio
                                         * geometry.eccentricity_m
                                         * geometry.disc_count),
        "largest_ring_pin_force_n": float(forces.max()),
    }


def orbit_couple_nm(geometry: CycloidalGeometry, input_speed_rad_s: float,
                    disc_mass_kg: float, disc_spacing_m: float) -> float:
    """The rocking couple two opposed discs leave behind as they orbit.

    Each disc's centrifugal force is m e omega squared, and two discs at 180
    degrees cancel the resultant, so what survives is the couple of two equal
    and opposite forces separated along the axis.
    """
    force = disc_mass_kg * geometry.eccentricity_m * input_speed_rad_s ** 2
    return force * disc_spacing_m


def output_pin_moment_arms(geometry: CycloidalGeometry, input_angle: float):
    """Signed moment arms of the output pins about the disc centre.

    A pin sits in a hole that is larger than it by the eccentricity, and the
    disc's centre is offset from the flange's by that same eccentricity, so
    every contact normal points along the SAME direction, the line of the
    offset. The moment arm of a hole at angle phi is then the pin circle
    radius times the sine of the angle between them, which is where the
    sinusoidal load share of a cycloidal output comes from. It is derived
    here rather than assumed.
    """
    phi = 2.0 * np.pi * np.arange(geometry.output_pin_count) / geometry.output_pin_count
    return geometry.output_pin_circle_radius_m * np.sin(phi - input_angle)


def pin_set_rotation(torque_nm: float, arms, length_m: float,
                     discs: int) -> float:
    """Disc rotation that carries a torque through one set of pin contacts.

    Only pins with a positive arm carry. Each one's approach is the rotation
    times its own arm, and its force follows the line contact law, so the
    share is solved rather than assumed: no fraction of engaged pins is put
    in by hand, it falls out of the arms.
    """
    driving = np.asarray(arms, dtype=float)
    driving = driving[driving > 0.0]
    if driving.size == 0:
        raise ValueError("no pin is on the driving side; check the arms")
    share = torque_nm / discs

    def carried(rotation: float) -> float:
        forces = line_contact_force_n(rotation * driving, length_m)
        return float(np.sum(forces * driving))

    low, high = 0.0, 1e-6
    while carried(high) < share:
        high *= 2.0
        if high > 1.0:
            raise ValueError("the contact law will not carry this torque")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if carried(middle) < share:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def pin_set_stiffness_nm_rad(torque_nm: float, arms, length_m: float,
                             discs: int) -> float:
    return torque_nm / pin_set_rotation(torque_nm, arms, length_m, discs)


def disc_shear_stiffness_nm_rad(geometry: CycloidalGeometry) -> float:
    """In plane torsion of the annulus between the two pin circles.

    Torque enters at the outer radius and leaves at the inner one, so the
    shear stress is T / (2 pi r^2 t) and the relative rotation integrates to
    T / (4 pi G t) times (1/a^2 - 1/b^2). The holes through the disc are
    ignored, which makes this an upper bound.
    """
    inner = geometry.output_pin_circle_radius_m
    outer = geometry.ring_pin_circle_radius_m
    return (4.0 * math.pi * geometry.shear_modulus_pa * geometry.disc_thickness_m
            / (1.0 / inner ** 2 - 1.0 / outer ** 2)) * geometry.disc_count


def eccentric_bearing_stiffness_nm_rad(radial_stiffness_n_m: float,
                                       geometry: CycloidalGeometry) -> float:
    """Output torsional stiffness contributed by the eccentric bearing.

    Derived, because the lever here is not obvious and getting it wrong
    changes the answer by an order of magnitude either way. The disc centre
    sits at e times the unit vector of the input angle, and the disc's own
    rotation is minus that angle over the ratio. If the bearing lets the
    centre shift tangentially by d, the disc behaves as though the input
    angle were larger by d / e, so its rotation errs by d / (e * N).

    The tangential force the bearing carries follows from power: the input
    torque is the output torque over the ratio, and it acts at the orbit
    radius, so the force is T / (N * e). Putting the two together the
    torsional stiffness is the bearing's radial stiffness times (e * N)
    squared, and e * N is the pitch radius, 25 mm here. Each disc has its
    own bearing and they act in parallel.
    """
    return (radial_stiffness_n_m * geometry.pitch_radius_m ** 2
            * geometry.disc_count)


def required_bearing_stiffness_n_m(target_nm_rad: float,
                                   geometry: CycloidalGeometry) -> float:
    """Turned round, the way the friction grip was: what the bearing must be."""
    return target_nm_rad / (geometry.pitch_radius_m ** 2 * geometry.disc_count)


def shell_torsion_nm_rad(diameter_m: float, wall_m: float, length_m: float,
                         shear_modulus_pa: float) -> float:
    """Torsion of a thin walled tube, G J / L with J = 2 pi r^3 t.

    This is the housing, and it is NOT the same calculation as the one in
    `joint_module_stiffness_stage`, which used E and a bending second moment.
    That one answers the out of plane question. Torsion needs G and J.
    """
    radius = 0.5 * diameter_m
    return shear_modulus_pa * 2.0 * math.pi * radius ** 3 * wall_m / length_m
