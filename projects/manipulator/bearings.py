"""The crossed roller ring the base yaw needs, and what its maker asks around it.

WHERE THESE NUMBERS COME FROM, because it is not where this project's other
sourced numbers come from. Every actuator dimension in `interfaces.py` was
read off a drawing by this session and can be re-read. THESE WERE NOT. They
were quoted out of THK 513-2E by the Fusion session, page by page, and this
session has not opened that catalogue. That is a weaker provenance than a
drawing this session read, it is stronger than an assumption, and it is
recorded as what it is rather than being levelled up to the other kind.

What the catalogue says, in its own words where it matters:

A18-19, on the moment rigidity diagrams: they are for the ring "as a separate
unit", and "Rigidity is affected by the deformation of the housing, presser
flange and bolts. Therefore, the strength of these parts must be taken into
account." So a catalogue moment stiffness is the BEARING ALONE and the
structure around it adds compliance on top. That is not this project's
inference any more, it is the maker's sentence.

A18-36, the housing: thickness T is (D - d) / 2 times 0.6 or greater.

A18-38, the presser flange: thickness F is between 0.5 and 1.2 times the ring
width B, land H equals B, clearance S is 0.5 mm, and a ring of D between 100
and 200 mm wants twelve or more bolts of M4 to M8. It also says model RU does
not require a presser flange at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: THK 513-2E, quoted by the Fusion session on 2026-09-05. Not read here.
THK_SOURCE = ("THK 513-2E cross roller ring catalogue, pages A18-19, A18-24, "
              "A18-36, A18-37 and A18-38, quoted by the Fusion session and "
              "NOT read by this one")

#: A18-36. The housing wall THK wants around a ring, as a fraction of the
#: ring's own radial section.
HOUSING_THICKNESS_FACTOR = 0.6

#: A18-38. The presser flange thickness, as a fraction of the ring width.
FLANGE_THICKNESS_FACTOR = (0.5, 1.2)


@dataclass(frozen=True)
class CrossRollerRing:
    """One RB ring's bore, outside diameter and width, all in metres."""

    model: str
    bore_m: float
    outer_m: float
    width_m: float

    @property
    def ring_section_m(self) -> float:
        return 0.5 * (self.outer_m - self.bore_m)

    @property
    def housing_thickness_m(self) -> float:
        """A18-36. The minimum, so a housing under this is not a thin one,
        it is one THK says will not hold the ring round."""
        return HOUSING_THICKNESS_FACTOR * self.ring_section_m

    @property
    def housing_outer_m(self) -> float:
        return self.outer_m + 2.0 * self.housing_thickness_m

    @property
    def flange_thickness_range_m(self) -> tuple[float, float]:
        return (FLANGE_THICKNESS_FACTOR[0] * self.width_m,
                FLANGE_THICKNESS_FACTOR[1] * self.width_m)


#: A18-24. The five sizes in the range this joint could use.
RB_RINGS = (
    CrossRollerRing("RB 8016", 0.080, 0.120, 0.016),
    CrossRollerRing("RB 9016", 0.090, 0.130, 0.016),
    CrossRollerRing("RB 10020", 0.100, 0.150, 0.020),
    CrossRollerRing("RB 11015", 0.110, 0.145, 0.015),
    CrossRollerRing("RB 12016", 0.120, 0.150, 0.016),
)

#: A18-38 Table 4, the presser flange bolt tightening torques in N m. THE
#: TABLE PRINTS NO BOLT GRADE AND NO NUT FACTOR, which is what stops these
#: being reconcilable with this project's own preloads: see
#: `flange_bolt_torque_disagreement`.
FLANGE_BOLT_TORQUE_NM = {"M3": 2.0, "M4": 4.0, "M5": 9.0, "M6": 14.0,
                         "M8": 30.0}

#: A18-38. Model RU carries mounting holes in both rings and needs no presser
#: flange, which would delete the flange term from the chain entirely. IT IS
#: NOT TAKEN, and the reason is about what can be known rather than what is
#: stiff. The Fusion session counted the moment rigidity diagrams: figure 4
#: carries sixteen RA curves and figures 5 to 7 carry twenty eight RB curves,
#: and NOT ONE of them is an RU. Choosing RU would delete a term this project
#: can compute, the flange, and replace it with a term it cannot read at all,
#: the ring's own moment stiffness. That is a worse position even if the
#: number turned out better.
RU_IS_REFUSED = (
    "model RU needs no presser flange and is refused anyway: THK prints no "
    "moment rigidity curve for it, so it trades a term that can be computed "
    "for one that cannot be read")


def flange_thickness_spread(ring: CrossRollerRing) -> float:
    """How much the flange term can move inside what THK allows.

    Needs no flange model at all, which is the point. Any plate's bending
    stiffness goes as its thickness cubed, and the catalogue allows 0.5 to
    1.2 times the ring width, so the term spans the cube of that ratio
    whatever the rest of the geometry is. Almost fourteen times.

    This is why choosing the bearing does not settle the flange. A design
    that names its ring still has a fourteen fold band open on this term
    until it names F as well.
    """
    low, high = ring.flange_thickness_range_m
    return (high / low) ** 3


def shell_bending_nm_rad(mean_diameter_m: float, wall_m: float,
                         length_m: float, modulus_pa: float) -> float:
    """E I / L for a thin walled tube, with I = pi r cubed t."""
    return (modulus_pa * math.pi * (0.5 * mean_diameter_m) ** 3 * wall_m
            / length_m)


def housing_meets_thk(ring: CrossRollerRing, wall_m: float) -> tuple[bool, str]:
    """Is a wall thick enough for the ring it is supposed to hold?"""
    needed = ring.housing_thickness_m
    if wall_m >= needed:
        return True, (f"{wall_m * 1000:.1f} mm against the {needed * 1000:.1f} "
                      f"mm A18-36 asks for {ring.model}")
    return False, (
        f"{wall_m * 1000:.1f} mm is {needed / wall_m:.1f} times UNDER the "
        f"{needed * 1000:.1f} mm A18-36 asks for {ring.model}. A stiffness "
        f"computed on this wall is computed on a housing the maker says will "
        f"not hold the ring round, so it is not a conservative number, it is "
        f"a number about a different part")


def flange_bolt_torque_disagreement(size: str, this_projects_preload_n: float,
                                    nut_factor: float = 0.2) -> dict:
    """Why THK's bolt torques and this project's preloads cannot be mixed.

    At the 0.2 nut factor this project uses, THK's table implies 1.3 to 1.5
    times the preload this project takes from 75 percent of an ISO 898-1
    class 8.8 proof load, and at M3, M5 and M6 that is ABOVE the proof load
    outright. At a nut factor of 0.25 to 0.3, which is the other end of the
    usual dry range, the same torques land within a fifth of this project's
    own figures.

    So the disagreement is most likely the nut factor and not the bolt grade,
    and it cannot be settled, because the table prints neither. The two sets
    of numbers belong to different joints in any case: THK's are for the
    bolts that hold a presser flange down and this project's are for the
    bolts that hold a housing to a link.
    """
    torque = FLANGE_BOLT_TORQUE_NM[size]
    diameter = float(size[1:]) / 1000.0
    implied = torque / (nut_factor * diameter)
    return {
        "size": size, "thk_torque_nm": torque,
        "implied_preload_n": implied,
        "this_projects_preload_n": this_projects_preload_n,
        "ratio": implied / this_projects_preload_n,
        "nut_factor_that_would_agree": torque / (this_projects_preload_n
                                                 * diameter),
    }
