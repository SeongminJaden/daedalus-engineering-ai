"""core.design_genome.section - cross-section design variables and properties.

Convention used throughout: the section lies in the local y-z plane of the
link, `b` (outer_width_m) runs horizontally and `h` (outer_height_m) runs
vertically. The bending axis of interest is the horizontal centroidal axis, so
`I_x` is the second moment of area about that axis and is the one that resists
a vertical tip load.

Nothing here needs a load. These are pure geometric quantities - Phase 2 turns
them into stress and deflection.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SectionProperties(BaseModel):
    """Pure geometric properties of a cross-section. SI units."""

    model_config = ConfigDict(extra="forbid")

    area_m2: float          # A   [m^2]
    i_x_m4: float           # I_x [m^4], about the horizontal centroidal axis
    i_y_m4: float           # I_y [m^4], about the vertical centroidal axis
    s_x_m3: float           # S_x [m^3], section modulus = I_x / (h/2)
    s_y_m3: float           # S_y [m^3], section modulus = I_y / (b/2)


class HollowRectangleSection(BaseModel):
    """A hollow rectangular tube of uniform wall thickness.

    Length is NOT here: it is inherited from the problem (the IR), because
    length is a fixed requirement rather than a design variable.
    """

    model_config = ConfigDict(extra="forbid")

    outer_width_m: float = Field(gt=0.0)    # b
    outer_height_m: float = Field(gt=0.0)   # h
    wall_thickness_m: float = Field(gt=0.0)  # t

    # --- validity ------------------------------------------------------- #
    def inner_width_m(self) -> float:
        """b_i = b - 2t. May be <= 0 for an invalid design."""
        return self.outer_width_m - 2.0 * self.wall_thickness_m

    def inner_height_m(self) -> float:
        """h_i = h - 2t. May be <= 0 for an invalid design."""
        return self.outer_height_m - 2.0 * self.wall_thickness_m

    def is_valid(self) -> bool:
        """t must leave a positive inner cavity: t < min(b, h) / 2."""
        b, h, t = self.outer_width_m, self.outer_height_m, self.wall_thickness_m
        if b <= 0.0 or h <= 0.0 or t <= 0.0:
            return False
        return t < min(b, h) / 2.0

    def validity_reason(self) -> str | None:
        """None when valid, else a human-readable reason. For error reporting."""
        b, h, t = self.outer_width_m, self.outer_height_m, self.wall_thickness_m
        if b <= 0.0:
            return f"outer_width_m must be > 0, got {b}"
        if h <= 0.0:
            return f"outer_height_m must be > 0, got {h}"
        if t <= 0.0:
            return f"wall_thickness_m must be > 0, got {t}"
        if t >= min(b, h) / 2.0:
            return (
                f"wall_thickness_m={t} leaves no cavity: needs t < min(b,h)/2 "
                f"= {min(b, h) / 2.0}"
            )
        return None

    # --- geometry ------------------------------------------------------- #
    def section_properties(self) -> SectionProperties:
        """Exact closed-form properties of the hollow rectangle.

            b_i = b - 2t          h_i = h - 2t
            A   = b*h - b_i*h_i
            I_x = (b*h^3 - b_i*h_i^3) / 12
            I_y = (h*b^3 - h_i*b_i^3) / 12
            S_x = I_x / (h/2)     S_y = I_y / (b/2)

        Verified against numerical integration of the actual section in
        tests/test_section_properties.py rather than trusted on sight.
        """
        if not self.is_valid():
            raise ValueError(
                f"invalid section: {self.validity_reason()}"
            )
        b, h = self.outer_width_m, self.outer_height_m
        bi, hi = self.inner_width_m(), self.inner_height_m()

        area = b * h - bi * hi
        i_x = (b * h**3 - bi * hi**3) / 12.0
        i_y = (h * b**3 - hi * bi**3) / 12.0
        return SectionProperties(
            area_m2=area,
            i_x_m4=i_x,
            i_y_m4=i_y,
            s_x_m3=i_x / (h / 2.0),
            s_y_m3=i_y / (b / 2.0),
        )

    def mass(self, length_m: float, density_kg_m3: float) -> float:
        """Mass of a prismatic bar of this section: A * L * rho.

        A geometry-derived quantity, not a simulation result: it assumes a
        perfectly prismatic part with no end caps, fillets, bosses or joints.
        Phase 2+ replaces this with the mass of the realised geometry.
        """
        if length_m <= 0.0:
            raise ValueError(f"length_m must be > 0, got {length_m}")
        if density_kg_m3 <= 0.0:
            raise ValueError(f"density_kg_m3 must be > 0, got {density_kg_m3}")
        return self.section_properties().area_m2 * length_m * density_kg_m3
