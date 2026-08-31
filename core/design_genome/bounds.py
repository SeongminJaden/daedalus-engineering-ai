"""core.design_genome.bounds - design-variable bounds and mutation.

Phase 3's optimizers own the real search strategy. What lives here is the part
they must not each reinvent: the legal range of every variable, and a mutation
that is guaranteed to stay inside it.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .genome import DesignGenome
from .section import HollowRectangleSection


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> "Interval":
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self

    def clamp(self, value: float) -> float:
        return min(self.max, max(self.min, value))

    def span(self) -> float:
        return self.max - self.min


class DesignBounds(BaseModel):
    """Per-variable legal ranges for a hollow-rectangle genome. SI units.

    The wall-thickness floor is a **[ASSUMED]** manufacturability limit, not a
    derived or measured one: t_min = 1.0 mm assumes a CNC-machined aluminium
    tube, where a thinner wall starts to chatter and distort. It is NOT a
    universal constant - additive manufacturing, extrusion and sheet forming
    all have different floors, and a different process must change this number.
    Because the optimum for a stiffness-limited link sits ON this bound (mass
    falls monotonically as the wall thins), the assumption directly sets the
    answer. Revisit it before treating any optimized mass as achievable.
    """

    model_config = ConfigDict(extra="forbid")

    outer_width_m: Interval = Field(
        default_factory=lambda: Interval(min=0.010, max=0.100)
    )
    outer_height_m: Interval = Field(
        default_factory=lambda: Interval(min=0.010, max=0.100)
    )
    # [ASSUMED] 1 mm minimum wall - CNC aluminium. See class docstring.
    wall_thickness_m: Interval = Field(
        default_factory=lambda: Interval(min=0.001, max=0.020)
    )

    def clamp_section(self, section: HollowRectangleSection) -> HollowRectangleSection:
        """Clamp every variable into range, then repair an impossible wall.

        Clamping each variable independently can still leave t >= min(b,h)/2,
        which is geometrically impossible. Thickness is reduced last so the
        result is always a valid section rather than merely an in-range one.
        """
        b = self.outer_width_m.clamp(section.outer_width_m)
        h = self.outer_height_m.clamp(section.outer_height_m)
        t = self.wall_thickness_m.clamp(section.wall_thickness_m)

        t_ceiling = min(b, h) / 2.0
        if t >= t_ceiling:
            # step just inside the ceiling, but never below the legal minimum
            t = max(self.wall_thickness_m.min, t_ceiling * 0.99)
        return HollowRectangleSection(
            outer_width_m=b, outer_height_m=h, wall_thickness_m=t
        )

    def mutate(
        self,
        genome: DesignGenome,
        rng: random.Random,
        scale: float = 0.1,
    ) -> DesignGenome:
        """Gaussian perturbation of each variable, kept inside the bounds.

        `scale` is a fraction of each variable's own range, so one setting
        behaves consistently across variables of very different magnitude.
        The result is always valid - see clamp_section.
        """
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}")

        s = genome.section
        proposed = HollowRectangleSection(
            outer_width_m=max(
                1e-12,
                s.outer_width_m + rng.gauss(0.0, scale * self.outer_width_m.span()),
            ),
            outer_height_m=max(
                1e-12,
                s.outer_height_m + rng.gauss(0.0, scale * self.outer_height_m.span()),
            ),
            wall_thickness_m=max(
                1e-12,
                s.wall_thickness_m
                + rng.gauss(0.0, scale * self.wall_thickness_m.span()),
            ),
        )
        return DesignGenome(
            section=self.clamp_section(proposed),
            material_id=genome.material_id,
            topology=genome.topology,
            structure=genome.structure,
        )

    def sample(self, rng: random.Random, material_id: str) -> DesignGenome:
        """Uniform random genome inside the bounds. Always valid."""
        section = HollowRectangleSection(
            outer_width_m=rng.uniform(self.outer_width_m.min, self.outer_width_m.max),
            outer_height_m=rng.uniform(
                self.outer_height_m.min, self.outer_height_m.max
            ),
            wall_thickness_m=rng.uniform(
                self.wall_thickness_m.min, self.wall_thickness_m.max
            ),
        )
        return DesignGenome(
            section=self.clamp_section(section), material_id=material_id
        )
