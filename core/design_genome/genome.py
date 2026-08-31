"""core.design_genome.genome - the Design Genome (the *design variables*).

Where the Engineering IR fixes the problem (how long, what load, which
material, what limits), the genome holds what a search is free to change. For
the MVP that is exactly the cross-section; `topology` and `structure` are
carried as stubs so the container shape does not change when later phases fill
them in.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .section import HollowRectangleSection, SectionProperties


class DesignGenome(BaseModel):
    """A candidate design. Only the fields a search may vary."""

    model_config = ConfigDict(extra="forbid")

    section: HollowRectangleSection
    material_id: str = Field(min_length=1)

    # Reserved for later phases; kept so the container shape is stable.
    topology: dict[str, Any] | None = None   # Phase 2+: density field / SDF
    structure: dict[str, Any] | None = None  # Phase 2+: ribs, joints, features

    def is_valid(self) -> bool:
        return self.section.is_valid()

    def validity_reason(self) -> str | None:
        return self.section.validity_reason()

    def section_properties(self) -> SectionProperties:
        return self.section.section_properties()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignGenome":
        return cls.model_validate(data)


def realize(genome: DesignGenome, problem) -> dict[str, Any]:
    """Combine the genome with its problem into a fully specified design.

    The genome alone is not simulatable: it has no length and no material
    properties. `realize` is the single place those are joined, so Phase 2
    physics receives one complete, self-contained description.

    Raises if the genome is invalid or if its material disagrees with the
    problem's - a mismatch there is a wiring bug, not a design to evaluate.
    """
    from core.materials import get_material

    if not genome.is_valid():
        raise ValueError(f"cannot realize invalid genome: {genome.validity_reason()}")

    if genome.material_id != problem.material_id:
        raise ValueError(
            f"material mismatch: genome has {genome.material_id!r}, "
            f"problem requires {problem.material_id!r}"
        )

    material = get_material(problem.material_id)
    props = genome.section_properties()
    length_m = problem.geometry.length_m

    return {
        "problem_name": problem.name,
        "length_m": length_m,
        "section_type": problem.geometry.section_type.value,
        "section": genome.section.model_dump(mode="json"),
        "section_properties": props.model_dump(mode="json"),
        "material": material.model_dump(mode="json"),
        "mass_kg": genome.section.mass(length_m, material.density_kg_m3),
        "loads": [load.model_dump(mode="json") for load in problem.loads],
        "boundary_conditions": [
            bc.model_dump(mode="json") for bc in problem.boundary_conditions
        ],
        "constraints": problem.constraints.model_dump(mode="json"),
        "objectives": [o.model_dump(mode="json") for o in problem.objectives],
    }
