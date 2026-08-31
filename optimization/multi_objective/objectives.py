"""The objective vector for a structural link, and the multi-material sweep.

Phase 3 minimised mass with everything else as a constraint. That answers one
question well and hides the interesting one: a titanium link is lighter than a
steel one and costs forty times as much per kilogram, and no single number
decides between them. This assembles the vector so the trade-off is visible
before anyone commits to weights.

VALIDITY, stated first:

* **Material is CATEGORICAL and therefore cannot be a search variable here.**
  The NSGA-II operators interpolate between values, and there is nothing
  halfway between aluminium and titanium. So the continuous search runs
  separately per material and the fronts are merged afterwards, which is what
  `sweep_materials` does. Putting a material index in the design vector would
  produce fractional materials and silently round them somewhere.

* **Cost here is a MATERIAL BILL, not a part cost.** mass times price per kg,
  with machining, finishing, assembly, tooling and quantity all excluded. For
  a small machined bracket those dominate the material several times over, so
  the cheapest design by this measure need not be the cheapest part. It ranks
  materials against each other under a fixed process, which is a question it
  can answer.

* **The objectives inherit the beam model's range.** Deflection and stress come
  from the Phase 2 and 7.5 beam path, so the slenderness caveat that applies
  there applies here: below L/h of about 12 the Euler-Bernoulli path is out of
  range and the Timoshenko one should be used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.materials import get_material
from optimization.constraints import build_optimization_problem, evaluate_batch

from .nsga2 import Sense, nsga2

# The objective set. Order is fixed because the senses, the weights a caller
# passes, and every column of the result are indexed by it.
OBJECTIVE_NAMES: tuple[str, ...] = ("mass_kg", "tip_deflection_m",
                                    "max_stress_pa", "material_cost_usd")
OBJECTIVE_SENSES: tuple[Sense, ...] = (Sense.MIN, Sense.MIN, Sense.MIN,
                                       Sense.MIN)


def material_cost_usd(mass_kg: np.ndarray, price_per_kg_usd: float
                      ) -> np.ndarray:
    """mass times price. See the module note: this is a material bill."""
    if price_per_kg_usd <= 0.0:
        raise ValueError("price per kg must be positive")
    return np.asarray(mass_kg, dtype=float) * price_per_kg_usd


def build_evaluator(op, price_per_kg_usd: float):
    """An NSGA-II evaluator over the batched GPU beam path.

    Returns objectives in their natural senses and a non-negative constraint
    violation. One batched beam evaluation per generation, not one per row and
    not one per objective: the metrics all come from the same launch.

    Geometrically impossible rows come back as NaN, and they are pushed to a
    large finite objective and violation rather than left as NaN. NaN compares
    false against everything, so such a row is neither dominated nor
    dominating and would sit in the front forever, crowding out real designs.
    """
    from physics.structural.beam import evaluate_beam

    from optimization.constraints import constraint_values

    def evaluate(design: np.ndarray):
        design = np.atleast_2d(np.asarray(design, dtype=float))
        n = design.shape[0]
        valid = np.array([op.is_geometrically_valid(row) for row in design])

        mass = np.full(n, np.nan)
        stress = np.full(n, np.nan)
        deflection = np.full(n, np.nan)
        if valid.any():
            metrics = evaluate_beam([op.genome(row) for row in design[valid]],
                                    op.problem)
            mass[valid] = metrics.mass_kg
            stress[valid] = metrics.max_bending_stress_pa
            deflection[valid] = metrics.tip_deflection_m

        constraints = constraint_values(op, stress, deflection, design)
        stacked = np.column_stack([np.ravel(v) for v in constraints.values()])
        with np.errstate(invalid="ignore"):
            worst = np.nanmin(stacked, axis=1)
        violation = np.maximum(0.0, -worst)

        objectives = np.column_stack([
            mass, deflection, stress,
            np.asarray(mass, dtype=float) * price_per_kg_usd])
        unusable = ~valid | ~np.isfinite(objectives).all(axis=1)
        objectives = np.where(unusable[:, None], 1e12, objectives)
        violation = np.where(unusable | ~np.isfinite(violation),
                             1e12, violation)
        return objectives, violation

    return evaluate


@dataclass
class MaterialFront:
    """One material's approximated front."""

    material_id: str
    design: np.ndarray
    objectives: np.ndarray

    def __len__(self) -> int:
        return int(self.objectives.shape[0])


def sweep_materials(problem, material_ids: "list[str]", population: int = 48,
                    generations: int = 40, seed: int = 0
                    ) -> list[MaterialFront]:
    """Run the continuous search once per material and keep each front.

    Separate runs rather than one search over a material index, because the
    variation operators are only defined for continuous variables. The fronts
    are merged by `merged_front`, which is where the cross-material trade-off
    becomes visible.
    """
    fronts: list[MaterialFront] = []
    for material_id in material_ids:
        candidate = problem.model_copy(deep=True)
        candidate.material_id = material_id
        material = get_material(material_id)
        if material.price_per_kg_usd is None:
            raise ValueError(
                f"{material_id} has no price, so material cost cannot be an "
                f"objective for it")
        op = build_optimization_problem(candidate)
        result = nsga2(build_evaluator(op, material.price_per_kg_usd),
                       op.lower, op.upper, list(OBJECTIVE_SENSES),
                       population=population, generations=generations,
                       seed=seed)
        fronts.append(MaterialFront(material_id=material_id,
                                    design=result.front_designs(),
                                    objectives=result.front_objectives()))
    return fronts


def merged_front(fronts: "list[MaterialFront]"
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Non-dominated set across all materials.

    A material whose entire front is dominated drops out completely, which is
    the useful answer: it says that material is not on the trade-off at all for
    this problem, rather than that it is merely worse on average.
    """
    from .pareto import non_dominated_mask

    if not fronts:
        return np.zeros((0, 0)), np.zeros((0, len(OBJECTIVE_NAMES))), []
    designs = np.vstack([f.design for f in fronts if len(f)])
    objectives = np.vstack([f.objectives for f in fronts if len(f)])
    labels = [f.material_id for f in fronts for _ in range(len(f))]
    keep = non_dominated_mask(objectives)
    return (designs[keep], objectives[keep],
            [label for label, k in zip(labels, keep) if k])
