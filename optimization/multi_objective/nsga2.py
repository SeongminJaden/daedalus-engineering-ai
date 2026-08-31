"""NSGA-II: non-dominated sorting genetic algorithm with crowding distance.

Single-objective optimisation answers "what is the lightest design that meets
the constraints". Real design questions are not that shape. A titanium link is
lighter than steel and costs forty times as much per kilogram; a CFRP one is
lighter still and costs sixty. None of those is the answer, and collapsing them
into one number with weights chosen up front hides the decision rather than
informing it. What a designer needs is the set of designs where nothing can be
improved without giving something else up, and then to choose.

VALIDITY, stated before the code rather than discovered afterwards:

* **Dominance here assumes every objective is MINIMISED.** Passing a column
  that should be maximised, natural frequency for instance, silently returns
  the wrong set: it would select the designs with the LOWEST frequency and
  report them as optimal. `to_minimisation` exists so the conversion is
  explicit and recorded, and nothing in this module accepts a mixed sense
  without it.

* **Crowding distance is meaningful only WITHIN one front.** It measures how
  isolated a solution is among its equals. Comparing distances across
  different fronts is meaningless, because rank already separates those.

* **The returned front is a finite-population approximation.** A true Pareto
  front is generally a continuum with infinitely many points. NSGA-II returns
  at most `population` of them, they are non-dominated only with respect to
  what was evaluated, and nothing here proves global optimality.

* **The variation operators are for box-bounded REAL variables.** Simulated
  binary crossover and polynomial mutation both assume a continuous variable
  between a lower and an upper bound. Applying them to an integer or a
  categorical choice, a material id for example, produces meaningless
  intermediate values, so a discrete variable needs a different operator and
  this module does not provide one.

* **Constraint handling is constrained domination**, which needs the violation
  to be a single meaningful non-negative scalar, zero when feasible. It ranks
  any feasible design above any infeasible one, so it does not apply where a
  small violation is genuinely acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np


class Sense(str, Enum):
    """Whether an objective is to be made small or large."""

    MIN = "min"
    MAX = "max"


def to_minimisation(objectives: np.ndarray,
                    senses: "list[Sense] | tuple[Sense, ...]") -> np.ndarray:
    """Flip maximised columns so every column is minimised.

    The conversion is a separate, named step because getting it wrong is
    silent: dominance on a maximised column returns exactly the worst designs
    and calls them optimal.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    if len(senses) != f.shape[1]:
        raise ValueError(
            f"{f.shape[1]} objective columns but {len(senses)} senses given; "
            f"an unstated sense would be assumed minimised and could invert "
            f"the answer")
    signs = np.array([-1.0 if s is Sense.MAX else 1.0 for s in senses])
    return f * signs


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True when `a` is no worse everywhere and strictly better somewhere.

    Minimisation. Equal rows do not dominate each other, which is what keeps
    duplicates from eliminating one another.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[np.ndarray]:
    """Partition rows into fronts: front 0 is non-dominated, front 1 is
    non-dominated once front 0 is removed, and so on.

    The bookkeeping version from the NSGA-II paper: count how many solutions
    dominate each one, then peel off the zero-count layer repeatedly. It is
    O(M N^2) like the direct comparison but computes every front in one pass
    rather than re-scanning for each.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    n = f.shape[0]
    if n == 0:
        return []

    dominated_by_count = np.zeros(n, dtype=int)
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        # Rows i dominates: f[i] <= f[j] everywhere and strictly less somewhere.
        i_dominates = np.all(f[i] <= f, axis=1) & np.any(f[i] < f, axis=1)
        i_dominates[i] = False
        dominates_list[i] = list(np.flatnonzero(i_dominates))
        # Rows that dominate i. This is the OTHER direction, and the count that
        # the peeling loop needs. Counting the ones i dominates instead reads
        # as plausible code and inverts the sort: the first front comes out as
        # the solutions dominated by nothing else being dominated, which is
        # nearly the worst layer rather than the best.
        dominate_i = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
        dominate_i[i] = False
        dominated_by_count[i] = int(np.count_nonzero(dominate_i))

    fronts: list[np.ndarray] = []
    current = np.flatnonzero(dominated_by_count == 0)
    remaining = dominated_by_count.copy()
    while current.size:
        fronts.append(current)
        nxt: list[int] = []
        for i in current:
            for j in dominates_list[i]:
                remaining[j] -= 1
                if remaining[j] == 0:
                    nxt.append(j)
        current = np.array(sorted(nxt), dtype=int)
    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Density estimate within ONE front: the perimeter of the box spanned by
    a solution's nearest neighbours on each objective.

    Boundary solutions get infinity so the extremes of the front are always
    kept. Without that the front collapses inward over generations and the
    interesting extremes, the lightest design and the cheapest one, are the
    first to be lost.

    An objective with zero range contributes nothing rather than dividing by
    zero: if every solution has the same value there, it cannot separate them.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    n, m = f.shape
    if n == 0:
        return np.zeros(0)
    distance = np.zeros(n)
    for axis in range(m):
        order = np.argsort(f[:, axis], kind="stable")
        values = f[order, axis]
        span = values[-1] - values[0]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if span <= 0.0 or n <= 2:
            continue
        interior = np.arange(1, n - 1)
        distance[order[interior]] += (values[interior + 1]
                                      - values[interior - 1]) / span
    return distance


def constrained_dominates(objectives_a: np.ndarray, violation_a: float,
                          objectives_b: np.ndarray,
                          violation_b: float) -> bool:
    """Deb's constrained domination.

    Feasible beats infeasible; between two infeasible designs the smaller
    violation wins; between two feasible ones the ordinary dominance applies.
    """
    if violation_a <= 0.0 and violation_b > 0.0:
        return True
    if violation_a > 0.0 and violation_b <= 0.0:
        return False
    if violation_a > 0.0 and violation_b > 0.0:
        return violation_a < violation_b
    return dominates(objectives_a, objectives_b)


def _rank_with_constraints(objectives: np.ndarray,
                           violations: np.ndarray) -> list[np.ndarray]:
    """Fronts under constrained domination.

    Infeasible designs are ordered purely by violation, so they form a chain of
    single-member fronts behind every feasible one. That is the intended
    behaviour: it drives the population toward feasibility before it starts
    trading objectives off.
    """
    feasible = violations <= 0.0
    fronts: list[np.ndarray] = []
    if np.any(feasible):
        index = np.flatnonzero(feasible)
        for front in fast_non_dominated_sort(objectives[index]):
            fronts.append(index[front])
    infeasible = np.flatnonzero(~feasible)
    if infeasible.size:
        order = infeasible[np.argsort(violations[infeasible], kind="stable")]
        fronts.extend(np.array([i]) for i in order)
    return fronts


# --- variation operators (box-bounded real variables only) -------------------

def simulated_binary_crossover(parent_a: np.ndarray, parent_b: np.ndarray,
                               lower: np.ndarray, upper: np.ndarray,
                               eta: float, rng: np.random.Generator
                               ) -> tuple[np.ndarray, np.ndarray]:
    """SBX. `eta` sets how close children stay to their parents.

    VALIDITY: continuous variables inside a box. There is no meaning to a
    value halfway between two categories.
    """
    u = rng.random(parent_a.shape)
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    child_a = 0.5 * ((1.0 + beta) * parent_a + (1.0 - beta) * parent_b)
    child_b = 0.5 * ((1.0 - beta) * parent_a + (1.0 + beta) * parent_b)
    return (np.clip(child_a, lower, upper), np.clip(child_b, lower, upper))


def polynomial_mutation(individual: np.ndarray, lower: np.ndarray,
                        upper: np.ndarray, eta: float, probability: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Polynomial mutation, bounded. VALIDITY: as for SBX."""
    span = upper - lower
    u = rng.random(individual.shape)
    delta = np.where(u < 0.5,
                     (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                     1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))
    mutate = rng.random(individual.shape) < probability
    moved = np.where(mutate, individual + delta * span, individual)
    return np.clip(moved, lower, upper)


@dataclass
class Nsga2Result:
    """The final population, its ranking, and the approximated front."""

    design: np.ndarray               # (n, n_variables)
    objectives: np.ndarray           # (n, n_objectives) in the ORIGINAL senses
    violation: np.ndarray            # (n,)
    front_index: np.ndarray          # (n,) 0 is the best front
    generations: int
    senses: tuple[Sense, ...]
    front_size_history: list[int] = field(default_factory=list)

    @property
    def front_mask(self) -> np.ndarray:
        return (self.front_index == 0) & (self.violation <= 0.0)

    def front_designs(self) -> np.ndarray:
        return self.design[self.front_mask]

    def front_objectives(self) -> np.ndarray:
        return self.objectives[self.front_mask]

    def choose(self, weights: np.ndarray) -> int:
        """Index of the front member preferred by a weighted sum.

        A preference applied AFTER the front is known, which is the point of
        computing a front at all: the trade-off is visible before anyone has to
        commit to weights. Objectives are normalised to their range across the
        front first, so a weight means the same thing regardless of whether an
        objective is measured in grams or dollars.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != self.objectives.shape[1]:
            raise ValueError(
                f"expected {self.objectives.shape[1]} weights, got {w.shape[0]}")
        front = to_minimisation(self.front_objectives(), list(self.senses))
        if front.shape[0] == 0:
            raise ValueError("no feasible front to choose from")
        low = front.min(axis=0)
        span = np.where(front.max(axis=0) - low > 0.0,
                        front.max(axis=0) - low, 1.0)
        normalised = (front - low) / span
        best_in_front = int(np.argmin(normalised @ w))
        return int(np.flatnonzero(self.front_mask)[best_in_front])


def nsga2(evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
          lower: np.ndarray, upper: np.ndarray, senses: "list[Sense]",
          population: int = 64, generations: int = 60, seed: int = 0,
          crossover_eta: float = 15.0, mutation_eta: float = 20.0,
          mutation_probability: float | None = None) -> Nsga2Result:
    """Approximate the Pareto front of a box-bounded problem.

    `evaluate` takes an (n, n_variables) array and returns objectives in the
    ORIGINAL senses plus a non-negative constraint violation per row. It is
    called once per generation with the whole population, so a batched
    evaluator does one launch rather than n.

    Deterministic for a given seed: every random draw comes from one generator
    seeded here.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("bounds must have the same shape")
    if np.any(upper <= lower):
        raise ValueError("every upper bound must exceed its lower bound")
    if population < 4:
        raise ValueError("a population under 4 cannot support tournament "
                         "selection and crossover")
    if population % 2:
        raise ValueError("population must be even so parents pair up")

    rng = np.random.default_rng(seed)
    n_variables = lower.shape[0]
    if mutation_probability is None:
        mutation_probability = 1.0 / n_variables

    design = lower + rng.random((population, n_variables)) * (upper - lower)
    raw, violation = evaluate(design)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    violation = np.asarray(violation, dtype=float).reshape(-1)
    history: list[int] = []

    for _ in range(generations):
        minimised = to_minimisation(raw, senses)
        fronts = _rank_with_constraints(minimised, violation)
        rank = np.empty(design.shape[0], dtype=int)
        crowding = np.zeros(design.shape[0])
        for order, front in enumerate(fronts):
            rank[front] = order
            crowding[front] = crowding_distance(minimised[front])
        history.append(int(np.count_nonzero((rank == 0) & (violation <= 0.0))))

        # Binary tournament on (rank, then crowding), the standard NSGA-II
        # selection. Crowding is compared only inside a front, which is the
        # only place it means anything.
        contenders = rng.integers(0, design.shape[0], size=(population, 2))
        first, second = contenders[:, 0], contenders[:, 1]
        pick_first = (rank[first] < rank[second]) | (
            (rank[first] == rank[second]) & (crowding[first] > crowding[second]))
        parents = design[np.where(pick_first, first, second)]

        children = np.empty_like(parents)
        for i in range(0, population, 2):
            a, b = simulated_binary_crossover(parents[i], parents[i + 1],
                                              lower, upper, crossover_eta, rng)
            children[i] = polynomial_mutation(a, lower, upper, mutation_eta,
                                              mutation_probability, rng)
            children[i + 1] = polynomial_mutation(b, lower, upper, mutation_eta,
                                                  mutation_probability, rng)

        child_raw, child_violation = evaluate(children)
        child_raw = np.atleast_2d(np.asarray(child_raw, dtype=float))
        child_violation = np.asarray(child_violation, dtype=float).reshape(-1)

        # Elitist survival: parents and children compete together, so a good
        # solution can never be lost to an unlucky generation.
        design = np.vstack([design, children])
        raw = np.vstack([raw, child_raw])
        violation = np.concatenate([violation, child_violation])

        merged = to_minimisation(raw, senses)
        survivors: list[int] = []
        for front in _rank_with_constraints(merged, violation):
            if len(survivors) + front.size <= population:
                survivors.extend(front.tolist())
                continue
            # The front that straddles the cut is trimmed by crowding, keeping
            # the most isolated members so the front stays spread out.
            room = population - len(survivors)
            distances = crowding_distance(merged[front])
            keep = front[np.argsort(-distances, kind="stable")[:room]]
            survivors.extend(keep.tolist())
            break
        keep = np.array(survivors, dtype=int)
        design, raw, violation = design[keep], raw[keep], violation[keep]

    minimised = to_minimisation(raw, senses)
    fronts = _rank_with_constraints(minimised, violation)
    front_index = np.empty(design.shape[0], dtype=int)
    for order, front in enumerate(fronts):
        front_index[front] = order
    history.append(int(np.count_nonzero((front_index == 0) & (violation <= 0.0))))

    return Nsga2Result(design=design, objectives=raw, violation=violation,
                       front_index=front_index, generations=generations,
                       senses=tuple(senses), front_size_history=history)
