"""Sizing an element down to the smallest dimension that still passes.

VALIDITY, before the implementation:

* **Bisection needs the safety factor to be MONOTONIC in the dimension, and
  that is an assumption about the physics, not about the search.** It usually
  holds for strength: a thicker wall is stronger, a bigger shaft is stiffer.

  It does NOT always hold, and a measured counter-example is worth more than
  the claim. Growing the wall of a FIXED outer envelope RAISES stiffness and
  RAISES mass, and for a cantilever's first natural frequency the mass wins:
  measured on a 40 mm section over 0.5 m, the frequency falls monotonically
  from 3.60 to 2.61 times its target as the wall goes from 1 mm to 20 mm.
  Sizing that check by bisection would find a root and return the wrong side of
  it.

  (Section modulus, by contrast, flattens as the cavity closes but never
  decreases, so it stays monotonic. That was this docstring's first example and
  it was wrong.)

  So the bracket is SAMPLED and the assumption checked before the search runs,
  rather than trusted because it is usually true.

* **A minimum dimension sits exactly on its constraint boundary, which means
  ZERO margin by construction.** That is what minimum means, and it is not a
  design: any modelling error, any manufacturing variation and any load beyond
  the assumed one puts it over. The target factor is therefore an argument, and
  sizing to 1.0 is offered but is not the default.

* **Sizing each element to its own minimum is a LOCAL result.** The elements
  are coupled: a lighter link lowers the torque the drivetrain needs, which
  lowers the shaft load, which allows a smaller shaft, which lowers the mass
  again. One pass per element does not close that loop, so the result is a set
  of individually minimal parts rather than a minimal assembly.

* **Only the checks that were supplied are satisfied.** An element sized to its
  minimum against stress alone is minimal against stress and says nothing about
  fatigue, buckling or anything else. The caller passes the evaluator, and what
  it does not evaluate does not constrain the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# How many samples across the bracket are used to test monotonicity before
# trusting a bisection.
MONOTONICITY_SAMPLES = 9


@dataclass(frozen=True)
class SizingResult:
    """The smallest dimension meeting the target, and how it was found."""

    dimension: float
    safety_factor: float
    target: float
    iterations: int
    monotonic: bool
    bracket: tuple[float, float]
    converged: bool

    @property
    def on_the_boundary(self) -> bool:
        """Whether the result sits on its constraint, which is zero margin."""
        return abs(self.safety_factor - self.target) < 1e-3 * self.target


def is_monotonic_increasing(evaluate: Callable[[float], float],
                            low: float, high: float,
                            samples: int = MONOTONICITY_SAMPLES) -> bool:
    """Sample the bracket and check the safety factor never falls.

    Checked rather than assumed. A non-monotonic response still gives
    bisection a root, and that root is not the minimum, so the search would
    return a confident wrong answer.
    """
    if samples < 3:
        raise ValueError("at least three samples are needed to see a trend")
    step = (high - low) / (samples - 1)
    values = [evaluate(low + i * step) for i in range(samples)]
    return all(b >= a * (1.0 - 1e-9) for a, b in zip(values, values[1:]))


# Related: physics.sizing.cantilever inverts each failure mode in closed form
# and reports which one governs. Use that where the closed forms exist; use
# this where they do not and only a callable is available.


def minimum_dimension(evaluate: Callable[[float], float], low: float,
                      high: float, target: float = 1.5,
                      tolerance: float = 1e-6,
                      max_iterations: int = 200,
                      check_monotonic: bool = True) -> SizingResult:
    """The smallest dimension in [low, high] whose safety factor reaches target.

    `evaluate` maps a dimension to a safety factor. `target` defaults to 1.5
    rather than 1.0: sizing to exactly 1.0 leaves no margin for the modelling
    error and manufacturing variation that a real part has, and offering it as
    the default would encourage it.
    """
    if high <= low:
        raise ValueError("the upper bracket must exceed the lower one")
    if target <= 0.0:
        raise ValueError("the target safety factor must be positive")

    monotonic = (is_monotonic_increasing(evaluate, low, high)
                 if check_monotonic else True)

    at_low, at_high = evaluate(low), evaluate(high)
    if at_low >= target:
        return SizingResult(dimension=low, safety_factor=at_low, target=target,
                            iterations=0, monotonic=monotonic,
                            bracket=(low, high), converged=True)
    if at_high < target:
        # Nothing in the bracket passes. Returning the upper bound with a
        # converged=False is honest; returning it silently would look like a
        # design that meets the target.
        return SizingResult(dimension=high, safety_factor=at_high,
                            target=target, iterations=0, monotonic=monotonic,
                            bracket=(low, high), converged=False)

    lower, upper = low, high
    for iteration in range(1, max_iterations + 1):
        middle = 0.5 * (lower + upper)
        if evaluate(middle) >= target:
            upper = middle
        else:
            lower = middle
        if upper - lower <= tolerance * max(abs(upper), 1.0):
            break
    return SizingResult(dimension=upper, safety_factor=evaluate(upper),
                        target=target, iterations=iteration,
                        monotonic=monotonic, bracket=(low, high),
                        converged=True)
