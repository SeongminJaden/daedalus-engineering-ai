"""physics.structural.beam - differentiable cantilever beam evaluator.

Turns a batch of design genomes plus one Engineering IR problem into the five
MVP metrics, on the GPU, differentiably.

    metrics = evaluate_beam(genomes, problem)
    grads   = beam_gradients(genomes, problem, "max_bending_stress_pa")

Fidelity: Euler-Bernoulli beam theory - see physics/warp_kernels/kernels.py
and physics/README.md. Root stress concentration, transverse shear deformation
and buckling are all outside this model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Sequence

import numpy as np

from core.design_genome import DesignGenome
from core.engineering_ir import (
    BoundaryConditionType,
    BoundaryLocation,
    LoadApplication,
    LoadType,
    SectionType,
)
from core.materials import get_material

# Metric name -> index of the kernel output array it comes from.
METRIC_NAMES = (
    "mass_kg",
    "max_bending_stress_pa",
    "tip_deflection_m",
    "safety_factor",
    "first_natural_frequency_hz",
    "mean_transverse_shear_stress_pa",
)

DESIGN_VARIABLES = ("outer_width_m", "outer_height_m", "wall_thickness_m")


@dataclass(frozen=True)
class BeamLoadCase:
    """Everything the kernel needs that is not a design variable. SI."""

    length_m: float
    tip_load_n: float
    youngs_modulus_pa: float
    density_kg_m3: float
    yield_strength_pa: float


@dataclass
class BeamMetrics:
    """One array per metric, indexed by candidate. SI units."""

    mass_kg: np.ndarray
    max_bending_stress_pa: np.ndarray
    tip_deflection_m: np.ndarray
    safety_factor: np.ndarray
    first_natural_frequency_hz: np.ndarray
    mean_transverse_shear_stress_pa: np.ndarray

    def __len__(self) -> int:
        return int(self.mass_kg.shape[0])

    def candidate(self, i: int) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)[i]) for f in fields(self)}

    @classmethod
    def concatenate(cls, parts: Sequence["BeamMetrics"]) -> "BeamMetrics":
        if not parts:
            raise ValueError("nothing to concatenate")
        return cls(**{
            f.name: np.concatenate([getattr(p, f.name) for p in parts])
            for f in fields(cls)
        })


def load_case_from_problem(problem) -> BeamLoadCase:
    """Extract the kernel's load case from an Engineering IR problem.

    Raises if the problem is not the cantilever/tip-load/hollow-rectangle case
    this model actually solves. Silently evaluating an unsupported problem
    would produce numbers that look fine and mean nothing.
    """
    if problem.geometry.section_type is not SectionType.HOLLOW_RECTANGLE:
        raise NotImplementedError(
            f"beam model handles {SectionType.HOLLOW_RECTANGLE.value}, "
            f"problem asks for {problem.geometry.section_type.value}"
        )
    if len(problem.loads) != 1:
        raise NotImplementedError(
            f"beam model handles exactly one load, got {len(problem.loads)}"
        )

    load = problem.loads[0]
    if load.type is not LoadType.POINT_FORCE:
        raise NotImplementedError(f"unsupported load type {load.type.value}")
    if load.application is not LoadApplication.TIP:
        raise NotImplementedError(
            f"beam model applies the load at the tip, got {load.application.value}"
        )

    if len(problem.boundary_conditions) != 1:
        raise NotImplementedError("beam model handles exactly one boundary condition")
    bc = problem.boundary_conditions[0]
    if bc.type is not BoundaryConditionType.FIXED or bc.location is not BoundaryLocation.ROOT:
        raise NotImplementedError(
            f"beam model is a root-fixed cantilever, got "
            f"{bc.type.value} at {bc.location.value}"
        )

    # The kernel bends the section about its horizontal axis, so the load must
    # act along -y (or +y). A sideways or axial load is a different problem.
    dx, dy, dz = load.direction.as_tuple()
    if abs(dx) > 1e-9 or abs(dz) > 1e-9:
        raise NotImplementedError(
            f"beam model takes a transverse (y-axis) tip load, got direction "
            f"({dx}, {dy}, {dz})"
        )

    material = get_material(problem.material_id)
    return BeamLoadCase(
        length_m=problem.geometry.length_m,
        tip_load_n=load.magnitude_n,
        youngs_modulus_pa=material.youngs_modulus_pa,
        density_kg_m3=material.density_kg_m3,
        yield_strength_pa=material.yield_strength_pa,
    )


def _design_arrays(genomes: Sequence[DesignGenome]) -> tuple[np.ndarray, ...]:
    if not genomes:
        raise ValueError("no genomes to evaluate")
    for i, g in enumerate(genomes):
        if not g.is_valid():
            raise ValueError(f"genome[{i}] is invalid: {g.validity_reason()}")
    b = np.array([g.section.outer_width_m for g in genomes], dtype=np.float32)
    h = np.array([g.section.outer_height_m for g in genomes], dtype=np.float32)
    t = np.array([g.section.wall_thickness_m for g in genomes], dtype=np.float32)
    return b, h, t


def _resolve_device(device: str | None) -> str:
    import warp as wp

    if device is not None:
        return device
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    return str(cuda[0]) if cuda else "cpu"


def _launch(b, h, t, case: BeamLoadCase, device: str, requires_grad: bool):
    """Run the kernel once. Returns (tape_or_None, inputs, outputs)."""
    import warp as wp

    from physics.warp_kernels import cantilever_hollow_rect_metrics

    n = b.shape[0]
    ins = [
        wp.array(b, dtype=wp.float32, device=device, requires_grad=requires_grad),
        wp.array(h, dtype=wp.float32, device=device, requires_grad=requires_grad),
        wp.array(t, dtype=wp.float32, device=device, requires_grad=requires_grad),
    ]
    outs = [
        wp.zeros(n, dtype=wp.float32, device=device, requires_grad=requires_grad)
        for _ in METRIC_NAMES
    ]
    scalars = [
        case.length_m, case.tip_load_n, case.youngs_modulus_pa,
        case.density_kg_m3, case.yield_strength_pa,
    ]

    def run():
        wp.launch(
            cantilever_hollow_rect_metrics,
            dim=n,
            inputs=[*ins, *scalars],
            outputs=outs,
            device=device,
        )

    tape = None
    if requires_grad:
        tape = wp.Tape()
        with tape:
            run()
    else:
        run()
    wp.synchronize_device(device)
    return tape, ins, outs


def evaluate_beam(
    genomes: Iterable[DesignGenome],
    problem,
    device: str | None = None,
) -> BeamMetrics:
    """Evaluate a batch of candidates in one GPU launch.

    No chunking here - physics.solver owns batching against the GPU profile.
    """
    genomes = list(genomes)
    case = load_case_from_problem(problem)
    dev = _resolve_device(device)
    b, h, t = _design_arrays(genomes)
    _, _, outs = _launch(b, h, t, case, dev, requires_grad=False)
    return BeamMetrics(**{
        name: out.numpy().astype(np.float64)
        for name, out in zip(METRIC_NAMES, outs)
    })


def beam_gradients(
    genomes: Iterable[DesignGenome],
    problem,
    metric: str,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """d(metric)/d(b, h, t) per candidate, via Warp autodiff.

    Each candidate is independent, so seeding every element of the chosen
    output's adjoint with 1 yields exactly the per-candidate derivative rather
    than a summed one.
    """
    import warp as wp

    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRIC_NAMES}")

    genomes = list(genomes)
    case = load_case_from_problem(problem)
    dev = _resolve_device(device)
    b, h, t = _design_arrays(genomes)
    tape, ins, outs = _launch(b, h, t, case, dev, requires_grad=True)

    idx = METRIC_NAMES.index(metric)
    tape.zero()
    outs[idx].grad = wp.array(
        np.ones(len(genomes), dtype=np.float32), dtype=wp.float32, device=dev
    )
    tape.backward()
    wp.synchronize_device(dev)

    return {
        var: arr.grad.numpy().astype(np.float64)
        for var, arr in zip(DESIGN_VARIABLES, ins)
    }


def beam_gradients_many(
    genomes: Iterable[DesignGenome],
    problem,
    metrics: Sequence[str],
    device: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Gradients of several metrics from a single forward launch.

    An optimizer needs d(mass), d(stress) and d(deflection) at the same point.
    Recording the tape once and replaying `backward()` per metric avoids
    re-running the forward pass for each, which is most of the cost.
    """
    import warp as wp

    unknown = [m for m in metrics if m not in METRIC_NAMES]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; expected {METRIC_NAMES}")

    genomes = list(genomes)
    case = load_case_from_problem(problem)
    dev = _resolve_device(device)
    b, h, t = _design_arrays(genomes)
    tape, ins, outs = _launch(b, h, t, case, dev, requires_grad=True)

    result: dict[str, dict[str, np.ndarray]] = {}
    for metric in metrics:
        tape.zero()
        # A fresh seed each pass: tape.zero() also zeros whatever array is
        # currently attached as an output's adjoint, so reusing one buffer
        # would silently make every pass after the first return zeros.
        outs[METRIC_NAMES.index(metric)].grad = wp.array(
            np.ones(len(genomes), dtype=np.float32), dtype=wp.float32, device=dev
        )
        tape.backward()
        wp.synchronize_device(dev)
        result[metric] = {
            var: arr.grad.numpy().astype(np.float64).copy()
            for var, arr in zip(DESIGN_VARIABLES, ins)
        }
    return result
