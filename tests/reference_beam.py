"""Independent closed-form reference for the cantilever beam metrics.

Deliberately standalone: this module imports **no Warp and no project code**.
It re-derives the section properties from the raw dimensions rather than
calling core.design_genome, so a mistake shared between the kernel and the rest
of the codebase cannot hide here too. Everything is float64.

Euler-Bernoulli cantilever, transverse point load P at the free tip, uniform
hollow rectangular section (h vertical, aligned with the load):

    b_i = b - 2t                    h_i = h - 2t
    A   = b*h - b_i*h_i
    I   = (b*h^3 - b_i*h_i^3) / 12
    c   = h/2
    m   = A*L*rho
    M   = P*L                       (root bending moment)
    sigma_max = M*c/I
    delta     = P*L^3 / (3*E*I)
    SF        = sigma_yield / sigma_max
    f1        = (beta1^2 / 2pi) * sqrt(E*I / (rho*A*L^4)),  beta1*L = 1.875104
    tau_avg   = P/A
"""

from __future__ import annotations

import math

import numpy as np

BETA1 = 1.875104


def reference_metrics(b, h, t, length_m, tip_load_n, youngs_modulus_pa,
                      density_kg_m3, yield_strength_pa) -> dict[str, np.ndarray]:
    """Closed-form metrics in float64. Arrays or scalars accepted."""
    b = np.asarray(b, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    L = float(length_m)
    P = float(tip_load_n)
    E = float(youngs_modulus_pa)
    rho = float(density_kg_m3)
    sy = float(yield_strength_pa)

    b_inner = b - 2.0 * t
    h_inner = h - 2.0 * t

    area = b * h - b_inner * h_inner
    inertia = (b * h**3 - b_inner * h_inner**3) / 12.0
    c = h / 2.0

    mass = area * L * rho
    moment = P * L
    sigma = moment * c / inertia
    delta = P * L**3 / (3.0 * E * inertia)
    safety = sy / sigma
    freq = (BETA1**2 / (2.0 * math.pi)) * np.sqrt(E * inertia / (rho * area * L**4))
    tau = P / area

    return {
        "mass_kg": mass,
        "max_bending_stress_pa": sigma,
        "tip_deflection_m": delta,
        "safety_factor": safety,
        "first_natural_frequency_hz": freq,
        "mean_transverse_shear_stress_pa": tau,
    }


def reference_metric_scalar(name, b, h, t, **case) -> float:
    """One metric as a plain float - convenient for finite differencing."""
    return float(reference_metrics(b, h, t, **case)[name])


def central_difference(name, b, h, t, variable, rel_step=1e-5, **case) -> float:
    """d(metric)/d(variable) by central difference on the float64 reference.

    Differentiating the independent reference rather than the kernel keeps the
    comparison honest: if the kernel's forward pass were wrong, a self-consistent
    autodiff would still match a finite difference taken on that same wrong
    function. This way both value and derivative are checked against outside
    algebra.
    """
    values = {"b": b, "h": h, "t": t}
    if variable not in values:
        raise ValueError(f"unknown variable {variable!r}")
    x = values[variable]
    step = rel_step * x

    lo = dict(values, **{variable: x - step})
    hi = dict(values, **{variable: x + step})
    f_lo = reference_metric_scalar(name, lo["b"], lo["h"], lo["t"], **case)
    f_hi = reference_metric_scalar(name, hi["b"], hi["h"], hi["t"], **case)
    return (f_hi - f_lo) / (2.0 * step)
