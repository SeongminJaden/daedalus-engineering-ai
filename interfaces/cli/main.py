"""interfaces.cli.main - main entrypoint.

    python -m interfaces.cli.main --help
    python -m interfaces.cli.main info --profile laptop_4gb
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer  # noqa: E402

from core import profile as profile_mod  # noqa: E402

app = typer.Typer(add_completion=False, help="engineering-ai design agent CLI")


@app.command()
def info(profile: str = typer.Option(None, "--profile", "-p",
                                     help="GPU profile; auto-detected if omitted")):
    """Show the resolved GPU profile and detected hardware."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    resolved = profile_mod.select_profile_name(profile)
    cfg = profile_mod.load_profile(resolved)
    vram = profile_mod.detect_vram_gb()

    table = Table(title="engineering-ai")
    table.add_column("key", style="bold cyan")
    table.add_column("value")
    table.add_row("detected VRAM", "-" if vram is None else f"{vram:.2f} GB")
    table.add_row("resolved profile", resolved)
    table.add_row("available profiles", ", ".join(profile_mod.available_profiles()))
    for section, body in cfg.items():
        if isinstance(body, dict):
            for k, v in body.items():
                table.add_row(f"{section}.{k}", str(v))
    console.print(table)


@app.command()
def profiles():
    """List available GPU profiles."""
    for name in profile_mod.available_profiles():
        typer.echo(name)


@app.command()
def run(profile: str = typer.Option(None, "--profile", "-p"),
        project: str = typer.Option("robotic_link", "--project")):
    """Run a design loop. Not implemented yet - scaffold only."""
    typer.echo(
        f"[stub] would run project={project} "
        f"under profile={profile_mod.select_profile_name(profile)}"
    )


@app.command()
def evaluate(
    profile: str = typer.Option(None, "--profile", "-p",
                                help="GPU profile; auto-detected if omitted"),
    width_mm: float = typer.Option(50.0, "--width", help="outer width b [mm]"),
    height_mm: float = typer.Option(80.0, "--height", help="outer height h [mm]"),
    thickness_mm: float = typer.Option(5.0, "--thickness", help="wall t [mm]"),
):
    """Evaluate one design against the MVP cantilever problem on the GPU."""
    from rich.console import Console
    from rich.table import Table

    from core.design_genome import DesignGenome, HollowRectangleSection
    from core.units import MM, to_mm, to_mpa
    from physics.solver import evaluate_population
    from projects.robotic_link.problem import build_mvp_problem

    console = Console()
    problem = build_mvp_problem()
    genome = DesignGenome(
        section=HollowRectangleSection(
            outer_width_m=width_mm * MM,
            outer_height_m=height_mm * MM,
            wall_thickness_m=thickness_mm * MM,
        ),
        material_id=problem.material_id,
    )
    if not genome.is_valid():
        typer.echo(f"invalid design: {genome.validity_reason()}")
        raise typer.Exit(code=1)

    metrics = evaluate_population([genome], problem, profile=profile).candidate(0)
    c = problem.constraints

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", justify="right")
    header.add_column()
    header.add_row("problem", problem.name)
    header.add_row("material", problem.material_id)
    header.add_row("length", f"{problem.geometry.length_m} m")
    header.add_row("tip load", f"{problem.loads[0].magnitude_n} N")
    header.add_row("section b x h x t",
                   f"{to_mm(genome.section.outer_width_m):.1f} x "
                   f"{to_mm(genome.section.outer_height_m):.1f} x "
                   f"{to_mm(genome.section.wall_thickness_m):.2f} mm")
    header.add_row("profile", profile or "(auto)")
    console.print(header)

    table = Table(title="evaluated metrics (Euler-Bernoulli beam theory)")
    table.add_column("quantity", style="bold")
    table.add_column("SI", justify="right")
    table.add_column("readable", justify="right")
    table.add_column("limit", justify="right")
    table.add_column("verdict", justify="center")

    def verdict(ok: bool | None) -> str:
        if ok is None:
            return "-"
        return "[green]PASS[/green]" if ok else "[red]FAIL[/red]"

    mass = metrics["mass_kg"]
    stress = metrics["max_bending_stress_pa"]
    defl = metrics["tip_deflection_m"]
    sf = metrics["safety_factor"]
    freq = metrics["first_natural_frequency_hz"]
    shear = metrics["mean_transverse_shear_stress_pa"]

    table.add_row("mass", f"{mass:.6g} kg", f"{mass:.4f} kg", "-", verdict(None))
    table.add_row(
        "max bending stress", f"{stress:.6g} Pa", f"{to_mpa(stress):.3f} MPa",
        f"{to_mpa(c.max_stress_pa):.0f} MPa" if c.max_stress_pa else "-",
        verdict(None if c.max_stress_pa is None else stress <= c.max_stress_pa))
    table.add_row(
        "tip deflection", f"{defl:.6g} m", f"{to_mm(defl):.4f} mm",
        f"{to_mm(c.max_deflection_m):.2f} mm" if c.max_deflection_m else "-",
        verdict(None if c.max_deflection_m is None else defl <= c.max_deflection_m))
    table.add_row(
        "safety factor", f"{sf:.6g}", f"{sf:.2f}",
        f"{c.min_safety_factor:.1f}" if c.min_safety_factor else "-",
        verdict(None if c.min_safety_factor is None else sf >= c.min_safety_factor))
    table.add_row("1st natural frequency", f"{freq:.6g} Hz", f"{freq:.1f} Hz",
                  "-", verdict(None))
    table.add_row("mean shear stress", f"{shear:.6g} Pa",
                  f"{to_mpa(shear):.4f} MPa", "-", verdict(None))
    console.print(table)
    console.print(
        "[dim]Beam theory: no root stress concentration, no shear deformation, "
        "no buckling. Peak real stress at the root will be higher.[/dim]"
    )


@app.command()
def optimize(
    seed: int = typer.Option(0, "--seed", help="differential evolution seed"),
    method: str = typer.Option("both", "--method",
                               help="slsqp | de | both (both cross-verifies)"),
    baseline_width_mm: float = typer.Option(50.0, "--baseline-width"),
    baseline_height_mm: float = typer.Option(80.0, "--baseline-height"),
    baseline_thickness_mm: float = typer.Option(5.0, "--baseline-thickness"),
):
    """Minimize link mass subject to the MVP constraints, on the GPU."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.units import MM, to_mm, to_mpa
    from optimization.constraints import build_optimization_problem, evaluate_design
    from optimization.evolutionary import optimize_differential_evolution
    from optimization.gradient import optimize_slsqp
    from projects.robotic_link.problem import build_mvp_problem

    console = Console()
    problem = build_mvp_problem()
    op = build_optimization_problem(problem)

    baseline_x = np.array([baseline_width_mm * MM, baseline_height_mm * MM,
                           baseline_thickness_mm * MM])
    baseline = evaluate_design(op, baseline_x)

    setup = Table.grid(padding=(0, 2))
    setup.add_column(style="bold cyan", justify="right")
    setup.add_column()
    setup.add_row("problem", problem.name)
    setup.add_row("objective", "minimize mass")
    setup.add_row("allowable stress",
                  f"{to_mpa(op.allowable_stress_pa):.1f} MPa  "
                  f"(min of ceiling and yield/SF)")
    setup.add_row("max deflection", f"{to_mm(op.max_deflection_m):.2f} mm")
    setup.add_row("bounds b,h", f"{to_mm(op.lower[0]):.0f}-{to_mm(op.upper[0]):.0f} mm")
    setup.add_row("bounds t",
                  f"{to_mm(op.lower[2]):.1f}-{to_mm(op.upper[2]):.0f} mm  "
                  f"(ASSUMED min wall: CNC aluminium)")
    console.print(setup)

    runs = []
    if method in ("slsqp", "both"):
        runs.append(optimize_slsqp(op))
    if method in ("de", "both"):
        runs.append(optimize_differential_evolution(op, seed=seed))

    table = Table(title="optimized designs")
    table.add_column("quantity", style="bold")
    table.add_column("baseline", justify="right")
    for r in runs:
        table.add_column(r.method, justify="right")

    def row(label, fmt, get):
        table.add_row(label, fmt(get(baseline)), *[fmt(get(r.evaluation)) for r in runs])

    # Units go in parentheses, never brackets: rich reads "[mm]" as a markup
    # tag and silently eats it.
    table.add_row("b (m)", f"{baseline_x[0]:.6g}",
                  *[f"{r.x[0]:.6g}" for r in runs])
    table.add_row("b (mm)", f"{to_mm(baseline_x[0]):.3f}",
                  *[f"{to_mm(r.x[0]):.3f}" for r in runs])
    table.add_row("h (m)", f"{baseline_x[1]:.6g}",
                  *[f"{r.x[1]:.6g}" for r in runs])
    table.add_row("h (mm)", f"{to_mm(baseline_x[1]):.3f}",
                  *[f"{to_mm(r.x[1]):.3f}" for r in runs])
    table.add_row("t (m)", f"{baseline_x[2]:.6g}",
                  *[f"{r.x[2]:.6g}" for r in runs])
    table.add_row("t (mm)", f"{to_mm(baseline_x[2]):.3f}",
                  *[f"{to_mm(r.x[2]):.3f}" for r in runs])
    row("mass (kg, SI)", lambda v: f"{v:.6f}", lambda e: e.mass_kg)
    row("sigma_max (Pa, SI)", lambda v: f"{v:.6g}",
        lambda e: e.max_bending_stress_pa)
    row("sigma_max (MPa)", lambda v: f"{to_mpa(v):.3f}",
        lambda e: e.max_bending_stress_pa)
    row("delta_tip (m, SI)", lambda v: f"{v:.6g}", lambda e: e.tip_deflection_m)
    row("delta_tip (mm)", lambda v: f"{to_mm(v):.5f}", lambda e: e.tip_deflection_m)
    row("safety factor (-)", lambda v: f"{v:.2f}", lambda e: e.safety_factor)
    row("f1 (Hz, SI)", lambda v: f"{v:.2f}", lambda e: e.first_natural_frequency_hz)
    table.add_row(
        "mass reduction", "-",
        *[f"[green]{1.0 - r.evaluation.mass_kg / baseline.mass_kg:.1%}[/green]"
          for r in runs])
    table.add_row(
        "active constraint", ", ".join(baseline.active_constraints()) or "none",
        *[", ".join(r.evaluation.active_constraints()) or "none" for r in runs])
    table.add_row(
        "all constraints",
        "[green]PASS[/green]" if baseline.is_feasible() else "[red]FAIL[/red]",
        *[("[green]PASS[/green]" if r.evaluation.is_feasible()
           else "[red]FAIL[/red]") for r in runs])
    table.add_row("evaluations", "1", *[str(r.n_evaluations) for r in runs])
    console.print(table)

    if len(runs) == 2:
        rel = abs(runs[0].mass_kg - runs[1].mass_kg) / runs[0].mass_kg
        verdict = "[green]AGREE[/green]" if rel < 0.01 else "[red]DISAGREE[/red]"
        console.print(
            f"cross-verification: |SLSQP - DE| / SLSQP = [bold]{rel:.3e}[/bold] "
            f"({rel:.4%})  {verdict}  [dim](independent local and global methods)[/dim]"
        )

    console.print(
        "[dim]Beam theory: no root stress concentration, no shear deformation, "
        "no buckling. Peak real stress at the root will be higher.[/dim]"
    )


if __name__ == "__main__":
    app()
