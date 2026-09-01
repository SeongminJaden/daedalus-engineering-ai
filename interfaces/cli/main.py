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

app = typer.Typer(add_completion=False,
                  help="Daedalus - autonomous engineering design agent")


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

    table = Table(title="Daedalus Engineering AI")
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


@app.command()
def run(
    iterations: int = typer.Option(6, "--iterations", "-n"),
    seed: int = typer.Option(1, "--seed"),
    target_mass_kg: float = typer.Option(None, "--target-mass",
                                         help="stop once a feasible design is this light"),
    max_evaluations: int = typer.Option(None, "--max-evaluations"),
    max_seconds: float = typer.Option(None, "--max-seconds"),
    profile: str = typer.Option(None, "--profile", "-p"),
    tui: bool = typer.Option(True, "--tui/--no-tui",
                             help="live terminal dashboard; --no-tui for headless"),
):
    """Run the autonomous design loop on the MVP problem."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from agent.experiment_manager import EpisodeLog
    from agent.loop import DesignLoop, LoopConfig
    from brain import Brain as EngineeringBrain
    from core.units import to_mm, to_mpa
    from monitoring.dashboard import Dashboard, RunState
    from optimization.constraints import build_optimization_problem, evaluate_design
    from projects.robotic_link.problem import build_mvp_problem

    console = Console()
    problem = build_mvp_problem()
    op = build_optimization_problem(problem)
    baseline = evaluate_design(op, np.array([0.05, 0.08, 0.005]))

    config = LoopConfig(
        max_iterations=iterations,
        seed=seed,
        target_mass_kg=target_mass_kg,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
        profile=profile,
    )

    run_dir = Path("runs") / f"loop-{seed}"
    log = EpisodeLog(run_dir / "episodes.jsonl")

    console.print(
        f"[bold cyan]autonomous design loop[/bold cyan]  problem={problem.name}  "
        f"seed={seed}  max_iterations={iterations}"
    )
    console.print(
        "[dim]reasoner: rule-based explore/exploit heuristic "
        "(not a language model)[/dim]"
    )

    memory = EngineeringBrain()
    warm = memory.warm_start()
    if warm is not None:
        console.print(
            f"[dim]warm start from brain: b/h/t = "
            f"{to_mm(warm[0]):.2f} / {to_mm(warm[1]):.2f} / "
            f"{to_mm(warm[2]):.2f} mm[/dim]"
        )

    if tui:
        state = RunState(profile=profile or "(auto)", status="starting")
        with Dashboard(state, refresh_hz=8) as dash:
            loop = DesignLoop(op, config, episode_log=log, dashboard=dash,
                              brain=memory, problem_name=problem.name)
            result = loop.run()
    else:
        loop = DesignLoop(op, config, episode_log=log, brain=memory,
                          problem_name=problem.name)
        result = loop.run()

    learned = memory.generalize()
    memory.close()

    # --- per-iteration trace ---
    trace = Table(title="iterations")
    trace.add_column("#", justify="right")
    trace.add_column("action")
    trace.add_column("strategy")
    trace.add_column("mass (kg)", justify="right")
    trace.add_column("feasible", justify="center")
    trace.add_column("best", justify="center")
    trace.add_column("evals", justify="right")
    for e in result.episodes:
        trace.add_row(
            str(e.iteration), e.action, e.strategy_used,
            f"{e.observation['mass_kg']:.6f}",
            "[green]yes[/green]" if e.feasible else "[red]no[/red]",
            "[green]NEW[/green]" if e.is_new_best else "-",
            str(e.evaluations),
        )
    console.print(trace)

    if result.episodes:
        first = result.episodes[0]
        console.print(f"[dim]hypothesis (iteration 0): {first.hypothesis}[/dim]")
        console.print(f"[dim]conclusion  (iteration 0): {first.conclusion}[/dim]")

    # --- outcome ---
    if result.best_evaluation is None:
        console.print("[red]no feasible design found[/red]")
    else:
        ev = result.best_evaluation
        out = Table(title="best design")
        out.add_column("quantity", style="bold")
        out.add_column("baseline", justify="right")
        out.add_column("best", justify="right")
        out.add_row("b (mm)", f"{to_mm(0.05):.3f}", f"{to_mm(result.best_x[0]):.3f}")
        out.add_row("h (mm)", f"{to_mm(0.08):.3f}", f"{to_mm(result.best_x[1]):.3f}")
        out.add_row("t (mm)", f"{to_mm(0.005):.3f}", f"{to_mm(result.best_x[2]):.3f}")
        out.add_row("mass (kg, SI)", f"{baseline.mass_kg:.6f}", f"{ev.mass_kg:.6f}")
        out.add_row("sigma_max (MPa)", f"{to_mpa(baseline.max_bending_stress_pa):.3f}",
                    f"{to_mpa(ev.max_bending_stress_pa):.3f}")
        out.add_row("delta_tip (mm)", f"{to_mm(baseline.tip_deflection_m):.5f}",
                    f"{to_mm(ev.tip_deflection_m):.5f}")
        out.add_row("safety factor (-)", f"{baseline.safety_factor:.2f}",
                    f"{ev.safety_factor:.2f}")
        out.add_row("f1 (Hz, SI)", f"{baseline.first_natural_frequency_hz:.2f}",
                    f"{ev.first_natural_frequency_hz:.2f}")
        out.add_row("mass reduction", "-",
                    f"[green]{1.0 - ev.mass_kg / baseline.mass_kg:.1%}[/green]")
        out.add_row("active constraint", "none",
                    ", ".join(ev.active_constraints()) or "none")
        out.add_row("all constraints", "PASS" if baseline.is_feasible() else "FAIL",
                    "[green]PASS[/green]" if ev.is_feasible() else "[red]FAIL[/red]")
        console.print(out)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column()
    summary.add_row("termination", f"{result.termination.value}")
    summary.add_row("detail", result.termination_detail)
    summary.add_row("iterations", str(result.iterations))
    summary.add_row("budget",
                    f"{result.budget['evaluations']}/{result.budget['max_evaluations']} "
                    f"evaluations, {result.budget['seconds']:.1f}/"
                    f"{result.budget['max_seconds']:.0f} s")
    summary.add_row("episodes", str(result.episode_log_path))
    summary.add_row("knowledge learned", str(len(learned)))
    console.print(summary)
    for k in learned:
        console.print(
            f"  [bold]{k.evidence_level.value}[/bold] "
            f"(conf {k.confidence:.2f}): {k.statement}"
        )
    console.print(
        "[dim]Beam theory: no root stress concentration, no shear deformation, "
        "no buckling. Peak real stress at the root will be higher.[/dim]"
    )


@app.command()
def brain(
    db: str = typer.Option(None, "--db", help="brain database path"),
    domain: str = typer.Option("cantilever_link", "--domain"),
    generalize: bool = typer.Option(False, "--generalize",
                                    help="run generalization before reporting"),
):
    """Inspect the engineering Brain: what it holds and how far to trust it."""
    from rich.console import Console
    from rich.table import Table

    from brain import Brain, EvidenceLevel

    console = Console()
    with Brain(db) as b:
        if generalize:
            b.generalize(domain)

        summary = b.summary()
        head = Table.grid(padding=(0, 2))
        head.add_column(style="bold cyan", justify="right")
        head.add_column()
        head.add_row("database", summary["path"])
        for table_name, n in summary["counts"].items():
            head.add_row(table_name, str(n))
        console.print(head)

        levels = Table(title="knowledge by evidence level")
        levels.add_column("level", style="bold")
        levels.add_column("items", justify="right")
        levels.add_column("ceiling", justify="right")
        from brain.semantic import LEVEL_CONFIDENCE_CEILING
        for level in EvidenceLevel:
            levels.add_row(level.value,
                           str(summary["knowledge_by_level"][level.value]),
                           f"{LEVEL_CONFIDENCE_CEILING[level]:.2f}")
        console.print(levels)

        items = b.knowledge(domain)
        if items:
            table = Table(title=f"knowledge: {domain}")
            table.add_column("level", style="bold")
            table.add_column("conf", justify="right")
            table.add_column("evidence", justify="right")
            table.add_column("runs", justify="right")
            table.add_column("statement")
            from brain.semantic import independent_runs
            for k in items:
                table.add_row(k.evidence_level.value, f"{k.confidence:.3f}",
                              str(len(k.evidence)),
                              str(independent_runs(k.evidence)),
                              k.statement)
            console.print(table)

        strategies = b.applicable_strategies()
        if strategies:
            table = Table(title="strategies")
            table.add_column("level", style="bold")
            table.add_column("conf", justify="right")
            table.add_column("name")
            table.add_column("statement")
            for st in strategies:
                table.add_row(st.evidence_level.value, f"{st.confidence:.3f}",
                              st.name, st.statement)
            console.print(table)

        console.print(
            "[dim]This is a store of evidence-graded EXPERIENCE, not validated "
            "fact. Everything above came from simulation at beam-theory "
            "fidelity; EXPERIMENTALLY_VALIDATED requires physical test "
            "evidence and is unreachable from simulation alone. Retrieval is "
            "numeric feature similarity, not semantic search.[/dim]"
        )


@app.command()
def verify(
    width_mm: float = typer.Option(10.0, "--width", help="outer width b in mm"),
    height_mm: float = typer.Option(80.96, "--height", help="outer height h in mm"),
    thickness_mm: float = typer.Option(1.0, "--thickness", help="wall t in mm"),
    profile: str = typer.Option(None, "--profile", "-p"),
    elements_through_wall: int = typer.Option(2, "--wall-elements"),
    record: bool = typer.Option(False, "--record",
                                help="write the result into the Brain"),
):
    """Verify one design with 3D FEM, the high fidelity gate of the funnel."""
    from rich.console import Console
    from rich.table import Table

    from core.design_genome import DesignGenome, HollowRectangleSection
    from core.units import MM, to_mm, to_mpa
    from optimization.constraints import build_optimization_problem, evaluate_design
    from physics.fem import high_fidelity_verify
    from projects.robotic_link.problem import build_mvp_problem

    console = Console()
    problem = build_mvp_problem()
    op = build_optimization_problem(problem)
    import numpy as np

    x = np.array([width_mm * MM, height_mm * MM, thickness_mm * MM])
    genome = DesignGenome(
        section=HollowRectangleSection(outer_width_m=x[0], outer_height_m=x[1],
                                       wall_thickness_m=x[2]),
        material_id=problem.material_id)
    if not genome.is_valid():
        typer.echo(f"invalid design: {genome.validity_reason()}")
        raise typer.Exit(code=1)

    beam = evaluate_design(op, x)
    console.print("[dim]running 3D FEM, this takes longer than beam theory[/dim]")
    result = high_fidelity_verify(genome, problem, profile=profile,
                                  elements_through_wall=elements_through_wall)

    table = Table(title="fidelity comparison: beam theory vs 3D FEM")
    table.add_column("quantity", style="bold")
    table.add_column("beam (Phase 2)", justify="right")
    table.add_column("3D FEM (Phase 7)", justify="right")
    table.add_column("limit", justify="right")
    table.add_column("verdict", justify="center")

    def verdict(ok):
        return "[green]PASS[/green]" if ok else "[red]FAIL[/red]"

    limit_d = op.max_deflection_m
    table.add_row("tip deflection (mm)",
                  f"{to_mm(beam.tip_deflection_m):.5f}",
                  f"{to_mm(result.tip_deflection_m):.5f}",
                  f"{to_mm(limit_d):.3f}",
                  verdict(result.tip_deflection_m <= limit_d))
    table.add_row("stress (MPa)",
                  f"{to_mpa(beam.max_bending_stress_pa):.3f}",
                  f"{to_mpa(result.gauge_von_mises_pa):.3f} gauge / "
                  f"{to_mpa(result.peak_von_mises_pa):.3f} peak",
                  f"{to_mpa(op.allowable_stress_pa):.0f}",
                  verdict(result.peak_von_mises_pa <= op.allowable_stress_pa))
    table.add_row("safety factor",
                  f"{beam.safety_factor:.2f}",
                  f"{result.safety_factor_peak:.2f} peak / "
                  f"{result.safety_factor_gauge:.2f} gauge",
                  f"{problem.constraints.min_safety_factor:.1f}",
                  verdict(result.safety_factor_peak
                          >= problem.constraints.min_safety_factor))
    table.add_row("mass (kg)", f"{beam.mass_kg:.6f}", "same geometry", "-", "-")
    console.print(table)

    detail = Table.grid(padding=(0, 2))
    detail.add_column(style="bold cyan", justify="right")
    detail.add_column()
    detail.add_row("fidelity", result.fidelity)
    detail.add_row("mesh", f"{result.mesh['grid']}  {result.n_dofs} DOFs")
    detail.add_row("CG", f"{result.iterations} iterations, "
                         f"converged={result.converged}")
    detail.add_row("deflection vs beam", f"{result.deflection_ratio:.4f}x")
    detail.add_row("gauge agreement",
                   f"{result.gauge_agreement:.4f} (FEM / beam at same fibre)")
    detail.add_row("stress concentration", f"{result.stress_concentration_factor:.3f}x")
    console.print(detail)

    for message in result.warnings:
        console.print(f"[yellow]![/yellow] {message}")

    if record:
        from brain import Brain, Evidence, EvidenceKind, Knowledge
        with Brain() as memory:
            statement = (
                f"3D FEM verification of a {to_mm(x[0]):.1f} x {to_mm(x[1]):.1f} x "
                f"{to_mm(x[2]):.2f} mm section: tip deflection "
                f"{to_mm(result.tip_deflection_m):.4f} mm, "
                f"{result.deflection_ratio:.3f}x the beam theory value."
            )
            memory.semantic.upsert_by_claim(Knowledge(
                claim_key=f"fem3d:{to_mm(x[0]):.1f}x{to_mm(x[1]):.1f}x{to_mm(x[2]):.2f}",
                statement=statement,
                domain="cantilever_link",
                source="physics.fem high fidelity verification",
                evidence=[Evidence(
                    kind=EvidenceKind.SIMULATION,
                    ref=f"fem3d-{result.n_dofs}dof",
                    note=f"fidelity={result.fidelity}, higher than beam theory "
                         f"but still simulation",
                )],
                assumptions=[
                    "3D linear elasticity, small strain, isotropic material.",
                    "Idealised fully clamped root face: a stress singularity, so "
                    "peak stress is mesh dependent.",
                    "No fillet, no fastener detail, no contact at the support.",
                ],
            ))
            console.print("[dim]recorded to the Brain as SIMULATION evidence: "
                          "higher fidelity than beam theory, still not a "
                          "physical test[/dim]")

    console.print(
        "[dim]3D FEM is still a simulation: linear elastic, small strain, "
        "idealised clamped boundary. Higher fidelity than beam theory, but a "
        "passing design remains a candidate, not a verified part.[/dim]"
    )


@app.command()
def export(
    width_mm: float = typer.Option(10.0, "--width", help="outer width b in mm"),
    height_mm: float = typer.Option(81.6185, "--height", help="outer height h in mm"),
    thickness_mm: float = typer.Option(1.0, "--thickness", help="wall t in mm"),
    out_dir: str = typer.Option("runs/cad", "--out-dir"),
    name: str = typer.Option("design", "--name"),
    stl: bool = typer.Option(False, "--stl", help="also write a tessellated STL"),
):
    """Export a design to STEP, checked against the analysed part."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.units import MM, to_mm
    from geometry.cad_export import INSTALL_HINT, find_kernel
    from optimization.constraints import build_optimization_problem, evaluate_design
    from projects.robotic_link.problem import build_mvp_problem

    console = Console()
    kernel = find_kernel()
    if kernel is None:
        console.print(f"[red]No CAD kernel available.[/red]\n{INSTALL_HINT}")
        raise typer.Exit(code=1)

    from geometry.cad_export import build_solid, export_step, export_stl

    problem = build_mvp_problem()
    op = build_optimization_problem(problem)
    from core.materials import get_material
    material = get_material(problem.material_id)

    x = np.array([width_mm * MM, height_mm * MM, thickness_mm * MM])
    if not op.is_geometrically_valid(x):
        console.print("[red]invalid geometry: the wall leaves no cavity[/red]")
        raise typer.Exit(code=1)

    analysis = evaluate_design(op, x)
    step_path = Path(out_dir) / f"{name}.step"

    report = export_step(
        length_m=problem.geometry.length_m,
        outer_width_m=x[0], outer_height_m=x[1], wall_thickness_m=x[2],
        path=step_path,
        density_kg_m3=material.density_kg_m3,
        analytic_mass_kg=analysis.mass_kg,
        kernel=kernel,
    )

    table = Table(title="STEP export, checked against the analysis")
    table.add_column("quantity", style="bold")
    table.add_column("CAD", justify="right")
    table.add_column("analysis", justify="right")
    table.add_column("relative error", justify="right")
    table.add_row("volume (m^3)", f"{report.volume_m3:.9g}",
                  f"{report.analytic_volume_m3:.9g}",
                  f"{report.volume_relative_error:.3e}")
    table.add_row("mass (kg)", f"{report.mass_kg:.9g}",
                  f"{report.analytic_mass_kg:.9g}",
                  f"{report.mass_relative_error:.3e}")
    table.add_row("bounding box (mm)",
                  " x ".join(f"{to_mm(v):.4f}" for v in report.bounding_box_m),
                  f"{problem.geometry.length_m * 1000:.4f} x "
                  f"{height_mm:.4f} x {width_mm:.4f}", "-")
    table.add_row("solids", str(report.solid_count), "1", "-")
    console.print(table)

    console.print(f"kernel: [bold]{report.kernel}[/bold]")
    console.print(f"wrote [bold]{step_path}[/bold] "
                  f"({step_path.stat().st_size} bytes)")

    if stl:
        solid = build_solid(problem.geometry.length_m, x[0], x[1], x[2], kernel)
        stl_path = Path(out_dir) / f"{name}.stl"
        export_stl(solid, stl_path, kernel)
        console.print(f"wrote [bold]{stl_path}[/bold] "
                      f"({stl_path.stat().st_size} bytes) "
                      f"[dim]tessellated approximation, not dimensionally "
                      f"exact[/dim]")

    console.print(
        "[dim]This is the ANALYSIS geometry, not a manufacturing-ready part. "
        "There are no fillets, no fastener features and no tolerances on it. "
        "In particular the sharp root corner is exactly where Phase 7 found the "
        "stress concentration, and a real part would need a fillet there.[/dim]"
    )


@app.command()
def assemble(
    payload_kg: float = typer.Option(2.0, "--payload", help="tip payload in kg"),
    shoulder_deg: float = typer.Option(None, "--shoulder",
                                       help="shoulder angle; worst gravity pose if omitted"),
    elbow_deg: float = typer.Option(0.0, "--elbow"),
    step: bool = typer.Option(False, "--step", help="write the posed assembly to STEP"),
    out_dir: str = typer.Option("runs/cad", "--out-dir"),
):
    """Analyse the two-link arm: kinematics, statics and per-link structure."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from core.units import to_mm, to_mpa
    from projects.robotic_arm.arm import analyse, build_arm

    console = Console()
    arm = build_arm()
    q = None
    if shoulder_deg is not None:
        q = np.array([np.deg2rad(shoulder_deg), np.deg2rad(elbow_deg)])
    result = analyse(arm, q, payload_kg=payload_kg)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("assembly", arm.name)
    head.add_row("degrees of freedom", str(arm.dof))
    head.add_row("material", arm.material_id)
    head.add_row("payload", f"{payload_kg} kg")
    head.add_row("pose (deg)", ", ".join(f"{np.rad2deg(v):.2f}" for v in result["q"]))
    head.add_row("tool position (m)",
                 ", ".join(f"{v:.5f}" for v in result["tool_position_m"]))
    head.add_row("structure mass", f"{result['total_mass_kg']:.4f} kg")
    console.print(head)

    torque_table = Table(title="joint torques to hold this pose")
    torque_table.add_column("joint", style="bold")
    torque_table.add_column("type")
    torque_table.add_column("torque (N m)", justify="right")
    for joint, torque in zip(arm.actuated_joints(), result["joint_torques_nm"]):
        torque_table.add_row(joint.name, joint.type.value, f"{torque:+.4f}")
    console.print(torque_table)

    table = Table(title="per-link load case and structural check")
    table.add_column("link", style="bold")
    table.add_column("root moment (N m)", justify="right")
    table.add_column("equiv tip load (N)", justify="right")
    table.add_column("sigma (MPa)", justify="right")
    table.add_column("allow (MPa)", justify="right")
    table.add_column("delta (mm)", justify="right")
    table.add_column("SF", justify="right")
    table.add_column("verdict", justify="center")
    for load, verdict in zip(result["link_loads"], result["verdicts"]):
        table.add_row(
            verdict.link, f"{load.root_bending_moment_nm:.4f}",
            f"{verdict.equivalent_tip_load_n:.3f}",
            f"{to_mpa(verdict.max_bending_stress_pa):.3f}",
            f"{to_mpa(verdict.allowable_stress_pa):.0f}",
            f"{to_mm(verdict.tip_deflection_m):.4f}",
            f"{verdict.safety_factor:.2f}",
            "[green]PASS[/green]" if verdict.passes else "[red]FAIL[/red]")
    console.print(table)

    if step:
        from geometry.cad_export import INSTALL_HINT, find_kernel
        if find_kernel() is None:
            console.print(f"[yellow]skipping STEP:[/yellow] {INSTALL_HINT}")
        else:
            from geometry.cad_export import export_assembly_step
            density = get_material(arm.material_id).density_kg_m3
            path = Path(out_dir) / "arm_assembly.step"
            report = export_assembly_step(arm, result["q"], density, path)
            console.print(
                f"wrote [bold]{path}[/bold] ({path.stat().st_size} bytes), "
                f"{report.part_count} parts, assembly mass "
                f"{report.total_mass_kg:.6f} kg vs sum of links "
                f"{report.analytic_mass_kg:.6f} kg "
                f"(relative {report.mass_relative_error:.2e})")

    console.print(
        "[dim]Statics only: no inertia, no Coriolis or acceleration torque, no "
        "friction, no backlash, no joint compliance. Rigid bodies on ideal "
        "joints. These torques size a link; they do NOT size a motor or a "
        "gearbox, which needs the dynamic terms. Still SIMULATED.[/dim]"
    )


@app.command()
def dynamics(
    nominal_payload_kg: float = typer.Option(2.0, "--payload"),
    max_payload_kg: float = typer.Option(5.0, "--max-payload"),
    max_accel: float = typer.Option(20.0, "--max-accel",
                                    help="peak joint acceleration, rad/s^2"),
):
    """Duty cycle for the two-link arm: torque and power per load case."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from physics.dynamics import evaluate_duty_cycle
    from projects.robotic_arm.arm import build_arm

    console = Console()
    arm = build_arm()
    density = get_material(arm.material_id).density_kg_m3
    duty = evaluate_duty_cycle(arm, density,
                               nominal_payload_kg=nominal_payload_kg,
                               max_payload_kg=max_payload_kg,
                               max_accel_rad_s2=max_accel)
    names = [j.name for j in arm.actuated_joints()]

    table = Table(title="load cases: torque, power, and how much is dynamic")
    table.add_column("case", style="bold")
    table.add_column("duty", justify="right")
    for name in names:
        table.add_column(f"{name} tau (N m)", justify="right")
        table.add_column(f"{name} P (W)", justify="right")
    table.add_column("dynamic share", justify="right")

    for result in duty.results:
        row = [result.case.name, f"{result.case.duty_fraction:.0%}"]
        for i in range(len(names)):
            row.append(f"{result.torque_nm[i]:+.4f}")
            row.append(f"{result.power_w[i]:+.3f}")
        row.append(", ".join(f"{v:+.1%}" for v in result.dynamic_share))
        table.add_row(*row)
    console.print(table)

    summary = Table(title="actuator requirements per joint")
    summary.add_column("joint", style="bold")
    summary.add_column("peak torque (N m)", justify="right")
    summary.add_column("continuous RMS (N m)", justify="right")
    summary.add_column("peak / continuous", justify="right")
    summary.add_column("peak power (W)", justify="right")
    peak = duty.peak_torque_nm()
    continuous = duty.continuous_torque_nm()
    ratio = duty.peak_to_continuous_ratio()
    power = duty.peak_power_w()
    for i, name in enumerate(names):
        summary.add_row(name, f"{peak[i]:.4f}", f"{continuous[i]:.4f}",
                        f"{ratio[i]:.2f}x", f"{power[i]:.3f}")
    console.print(summary)

    console.print(
        "[dim]Peak and continuous are different ratings and both are needed: "
        "sizing to the peak alone over-specifies the drive, sizing to the "
        "continuous value alone overheats it on every acceleration. A negative "
        "dynamic share means acceleration is relieving gravity in that pose, "
        "not adding to it.[/dim]")
    console.print(
        "[dim]Rigid bodies on ideal joints. Friction, backlash and joint "
        "compliance are all zero: the terms exist, the data does not. These are "
        "the torques a motor must supply; SELECTING that motor and its gearbox "
        "is a later phase. Still SIMULATED.[/dim]")


@app.command()
def drivetrain(
    nominal_payload_kg: float = typer.Option(2.0, "--payload"),
    max_payload_kg: float = typer.Option(5.0, "--max-payload"),
    joint_speed: float = typer.Option(1.0, "--joint-speed",
                                      help="maximum joint speed, rad/s"),
    safety_factor: float = typer.Option(1.0, "--safety-factor"),
    max_backlash: float = typer.Option(None, "--max-backlash",
                                       help="backlash limit in arcminutes"),
):
    """Select a motor and gearbox for each joint from the duty cycle."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from drivetrain.selection import (
        Requirement, compare_alternatives, evaluate_candidate,
        infeasibility_report, select_drivetrain,
    )
    from drivetrain.gearboxes import gearboxes as all_gearboxes
    from drivetrain.motors import motors as all_motors
    from physics.dynamics import evaluate_duty_cycle, mass_matrix
    from projects.robotic_arm.arm import build_arm

    console = Console()
    arm = build_arm()
    density = get_material(arm.material_id).density_kg_m3
    duty = evaluate_duty_cycle(arm, density,
                               nominal_payload_kg=nominal_payload_kg,
                               max_payload_kg=max_payload_kg)
    inertia = mass_matrix(arm, [np.pi / 4, 0.2], density)
    peak, continuous = duty.peak_torque_nm(), duty.continuous_torque_nm()

    for i, joint in enumerate(arm.actuated_joints()):
        req = Requirement(joint=joint.name,
                          continuous_torque_nm=float(continuous[i]),
                          peak_torque_nm=float(peak[i]),
                          max_speed_rad_s=joint_speed,
                          load_inertia_kg_m2=float(inertia[i, i]),
                          max_backlash_arcmin=max_backlash)
        best, feasible = select_drivetrain(req, safety_factor=safety_factor)

        console.print(f"\n[bold cyan]{joint.name}[/bold cyan]  required: "
                      f"continuous {req.continuous_torque_nm:.3f} N m, peak "
                      f"{req.peak_torque_nm:.3f} N m, speed "
                      f"{req.max_speed_rad_s:.2f} rad/s, load inertia "
                      f"{req.load_inertia_kg_m2:.4e} kg m^2")

        if best is None:
            candidates = [evaluate_candidate(m, g, req, safety_factor)
                          for m in all_motors() for g in all_gearboxes()]
            console.print("[red]INFEASIBLE[/red]")
            console.print(infeasibility_report(req, candidates))
            continue

        table = Table(title=f"{best.motor.id} + {best.gearbox.id} "
                            f"(ratio {best.gearbox.ratio:.0f}, "
                            f"{best.total_mass_kg:.2f} kg)")
        table.add_column("check", style="bold")
        table.add_column("required", justify="right")
        table.add_column("available", justify="right")
        table.add_column("unit")
        table.add_column("margin", justify="right")
        table.add_column("status", justify="center")
        for check in best.checks:
            table.add_row(check.name, f"{check.required:.4g}",
                          f"{check.available:.4g}", check.unit,
                          f"{check.margin:.2f}x",
                          "[green]PASS[/green]" if check.passes
                          else "[red]FAIL[/red]")
        table.add_row("inertia ratio (load/rotor)", "-",
                      f"{best.inertia_ratio:.2f}", "-", "-",
                      "[green]OK[/green]" if best.inertia_ratio < 10
                      else "[yellow]HIGH[/yellow]")
        console.print(table)
        console.print(f"  limited by [bold]{best.limiting_check.name}[/bold] "
                      f"at {best.limiting_check.margin:.2f}x")

        alternatives = compare_alternatives(feasible, count=3)
        if len(alternatives) > 1:
            alt_table = Table(title="alternatives considered")
            alt_table.add_column("option", style="bold")
            alt_table.add_column("mass (kg)", justify="right")
            alt_table.add_column("backlash", justify="right")
            alt_table.add_column("why")
            for candidate, reason in alternatives:
                alt_table.add_row(
                    f"{candidate.motor.id} + {candidate.gearbox.id}",
                    f"{candidate.total_mass_kg:.2f}",
                    f"{candidate.gearbox.backlash_arcmin:.0f}'", reason)
            console.print(alt_table)

    console.print(
        "\n[dim]Status: PASS subject to thermal validation. The thermal check "
        "here is a continuous-torque proxy; a real one needs the duty profile, "
        "ambient temperature and thermal resistance.[/dim]")
    console.print(
        "[dim]The motor and gearbox catalogues are ILLUSTRATIVE ARCHETYPES, not "
        "real parts, and carry no vendor part numbers on purpose. The selection "
        "logic is the deliverable. Replace the catalogue with datasheet values "
        "before ordering. Friction and joint compliance are still zero. This is "
        "first-pass screening, not a final component decision.[/dim]")


@app.command()
def topology(
    volume_fraction: float = typer.Option(0.4, "--volume-fraction"),
    nx: int = typer.Option(32, "--nx"), ny: int = typer.Option(10, "--ny"),
    nz: int = typer.Option(2, "--nz"),
    iterations: int = typer.Option(80, "--iterations"),
    no_filter: bool = typer.Option(False, "--no-filter",
                                   help="disable the density filter, to see checkerboarding"),
    stl: bool = typer.Option(False, "--stl", help="write the thresholded shape"),
    out_dir: str = typer.Option("runs/topology", "--out-dir"),
    stress: bool = typer.Option(False, "--stress",
                                help="stress-constrained run on the L-bracket, "
                                     "compared against pure compliance"),
    stress_limit_mpa: float = typer.Option(
        16.0, "--stress-limit-mpa",
        help="von Mises limit for --stress. The compliance design reaches "
             "about 18 MPa, so lower values bind and higher ones do not."),
    p_norm: float = typer.Option(12.0, "--p-norm",
                                 help="stress aggregation exponent"),
    projection: bool = typer.Option(
        False, "--projection",
        help="three-field SIMP: filter, Heaviside projection and beta "
             "continuation, which drives the design black and white"),
    beta_max: float = typer.Option(64.0, "--beta-max",
                                   help="final projection sharpness"),
):
    """Topology-optimize a cantilever domain with SIMP on the GPU FEM.

    With --stress the domain changes to the L-bracket benchmark, whose
    re-entrant corner carries a stress concentration that a compliance
    objective has no reason to care about.
    """
    if stress:
        _topology_stress(volume_fraction, nx, iterations, stress_limit_mpa,
                         p_norm, not no_filter)
        return
    if projection:
        _topology_projected(volume_fraction, nx, ny, nz, iterations, beta_max,
                            stl, out_dir)
        return

    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from optimization.topology import (
        SimpProblem, checkerboard_metric, export_stl, grey_fraction, optimize,
    )
    from physics.fem.mesh import solid_box_mesh

    console = Console()
    material = get_material("al_7075_t6")
    length, height, width, load = 0.16, 0.05, 0.02, 200.0
    mesh = solid_box_mesh(length, height, width, nx, ny, nz)
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        fixed_nodes=mesh.nodes_at_x(0.0), load_nodes=mesh.nodes_at_x(length),
        total_load_n=-load, load_direction=1,
        volume_fraction=volume_fraction, filter_radius_elements=1.5)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("domain", f"{length} x {height} x {width} m")
    head.add_row("mesh", f"{nx} x {ny} x {nz} = {mesh.n_elements} elements, "
                         f"{mesh.n_dofs} dofs")
    head.add_row("volume fraction", f"{volume_fraction}")
    head.add_row("penalty p", "3.0")
    head.add_row("filter", "off" if no_filter else "on, radius 1.5 elements")
    console.print(head)

    with console.status("optimizing"):
        result = optimize(problem, max_iterations=iterations,
                          use_filter=not no_filter)

    history = result.compliance_history
    table = Table(title="convergence")
    table.add_column("iteration", justify="right")
    table.add_column("compliance (J)", justify="right")
    table.add_column("volume", justify="right")
    table.add_column("max density change", justify="right")
    shown = sorted(set([0, 1, 2, 4, 9, 19, 39, len(history) - 1]))
    for i in shown:
        if 0 <= i < len(history):
            table.add_row(str(i + 1), f"{history[i]:.6e}",
                          f"{result.volume_history[i]:.4f}",
                          f"{result.change_history[i]:.5f}")
    console.print(table)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column()
    summary.add_row("iterations", f"{result.iterations} "
                                  f"(converged={result.converged})")
    summary.add_row("compliance", f"{history[0]:.6e} -> {history[-1]:.6e} J "
                                  f"({history[-1] / history[0]:.4f}x)")
    summary.add_row("final volume fraction", f"{result.volume_fraction:.6f}")
    summary.add_row("checkerboard metric",
                    f"{checkerboard_metric(mesh, result.density):.4f}")
    summary.add_row("grey fraction (0.1 to 0.9)",
                    f"{grey_fraction(result.density):.3f}")
    console.print(summary)

    grid = result.density.reshape(nx, ny, nz)
    profile = Table(title="density through the section height "
                          "(bending puts material at the extremes)")
    profile.add_column("row (bottom to top)", justify="right")
    profile.add_column("mean density", justify="right")
    profile.add_column("", justify="left")
    for j, value in enumerate(grid.mean(axis=(0, 2))):
        profile.add_row(str(j), f"{value:.3f}", "#" * int(round(value * 40)))
    console.print(profile)

    if stl:
        report = export_stl(mesh, result.density, Path(out_dir) / "topology.stl")
        console.print(
            f"wrote [bold]{report.path}[/bold] "
            f"({report.path.stat().st_size} bytes): "
            f"{report.retained_elements}/{report.total_elements} voxels "
            f"({report.retained_fraction:.1%}), volume "
            f"{report.volume_m3:.6e} m^3, watertight={report.watertight}")
        if not report.watertight:
            console.print(
                "[yellow]![/yellow] the surface is not watertight: voxels that "
                "meet only along an edge leave non-manifold edges. The volume "
                "above is exact from the voxel count, but closing those "
                "contacts needs smoothing, which changes the shape.")

    console.print(
        "\n[dim]SIMP leaves intermediate densities, and no material is 40% "
        "present, so the field must be thresholded and the thresholded shape "
        "differs from what was optimized. The output is a voxel model, blocky "
        "by construction, and an STL rather than a clean STEP: recovering "
        "analytic faces from a density field is surface reconstruction. This "
        "minimises COMPLIANCE, not stress, so it carries no stress constraint "
        "and says nothing about peak stress. A design concept, not a verified "
        "part: it still has to pass the 3D FEM gate and gain manufacturing "
        "features. Still SIMULATED.[/dim]")


def _topology_projected(volume_fraction: float, nx: int, ny: int, nz: int,
                        iterations: int, beta_max: float, stl: bool,
                        out_dir: str) -> None:
    """Compare plain SIMP against the three-field formulation.

    Same mesh, same volume, same load. The difference is whether the optimiser
    is allowed to settle on intermediate densities.
    """
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from optimization.topology import (SimpProblem, checkerboard_metric,
                                       connected_fraction, export_stl,
                                       grey_fraction, optimize)
    from optimization.topology.projection import BetaSchedule
    from optimization.topology.threefield import optimize_projected
    from physics.fem.mesh import solid_box_mesh

    console = Console()
    material = get_material("al_7075_t6")
    length, height, width, load = 0.16, 0.05, 0.02, 200.0
    mesh = solid_box_mesh(length, height, width, nx, ny, nz)
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        fixed_nodes=mesh.nodes_at_x(0.0), load_nodes=mesh.nodes_at_x(length),
        total_load_n=-load, load_direction=1,
        volume_fraction=volume_fraction, filter_radius_elements=1.5)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("mesh", f"{nx} x {ny} x {nz} = {mesh.n_elements} elements, "
                         f"{mesh.n_dofs} dofs")
    head.add_row("volume fraction", f"{volume_fraction}")
    head.add_row("projection", f"tanh Heaviside, eta 0.5, beta 1 to {beta_max:g}")
    console.print(head)

    with console.status("plain SIMP"):
        plain = optimize(problem, max_iterations=iterations)
    with console.status("three-field SIMP"):
        projected = optimize_projected(
            problem, max_iterations=iterations, move_limit=0.1,
            schedule=BetaSchedule(start=1.0, maximum=beta_max, every=20))

    table = Table(title="plain SIMP against the three-field formulation")
    table.add_column("formulation")
    table.add_column("compliance (J)", justify="right")
    table.add_column("volume", justify="right")
    table.add_column("grey fraction", justify="right")
    table.add_column("connected", justify="right")
    table.add_column("checkerboard", justify="right")
    for name, density, compliance in (
            ("plain", plain.density, plain.final_compliance),
            ("three-field", projected.density, projected.final_compliance)):
        table.add_row(name, f"{compliance:.6e}", f"{density.mean():.6f}",
                      f"{grey_fraction(density):.4f}",
                      f"{connected_fraction(mesh, density):.3f}",
                      f"{checkerboard_metric(mesh, density):.4f}")
    console.print(table)
    console.print(
        "read grey and connected together. The projection lowers the grey "
        "fraction, and at a small filter radius it also breaks the design into "
        "pieces; material that is not connected carries no load, so a blacker "
        "but fragmented design is worse, not better. A radius of about 2 "
        "elements holds it together here, 3 fully.")
    console.print(
        "the checkerboard metric RISES for a black and white design and that "
        "is expected: it measures the density difference between neighbours, "
        "which is large across any sharp solid-to-void boundary. It detects "
        "alternating artifacts on a grey field and cannot tell one from a "
        "genuine edge, so connectivity is the metric that matters here.")

    if stl:
        for name, density in (("plain", plain.density),
                              ("projected", projected.density)):
            report = export_stl(mesh, density,
                                Path(out_dir) / f"topology_{name}.stl")
            console.print(f"wrote [bold]{report.path}[/bold]: "
                          f"{report.retained_elements}/{report.total_elements} "
                          f"voxels, watertight={report.watertight}")


def _topology_stress(volume_fraction: float, n: int, iterations: int,
                     limit_mpa: float, p_norm: float, use_filter: bool) -> None:
    """Compare compliance minimisation against a stress-constrained run.

    Both at the same volume on the same mesh, because the comparison is only a
    comparison if the amount of material is equal.
    """
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from optimization.topology import SimpProblem, grey_fraction, optimize
    from optimization.topology.simp import compliance_and_sensitivity
    from optimization.topology.stress import (StressProblem, evaluate,
                                              optimize_constrained)
    from physics.fem.mesh import (l_bracket_mesh,
                                  realised_arm_thickness)

    console = Console()
    material = get_material("al_7075_t6")
    size, thickness, width, load = 0.10, 0.4, 0.01, 300.0
    # The arm is a whole number of cells, so not every grid can build the
    # requested thickness. Snapping is accepted here rather than refusing a
    # grid the user chose, but the bracket actually meshed is reported: a
    # silently different benchmark shape is what this exists to prevent.
    realised = realised_arm_thickness(size, thickness, n)
    mesh = l_bracket_mesh(size, thickness, width, n, nz=2, allow_snapping=True)
    if abs(realised - size * thickness) > 1e-9:
        console.print(
            f"[yellow]arm snapped to the grid: {to_mm(realised):.3f} mm "
            f"instead of the requested {to_mm(size * thickness):.3f} mm "
            f"(n={n}). Pick n so that n x {thickness} is a whole number to "
            f"get the exact shape.[/yellow]")
    top = mesh.nodes_where(np.abs(mesh.node_coords[:, 1] - size) < 1e-9)
    tip = mesh.nodes_where(np.abs(mesh.node_coords[:, 0] - size) < 1e-9)
    base = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio, fixed_nodes=top, load_nodes=tip,
        total_load_n=-load, load_direction=1, volume_fraction=volume_fraction,
        filter_radius_elements=1.5)
    problem = StressProblem(base=base, stress_limit_pa=limit_mpa * 1e6,
                            p_norm=p_norm)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("domain", f"L-bracket {size} m, arm {thickness:.0%}, "
                           f"width {width} m")
    head.add_row("mesh", f"{mesh.nx} x {mesh.ny} x {mesh.nz} = "
                         f"{mesh.n_elements} elements, {mesh.n_dofs} dofs")
    head.add_row("volume fraction", f"{volume_fraction}")
    head.add_row("stress limit", f"{limit_mpa} MPa von Mises")
    head.add_row("aggregation", f"P-norm, P = {p_norm:g}")
    console.print(head)

    with console.status("compliance minimisation"):
        free = optimize(base, max_iterations=iterations, use_filter=use_filter)
    with console.status("stress-constrained minimisation"):
        held = optimize_constrained(problem, max_iterations=iterations,
                                    use_filter=use_filter)

    if not held.found_feasible:
        console.print(
            f"[red]![/red] no iterate satisfied the {limit_mpa} MPa limit "
            f"(best p-norm {min(held.p_norm_history):.4f}). The limit may be "
            f"unreachable at this volume and mesh; nothing is reported as a "
            f"design.")
        return
    design = held.best_feasible_density

    centroids = mesh.element_centroids()
    cell = size / n
    arm = max(1, int(round(n * thickness)))
    corner = np.array([arm * cell, arm * cell])
    near = np.linalg.norm(centroids[:, :2] - corner, axis=1) < 2.5 * cell

    table = Table(title=f"both designs at volume {volume_fraction}")
    table.add_column("design")
    table.add_column("compliance (J)", justify="right")
    table.add_column("max stress (MPa)", justify="right")
    table.add_column("p-norm", justify="right")
    table.add_column("corner density", justify="right")
    table.add_column("grey", justify="right")
    for name, density in (("compliance-min", free.density),
                          ("stress-constrained", design)):
        ev = evaluate(problem, density)
        compliance, _, _ = compliance_and_sensitivity(base, density)
        verdict = "" if ev.p_norm <= 1.0 else " [red](violates)[/red]"
        table.add_row(name, f"{compliance:.4e}",
                      f"{ev.max_relaxed_stress_pa / 1e6:.2f}{verdict}",
                      f"{ev.p_norm:.4f}", f"{density[near].mean():.4f}",
                      f"{grey_fraction(density):.3f}")
    console.print(table)

    console.print(
        "the p-norm over-estimates the true maximum, so satisfying it "
        "satisfies the elementwise limit; the gap is what a finite P costs "
        "in conservatism")
    console.print(
        "[yellow]note[/yellow] the stress constraint lowers the peak and pulls "
        "material off the re-entrant corner, but it does not produce a clean "
        "fillet: the constrained design is greyer than the compliance one. "
        "Reading a manufacturable shape off it needs a projection scheme that "
        "is not implemented here.")



@app.command()
def methods(
    category: str = typer.Option(None, "--category",
                                 help="design_generation, analysis, "
                                      "optimization or selection"),
    geometry: str = typer.Option(None, "--geometry",
                                 help="prismatic_beam, voxel_domain, assembly"),
    slenderness: float = typer.Option(None, "--slenderness",
                                      help="L/h, the ratio that decides "
                                           "whether a beam model is valid"),
    stress_constraint: bool = typer.Option(False, "--stress-constraint"),
    stress_field: bool = typer.Option(False, "--stress-field",
                                      help="a full stress field is required"),
    gradients: bool = typer.Option(False, "--gradients"),
    show_excluded: bool = typer.Option(True, "--excluded/--no-excluded"),
):
    """List registered methods and which of them apply to a problem.

    Without any problem options this is a catalogue. Give it a problem and it
    becomes the routing decision, including which methods were ruled out and
    which declared condition ruled them out.
    """
    from rich.console import Console
    from rich.table import Table

    from core.registry import DEFAULT_REGISTRY, Category, ProblemContext

    console = Console()
    chosen = None
    if category is not None:
        try:
            chosen = Category(category)
        except ValueError:
            console.print(f"[red]unknown category {category!r}[/red]. Known: "
                          f"{', '.join(c.value for c in Category)}")
            raise typer.Exit(code=2)

    context = ProblemContext(
        geometry=geometry,
        representations=(geometry,) if geometry else None,
        slenderness=slenderness,
        has_stress_constraint=stress_constraint if stress_constraint else None,
        needs_stress_field=stress_field,
        needs_gradients=gradients if gradients else None)
    candidates = DEFAULT_REGISTRY.query(context, chosen)

    table = Table(title="applicable methods"
                        if geometry else "registered methods (no problem given)")
    table.add_column("name")
    table.add_column("category")
    table.add_column("fidelity")
    table.add_column("cost")
    table.add_column("implementation")
    listing = candidates.applicable if geometry else DEFAULT_REGISTRY.all()
    for method in listing:
        table.add_row(method.name, method.category.value,
                      method.fidelity.name.lower(), method.cost.name.lower(),
                      method.implementation)
    console.print(table)

    if geometry and show_excluded and candidates.excluded:
        ruled = Table(title="excluded, and the declared condition that "
                            "excluded them")
        ruled.add_column("name")
        ruled.add_column("failed condition")
        for exclusion in candidates.excluded:
            ruled.add_row(exclusion.method.name, "; ".join(exclusion.failed))
        console.print(ruled)

    if geometry and not candidates.applicable:
        console.print("[yellow]![/yellow] nothing applies to this problem. "
                      "That is a refusal, not a failure: running a method "
                      "outside its declared range is how Phase 7 shipped an "
                      "optimum that the 3D FEM gate then rejected.")


@app.command()
def nodes(
    geometry: str = typer.Option(None, "--geometry",
                                 help="prismatic_beam, voxel_domain, assembly, brep"),
    slenderness: float = typer.Option(None, "--slenderness"),
    stress_field: bool = typer.Option(False, "--stress-field"),
):
    """List the nodes and the capabilities they provide.

    A node that cannot serve its capabilities is listed anyway, with the reason.
    An unbuilt capability and a blocked one need different responses, and a
    roster that hides the blocked ones cannot tell them apart.
    """
    from rich.console import Console
    from rich.table import Table

    from core.registry import ProblemContext
    from nodes import build_roster

    console = Console()
    registry = build_roster()

    roster = Table(title="nodes")
    roster.add_column("node")
    roster.add_column("transport")
    roster.add_column("address")
    roster.add_column("status")
    for descriptor in registry.nodes():
        status = ("[green]available[/green]" if descriptor.available
                  else f"[yellow]{descriptor.unavailable_reason}[/yellow]")
        roster.add_row(descriptor.name, descriptor.transport.value,
                       descriptor.address or "(in process)", status)
    console.print(roster)

    context = ProblemContext(
        geometry=geometry,
        representations=(geometry,) if geometry else None,
        slenderness=slenderness, needs_stress_field=stress_field)
    candidates = registry.query(context)

    table = Table(title="capabilities"
                        + (f" applicable to this problem" if geometry else ""))
    table.add_column("capability")
    table.add_column("node")
    table.add_column("category")
    table.add_column("fidelity")
    table.add_column("cost")
    listing = candidates.applicable if geometry else registry.all()
    for capability in listing:
        table.add_row(capability.name, capability.node.name,
                      capability.method.category.value,
                      capability.method.fidelity.name.lower(),
                      capability.method.cost.name.lower())
    console.print(table)

    if geometry and candidates.excluded:
        ruled = Table(title="excluded, with every reason")
        ruled.add_column("capability")
        ruled.add_column("reasons")
        for exclusion in candidates.excluded:
            ruled.add_row(exclusion.capability.name, "; ".join(exclusion.failed))
        console.print(ruled)

    console.print(
        "external verification is not wired to a running node, so designs "
        "carry the status [bold]self_fem_only[/bold]: checked by this "
        "project's own FEM and by nothing else")


def _mean_stress_note(fatigue) -> str:
    """Say what happened to the mean stress, accurately for each sign.

    A fully reversed cycle has a mean of exactly zero, which takes the same
    branch as a compressive mean but for a different reason: there is nothing
    to credit rather than something withheld.
    """
    if fatigue.mean_stress_charged:
        return ""
    if fatigue.mean_pa < 0.0:
        return " (compressive mean, not credited)"
    return " (fully reversed)"


@app.command(name="failure-modes")
def failure_modes(
    material_id: str = typer.Option("al_7075_t6", "--material"),
    width_mm: float = typer.Option(40.0, "--width-mm"),
    height_mm: float = typer.Option(40.0, "--height-mm"),
    thickness_mm: float = typer.Option(2.0, "--thickness-mm"),
    length_m: float = typer.Option(0.5, "--length-m"),
    load_max_n: float = typer.Option(0.0, "--load-max-n",
                                     help="peak transverse load"),
    load_min_n: float = typer.Option(0.0, "--load-min-n",
                                     help="lowest transverse load in the cycle; "
                                          "set equal to the max for a static duty"),
    compression_n: float = typer.Option(0.0, "--compression-n",
                                        help="axial compression, positive"),
    ends: str = typer.Option("fixed_free", "--ends",
                             help="fixed_free, pinned_pinned, fixed_pinned, "
                                  "fixed_fixed"),
    soderberg: bool = typer.Option(False, "--soderberg",
                                   help="charge the mean stress against yield "
                                        "instead of ultimate"),
):
    """Check a section against static, fatigue and buckling failure.

    A design that passes the yield check can still be governed by fatigue or by
    buckling, and this reports whichever is closest to failing.
    """
    from rich.console import Console
    from rich.table import Table

    from core.design_genome.section import HollowRectangleSection
    from core.materials import get_material
    from physics.buckling import EndCondition
    from physics.failure_modes import DutyCycle, check_design
    from physics.fatigue import MeanStressCriterion

    console = Console()
    try:
        condition = EndCondition(ends)
    except ValueError:
        console.print(f"[red]unknown end condition {ends!r}[/red]. Known: "
                      f"{', '.join(c.value for c in EndCondition)}")
        raise typer.Exit(code=2)

    material = get_material(material_id)
    section = HollowRectangleSection(outer_width_m=width_mm / 1000.0,
                                     outer_height_m=height_mm / 1000.0,
                                     wall_thickness_m=thickness_mm / 1000.0)
    if not section.is_valid():
        console.print(f"[red]invalid section[/red]: {section.validity_reason()}")
        raise typer.Exit(code=2)

    duty = DutyCycle(bending_load_max_n=load_max_n,
                     bending_load_min_n=load_min_n,
                     compressive_load_n=compression_n,
                     end_condition=condition)
    report = check_design(
        section, material, length_m, duty,
        MeanStressCriterion.SODERBERG if soderberg
        else MeanStressCriterion.GOODMAN)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("material", f"{material.name} ({material.status.value})")
    head.add_row("section", f"{width_mm} x {height_mm} x {thickness_mm} mm, "
                            f"length {length_m} m")
    head.add_row("duty", f"bending {load_min_n} to {load_max_n} N, "
                         f"compression {compression_n} N, ends {ends}")
    console.print(head)

    table = Table(title="failure modes")
    table.add_column("mode")
    table.add_column("safety factor", justify="right")
    table.add_column("basis")
    table.add_row("static bending", f"{report.static_safety_factor:.3f}",
                  f"peak {report.max_bending_stress_pa / 1e6:.1f} MPa against "
                  f"yield {material.yield_strength_pa / 1e6:.0f} MPa")
    if report.axial_safety_factor is not None:
        table.add_row("axial yield", f"{report.axial_safety_factor:.3f}",
                      f"{report.axial_stress_pa / 1e6:.1f} MPa against yield")
    if report.fatigue is not None:
        fatigue = report.fatigue
        table.add_row(f"fatigue ({fatigue.criterion.value})",
                      f"{fatigue.safety_factor:.3f}",
                      f"alternating {fatigue.alternating_pa / 1e6:.1f} MPa, "
                      f"mean {fatigue.mean_pa / 1e6:.1f} MPa"
                      + _mean_stress_note(fatigue))
    else:
        table.add_row("fatigue", "n/a", "the duty is not cyclic")
    if report.buckling is not None:
        buckling = report.buckling
        basis = (f"P_cr {buckling.critical_load_n:.0f} N, slenderness "
                 f"{buckling.slenderness:.1f} against transition "
                 f"{buckling.critical_slenderness:.1f}")
        table.add_row("buckling (euler)",
                      f"{buckling.safety_factor:.3f}"
                      + ("" if buckling.euler_valid else " [yellow]invalid[/yellow]"),
                      basis)
    else:
        table.add_row("buckling", "n/a", "no axial compression")
    console.print(table)

    verdict = ("[green]PASS[/green]" if report.passes else "[red]FAIL[/red]")
    console.print(f"{verdict}: {report.summary()}")

    if report.fatigue is not None:
        console.print(f"[yellow]note[/yellow] {report.fatigue.notes}")
        console.print(
            "stress-life is high-cycle only, and carries no notch, surface, "
            "size or temperature factor. Each of those lowers a real endurance "
            "limit, so this number is optimistic.")
    if report.buckling is not None and not report.buckling.euler_valid:
        console.print(f"[yellow]note[/yellow] {report.buckling.notes}")
    elif report.buckling is not None:
        console.print(
            "[yellow]note[/yellow] the Euler load assumes a perfectly straight, "
            "centrally loaded column. Crookedness, eccentricity and residual "
            "stress all lower the real collapse load and no knock-down factor "
            "is applied.")


@app.command()
def loadpath(
    torque_nm: float = typer.Option(25.0, "--torque-nm",
                                    help="continuous joint torque required"),
    peak_torque_nm: float = typer.Option(60.0, "--peak-torque-nm"),
    speed_rad_s: float = typer.Option(3.0, "--speed-rad-s"),
    radial_load_n: float = typer.Option(400.0, "--radial-load-n",
                                        help="transverse load at the overhang"),
    span_mm: float = typer.Option(80.0, "--span-mm",
                                  help="bearing to bearing"),
    overhang_mm: float = typer.Option(50.0, "--overhang-mm",
                                      help="near bearing to the load"),
    shaft_material: str = typer.Option("steel_scm440", "--shaft-material"),
    target_safety: float = typer.Option(2.0, "--target-safety"),
    required_hours: float = typer.Option(20000.0, "--required-hours"),
):
    """Carry a selected drivetrain's torque through to its shaft and bearings.

    Phase 12 picks a motor and gearbox. The torque does not stop there: it goes
    out through a shaft held by bearings, and this shows what those carry.
    """
    import math

    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from drivetrain.bearings import all_bearings, rate_bearing
    from drivetrain.loadpath import ShaftLayout, trace
    from drivetrain.selection.select import Requirement, select_drivetrain
    from physics.shaft import (analyze_shaft, de_goodman_diameter_m,
                               first_critical_speed_rad_s)

    console = Console()
    requirement = Requirement(joint="joint", continuous_torque_nm=torque_nm,
                              peak_torque_nm=peak_torque_nm,
                              max_speed_rad_s=speed_rad_s)
    best, _ = select_drivetrain(requirement)
    if best is None:
        console.print("[red]no feasible drivetrain[/red] for that duty; "
                      "nothing to carry through to a shaft")
        raise typer.Exit(code=1)

    layout = ShaftLayout(bearing_span_m=span_mm / 1000.0,
                         overhang_m=overhang_mm / 1000.0,
                         radial_load_n=radial_load_n)
    path = trace(best, layout)
    material = get_material(shaft_material)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("drivetrain", f"{best.motor.name} + {best.gearbox.id}, "
                               f"ratio {best.gearbox.ratio:g}")
    head.add_row("output torque", f"{path.output_torque_nm:.2f} N m "
                                  f"(motor {best.motor.continuous_torque_nm:.2f} "
                                  f"x ratio x efficiency)")
    head.add_row("bending moment", f"{path.bending_moment_nm:.2f} N m "
                                   f"({radial_load_n:g} N at {overhang_mm:g} mm)")
    head.add_row("bearing loads", f"near {path.near_bearing_load_n:.1f} N, "
                                  f"far {path.far_bearing_load_n:.1f} N")
    head.add_row("shaft material", f"{material.name} ({material.status.value})")
    console.print(head)
    console.print(
        "the near bearing carries MORE than the applied load, not a share of "
        "it: an overhung load lifts the far bearing")

    loads = path.shaft_loads()
    sized = de_goodman_diameter_m(loads, material, target_safety)
    console.print(f"\nDE-Goodman diameter for n = {target_safety:g}: "
                  f"[bold]{sized * 1000:.2f} mm[/bold]")

    table = Table(title="shaft diameters")
    table.add_column("diameter (mm)", justify="right")
    table.add_column("static SF", justify="right")
    table.add_column("fatigue SF", justify="right")
    table.add_column("governs")
    table.add_column("verdict")
    for diameter in sorted({round(sized * 1000) - 2, round(sized * 1000),
                            round(sized * 1000) + 4}):
        if diameter <= 0:
            continue
        result = analyze_shaft(loads, material, diameter / 1000.0)
        table.add_row(f"{diameter:.0f}",
                      f"{result.static_safety_factor:.2f}",
                      f"{result.fatigue_safety_factor:.3f}",
                      result.governing_mode,
                      "[green]pass[/green]" if result.passes
                      else "[red]fail[/red]")
    console.print(table)
    console.print(
        "fatigue governs a rotating shaft at every diameter, because a steady "
        "transverse load is fully reversed in the material once per "
        "revolution. Sizing on the static stress alone is wrong at every size, "
        "not just near the limit.")

    critical = first_critical_speed_rad_s(material, sized,
                                          (span_mm + overhang_mm) / 1000.0)
    console.print(f"first critical speed of a bare shaft that size: "
                  f"{critical:.0f} rad/s ({critical * 60 / (2 * math.pi):.0f} "
                  f"rpm) against an operating {path.speed_rad_s:g} rad/s. "
                  f"[yellow]This ignores the gear or pulley the shaft carries, "
                  f"which lowers it substantially.[/yellow]")

    bearings = Table(title=f"bearings at the near support, "
                           f"{path.near_bearing_load_n:.0f} N, required "
                           f"{required_hours:g} h")
    bearings.add_column("designation")
    bearings.add_column("C (kN)", justify="right")
    bearings.add_column("L10 (h)", justify="right")
    bearings.add_column("margin", justify="right")
    bearings.add_column("static SF", justify="right")
    bearings.add_column("verdict")
    for bearing in all_bearings():
        if bearing.bore_m > sized * 1.6:
            continue
        try:
            result = rate_bearing(bearing, path.near_bearing_load_n,
                                  path.speed_rad_s,
                                  required_hours=required_hours)
        except ValueError:
            continue
        bearings.add_row(
            bearing.designation, f"{bearing.dynamic_rating_n / 1000:.2f}",
            f"{result.l10_hours:,.0f}", f"{result.life_margin:.2f}",
            f"{result.static_safety_factor:.1f}",
            "[green]pass[/green]" if result.passes else "[red]fail[/red]")
    console.print(bearings)
    console.print(
        "[yellow]note[/yellow] L10 is a statistic: one bearing in ten is "
        "expected to fail before it, and a specific bearing has no promised "
        "life. The ISO 281 reliability, lubrication, contamination and "
        "temperature factors are not applied and each can move this by more "
        "than an order of magnitude.")
    console.print(
        "[yellow]note[/yellow] the bearing ratings are representative values "
        "for the size class, tagged illustrative. The designations are ISO "
        "boundary dimensions, not vendor part numbers.")


@app.command()
def thermal(
    motor_id: str = typer.Option("bldc_100w", "--motor"),
    torque_nm: float = typer.Option(0.60, "--torque-nm",
                                    help="torque during the active part"),
    speed_rad_s: float = typer.Option(300.0, "--speed-rad-s"),
    duty_fraction: float = typer.Option(0.5, "--duty-fraction",
                                        help="share of the cycle spent active"),
    ambient_c: float = typer.Option(40.0, "--ambient-c"),
    material_id: str = typer.Option("al_7075_t6", "--material"),
    delta_t_k: float = typer.Option(60.0, "--delta-t-k",
                                    help="temperature change, signed"),
    constraint: float = typer.Option(1.0, "--constraint",
                                     help="0 free, 1 fully restrained"),
    mechanical_mpa: float = typer.Option(0.0, "--mechanical-mpa",
                                         help="mechanical stress, signed"),
):
    """Motor winding temperature under a duty, and restrained thermal stress."""
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from drivetrain.motors.catalog import get_motor
    from physics.thermal import (DutySegment, check_motor_thermal,
                                 check_thermal_stress, rms_torque_nm,
                                 stress_per_kelvin_pa)

    console = Console()
    if not 0.0 < duty_fraction <= 1.0:
        console.print("[red]--duty-fraction must be in (0, 1][/red]")
        raise typer.Exit(code=2)

    motor = get_motor(motor_id)
    duty = [DutySegment(torque_nm, speed_rad_s, duty_fraction)]
    if duty_fraction < 1.0:
        duty.append(DutySegment(0.0, 0.0, 1.0 - duty_fraction))
    result = check_motor_thermal(motor, duty, ambient_c=ambient_c)
    mean_torque = torque_nm * duty_fraction

    table = Table(title=f"{motor.name}, {duty_fraction:.0%} of the cycle at "
                        f"{torque_nm} N m")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_column("note")
    table.add_row("mean torque", f"{mean_torque:.4f} N m",
                  f"against the {motor.continuous_torque_nm} N m rating: "
                  + ("within" if mean_torque <= motor.continuous_torque_nm
                     else "over"))
    table.add_row("RMS torque", f"{result.rms_torque_nm:.4f} N m",
                  "what actually heats it, since copper loss goes as torque "
                  "squared")
    table.add_row("copper loss", f"{result.losses.copper_w:.2f} W", "k T_rms^2")
    table.add_row("iron loss", f"{result.losses.iron_w:.2f} W",
                  "k omega, linear stand-in for core loss")
    table.add_row("temperature rise", f"{result.temperature_rise_k:.1f} K",
                  f"P_loss x R_th ({motor.thermal_resistance_k_w} K/W)")
    table.add_row("winding", f"{result.winding_c:.1f} C",
                  f"ambient {ambient_c:.0f} C plus the rise")
    table.add_row("limit", f"{result.limit_c:.0f} C",
                  f"insulation class {motor.insulation_class.value}")
    table.add_row("margin", f"{result.margin_k:.1f} K",
                  "[green]pass[/green]" if result.passes
                  else "[red]fail[/red]")
    console.print(table)
    if mean_torque <= motor.continuous_torque_nm and not result.passes:
        console.print(
            "[yellow]![/yellow] the continuous-torque proxy accepts this duty "
            "and the thermal model rejects it. A proxy sees only the average, "
            "not the shape of the cycle.")
    console.print(
        "[yellow]note[/yellow] steady state only. A brief overload this rejects "
        "may be harmless, and a long one it accepts on average may not be. "
        "The thermal resistance is one lumped number and the mounting "
        "dominates it, so it can be out by a factor of two either way.")

    material = get_material(material_id)
    if material.thermal_expansion_1_k is None:
        console.print(f"[red]{material_id} has no expansion coefficient[/red]")
        raise typer.Exit(code=2)
    stress = check_thermal_stress(material, delta_t_k=delta_t_k,
                                  mechanical_stress_pa=mechanical_mpa * 1e6,
                                  constraint=constraint)

    second = Table(title=f"{material.name}, dT = {delta_t_k:+g} K, constraint "
                         f"{constraint:g}")
    second.add_column("quantity")
    second.add_column("value", justify="right")
    second.add_column("note")
    second.add_row("expansion coefficient", f"{stress.alpha_1_k:.3e} 1/K",
                   f"{material.status.value}, room temperature")
    second.add_row("stress per kelvin",
                   f"{stress_per_kelvin_pa(material.youngs_modulus_pa, stress.alpha_1_k) / 1e6:.4f} MPa/K",
                   "E alpha, at full restraint")
    second.add_row("thermal stress",
                   f"{stress.thermal_stress_pa / 1e6:+.2f} MPa",
                   "compression when heated, tension when cooled")
    second.add_row("mechanical stress", f"{mechanical_mpa:+.2f} MPa", "as given")
    second.add_row("combined", f"{stress.combined_stress_pa / 1e6:+.2f} MPa",
                   f"superposed with signs; {stress.governing_contribution} "
                   f"dominates")
    second.add_row("safety factor", f"{stress.safety_factor:.2f}",
                   "[green]pass[/green]" if stress.passes
                   else "[red]fail[/red]")
    console.print(second)
    console.print(
        "[yellow]note[/yellow] a uniform temperature change on a uniformly "
        "restrained member. The constraint factor is your judgement and the "
        "answer is proportional to it. Real parts have gradients, which "
        "produce stress even with no restraint at all.")


@app.command()
def bolt(
    size: str = typer.Option("M8", "--size", help="M3 to M12"),
    grade: str = typer.Option("8.8", "--grade", help="8.8, 10.9 or 12.9"),
    grip_mm: float = typer.Option(20.0, "--grip-mm"),
    load_n: float = typer.Option(8000.0, "--load-n",
                                 help="maximum external tensile load"),
    load_min_n: float = typer.Option(0.0, "--load-min-n",
                                     help="minimum over the cycle; equal to "
                                          "the max for a static load"),
    preload_fraction: float = typer.Option(0.75, "--preload-fraction"),
):
    """Check a preloaded bolted joint, and show what preload buys.

    A bolted joint is a preload problem. A properly tightened bolt sees only a
    small share of an external load; an under-tightened one lets the joint
    separate and then carries all of it.
    """
    from rich.console import Console
    from rich.table import Table

    from physics.joints import PropertyClass, analyze_joint

    console = Console()
    try:
        property_class = PropertyClass(grade)
    except ValueError:
        console.print(f"[red]unknown grade {grade!r}[/red]. Known: "
                      f"{', '.join(c.value for c in PropertyClass)}")
        raise typer.Exit(code=2)

    sweep = Table(title=f"{size} class {grade}, {grip_mm:g} mm grip, "
                        f"{load_n:g} N external")
    sweep.add_column("preload", justify="right")
    sweep.add_column("F_i (N)", justify="right")
    sweep.add_column("separates at (N)", justify="right")
    sweep.add_column("bolt load (N)", justify="right")
    sweep.add_column("fatigue", justify="right")
    sweep.add_column("governs")
    sweep.add_column("verdict")

    fractions = sorted({0.90, 0.75, 0.50, 0.30, 0.15, preload_fraction},
                       reverse=True)
    for fraction in fractions:
        try:
            result = analyze_joint(size, property_class, grip_mm / 1000.0,
                                   load_n, load_min_n,
                                   preload_fraction=fraction)
        except KeyError as unknown:
            console.print(f"[red]{unknown}[/red]")
            raise typer.Exit(code=2)
        fatigue = ("n/a" if result.fatigue_safety_factor is None
                   else f"{result.fatigue_safety_factor:.2f}")
        mark = " <-" if abs(fraction - preload_fraction) < 1e-9 else ""
        sweep.add_row(
            f"{fraction:.0%}{mark}", f"{result.preload_n:,.0f}",
            f"{result.separation_load_n:,.0f}"
            + (" [red]separated[/red]" if result.separated else ""),
            f"{result.bolt_load_n:,.0f}", fatigue, result.governing_mode,
            "[green]pass[/green]" if result.passes else "[red]fail[/red]")
    console.print(sweep)

    chosen = analyze_joint(size, property_class, grip_mm / 1000.0, load_n,
                           load_min_n, preload_fraction=preload_fraction)
    console.print(
        f"load factor C = {chosen.load_factor:.3f}: while the joint stays "
        f"clamped the bolt feels {chosen.load_factor:.0%} of an external load "
        f"and the rest merely relieves compression in the members. Past "
        f"separation the bolt carries all of it.")
    console.print(
        f"tightening torque {chosen.tightening_torque_nm:.1f} N m at a nut "
        f"factor of 0.2. [yellow]That factor scatters by about 30 percent with "
        f"lubrication and surface condition, so the achieved preload is "
        f"uncertain by the same amount even with a perfect wrench.[/yellow] "
        f"Angle control is better and is not modelled here.")


@app.command()
def gear(
    module_mm: float = typer.Option(2.0, "--module-mm"),
    pinion_teeth: int = typer.Option(20, "--pinion-teeth"),
    gear_teeth: int = typer.Option(60, "--gear-teeth"),
    face_mm: float = typer.Option(20.0, "--face-mm"),
    torque_nm: float = typer.Option(25.0, "--torque-nm",
                                    help="torque at the pinion"),
    bending_allowable_mpa: float = typer.Option(200.0, "--bending-mpa"),
    contact_allowable_mpa: float = typer.Option(700.0, "--contact-mpa"),
    bending_correction: float = typer.Option(
        1.0, "--bending-correction",
        help="stands for the AGMA factors this does not compute; 1.0 is "
             "optimistic"),
    contact_correction: float = typer.Option(1.0, "--contact-correction"),
):
    """Check a gear mesh against tooth bending and surface pitting.

    Two independent failure modes sized by different things. A fine-pitch gear
    breaks its teeth; a coarse-pitch one pits.
    """
    from rich.console import Console
    from rich.table import Table

    from physics.gears import GearMesh, analyze_mesh, lewis_form_factor

    console = Console()
    mesh = GearMesh(module_m=module_mm / 1000.0, pinion_teeth=pinion_teeth,
                    gear_teeth=gear_teeth, face_width_m=face_mm / 1000.0,
                    torque_nm=torque_nm)
    result = analyze_mesh(mesh, bending_allowable_mpa * 1e6,
                          contact_allowable_mpa * 1e6,
                          bending_correction=bending_correction,
                          contact_correction=contact_correction)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("mesh", f"module {module_mm:g} mm, {pinion_teeth}/{gear_teeth} "
                         f"teeth, ratio {mesh.ratio:.2f}")
    head.add_row("pinion pitch diameter",
                 f"{mesh.pinion_pitch_diameter_m * 1000:.1f} mm")
    head.add_row("face width", f"{face_mm:g} mm")
    head.add_row("tangential load", f"{result.tangential_load_n:.1f} N "
                                    f"(2T/d)")
    head.add_row("Lewis form factor", f"{lewis_form_factor(pinion_teeth):.3f}")
    console.print(head)

    table = Table(title="failure modes")
    table.add_column("mode")
    table.add_column("stress (MPa)", justify="right")
    table.add_column("allowable (MPa)", justify="right")
    table.add_column("safety factor", justify="right")
    table.add_row("tooth bending (Lewis)",
                  f"{result.bending_stress_pa / 1e6:.1f}",
                  f"{bending_allowable_mpa:g}",
                  f"{result.bending_safety_factor:.2f}")
    table.add_row("surface pitting (Hertz)",
                  f"{result.contact_stress_pa / 1e6:.1f}",
                  f"{contact_allowable_mpa:g}",
                  f"{result.contact_safety_factor:.2f}")
    console.print(table)

    verdict = "[green]PASS[/green]" if result.passes else "[red]FAIL[/red]"
    console.print(f"{verdict}: {result.governing_mode} governs at "
                  f"{result.governing_safety_factor:.2f}")
    console.print(
        "bending goes linearly with load and contact as its square root, so "
        "which mode governs can flip as a design is scaled. A fine-pitch gear "
        "is bending-critical and a coarse-pitch one pitting-critical.")
    if bending_correction == 1.0 and contact_correction == 1.0:
        console.print(
            "[yellow]note[/yellow] this is Lewis and elementary Hertz, not "
            "AGMA. The dynamic, load distribution, application and size "
            "factors are all at least 1.0 and none is applied here, so this "
            "result runs OPTIMISTIC against a real gear. Pass "
            "--bending-correction and --contact-correction to include them.")


@app.command()
def pareto(
    materials: str = typer.Option("steel_s45c,al_6061_t6,al_7075_t6,ti_6al_4v",
                                  "--materials",
                                  help="comma separated material ids"),
    population: int = typer.Option(32, "--population"),
    generations: int = typer.Option(30, "--generations"),
    seed: int = typer.Option(0, "--seed"),
    weight_mass: float = typer.Option(1.0, "--weight-mass"),
    weight_deflection: float = typer.Option(0.0, "--weight-deflection"),
    weight_stress: float = typer.Option(0.0, "--weight-stress"),
    weight_cost: float = typer.Option(1.0, "--weight-cost"),
):
    """Map the mass, stiffness, stress and cost trade-off across materials.

    Single-objective optimisation answers "what is the lightest design". This
    answers "what are the designs where nothing improves without something
    else getting worse", and only then applies a preference.
    """
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from optimization.multi_objective.objectives import (OBJECTIVE_NAMES,
                                                         merged_front,
                                                         sweep_materials)
    from projects.robotic_link.problem import build_mvp_problem

    console = Console()
    ids = [m.strip() for m in materials.split(",") if m.strip()]
    if len(ids) < 1:
        console.print("[red]give at least one material[/red]")
        raise typer.Exit(code=2)

    with console.status(f"searching {len(ids)} materials"):
        try:
            fronts = sweep_materials(build_mvp_problem(), ids,
                                     population=population,
                                     generations=generations, seed=seed)
        except (KeyError, ValueError) as problem:
            console.print(f"[red]{problem}[/red]")
            raise typer.Exit(code=2)
    designs, objectives, labels = merged_front(fronts)

    if objectives.shape[0] == 0:
        console.print("[yellow]no feasible design was found for any "
                      "material[/yellow]")
        raise typer.Exit(code=1)

    per_material = Table(title="approximated Pareto front, by material")
    per_material.add_column("material")
    per_material.add_column("own front", justify="right")
    per_material.add_column("on merged front", justify="right")
    per_material.add_column("lightest (kg)", justify="right")
    per_material.add_column("cheapest (USD)", justify="right")
    for front in fronts:
        survivors = sum(1 for label in labels if label == front.material_id)
        if len(front) == 0:
            per_material.add_row(front.material_id, "0", "0", "n/a", "n/a")
            continue
        per_material.add_row(
            front.material_id, str(len(front)), str(survivors),
            f"{front.objectives[:, 0].min():.4f}",
            f"{front.objectives[:, 3].min():.3f}")
    console.print(per_material)
    console.print(
        "a material with nothing on the merged front is not on this "
        "trade-off at all for this problem, which is a stronger statement "
        "than being worse on average")

    extremes = Table(title="who wins each objective outright")
    extremes.add_column("objective")
    extremes.add_column("best value", justify="right")
    extremes.add_column("material")
    for column, name in enumerate(OBJECTIVE_NAMES):
        best = int(np.argmin(objectives[:, column]))
        extremes.add_row(name, f"{objectives[best, column]:.6g}", labels[best])
    console.print(extremes)

    weights = np.array([weight_mass, weight_deflection, weight_stress,
                        weight_cost])
    low = objectives.min(axis=0)
    span = np.where(objectives.max(axis=0) - low > 0.0,
                    objectives.max(axis=0) - low, 1.0)
    scores = ((objectives - low) / span) @ weights
    chosen = int(np.argmin(scores))
    console.print(
        f"\nwith weights mass {weight_mass:g}, deflection "
        f"{weight_deflection:g}, stress {weight_stress:g}, cost "
        f"{weight_cost:g}, the preferred design is [bold]{labels[chosen]}"
        f"[/bold] at {objectives[chosen, 0]:.4f} kg and "
        f"{objectives[chosen, 3]:.3f} USD of material")
    console.print(
        "the weights are applied AFTER the front is known, which is the point "
        "of computing one: the trade-off is visible before anyone commits to "
        "a preference")
    console.print(
        "[yellow]note[/yellow] this front is a finite-population "
        "approximation, not the true front, and it is non-dominated only with "
        "respect to what was evaluated. Cost is RAW MATERIAL only: machining, "
        "finishing and assembly are excluded and usually dominate a small "
        "part, so the cheapest design here need not be the cheapest part.")


def _compact_layup(angles: list[float]) -> str:
    """The shorthand engineers actually write: [0]8, [0/90]2s and so on.

    A symmetric stack is written as its half with an s, and a repeated block as
    the block with a count, because a laminate is specified that way and
    spelling every ply out makes two layups hard to tell apart at a glance.
    """
    def render(sequence: list[float]) -> str:
        return "/".join(f"{a:g}" for a in sequence)

    half = len(angles) // 2
    if len(angles) % 2 == 0 and angles == angles[::-1]:
        body, suffix = angles[:half], "s"
    else:
        body, suffix = list(angles), ""

    for size in range(1, len(body) // 2 + 1):
        if len(body) % size:
            continue
        block = body[:size]
        if all(body[i:i + size] == block for i in range(0, len(body), size)):
            repeats = len(body) // size
            count = "" if repeats == 1 else str(repeats)
            return f"[{render(block)}]{count}{suffix}"
    return f"[{render(body)}]{suffix}"


@app.command()
def laminate(
    material_id: str = typer.Option("cfrp_ud", "--material"),
    ply_thickness_mm: float = typer.Option(0.125, "--ply-thickness-mm"),
    load_n_per_m: float = typer.Option(2.0e5, "--load-n-per-m",
                                       help="in-plane force resultant Nx"),
    layups: str = typer.Option(
        "0,0,0,0,0,0,0,0 | 0,90,0,90,90,0,90,0 | "
        "45,-45,45,-45,-45,45,-45,45 | 0,45,-45,90,90,-45,45,0",
        "--layups",
        help="stacking sequences separated by |, angles by comma"),
):
    """Compare laminate stacking sequences: stiffness, strength, failing ply.

    The same plies stacked differently are a different material. This shows how
    differently.
    """
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from core.materials import get_material
    from physics.composite import (LaminaStrength, Laminate, abd_matrices,
                                   first_ply_failure)

    console = Console()
    material = get_material(material_id)
    try:
        strength = LaminaStrength.from_material(material)
    except ValueError as missing:
        console.print(f"[red]{missing}[/red]")
        raise typer.Exit(code=2)

    thickness = ply_thickness_mm / 1000.0
    load = np.array([load_n_per_m, 0.0, 0.0])

    table = Table(title=f"{material.name}, {ply_thickness_mm:g} mm plies, "
                        f"Nx = {load_n_per_m:g} N/m")
    table.add_column("layup", no_wrap=True)
    table.add_column("Ex (GPa)", justify="right")
    table.add_column("Ey (GPa)", justify="right")
    table.add_column("sym", justify="center")
    table.add_column("strength ratio", justify="right")
    table.add_column("first ply")
    table.add_column("mode")

    for sequence in layups.split("|"):
        angles = [float(a) for a in sequence.strip().split(",") if a.strip()]
        if not angles:
            continue
        stack = Laminate.from_material(material, angles, thickness)
        abd = abd_matrices(stack)
        failure = first_ply_failure(stack, strength, load)
        ex = abd.a[0, 0] / stack.thickness_m / 1e9
        ey = abd.a[1, 1] / stack.thickness_m / 1e9
        table.add_row(
            _compact_layup(angles), f"{ex:.1f}", f"{ey:.1f}",
            "yes" if stack.is_symmetric() else "[yellow]no[/yellow]",
            (f"[green]{failure.strength_ratio:.3f}[/green]"
             if failure.passes else f"[red]{failure.strength_ratio:.3f}[/red]"),
            f"{failure.ply_index} at {failure.angle_deg:g} deg",
            failure.governing_mode.value)
    console.print(table)

    console.print(
        "the strength ratio is a LOAD MULTIPLIER to first-ply failure. It is "
        "not the reciprocal of the Tsai-Wu index, which has linear terms and "
        "so does not scale that way.")
    console.print(
        "[yellow]note[/yellow] this is FIRST-ply failure. A laminate usually "
        "carries more load after its first ply goes, so as an ultimate "
        "strength this is conservative. The exception is a fibre failure in "
        "the load direction, where first-ply really is the end, which is what "
        "the mode column is for.")
    console.print(
        "[yellow]note[/yellow] plane stress in every ply, so interlaminar "
        "stresses are invisible. Those are what delaminate real laminates at "
        "their free edges, and nothing here will warn about it. No progressive "
        "damage and no cure or moisture stresses.")


@app.command()
def design(
    payload_kg: float = typer.Option(5.0, "--payload-kg"),
    link_length_m: float = typer.Option(0.35, "--link-length-m"),
    speed_rad_s: float = typer.Option(3.0, "--speed-rad-s"),
    ambient_c: float = typer.Option(40.0, "--ambient-c"),
    bearing_life_h: float = typer.Option(20000.0, "--bearing-life-h"),
    shaft_diameter_mm: float = typer.Option(16.0, "--shaft-diameter-mm"),
    bolt_size: str = typer.Option("M6", "--bolt-size"),
    material: str = typer.Option("al_7075_t6", "--material"),
    step: bool = typer.Option(False, "--step",
                              help="write the structural link as STEP"),
    out_dir: str = typer.Option("runs/capstone", "--out-dir"),
):
    """Design a driven revolute joint end to end and verify it conjunctively.

    Every method the project has, applied to one assembly. The joint passes
    only if EVERY applicable check passes, and every failure mode with no
    applicable method is reported as unassessed rather than passed over.
    """
    from rich.console import Console
    from rich.table import Table

    from integration import CheckStatus, JointSpec, design_joint, review

    console = Console()
    spec = JointSpec(payload_kg=payload_kg, link_length_m=link_length_m,
                     max_speed_rad_s=speed_rad_s, ambient_c=ambient_c,
                     required_bearing_life_h=bearing_life_h,
                     shaft_diameter_m=shaft_diameter_mm / 1000.0,
                     mount_bolt_size=bolt_size, material_id=material)

    with console.status("designing and verifying"):
        result = design_joint(spec)
    report = review(result.verdict)

    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold cyan", justify="right")
    head.add_column()
    head.add_row("goal", f"revolute joint carrying {payload_kg:g} kg at "
                         f"{link_length_m:g} m, {speed_rad_s:g} rad/s")
    head.add_row("static torque", f"{spec.static_torque_nm:.2f} N m")
    head.add_row("link", f"{result.link_design.get('mass_kg', 0):.4f} kg of "
                         f"{material}")
    if result.drivetrain:
        head.add_row("drivetrain", f"{result.drivetrain['motor']} + "
                                   f"{result.drivetrain['gearbox']} "
                                   f"(ratio {result.drivetrain['ratio']:g})")
    if result.selected_bearing:
        head.add_row("bearing", result.selected_bearing)
    head.add_row("total mass", f"{result.total_mass_kg:.3f} kg "
                               f"(link plus drivetrain)")
    console.print(head)

    table = Table(title="every applicable check")
    table.add_column("component")
    table.add_column("failure mode")
    table.add_column("method")
    table.add_column("SF", justify="right")
    table.add_column("status")
    for check in result.verdict.results:
        if check.status is CheckStatus.NOT_ASSESSED:
            continue
        mark = {CheckStatus.PASSED: "[green]pass[/green]",
                CheckStatus.FAILED: "[red]FAIL[/red]",
                CheckStatus.NOT_APPLICABLE: "[dim]n/a[/dim]"}[check.status]
        table.add_row(check.component, check.failure_mode,
                      check.method or "",
                      "" if check.safety_factor is None
                      else f"{check.safety_factor:.3f}", mark)
    console.print(table)

    verdict_colour = {"passed": "green", "passed_with_gaps": "yellow",
                      "failed": "red"}[report.status.value]
    console.print(f"\n[bold {verdict_colour}]{report.headline}[/bold "
                  f"{verdict_colour}]")

    if report.governing is not None:
        console.print(f"\n[bold]governing constraint[/bold]: "
                      f"{report.governing.detail}")
        console.print(f"[bold]most load-bearing assumption[/bold]: "
                      f"{report.weakest_assumption}")

    if report.unassessed:
        gaps = Table(title=f"NOT ASSESSED: {len(report.unassessed)} failure "
                           f"modes with no applicable method")
        gaps.add_column("component")
        gaps.add_column("failure mode")
        for check in result.verdict.unassessed():
            gaps.add_row(check.component, check.failure_mode)
        console.print(gaps)
        console.print(
            "[yellow]none of these is known to be satisfied.[/yellow] They are "
            "listed because an unchecked mode nobody names reads exactly like "
            "a mode that passed")

    console.print("\n[bold]what to do next[/bold]")
    for line in report.recommendations:
        console.print(f"  - {line}")

    if step:
        from geometry.cad_export.hollow_rect import export_step

        vector = result.link_design.get("design_vector")
        if vector is None:
            console.print("[red]no link geometry to export[/red]")
        else:
            path = Path(out_dir) / f"{spec.name}_link.step"
            path.parent.mkdir(parents=True, exist_ok=True)
            written = export_step(link_length_m, vector[0], vector[1],
                                  vector[2], path)
            console.print(f"\nwrote [bold]{written.path}[/bold] "
                          f"({written.path.stat().st_size} bytes)")
            console.print(
                "[yellow]note[/yellow] this is the STRUCTURAL LINK only. The "
                "motor, gearbox, bearings and fasteners are selected catalogue "
                "items with no geometry in this model, and inventing solids "
                "for them would put shapes in a STEP file that nothing "
                "designed.")


@app.command()
def elements(
    shaft_mm: float = typer.Option(30.0, "--shaft-mm"),
    torque_nm: float = typer.Option(150.0, "--torque-nm"),
    key_length_mm: float = typer.Option(40.0, "--key-length-mm"),
    hub_outer_mm: float = typer.Option(60.0, "--hub-outer-mm"),
    interference_um: float = typer.Option(30.0, "--interference-um"),
    engagement_mm: float = typer.Option(40.0, "--engagement-mm"),
    weld_leg_mm: float = typer.Option(6.0, "--weld-leg-mm"),
    weld_length_mm: float = typer.Option(80.0, "--weld-length-mm"),
    weld_force_n: float = typer.Option(50000.0, "--weld-force-n"),
):
    """Shaft-to-hub connections, welds and ISO 286 fits at one diameter."""
    from rich.console import Console
    from rich.table import Table

    from physics.elements import (analyze_fillet_weld, analyze_key,
                                  analyze_press_fit, fit,
                                  standard_key_section)

    console = Console()
    shaft_m = shaft_mm / 1000.0
    steel_e, steel_nu, steel_yield = 207e9, 0.29, 655e6

    try:
        width, height = standard_key_section(shaft_m)
    except ValueError as refused:
        console.print(f"[red]{refused}[/red]")
        raise typer.Exit(code=2)
    key = analyze_key(shaft_m, key_length_mm / 1000.0, torque_nm, 90e6, 180e6)
    press = analyze_press_fit(interference_um * 1e-6, shaft_m,
                              hub_outer_mm / 1000.0, engagement_mm / 1000.0,
                              steel_e, steel_nu, steel_e, steel_nu,
                              hub_yield_pa=steel_yield)
    weld = analyze_fillet_weld(weld_force_n, weld_leg_mm / 1000.0,
                               weld_length_mm / 1000.0, 120e6)

    table = Table(title=f"{shaft_mm:g} mm shaft carrying {torque_nm:g} N m")
    table.add_column("element")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_column("note")
    table.add_row("key", f"{width * 1e3:g} x {height * 1e3:g} mm section",
                  f"SF {key.safety_factor:.2f}",
                  f"{key.governing_mode} governs"
                  + (f", length capped at {key.effective_length_m * 1e3:.0f} mm"
                     if key.length_was_capped else ""))
    table.add_row("press fit", "contact pressure",
                  f"{press.contact_pressure_pa / 1e6:.1f} MPa",
                  f"holds {press.torque_capacity_nm:.0f} N m at friction "
                  f"{press.friction_coefficient:g}")
    table.add_row("", "hub hoop stress",
                  f"{press.hub_hoop_stress_pa / 1e6:.1f} MPa",
                  ("[red]hub yields[/red]" if press.hub_yields
                   else f"hub SF {press.hub_yield_safety_factor:.2f}"))
    table.add_row("weld", f"{weld_leg_mm:g} mm fillet",
                  f"SF {weld.safety_factor:.2f}",
                  f"throat {weld.throat_m * 1e3:.2f} mm, "
                  f"{weld.stress_pa / 1e6:.0f} MPa")
    console.print(table)

    fits = Table(title="ISO 286 hole-basis fits at this diameter")
    fits.add_column("designation")
    fits.add_column("type")
    fits.add_column("clearance (um)", justify="right")
    for letter in ("g", "h", "k", "n"):
        try:
            chosen = fit(shaft_mm, 7, letter, 6)
        except ValueError:
            continue
        fits.add_row(chosen.designation, chosen.fit_type.value,
                     f"{chosen.min_clearance_mm * 1000:+.1f} to "
                     f"{chosen.max_clearance_mm * 1000:+.1f}")
    console.print(fits)
    console.print(
        "[yellow]note[/yellow] the tolerances are computed from the ISO 286 "
        "expression, which does NOT reproduce the published table exactly "
        "because the standard rounds to preferred numbers. Measured agreement "
        "is 1.2% mean and 8.4% worst above 3 mm. Good enough to compare fits, "
        "not to put on a drawing.")
    console.print(
        "[yellow]note[/yellow] the press fit deducts nothing for surface "
        "roughness, which always reduces the effective interference, and its "
        "friction coefficient is the weakest number in the chain with torque "
        "capacity directly proportional to it.")
    console.print(
        "[yellow]note[/yellow] the weld figure is STATIC. A welded joint's "
        "endurance strength is a fraction of the parent metal's, so a "
        "parent-metal fatigue check badly overstates it.")


@app.command()
def monitor(
    refresh_hz: int = typer.Option(2, "--refresh-hz",
                                   help="redraws per second"),
    seconds: float = typer.Option(None, "--seconds",
                                  help="stop after this long; default runs "
                                       "until interrupted"),
    once: bool = typer.Option(False, "--once",
                              help="print one snapshot and exit"),
    as_json: bool = typer.Option(False, "--json",
                                 help="print one snapshot as JSON and exit"),
):
    """Live CPU, memory and GPU readings.

    A value the driver will not report shows as "unavailable" rather than as
    zero: this machine's GPU reports power draw but not its power limit, and
    a zero there would look like a broken device instead of a missing field.
    """
    from rich.console import Console

    from monitoring.resources import live, render_panel, snapshot

    if as_json:
        import json

        print(json.dumps(snapshot().as_dict(), indent=2))
        return
    if once:
        Console().print(render_panel())
        return
    live(refresh_hz=refresh_hz, duration_s=seconds)


if __name__ == "__main__":
    app()
