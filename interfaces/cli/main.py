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


if __name__ == "__main__":
    app()
