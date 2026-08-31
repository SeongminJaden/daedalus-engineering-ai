"""OpenFOAM as an external CFD solver, for cross-validating the fluid work.

Everything in :mod:`physics.fluids` is a closed-form correlation. Nothing in
this project has ever solved the Navier-Stokes equations, so the fluid results
have had no independent check at all: a sign error in a friction factor would
have propagated silently. This node runs a second, independently written
solver on flows whose exact solutions are known and compares all three.

WHAT THIS IS NOT
================
OpenFOAM is a SIMULATION, not a wind tunnel. It carries its own assumptions
(incompressible, Newtonian, a laminar closure here) and its own discretisation
error. Agreement means two implementations of the same mathematics agree, not
that the mathematics describes a real fluid. Nothing here is physically
validated.

There is a sharper trap specific to this comparison, and it was found while
writing this module rather than assumed. :func:`physics.fluids.internal.
solve_pipe_flow` computes the laminar pressure drop as Darcy-Weisbach with
``f = 64/Re``. Substituting ``f = 64 nu / (V D)`` into
``dp = f (L/D) rho V^2 / 2`` gives ``dp = 32 mu L V / D^2``, which IS the
Hagen-Poiseuille equation. The project formula and the "analytical reference"
are one equation written two ways, so comparing them proves only that the
algebra was done correctly. Of the three sources in the three-way comparison
only OpenFOAM is independent, and this module says so rather than presenting a
tautology as a confirmation.

VALIDITY DOMAIN OF EACH BENCHMARK
=================================
Stated before implementing, per the standing discipline. Every case below is
steady, incompressible, Newtonian, isothermal, laminar and fully developed,
with gravity and entrance effects absent by construction. Streamwise
periodicity is imposed with cyclic boundaries, which makes "fully developed"
exact rather than approximate and removes the entrance length
(``L_e = 0.05 Re D``) from the problem entirely.

Plane Couette, ``u(y) = U y / h``
    Valid for infinite parallel plates with no streamwise pressure gradient.
    Plane Couette flow is linearly stable at every Reynolds number, yet
    experiment finds turbulence above roughly Re = 360 based on gap and wall
    speed. These cases are meshed one cell deep, so the three-dimensional
    disturbance that causes real transition cannot exist. The flow is laminar
    BY CONSTRUCTION, and agreement therefore says nothing whatever about
    whether a real flow at the same Reynolds number would stay laminar.

Plane Poiseuille, ``u(y) = (G / 2 nu) y (h - y)``
    Driven by a uniform body force G (kinematic, m/s^2). Requires
    ``Re = V_mean (2h) / nu`` below about 2300. The discretisation error is
    predictable in closed form: the wall face gradient of a parabola is
    approximated one-sidedly, which shifts the whole profile by exactly
    ``dy^2 |u''| / 8`` with no change of shape. That is a testable prediction,
    and :mod:`tests.test_openfoam` tests it rather than accepting whatever
    number comes out.

Hagen-Poiseuille in a round pipe, ``u(r) = (G / 4 nu) (R^2 - r^2)``
    Mean velocity ``G R^2 / (8 nu)``, pressure drop ``32 mu L V / D^2``.
    Requires ``Re = V D / nu`` below about 2300. Solved on an axisymmetric
    wedge, which introduces a geometric error that is NOT reduced by radial
    refinement: the wedge's outer face is a flat chord, so the wall sits at
    ``R cos(theta/2)`` at mid azimuth instead of R. Poiseuille velocity scales
    as R^2, so the mean velocity is low by about ``theta^2 / 4`` in radians.
    At the OpenFOAM tutorials' customary 5 degrees that is 0.19 percent and it
    survives every level of mesh refinement. This module therefore defaults to
    1 degree, where the term falls to 0.008 percent.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

OPENFOAM_NODE_NAME = "openfoam.local"
OPENFOAM_CAPABILITY = "analysis.cfd.openfoam"

#: Bessel zero squared, the decay rate of the slowest transient in a pipe.
_PIPE_DECAY = 5.7831859629
#: Decay rate of the slowest transient between parallel plates.
_CHANNEL_DECAY = math.pi ** 2
#: exp(-25) is far below the discretisation error, so the run is steady.
_DECAY_DECADES = 25.0

_SEARCH_PREFIXES = (
    Path.home() / ".local/opt/openfoam/usr",
    Path("/usr"),
    Path("/opt/openfoam/usr"),
)


def _prefix() -> Path | None:
    """The install prefix holding bin/blockMesh, or None."""
    for prefix in _SEARCH_PREFIXES:
        if (prefix / "bin" / "blockMesh").exists():
            return prefix
    return None


def executable(name: str = "blockMesh") -> str | None:
    """A named OpenFOAM binary, or None when OpenFOAM is not installed."""
    prefix = _prefix()
    if prefix is not None and (prefix / "bin" / name).exists():
        return str(prefix / "bin" / name)
    return shutil.which(name)


def is_available() -> bool:
    return executable("blockMesh") is not None and executable("pisoFoam") is not None


def _environment() -> dict[str, str]:
    """The environment OpenFOAM needs to find its own etc directory.

    The distribution packages are extracted into a user prefix rather than
    installed, so nothing set these globally. Resolving them here keeps the
    node working without a shell wrapper.
    """
    prefix = _prefix()
    env = dict(os.environ)
    if prefix is None:
        return env
    project = prefix / "share" / "openfoam"
    env["WM_PROJECT_DIR"] = str(project)
    env["FOAM_ETC"] = str(project / "etc")
    libs = [str(prefix / "lib"), str(prefix / "lib" / "openmpi-system"),
            str(prefix / "lib" / "dummy")]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(libs + ([existing] if existing else []))
    return env


def version() -> str | None:
    """The reported version, or None when OpenFOAM is absent.

    This Debian build leaves the banner's Version field blank and puts the
    real number in etc/bashrc, so that is read directly. The patch level comes
    from the binary itself, which keeps the two from drifting apart silently.
    """
    binary = executable("blockMesh")
    if binary is None:
        return None
    name = "OpenFOAM"
    prefix = _prefix()
    if prefix is not None:
        bashrc = prefix / "share" / "openfoam" / "etc" / "bashrc"
        if bashrc.exists():
            match = re.search(r"WM_PROJECT_VERSION=(\S+)", bashrc.read_text())
            if match:
                name = f"OpenFOAM {match.group(1)}"
    result = subprocess.run([binary, "-help"], capture_output=True, text=True,
                            timeout=60, env=_environment())
    patch = re.search(r"patch=(\d+)", result.stdout + result.stderr)
    return f"{name} patch {patch.group(1)}" if patch else name


def openfoam_descriptor(available: bool | None = None) -> NodeDescriptor:
    """The node as the registry sees it, read from the filesystem."""
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=OPENFOAM_NODE_NAME, transport=Transport.STDIO,
        address=executable("pisoFoam") or "pisoFoam",
        available=present,
        unavailable_reason="" if present else
        "unavailable: the OpenFOAM binaries were not found")


def openfoam_capability_method():
    """The capability declaration, in the registry's schema."""
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=OPENFOAM_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Laminar incompressible CFD in OpenFOAM, for cross-validating "
                "this project's closed-form fluid correlations.",
        inputs=("geometry", "fluid", "driving_condition"),
        outputs=("velocity_profile", "mean_velocity", "pressure_gradient"),
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("fluid is carried through a conduit",
                      lambda c: c.require("has_internal_flow")),
            Condition("the flow is laminar, since no turbulence model is "
                      "configured",
                      lambda c: c.require("flow_is_laminar")),
        ),
        implementation="nodes.openfoam.solve",
        evidence="SIMULATED",
        notes="A finite volume Navier-Stokes solver, not a wind tunnel. It "
              "carries its own discretisation error and its own laminar "
              "closure, so agreement means two implementations of the same "
              "mathematics agree. The project's own laminar pipe formula is "
              "Darcy-Weisbach with f = 64/Re, which is algebraically identical "
              "to Hagen-Poiseuille, so the closed form is NOT an independent "
              "check on it and only OpenFOAM is. Cases are meshed one cell "
              "deep or on an axisymmetric wedge and are therefore laminar by "
              "construction, which says nothing about whether a real flow "
              "would stay laminar.")


@dataclass(frozen=True)
class ChannelCase:
    """Flow between two parallel plates, one cell deep and streamwise cyclic.

    A non-zero ``wall_velocity_m_s`` drives Couette flow; a non-zero
    ``body_force_m_s2`` drives Poiseuille flow. Both may be set, in which case
    the exact solution is their superposition, because the Navier-Stokes
    equations are linear for unidirectional fully developed flow.
    """

    gap_m: float
    kinematic_viscosity_m2_s: float
    cells: int = 20
    wall_velocity_m_s: float = 0.0
    body_force_m_s2: float = 0.0
    length_m: float = 0.1

    def analytical_velocity(self, y_m: float) -> float:
        """The exact profile, valid only inside the domain stated above."""
        h, nu = self.gap_m, self.kinematic_viscosity_m2_s
        couette = self.wall_velocity_m_s * y_m / h
        poiseuille = self.body_force_m_s2 / (2.0 * nu) * y_m * (h - y_m)
        return couette + poiseuille

    def analytical_mean_velocity(self) -> float:
        h, nu = self.gap_m, self.kinematic_viscosity_m2_s
        return (0.5 * self.wall_velocity_m_s
                + self.body_force_m_s2 * h * h / (12.0 * nu))

    def reynolds_number(self) -> float:
        """Based on the hydraulic diameter of a plane channel, which is 2h."""
        return (abs(self.analytical_mean_velocity()) * 2.0 * self.gap_m
                / self.kinematic_viscosity_m2_s)

    def wall_offset_prediction(self) -> float:
        """The predicted uniform shift of the discrete Poiseuille profile.

        The wall face gradient of a parabola is taken one-sidedly, which is
        wrong by a constant. The whole profile moves by dy^2 |u''| / 8 without
        changing shape, and the shape error stays at round-off.
        """
        dy = self.gap_m / self.cells
        curvature = abs(self.body_force_m_s2 / self.kinematic_viscosity_m2_s)
        return dy * dy * curvature / 8.0


@dataclass(frozen=True)
class PipeCase:
    """Fully developed laminar flow in a round pipe, on an axisymmetric wedge.

    The wedge's outer face is a flat chord rather than an arc, so the wall
    sits at ``R cos(theta/2)`` at mid azimuth. See the module docstring: this
    costs about ``theta^2 / 4`` in mean velocity and radial refinement does
    not remove it.
    """

    radius_m: float
    kinematic_viscosity_m2_s: float
    body_force_m_s2: float
    radial_cells: int = 40
    length_m: float = 0.05
    wedge_angle_deg: float = 1.0

    def analytical_velocity(self, r_m: float) -> float:
        return (self.body_force_m_s2 / (4.0 * self.kinematic_viscosity_m2_s)
                * (self.radius_m ** 2 - r_m ** 2))

    def analytical_mean_velocity(self) -> float:
        return (self.body_force_m_s2 * self.radius_m ** 2
                / (8.0 * self.kinematic_viscosity_m2_s))

    def reynolds_number(self) -> float:
        return (self.analytical_mean_velocity() * 2.0 * self.radius_m
                / self.kinematic_viscosity_m2_s)

    def wedge_deficit_prediction(self) -> float:
        """Relative shortfall in mean velocity from the flat outer face."""
        theta = math.radians(self.wedge_angle_deg)
        return 1.0 - math.cos(theta / 2.0) ** 2

    def pressure_drop_pa(self, density_kg_m3: float) -> float:
        """Hagen-Poiseuille drop over the case length, from the body force.

        The kinematic body force IS the pressure gradient divided by density,
        so this is exact for the driven problem rather than a correlation.
        """
        return density_kg_m3 * self.body_force_m_s2 * self.length_m


@dataclass(frozen=True)
class FlowResult:
    """What the external solver returned, with nothing inferred."""

    solver: str
    solver_version: str
    coordinates_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    mean_velocity_m_s: float

    def max_velocity_m_s(self) -> float:
        return max(self.velocity_m_s)


_STEPS = 3000
_HEADER = ("FoamFile {{ version 2.0; format ascii; class {cls}; "
           "object {obj}; }}\n")


def _write(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER.format(cls=cls, obj=obj) + body)


def _write_common(case_dir: Path, end_time: float, nu: float,
                  body_force: float) -> None:
    """The dictionaries that do not depend on the geometry."""
    delta_t = end_time / _STEPS
    _write(case_dir / "system/controlDict", "dictionary", "controlDict",
           f"application pisoFoam; startFrom startTime; startTime 0;\n"
           f"stopAt endTime; endTime {end_time!r}; deltaT {delta_t!r};\n"
           f"writeControl timeStep; writeInterval {_STEPS}; purgeWrite 0;\n"
           "writeFormat ascii; writePrecision 12; writeCompression off;\n"
           "timeFormat general; timePrecision 8; runTimeModifiable false;\n")
    _write(case_dir / "system/fvSchemes", "dictionary", "fvSchemes",
           "ddtSchemes { default Euler; }\n"
           "gradSchemes { default Gauss linear; }\n"
           "divSchemes { default none; div(phi,U) Gauss linear;\n"
           "  div((nuEff*dev2(T(grad(U))))) Gauss linear; }\n"
           "laplacianSchemes { default Gauss linear orthogonal; }\n"
           "interpolationSchemes { default linear; }\n"
           "snGradSchemes { default orthogonal; }\n")
    _write(case_dir / "system/fvSolution", "dictionary", "fvSolution",
           "solvers {\n"
           "  p { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }\n"
           "  pFinal { solver PCG; preconditioner DIC; tolerance 1e-12; "
           "relTol 0; }\n"
           "  U { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; "
           "relTol 0; } }\n"
           "PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; "
           "pRefValue 0; }\n")
    _write(case_dir / "constant/transportProperties", "dictionary",
           "transportProperties", f"transportModel Newtonian;\nnu {nu!r};\n")
    _write(case_dir / "constant/turbulenceProperties", "dictionary",
           "turbulenceProperties", "simulationType laminar;\n")
    if body_force:
        _write(case_dir / "system/fvOptions", "dictionary", "fvOptions",
               "pressureGradient { type vectorSemiImplicitSource;\n"
               "  selectionMode all; volumeMode specific;\n"
               f"  injectionRateSuSp {{ U (({body_force!r} 0 0) 0); }} }}\n")


def _write_channel(case_dir: Path, case: ChannelCase) -> None:
    h, length, span = case.gap_m, case.length_m, case.gap_m
    verts = [(0, 0, 0), (length, 0, 0), (length, h, 0), (0, h, 0),
             (0, 0, span), (length, 0, span), (length, h, span), (0, h, span)]
    block = "\n".join(f"    ({x} {y} {z})" for x, y, z in verts)
    _write(case_dir / "system/blockMeshDict", "dictionary", "blockMeshDict",
           f"scale 1;\nvertices\n(\n{block}\n);\n"
           f"blocks ( hex (0 1 2 3 4 5 6 7) (1 {case.cells} 1) "
           "simpleGrading (1 1 1) );\nedges ();\nboundary\n(\n"
           "    movingWall { type wall; faces ((3 7 6 2)); }\n"
           "    fixedWall  { type wall; faces ((0 1 5 4)); }\n"
           "    leftRight  { type cyclic; neighbourPatch rightLeft; "
           "faces ((0 4 7 3)); }\n"
           "    rightLeft  { type cyclic; neighbourPatch leftRight; "
           "faces ((1 2 6 5)); }\n"
           "    frontBack  { type empty; faces ((0 3 2 1) (4 5 6 7)); }\n);\n")
    moving = (f"{{ type fixedValue; value uniform "
              f"({case.wall_velocity_m_s!r} 0 0); }}"
              if case.wall_velocity_m_s else "{ type noSlip; }")
    _write(case_dir / "0/U", "volVectorField", "U",
           "dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);\n"
           f"boundaryField {{ movingWall {moving}\n"
           "  fixedWall { type noSlip; } leftRight { type cyclic; }\n"
           "  rightLeft { type cyclic; } frontBack { type empty; } }\n")
    _write(case_dir / "0/p", "volScalarField", "p",
           "dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;\n"
           "boundaryField { movingWall { type zeroGradient; }\n"
           "  fixedWall { type zeroGradient; } leftRight { type cyclic; }\n"
           "  rightLeft { type cyclic; } frontBack { type empty; } }\n")


def _write_pipe(case_dir: Path, case: PipeCase) -> None:
    r, length = case.radius_m, case.length_m
    half = math.radians(case.wedge_angle_deg) / 2.0
    cy, cz = r * math.cos(half), r * math.sin(half)
    verts = [(0, 0, 0), (length, 0, 0), (0, cy, -cz), (length, cy, -cz),
             (0, 0, 0), (length, 0, 0), (0, cy, cz), (length, cy, cz)]
    block = "\n".join(f"    ({x!r} {y!r} {z!r})" for x, y, z in verts)
    _write(case_dir / "system/blockMeshDict", "dictionary", "blockMeshDict",
           f"scale 1;\nvertices\n(\n{block}\n);\n"
           f"blocks ( hex (0 1 3 2 4 5 7 6) (1 {case.radial_cells} 1) "
           "simpleGrading (1 1 1) );\nedges ();\nboundary\n(\n"
           "    wall   { type wall; faces ((2 3 7 6)); }\n"
           "    inlet  { type cyclic; neighbourPatch outlet; "
           "faces ((0 2 6 4)); }\n"
           "    outlet { type cyclic; neighbourPatch inlet; "
           "faces ((1 5 7 3)); }\n"
           "    wedgeBack  { type wedge; faces ((0 1 3 2)); }\n"
           "    wedgeFront { type wedge; faces ((4 6 7 5)); }\n);\n")
    _write(case_dir / "0/U", "volVectorField", "U",
           "dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);\n"
           "boundaryField { wall { type noSlip; } inlet { type cyclic; }\n"
           "  outlet { type cyclic; } wedgeBack { type wedge; }\n"
           "  wedgeFront { type wedge; } }\n")
    _write(case_dir / "0/p", "volScalarField", "p",
           "dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;\n"
           "boundaryField { wall { type zeroGradient; } inlet { type cyclic; }\n"
           "  outlet { type cyclic; } wedgeBack { type wedge; }\n"
           "  wedgeFront { type wedge; } }\n")


def _run(binary_name: str, case_dir: Path) -> None:
    binary = executable(binary_name)
    if binary is None:
        raise CapabilityUnavailable(
            OPENFOAM_CAPABILITY, OPENFOAM_NODE_NAME,
            f"the OpenFOAM binary {binary_name} was not found")
    result = subprocess.run([binary, "-case", str(case_dir)],
                            capture_output=True, text=True, timeout=1800,
                            env=_environment(), cwd=str(case_dir))
    output = result.stdout + result.stderr
    if result.returncode != 0 or "FOAM FATAL" in output:
        tail = "\n".join(output.strip().splitlines()[-12:])
        raise RuntimeError(f"{binary_name} failed in {case_dir}:\n{tail}")


def _latest_time(case_dir: Path) -> Path:
    """The last written time directory, which is the converged state."""
    times = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            value = float(entry.name)
        except ValueError:
            continue
        if value > 0.0:
            times.append((value, entry))
    if not times:
        raise RuntimeError(f"no time directory was written in {case_dir}")
    return max(times)[1]


def _parse_streamwise_velocity(path: Path, expected: int) -> tuple[float, ...]:
    """The x component of U in every cell, in mesh order.

    The boundaryField carries vectors of the same shape, so it is cut away
    before parsing rather than filtered afterwards.
    """
    text = path.read_text()
    interior = text.split("internalField", 1)[1].split("boundaryField", 1)[0]
    uniform = re.match(r"\s*uniform\s*\(([-\d.eE+]+)", interior)
    if uniform:
        return tuple([float(uniform.group(1))] * expected)
    values = [float(m.group(1)) for m in re.finditer(
        r"\(([-\d.eE+]+)\s+[-\d.eE+]+\s+[-\d.eE+]+\)", interior)]
    if len(values) != expected:
        raise RuntimeError(
            f"expected {expected} cell values in {path}, parsed {len(values)}")
    return tuple(values)


def _end_time(case: ChannelCase | PipeCase) -> float:
    """Long enough that the starting transient is far below round-off.

    Chosen from the physics rather than by trial: the slowest mode decays as
    exp(-lambda nu t / L^2), so the run length follows from lambda.
    """
    nu = case.kinematic_viscosity_m2_s
    if isinstance(case, PipeCase):
        return _DECAY_DECADES * case.radius_m ** 2 / (_PIPE_DECAY * nu)
    return _DECAY_DECADES * case.gap_m ** 2 / (_CHANNEL_DECAY * nu)


def solve(case: ChannelCase | PipeCase, work_dir: Path) -> FlowResult:
    """Mesh, run and parse one benchmark case.

    Raises CapabilityUnavailable when OpenFOAM is not installed, so a missing
    solver can never be mistaken for a passing comparison.
    """
    if not is_available():
        raise CapabilityUnavailable(
            OPENFOAM_CAPABILITY, OPENFOAM_NODE_NAME,
            "the OpenFOAM binaries were not found")
    case_dir = Path(work_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    end_time = _end_time(case)
    if isinstance(case, PipeCase):
        _write_common(case_dir, end_time, case.kinematic_viscosity_m2_s,
                      case.body_force_m_s2)
        _write_pipe(case_dir, case)
        cells = case.radial_cells
        step = case.radius_m / cells
    else:
        _write_common(case_dir, end_time, case.kinematic_viscosity_m2_s,
                      case.body_force_m_s2)
        _write_channel(case_dir, case)
        cells = case.cells
        step = case.gap_m / cells
    _run("blockMesh", case_dir)
    _run("pisoFoam", case_dir)
    velocity = _parse_streamwise_velocity(
        _latest_time(case_dir) / "U", cells)
    coordinates = tuple((i + 0.5) * step for i in range(cells))
    if isinstance(case, PipeCase):
        # Area weighted, because an annulus far from the axis carries more
        # flow than one of the same thickness near it.
        total = sum(u * (((i + 1) * step) ** 2 - (i * step) ** 2)
                    for i, u in enumerate(velocity))
        mean = total / case.radius_m ** 2
    else:
        mean = sum(velocity) / cells
    return FlowResult(solver="OpenFOAM", solver_version=version() or "unknown",
                      coordinates_m=coordinates, velocity_m_s=velocity,
                      mean_velocity_m_s=mean)
