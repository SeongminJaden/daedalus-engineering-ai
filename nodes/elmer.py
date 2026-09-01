"""nodes.elmer - Elmer FEM, for the physics nothing else in the stack does.

Elmer is added for ONE capability: electromagnetics, and the coupled solves
where a field computed by one equation becomes a source term in the next.
Its structural and conduction solvers overlap CalculiX's, and overlap is a
cross-check rather than a new capability, so nothing here registers itself as
a second way to do stress.

The validity domain of every equation used was written down before any of this
existed; see docs/scoping_elmer.md. The short version:

* Magnetostatics holds while the skin depth exceeds the conductor size. In
  copper at 50 Hz that is about 9 mm, so a 20 mm bar is already marginal.
* Joule heating needs conductivity at the temperature actually reached. A
  100 K rise raises copper's resistivity by about 39 percent, and the feedback
  compounds.
* A conduction result is a statement about an ASSUMED cooling condition, not
  about the part, because the convection coefficient carries a factor of two
  spread in the honest literature. The assumed value travels with the result.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Where the local build puts itself. Nothing was installed system wide.
ELMER_HOME = Path(os.environ.get("ELMER_HOME", Path.home() / "opt" / "elmer"))

#: Elmer's "Heat Source" is power per unit MASS, W/kg, and is multiplied by
#: density internally. A volumetric source in W/m^3 must therefore be divided
#: by the density before being written into the sif. Getting this backwards
#: scales the answer by the density and still produces a plausible field.
HEAT_SOURCE_IS_PER_UNIT_MASS = True


class ElmerUnavailable(RuntimeError):
    """Elmer is not installed where this module expects it."""


def solver_path() -> Path:
    exe = ELMER_HOME / "bin" / "ElmerSolver"
    if not exe.exists():
        raise ElmerUnavailable(
            f"ElmerSolver not found at {exe}. This module expects the local "
            f"build under {ELMER_HOME}; set ELMER_HOME to point elsewhere.")
    return exe


def is_available() -> bool:
    return (ELMER_HOME / "bin" / "ElmerSolver").exists()


def _environment() -> dict:
    env = dict(os.environ)
    env["ELMER_HOME"] = str(ELMER_HOME)
    lib = str(ELMER_HOME / "lib")
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env.pop("PYTHONPATH", None)
    return env


# ------------------------------------------------------------------ meshing

@dataclass(frozen=True)
class BoxMesh:
    """A structured hexahedral box, written in Elmer's native mesh format.

    Boundary numbering, fixed so callers do not have to guess:
        1 = x low,  2 = x high,  3 = y low,
        4 = y high, 5 = z low,   6 = z high
    """

    nodes: np.ndarray            # (n, 3)
    elements: np.ndarray         # (m, 8), 1-based
    boundary: list               # (bc_id, parent_element, (4 node ids))
    size_m: tuple
    bodies: tuple = ()           # per element body number, defaults to all 1

    def write(self, directory: Path) -> Path:
        mesh = Path(directory) / "mesh"
        mesh.mkdir(parents=True, exist_ok=True)

        (mesh / "mesh.nodes").write_text("".join(
            f"{i + 1} -1 {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n"
            for i, p in enumerate(self.nodes)))

        bodies = self.bodies or (1,) * len(self.elements)
        (mesh / "mesh.elements").write_text("".join(
            f"{i + 1} {bodies[i]} 808 " + " ".join(str(n) for n in e) + "\n"
            for i, e in enumerate(self.elements)))

        (mesh / "mesh.boundary").write_text("".join(
            f"{i + 1} {bc} {parent} 0 404 " + " ".join(str(n) for n in quad)
            + "\n" for i, (bc, parent, quad) in enumerate(self.boundary)))

        (mesh / "mesh.header").write_text(
            f"{len(self.nodes)} {len(self.elements)} {len(self.boundary)}\n"
            f"2\n808 {len(self.elements)}\n404 {len(self.boundary)}\n")
        return mesh


def box_mesh(length_m: float, width_m: float, height_m: float,
             nx: int, ny: int, nz: int, body_of=None) -> BoxMesh:
    """A uniform hex grid over a box, with the six faces tagged.

    `body_of` maps an element centroid (x, y, z) to a body number, for meshes
    that carry more than one material. It defaults to a single body.
    """
    if min(nx, ny, nz) < 1:
        raise ValueError("need at least one element in each direction")

    xs = np.linspace(0.0, length_m, nx + 1)
    ys = np.linspace(0.0, width_m, ny + 1)
    zs = np.linspace(0.0, height_m, nz + 1)
    nodes = np.array([(x, y, z) for z in zs for y in ys for x in xs])

    def nid(i, j, k):                      # 1-based
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i + 1

    elements = []
    bodies = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append([
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])
                centre = (0.5 * (xs[i] + xs[i + 1]),
                          0.5 * (ys[j] + ys[j + 1]),
                          0.5 * (zs[k] + zs[k + 1]))
                bodies.append(1 if body_of is None else int(body_of(*centre)))
    elements = np.array(elements, dtype=np.int64)

    def eid(i, j, k):
        return k * ny * nx + j * nx + i + 1

    boundary = []
    for k in range(nz):
        for j in range(ny):
            boundary.append((1, eid(0, j, k),
                             (nid(0, j, k), nid(0, j + 1, k),
                              nid(0, j + 1, k + 1), nid(0, j, k + 1))))
            boundary.append((2, eid(nx - 1, j, k),
                             (nid(nx, j, k), nid(nx, j + 1, k),
                              nid(nx, j + 1, k + 1), nid(nx, j, k + 1))))
    for k in range(nz):
        for i in range(nx):
            boundary.append((3, eid(i, 0, k),
                             (nid(i, 0, k), nid(i + 1, 0, k),
                              nid(i + 1, 0, k + 1), nid(i, 0, k + 1))))
            boundary.append((4, eid(i, ny - 1, k),
                             (nid(i, ny, k), nid(i + 1, ny, k),
                              nid(i + 1, ny, k + 1), nid(i, ny, k + 1))))
    for j in range(ny):
        for i in range(nx):
            boundary.append((5, eid(i, j, 0),
                             (nid(i, j, 0), nid(i + 1, j, 0),
                              nid(i + 1, j + 1, 0), nid(i, j + 1, 0))))
            boundary.append((6, eid(i, j, nz - 1),
                             (nid(i, j, nz), nid(i + 1, j, nz),
                              nid(i + 1, j + 1, nz), nid(i, j + 1, nz))))

    return BoxMesh(nodes=nodes, elements=elements, boundary=boundary,
                   size_m=(length_m, width_m, height_m),
                   bodies=tuple(bodies))


# ------------------------------------------------------------------ running

def run(directory: Path, sif: str) -> str:
    """Write the sif, run ElmerSolver, return its stdout.

    Raises on a non-zero exit, and also on the silent failure where the solver
    exits cleanly having written nothing.
    """
    directory = Path(directory)
    (directory / "case.sif").write_text(sif)
    (directory / "ELMERSOLVER_STARTINFO").write_text("case.sif\n")

    result = subprocess.run(
        [str(solver_path())], cwd=directory, capture_output=True, text=True,
        env=_environment(), timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"ElmerSolver exited {result.returncode}\n"
            f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}")
    if "ALL DONE" not in result.stdout:
        raise RuntimeError(
            "ElmerSolver exited cleanly but did not report ALL DONE, which "
            "means it stopped early and any result files are stale:\n"
            + result.stdout[-3000:])
    return result.stdout


def read_scalars(directory: Path) -> dict:
    """Read a SaveScalars output into {name: value}.

    Raises if the file is missing or empty rather than returning a default. An
    empty result read as zero is the failure mode that makes a broken solve
    look like a converged one.
    """
    directory = Path(directory)
    data = directory / "scalars.dat"
    names = directory / "scalars.dat.names"
    if not data.exists():
        raise RuntimeError(
            f"Elmer wrote no scalars.dat in {directory}. The solve produced "
            f"no saved values, so there is nothing to check.")
    values = data.read_text().split()
    if not values:
        raise RuntimeError(f"{data} is empty; the solve saved nothing.")

    labels = []
    if names.exists():
        for line in names.read_text().splitlines():
            match = re.match(r"\s*\d+:\s*(.+?)\s*$", line)
            if match:
                labels.append(match.group(1))
    if len(labels) != len(values):
        labels = [f"column {i + 1}" for i in range(len(values))]
    return {label: float(value) for label, value in zip(labels, values)}


# --------------------------------------------------- magnetostatics: a wire

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class WireField:
    """The solved field of a round wire, with the answer it is checked against."""

    wire_radius_m: float
    domain_radius_m: float
    current_a: float
    elements: int
    potential_centre_wb_per_m: float
    flux_density_max_t: float

    @property
    def exact_potential_centre_wb_per_m(self) -> float:
        """A on the axis when A is zero on the outer circle.

        A(r) = (mu0 I / 4 pi)(1 - r^2/a^2) + (mu0 I / 2 pi) ln(b/a), so on the
        axis A = (mu0 I / 2 pi) (1/2 + ln(b/a)).

        This is EXACT, not an approximation of an open boundary. Setting A = 0
        on r = b is a gauge choice, and B, which is the curl of A, is
        unaffected by it: outside the wire B = mu0 I / (2 pi r) either way.
        """
        return (MU0 * self.current_a / (2.0 * np.pi)) * (
            0.5 + np.log(self.domain_radius_m / self.wire_radius_m))

    @property
    def exact_flux_density_max_t(self) -> float:
        """|B| is largest at the wire surface: mu0 I / (2 pi a)."""
        return MU0 * self.current_a / (2.0 * np.pi * self.wire_radius_m)

    @property
    def potential_error(self) -> float:
        exact = self.exact_potential_centre_wb_per_m
        return abs(self.potential_centre_wb_per_m - exact) / abs(exact)

    @property
    def flux_density_error(self) -> float:
        exact = self.exact_flux_density_max_t
        return abs(self.flux_density_max_t - exact) / abs(exact)


def _write_triangle_mesh(directory: Path, coords, triangles, boundary_edges):
    mesh = Path(directory) / "mesh"
    mesh.mkdir(parents=True, exist_ok=True)
    (mesh / "mesh.nodes").write_text("".join(
        f"{i + 1} -1 {p[0]:.17g} {p[1]:.17g} 0\n"
        for i, p in enumerate(coords)))
    (mesh / "mesh.elements").write_text("".join(
        f"{i + 1} {body} 303 " + " ".join(map(str, ns)) + "\n"
        for i, (body, ns) in enumerate(triangles)))
    (mesh / "mesh.boundary").write_text("".join(
        f"{i + 1} 1 0 0 202 " + " ".join(map(str, ns)) + "\n"
        for i, ns in enumerate(boundary_edges)))
    # The type count must match the number of type lines that follow. Writing
    # a count of three with two lines makes Elmer stop with a header error.
    (mesh / "mesh.header").write_text(
        f"{len(coords)} {len(triangles)} {len(boundary_edges)}\n"
        f"2\n303 {len(triangles)}\n202 {len(boundary_edges)}\n")
    return mesh


def _wire_mesh(directory: Path, wire_radius_m: float, domain_radius_m: float,
               divisions: int):
    """A disc of wire inside a disc of air, meshed with triangles."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wire")
        wire = gmsh.model.occ.addDisk(0, 0, 0, wire_radius_m, wire_radius_m)
        air = gmsh.model.occ.addDisk(0, 0, 0, domain_radius_m,
                                     domain_radius_m)
        gmsh.model.occ.fragment([(2, air)], [(2, wire)])
        gmsh.model.occ.synchronize()

        areas = {s[1]: gmsh.model.occ.getMass(2, s[1])
                 for s in gmsh.model.getEntities(2)}
        wire_tag = min(areas, key=areas.get)
        air_tag = max(areas, key=areas.get)
        lengths = {c[1]: gmsh.model.occ.getMass(1, c[1])
                   for c in gmsh.model.getEntities(1)}
        outer = max(lengths, key=lengths.get)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),
                                domain_radius_m / divisions)
        gmsh.model.mesh.setSize(
            [(0, p[1]) for p in gmsh.model.getBoundary([(2, wire_tag)],
                                                       recursive=True)],
            wire_radius_m / max(2, divisions // 3))
        gmsh.model.mesh.generate(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags = tags[order]
        coords = xyz.reshape(-1, 3)[order]
        remap = {int(t): i + 1 for i, t in enumerate(tags)}

        triangles = []
        for body, tag in ((1, wire_tag), (2, air_tag)):
            types, ids, nodes = gmsh.model.mesh.getElements(2, tag)
            for _, batch, flat in zip(types, ids, nodes):
                for row in flat.reshape(len(batch), -1):
                    triangles.append((body, [remap[int(v)] for v in row]))

        edges = []
        types, ids, nodes = gmsh.model.mesh.getElements(1, outer)
        for _, batch, flat in zip(types, ids, nodes):
            for row in flat.reshape(len(batch), -1):
                edges.append([remap[int(v)] for v in row])
    finally:
        gmsh.finalize()

    _write_triangle_mesh(directory, coords, triangles, edges)
    return len(triangles)


def wire_magnetostatics(directory, current_a: float = 100.0,
                        wire_radius_m: float = 0.002,
                        domain_radius_m: float = 0.05,
                        divisions: int = 25) -> WireField:
    """Solve for the field of a round wire carrying a uniform current.

    Validity, from docs/scoping_elmer.md: this is MAGNETOSTATICS, so it holds
    while the skin depth exceeds the wire radius. At DC that is unconditional;
    at 50 Hz in copper the skin depth is about 9 mm, so a wire thicker than
    that is already outside this model and the answer would flatter it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    elements = _wire_mesh(directory, wire_radius_m, domain_radius_m, divisions)
    current_density = current_a / (np.pi * wire_radius_m ** 2)

    sif = f"""
Check Keywords "Warn"
Header
  Mesh DB "." "mesh"
End
Simulation
  Max Output Level = 3
  Coordinate System = Cartesian 2D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 0
End
Body 1
  Equation = 1
  Material = 1
  Body Force = 1
End
Body 2
  Equation = 1
  Material = 1
End
Equation 1
  Active Solvers(3) = 1 2 3
End
Solver 1
  Equation = "Mag"
  Variable = Potential
  Procedure = "MagnetoDynamics2D" "MagnetoDynamics2D"
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-12
  Linear System Preconditioning = ILU1
  Nonlinear System Max Iterations = 1
  Steady State Convergence Tolerance = 1.0e-10
End
Solver 2
  Equation = "MagFields"
  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"
  Potential Variable = Potential
  Calculate Magnetic Flux Density = Logical True
  Linear System Solver = Iterative
  Linear System Iterative Method = CG
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-12
  Linear System Preconditioning = ILU0
End
Solver 3
  Equation = SaveScalars
  Procedure = "SaveData" "SaveScalars"
  Filename = "scalars.dat"
  Variable 1 = Potential
  Operator 1 = max
  Variable 2 = Magnetic Flux Density 1
  Operator 2 = max
  Variable 3 = Magnetic Flux Density 2
  Operator 3 = max
End
Material 1
  Relative Permeability = 1.0
  Relative Permittivity = 1.0
End
Body Force 1
  Current Density = {current_density!r}
End
Boundary Condition 1
  Target Boundaries(1) = 1
  Potential = 0.0
End
"""
    run(directory, sif)
    scalars = read_scalars(directory)

    def pick(fragment):
        for name, value in scalars.items():
            if fragment in name.lower():
                return value
        raise RuntimeError(
            f"Elmer saved no column matching {fragment!r}; got "
            f"{sorted(scalars)}")

    flux = max(abs(pick("magnetic flux density 1")),
               abs(pick("magnetic flux density 2")))
    return WireField(
        wire_radius_m=wire_radius_m, domain_radius_m=domain_radius_m,
        current_a=current_a, elements=elements,
        potential_centre_wb_per_m=pick("potential"),
        flux_density_max_t=flux)


# ------------------------------------------------------------- registration

ELMER_NODE_NAME = "elmer"
ELMER_MAGNETOSTATICS_CAPABILITY = "analysis.electromagnetics.magnetostatics"


def elmer_descriptor(available: bool | None = None):
    from .descriptor import NodeDescriptor, Transport

    present = is_available() if available is None else available
    return NodeDescriptor(
        name=ELMER_NODE_NAME, transport=Transport.STDIO,
        address=str(ELMER_HOME / "bin" / "ElmerSolver"),
        available=present,
        unavailable_reason="" if present else
        f"unavailable: ElmerSolver was not found under {ELMER_HOME}")


def elmer_capability_method():
    """Magnetostatics only.

    Elmer also solves conduction and elasticity, and those are deliberately
    NOT registered. CalculiX already covers them and is already verified, so a
    second implementation of the same equations is a cross-check rather than a
    capability the engine gains.
    """
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=ELMER_MAGNETOSTATICS_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Two dimensional magnetostatics in Elmer, for the magnetic "
                "field of current carrying conductors.",
        inputs=("conductor_geometry", "current", "permeability"),
        outputs=("vector_potential", "flux_density"),
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("the conductor carries a known current",
                      lambda c: c.require("has_conductor_current")),
            Condition("the skin depth exceeds the conductor size, so the "
                      "current fills it and a static solve is valid",
                      lambda c: c.require("skin_depth_exceeds_conductor")),
            Condition("the permeability is linear, since saturation is not "
                      "modelled",
                      lambda c: c.require("magnetically_linear")),
        ),
        implementation="nodes.elmer.wire_magnetostatics",
        evidence="SIMULATED",
        notes="Verified against a closed form rather than against a "
              "tolerance: for a round wire with the vector potential set to "
              "zero on an outer circle, A on the axis is "
              "(mu0 I / 2 pi)(1/2 + ln(b/a)) and the peak flux density is "
              "mu0 I / (2 pi a), both exact. Measured convergence orders were "
              "about 2 for the potential and about 1 for the flux density, "
              "which is what linear triangles must give since the flux "
              "density is a derivative of the potential and loses an order. A "
              "rate is stronger evidence than a threshold, because a wrong "
              "solve can meet a threshold on a fine enough mesh. Two "
              "assumptions bound this hard and both fail in the direction "
              "that flatters a design: at mains frequency the skin depth in "
              "copper is about 9 mm, so a thicker conductor is already "
              "outside the model, and a linear permeability overpredicts flux "
              "once the steel is past the knee of its B-H curve. Nothing here "
              "has been measured against a physical magnet.")


# ------------------------------------------- the coupled check: Joule heating

@dataclass(frozen=True)
class JouleRun:
    """Ohmic heating in a bar, with the answer computed independently.

    The point of this case is NOT that Elmer can solve a conduction problem.
    It is that the coupling between the electric solve and the heat source can
    be wired backwards and both physics will still pass their own separate
    checks. Comparing the solver's INTEGRATED heating against I squared R,
    computed from the geometry and the conductivity alone, is what catches it.
    """

    applied_volts: float
    resistance_ohm: float
    total_heating_w: float

    @property
    def current_a(self) -> float:
        return self.applied_volts / self.resistance_ohm

    @property
    def exact_heating_w(self) -> float:
        """I^2 R, from Ohm's law and the bar's dimensions. No solver involved."""
        return self.current_a ** 2 * self.resistance_ohm

    @property
    def heating_error(self) -> float:
        return abs(self.total_heating_w - self.exact_heating_w) \
            / self.exact_heating_w


_HEATING = re.compile(r"Total Heating Power\s*:\s*([-+0-9.eEdD]+)")


def joule_heating(directory, applied_volts: float = 0.5,
                  length_m: float = 0.1, width_m: float = 0.01,
                  height_m: float = 0.01,
                  conductivities: tuple = (5.96e7,),
                  divisions: int = 20) -> JouleRun:
    """Drive a bar with a potential difference and integrate the heating.

    `conductivities` may hold one value for a uniform bar, or two for a bar
    made of two equal halves in series. The series case is the discriminating
    one: resistances in series add, so the answer depends on the harmonic
    combination of the two conductivities. A coupling that averaged them, or
    that used conductivity where resistivity belongs, lands somewhere else.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    area = width_m * height_m

    if len(conductivities) == 1:
        resistance = length_m / (conductivities[0] * area)
        mesh = box_mesh(length_m, width_m, height_m, divisions, 2, 2)
        bodies = "Body 1\n  Equation = 1\n  Material = 1\nEnd\n"
        materials = f"Material 1\n  Electric Conductivity = " \
                    f"{conductivities[0]!r}\n  Density = 1.0\n" \
                    f"  Heat Conductivity = 1.0\nEnd\n"
    elif len(conductivities) == 2:
        half = 0.5 * length_m
        resistance = half / (conductivities[0] * area) \
            + half / (conductivities[1] * area)
        mesh = box_mesh(length_m, width_m, height_m, divisions, 2, 2,
                        body_of=lambda x, y, z: 1 if x < half else 2)
        bodies = ("Body 1\n  Equation = 1\n  Material = 1\nEnd\n"
                  "Body 2\n  Equation = 1\n  Material = 2\nEnd\n")
        materials = "".join(
            f"Material {i + 1}\n  Electric Conductivity = {s!r}\n"
            f"  Density = 1.0\n  Heat Conductivity = 1.0\nEnd\n"
            for i, s in enumerate(conductivities))
    else:
        raise ValueError("conductivities must hold one or two values")

    mesh.write(directory)
    sif = f"""
Check Keywords "Warn"
Header
  Mesh DB "." "mesh"
End
Simulation
  Max Output Level = 5
  Coordinate System = Cartesian 3D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 0
End
{bodies}Equation 1
  Active Solvers(1) = 1
End
Solver 1
  Equation = Static Current Conduction
  Procedure = "StatCurrentSolve" "StatCurrentSolver"
  Variable = Potential
  Variable DOFs = 1
  Calculate Joule Heating = Logical True
  Calculate Volume Current = Logical True
  Linear System Solver = Iterative
  Linear System Iterative Method = CG
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-14
  Linear System Preconditioning = ILU1
  Nonlinear System Max Iterations = 1
  Steady State Convergence Tolerance = 1.0e-12
End
{materials}Boundary Condition 1
  Target Boundaries(1) = 1
  Potential = {applied_volts!r}
End
Boundary Condition 2
  Target Boundaries(1) = 2
  Potential = 0.0
End
"""
    output = run(directory, sif)
    matches = _HEATING.findall(output.replace("D+", "E+").replace("D-", "E-"))
    if not matches:
        raise RuntimeError(
            "Elmer reported no Total Heating Power. Without it there is "
            "nothing to compare against I squared R, and a silent zero would "
            "look like agreement for a bar carrying no current.")
    return JouleRun(applied_volts=applied_volts, resistance_ohm=resistance,
                    total_heating_w=float(matches[-1]))
