"""CalculiX as an external FEA node, for cross-validating the Warp solver.

Every FEM result this project has produced so far was checked against beam
theory, against a patch test, or against itself. Those catch a lot and they do
not catch a shared misunderstanding of the element formulation. CalculiX is an
independently written solver, in Fortran and C, by people who made different
choices, so agreeing with it is evidence of a different kind.

VALIDITY, before the implementation, and this is the part most easily
overstated:

* **CalculiX is a SIMULATION, not a physical test.** Agreement means "this
  implementation gives the same answer as an established solver", not "this
  answer matches reality". A design that agrees with CalculiX has still never
  been built. The verification status this can grant is therefore still
  SIMULATED, and EXPERIMENTALLY_VALIDATED remains unreachable by any amount of
  cross-solver agreement.

* **The two solvers share their modelling assumptions, so common-mode errors
  survive.** Both are linear elastic, small strain, and run on the same mesh
  with the same element family. If the assumption itself is wrong for the
  problem, both are wrong together and agree beautifully. This checks the
  IMPLEMENTATION, not the model.

* **The deck is generated from OUR mesh, which is deliberate and limiting.**
  Using the same nodes and elements isolates the solver and the element
  formulation from the discretisation. It means a meshing error would be
  invisible, and it means the comparison is much tighter than two independent
  analysts would achieve.

* **The element type must MATCH or the comparison is meaningless.** Our
  formulation uses Wilson incompatible modes, whose CalculiX equivalent is
  C3D8I. Comparing against the fully integrated C3D8 shows a large
  disagreement that is a property of that element, not a bug in either solver:
  C3D8 shear-locks in bending, which is the whole reason the incompatible
  modes are there. Both are offered so the difference can be seen rather than
  stumbled into.

* **CalculiX has its own assumptions and its own defaults**, and being
  established does not make it a reference standard. It is a second opinion
  from a different implementation.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from physics.fem.mesh import Mesh

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

CALCULIX_NODE_NAME = "calculix.local"
CALCULIX_CAPABILITY = "analysis.fea.calculix"


class ElementType(str, Enum):
    """The brick elements worth comparing against.

    C3D8I carries incompatible modes and is the counterpart of this project's
    formulation. C3D8 is fully integrated and shear-locks in bending; it is
    here so the locking can be measured rather than assumed.
    """

    C3D8 = "C3D8"
    C3D8I = "C3D8I"
    C3D20R = "C3D20R"
    # The tetrahedra, for shapes this project's structured mesher cannot
    # cover. C3D10 is the default of the two: a linear tetrahedron is very
    # stiff in bending and would report a confidently wrong deflection.
    C3D4 = "C3D4"
    C3D10 = "C3D10"

    @property
    def nodes_per_element(self) -> int:
        return {"C3D8": 8, "C3D8I": 8, "C3D20R": 20,
                "C3D4": 4, "C3D10": 10}[self.value]


def executable() -> str | None:
    """The ccx binary, or None when it is not installed."""
    return shutil.which("ccx")


def is_available() -> bool:
    return executable() is not None


def version() -> str | None:
    """The reported version string, or None when ccx is absent."""
    binary = executable()
    if binary is None:
        return None
    result = subprocess.run([binary, "-v"], capture_output=True, text=True,
                            timeout=30)
    for line in (result.stdout + result.stderr).splitlines():
        if "Version" in line:
            return line.strip()
    return None


def calculix_descriptor(available: bool | None = None) -> NodeDescriptor:
    """The node as the registry sees it.

    Unlike the Fusion node this one is genuinely available when the binary is
    present, and it says so from the filesystem rather than from a flag.
    """
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=CALCULIX_NODE_NAME, transport=Transport.STDIO,
        address=executable() or "ccx",
        available=present,
        unavailable_reason="" if present else
        "unavailable: the ccx binary is not on PATH")


def calculix_capability_method():
    """The capability declaration, in the registry's schema.

    Unlike the Fusion node this one is genuinely usable, so its applicability
    is about the problem rather than about an entitlement.
    """
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=CALCULIX_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Independent 3D FEA in CalculiX, for cross-validating this "
                "project's solver.",
        inputs=("mesh", "material", "boundary_conditions", "load"),
        outputs=("displacements", "element_stress"),
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("the problem can be posed on a structured hex grid",
                      lambda c: c.supports("prismatic_beam")
                      or c.supports("voxel_domain")),
        ),
        implementation="nodes.calculix.solve",
        evidence="SIMULATED",
        notes="A second independently written solver, not a reference "
              "standard and not a physical test. Agreement means this "
              "implementation gives the same answer as an established one, not "
              "that the answer matches reality, and both share the same linear "
              "elastic small-strain assumptions so a wrong MODEL agrees "
              "beautifully. The deck is generated from this project's own "
              "mesh, which isolates the solver and element formulation and "
              "makes a meshing error invisible. C3D8I is the counterpart of "
              "this project's incompatible-mode element; comparing against the "
              "fully integrated C3D8 shows that element's shear locking rather "
              "than a bug.")


CALCULIX_GENERAL_CAPABILITY = "analysis.fea.general_shape"


def calculix_general_capability_method():
    """FEA on a shape this project's own solver cannot mesh.

    Separate from analysis.fea.calculix because the two answer different
    questions. That one cross-checks the Warp solver on a mesh both share.
    This one is the ONLY route for a domain the structured mesher cannot
    cover, so there is nothing to cross-check it against and it must say so.
    """
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=CALCULIX_GENERAL_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Tetrahedral FEA in CalculiX for general shapes, which this "
                "project's structured hex solver cannot mesh at all.",
        inputs=("geometry", "material", "boundary_conditions", "load"),
        outputs=("displacements", "element_stress"),
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("the domain is not a structured hex grid, which is "
                      "exactly when this route is needed",
                      lambda c: not (c.supports("prismatic_beam")
                                     or c.supports("voxel_domain"))),
        ),
        implementation="nodes.calculix.solve",
        evidence="SIMULATED",
        notes="Gmsh meshes the shape with tetrahedra and CalculiX solves it. "
              "This project's Warp solver has NO tetrahedral element and "
              "cannot check the answer, so unlike the hex route there is no "
              "second opinion here: it is a single solver on a single mesh. "
              "Quadratic C3D10 is used because linear C3D4 is far too stiff "
              "in bending, measured at 11 to 18 percent low on a cantilever "
              "against 0.5 percent for C3D10. Gmsh and CalculiX order the mid "
              "edge nodes differently and the permutation is applied "
              "explicitly; getting it wrong makes CalculiX write an empty "
              "result rather than a wrong one, which is now raised instead of "
              "being returned as zeros.")


def write_deck(path: Path, mesh: "Mesh | TetMesh", youngs_modulus_pa: float,
               poisson_ratio: float, fixed_nodes: np.ndarray,
               load_nodes: np.ndarray, total_load_n: float,
               load_direction: int = 1,
               element_type: ElementType = ElementType.C3D8I,
               nodal_forces: np.ndarray | None = None,
               thermal: "ThermalLoad | None" = None) -> Path:
    """Write a CalculiX input deck for the same problem the Warp solver runs.

    Node numbering is one-based in the deck and zero-based in the mesh, and the
    corner ordering already matches: this project's NODE_OFFSETS run round the
    bottom face and then the top, which is what C3D8 expects.

    The load is divided equally over the loaded nodes, exactly as the Warp
    solver does, so any difference in the answer is not a difference in how the
    load was applied.

    `nodal_forces`, when given, is an (n_loaded, 3) array of forces in newtons
    for `load_nodes` in order and REPLACES the equal division: it is how a
    torque or a combined load is applied, since a single direction and a total
    cannot describe either. `thermal` adds an expansion coefficient to the
    material and a nodal temperature field to the step; it may be the only
    load, in which case `total_load_n` is ignored and may be zero.
    """
    if nodal_forces is not None:
        nodal_forces = np.asarray(nodal_forces, dtype=float)
        if nodal_forces.shape != (len(load_nodes), 3):
            raise ValueError(
                f"nodal_forces must be ({len(load_nodes)}, 3) to match "
                f"load_nodes, got {nodal_forces.shape}")
        if not np.any(nodal_forces):
            raise ValueError("nodal_forces are all zero and check nothing")
    elif thermal is None:
        if total_load_n == 0.0:
            raise ValueError("a zero load gives a zero answer and checks nothing")
    if len(load_nodes) == 0 and thermal is None:
        raise ValueError("no loaded nodes")
    if len(fixed_nodes) == 0:
        raise ValueError(
            "no fixed nodes, so the problem is singular and CalculiX will "
            "refuse it as this project's solver would")
    expected = element_type.nodes_per_element
    actual = int(np.asarray(mesh.connectivity).shape[1])
    if actual != expected:
        raise ValueError(
            f"element type {element_type.value} takes {expected} nodes per "
            f"element but the mesh supplies {actual}; a mismatched deck "
            f"either fails to read or silently describes a different solid")

    per_node = total_load_n / len(load_nodes) if len(load_nodes) else 0.0
    lines: list[str] = ["*HEADING", "cross-validation deck", "*NODE, NSET=Nall"]
    for index, (x, y, z) in enumerate(mesh.node_coords, start=1):
        lines.append(f"{index}, {x:.10e}, {y:.10e}, {z:.10e}")

    lines.append(f"*ELEMENT, TYPE={element_type.value}, ELSET=Eall")
    for index, nodes in enumerate(mesh.connectivity, start=1):
        joined = ", ".join(str(int(n) + 1) for n in nodes)
        lines.append(f"{index}, {joined}")

    lines += [
        "*MATERIAL, NAME=MAT",
        "*ELASTIC",
        f"{youngs_modulus_pa:.10e}, {poisson_ratio:.10f}",
    ]
    if thermal is not None:
        lines += ["*EXPANSION", f"{thermal.expansion_1_k:.10e}"]
    lines += ["*SOLID SECTION, ELSET=Eall, MATERIAL=MAT"]
    if thermal is not None:
        lines += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE",
                  f"Nall, {thermal.reference_k:.10e}"]
    lines += ["*BOUNDARY"]
    for node in fixed_nodes:
        lines.append(f"{int(node) + 1}, 1, 3, 0.0")

    lines += ["*STEP", "*STATIC"]
    if nodal_forces is not None:
        lines.append("*CLOAD")
        for node, force in zip(load_nodes, nodal_forces):
            for axis in range(3):
                if force[axis] != 0.0:
                    lines.append(f"{int(node) + 1}, {axis + 1}, "
                                 f"{force[axis]:.10e}")
    elif len(load_nodes) and total_load_n != 0.0:
        lines.append("*CLOAD")
        for node in load_nodes:
            lines.append(f"{int(node) + 1}, {load_direction + 1}, "
                         f"{per_node:.10e}")
    if thermal is not None:
        lines.append("*TEMPERATURE")
        for index, temperature in enumerate(thermal.node_temperatures_k(mesh),
                                            start=1):
            lines.append(f"{index}, {temperature:.10e}")
    lines += [
        "*NODE PRINT, NSET=Nall",
        "U",
        "*EL PRINT, ELSET=Eall",
        "S",
        "*END STEP",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


@dataclass(frozen=True)
class ThermalLoad:
    """A temperature field for a static step, with the material's expansion.

    `gradient_k_per_m` is along `gradient_axis`, measured from the mesh's
    minimum on that axis, on top of a uniform rise `delta_k` above the
    reference. Uniform with a free end is free expansion and zero stress;
    a through-thickness gradient on a clamped cantilever bends it, and both
    have closed forms the tests use.
    """

    expansion_1_k: float
    delta_k: float = 0.0
    gradient_k_per_m: float = 0.0
    gradient_axis: int = 1
    reference_k: float = 293.15

    def node_temperatures_k(self, mesh) -> np.ndarray:
        coords = np.asarray(mesh.node_coords)
        along = coords[:, self.gradient_axis]
        return (self.reference_k + self.delta_k
                + self.gradient_k_per_m * (along - along.min()))


@dataclass(frozen=True)
class CalculixResult:
    """What CalculiX returned, in the same terms the Warp solver reports."""

    displacements: np.ndarray          # (n_nodes, 3)
    element_stress: np.ndarray         # (n_elements, 6) Voigt
    element_type: ElementType
    converged: bool
    deck_path: str

    def max_displacement_magnitude(self) -> float:
        return float(np.linalg.norm(self.displacements, axis=1).max())

    def max_von_mises_pa(self) -> float:
        s = self.element_stress
        xx, yy, zz, xy, yz, zx = (s[:, i] for i in range(6))
        return float(np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2
                                    + (zz - xx) ** 2)
                             + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2)).max())


def _parse_dat(path: Path, n_nodes: int, n_elements: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Read displacements and element stresses from a .dat file.

    The .dat format is a sequence of labelled blocks. Stresses are printed per
    integration point, so the eight values belonging to one element are
    averaged, which is what makes them comparable to this project's
    element-centre stress.
    """
    displacements = np.zeros((n_nodes, 3))
    stress_sums = np.zeros((n_elements, 6))
    stress_counts = np.zeros(n_elements)
    seen_displacement = False

    section = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("displacements"):
            section = "u"
            continue
        if lowered.startswith("stresses"):
            section = "s"
            continue
        if line[0].isalpha():
            section = None
            continue

        parts = line.split()
        if section == "u" and len(parts) >= 4:
            node = int(parts[0]) - 1
            if 0 <= node < n_nodes:
                displacements[node] = [float(v) for v in parts[1:4]]
                seen_displacement = True
        elif section == "s" and len(parts) >= 8:
            element = int(parts[0]) - 1
            if 0 <= element < n_elements:
                stress_sums[element] += [float(v) for v in parts[2:8]]
                stress_counts[element] += 1

    if not seen_displacement:
        # A run that failed writes an empty .dat, and the arrays above are
        # still full of zeros. Returning them would hand back a plausible
        # looking answer for a solve that did not happen: a wrong element
        # ordering, for instance, produces exactly this and a caller who did
        # not check `converged` would read it as a rigid structure.
        raise RuntimeError(
            f"CalculiX wrote no displacements to {path.name}. The solve "
            f"failed; a common cause is an element node ordering the solver "
            f"rejects, which yields an empty result rather than a bad one")

    counts = np.where(stress_counts > 0, stress_counts, 1.0)
    return displacements, stress_sums / counts[:, None]


def solve(mesh: "Mesh | TetMesh", youngs_modulus_pa: float, poisson_ratio: float,
          fixed_nodes: np.ndarray, load_nodes: np.ndarray,
          total_load_n: float, load_direction: int = 1,
          element_type: ElementType = ElementType.C3D8I,
          timeout_s: float = 600.0,
          keep_directory: Path | None = None,
          nodal_forces: np.ndarray | None = None,
          thermal: ThermalLoad | None = None) -> CalculixResult:
    """Run the same problem through CalculiX and return its answer.

    Raises `CapabilityUnavailable` when ccx is absent, matching how the Fusion
    node behaves, so a caller handles both the same way.
    """
    binary = executable()
    if binary is None:
        raise CapabilityUnavailable(
            capability=CALCULIX_CAPABILITY, node=CALCULIX_NODE_NAME,
            reason="unavailable: the ccx binary is not on PATH")

    context = (tempfile.TemporaryDirectory() if keep_directory is None
               else None)
    directory = Path(context.name) if context else Path(keep_directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        job = directory / "job"
        write_deck(job.with_suffix(".inp"), mesh, youngs_modulus_pa,
                   poisson_ratio, fixed_nodes, load_nodes, total_load_n,
                   load_direction, element_type, nodal_forces=nodal_forces,
                   thermal=thermal)
        completed = subprocess.run([binary, str(job)], capture_output=True,
                                   text=True, timeout=timeout_s,
                                   cwd=str(directory))
        dat = job.with_suffix(".dat")
        if not dat.exists():
            raise RuntimeError(
                f"CalculiX produced no .dat output. stdout tail:\n"
                f"{completed.stdout[-2000:]}")
        displacements, stress = _parse_dat(dat, mesh.n_nodes,
                                           mesh.n_elements)
        return CalculixResult(
            displacements=displacements, element_stress=stress,
            element_type=element_type,
            converged="*ERROR" not in completed.stdout,
            deck_path=str(job.with_suffix(".inp")))
    finally:
        if context is not None:
            context.cleanup()
