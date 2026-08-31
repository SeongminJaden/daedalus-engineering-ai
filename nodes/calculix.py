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


def write_deck(path: Path, mesh: Mesh, youngs_modulus_pa: float,
               poisson_ratio: float, fixed_nodes: np.ndarray,
               load_nodes: np.ndarray, total_load_n: float,
               load_direction: int = 1,
               element_type: ElementType = ElementType.C3D8I) -> Path:
    """Write a CalculiX input deck for the same problem the Warp solver runs.

    Node numbering is one-based in the deck and zero-based in the mesh, and the
    corner ordering already matches: this project's NODE_OFFSETS run round the
    bottom face and then the top, which is what C3D8 expects.

    The load is divided equally over the loaded nodes, exactly as the Warp
    solver does, so any difference in the answer is not a difference in how the
    load was applied.
    """
    if total_load_n == 0.0:
        raise ValueError("a zero load gives a zero answer and checks nothing")
    if len(load_nodes) == 0:
        raise ValueError("no loaded nodes")
    if len(fixed_nodes) == 0:
        raise ValueError(
            "no fixed nodes, so the problem is singular and CalculiX will "
            "refuse it as this project's solver would")

    per_node = total_load_n / len(load_nodes)
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
        "*SOLID SECTION, ELSET=Eall, MATERIAL=MAT",
        "*BOUNDARY",
    ]
    for node in fixed_nodes:
        lines.append(f"{int(node) + 1}, 1, 3, 0.0")

    lines += ["*STEP", "*STATIC", "*CLOAD"]
    for node in load_nodes:
        lines.append(f"{int(node) + 1}, {load_direction + 1}, {per_node:.10e}")
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
        elif section == "s" and len(parts) >= 8:
            element = int(parts[0]) - 1
            if 0 <= element < n_elements:
                stress_sums[element] += [float(v) for v in parts[2:8]]
                stress_counts[element] += 1

    counts = np.where(stress_counts > 0, stress_counts, 1.0)
    return displacements, stress_sums / counts[:, None]


def solve(mesh: Mesh, youngs_modulus_pa: float, poisson_ratio: float,
          fixed_nodes: np.ndarray, load_nodes: np.ndarray,
          total_load_n: float, load_direction: int = 1,
          element_type: ElementType = ElementType.C3D8I,
          timeout_s: float = 600.0,
          keep_directory: Path | None = None) -> CalculixResult:
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
                   load_direction, element_type)
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
