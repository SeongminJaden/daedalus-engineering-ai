"""Gmsh as an external mesher, for cross-checking this project's own meshing.

The CalculiX node left one gap open on purpose. Its deck is generated from
THIS project's mesh, which isolates the solver and the element formulation and
in exchange makes a meshing error invisible: both solvers would agree
beautifully while computing the wrong geometry. Gmsh closes that gap. It
generates a mesh independently, and it measures geometry with an independent
CAD kernel (OpenCASCADE) rather than by counting cells.

VALIDITY DOMAIN
===============
Stated before implementing, per the standing discipline, and it is narrow.

This project's FEM solver is an 8-node hexahedron on a STRUCTURED UNIFORM
grid. `solve_linear_elasticity` builds ONE element stiffness matrix from
`mesh.dx, mesh.dy, mesh.dz` and reuses it for every element. Two consequences
follow, and neither is a limitation of Gmsh:

Consumable by this project's solver
    Axis-aligned structured hexahedral meshes with uniform spacing, which
    Gmsh produces on a box or an extruded prism via transfinite meshing
    followed by recombination.

NOT consumable, and this module does not pretend otherwise
    Tetrahedra, which is what Gmsh produces for a general shape. There is no
    tetrahedral element in this project, so a tet mesh cannot be run at all.
    Graded or otherwise non-uniform hexahedra are equally unusable, because
    every element sharing one stiffness matrix is precisely the assumption a
    graded mesh breaks. Feeding either in would not produce a slightly worse
    answer, it would produce a wrong one.

So the useful reach of this node is not "mesh anything". It is: generate the
one mesh family the solver can consume and check that this project builds the
same one, and measure geometry exactly where this project only approximates
it.

WHAT THIS IS NOT
================
Gmsh is a mesher, not a measurement of a real part. Agreement on volume means
two descriptions of the same idealised solid agree. A part that was actually
machined has tolerances, fillets and tool marks that neither describes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from physics.fem.mesh import NODE_OFFSETS, Mesh

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

GMSH_NODE_NAME = "gmsh.local"
GMSH_CAPABILITY = "meshing.gmsh"

#: Gmsh's element type number for an 8-node hexahedron.
_HEX8 = 5


def _gmsh():
    try:
        import gmsh
    except ImportError:
        return None
    return gmsh


def is_available() -> bool:
    return _gmsh() is not None


def version() -> str | None:
    """The reported version, or None when Gmsh is absent."""
    module = _gmsh()
    if module is None:
        return None
    module.initialize()
    try:
        return f"Gmsh {module.option.getString('General.Version')}"
    finally:
        module.finalize()


def _require():
    module = _gmsh()
    if module is None:
        raise CapabilityUnavailable(
            GMSH_CAPABILITY, GMSH_NODE_NAME,
            "the gmsh package is not installed")
    return module


@contextmanager
def _session(name: str):
    """A Gmsh session that is always finalised.

    Gmsh holds global state, so leaving it initialised after an exception
    makes the NEXT call fail somewhere unrelated.
    """
    module = _require()
    module.initialize()
    module.option.setNumber("General.Terminal", 0)
    module.model.add(name)
    try:
        yield module
    finally:
        module.finalize()


def gmsh_descriptor(available: bool | None = None) -> NodeDescriptor:
    """The node as the registry sees it, read from the import system."""
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=GMSH_NODE_NAME, transport=Transport.STDIO, address="gmsh",
        available=present,
        unavailable_reason="" if present else
        "unavailable: the gmsh package is not installed")


def gmsh_capability_method():
    """The capability declaration, in the registry's schema."""
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=GMSH_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Independent structured hex meshing and exact CAD volume, for "
                "cross-validating this project's own mesh generation.",
        inputs=("geometry",),
        outputs=("mesh", "volume"),
        fidelity=Fidelity.ANALYTICAL,
        cost=Cost.CHEAP,
        conditions=(
            Condition("the domain is a box or an extruded prism, which is the "
                      "only family a structured uniform hex mesh covers",
                      lambda c: c.supports("prismatic_beam")
                      or c.supports("voxel_domain")),
        ),
        implementation="nodes.gmsh_node.structured_box_mesh",
        evidence="SIMULATED",
        notes="A mesher, not a measurement of a real part. It closes the gap "
              "the CalculiX node leaves open, since that node meshes with this "
              "project's own mesh and so cannot see a meshing error. Its reach "
              "is narrow on purpose: this project's solver builds ONE element "
              "stiffness matrix from the cell size and reuses it, so only "
              "axis-aligned uniform hexahedra can be consumed. Tetrahedra, "
              "which is what Gmsh produces for a general shape, cannot be run "
              "at all because there is no tetrahedral element here, and a "
              "graded hex mesh breaks the shared-stiffness assumption "
              "outright.")


def structured_box_mesh(length_m: float, height_m: float, width_m: float,
                        nx: int, ny: int, nz: int) -> Mesh:
    """Mesh a box in Gmsh and return it in this project's Mesh form.

    The conversion identifies each element's local nodes GEOMETRICALLY, by
    position relative to the element's own minimum corner, rather than by
    trusting Gmsh's node ordering to match this project's NODE_OFFSETS. A
    silent change in either convention would otherwise produce an inverted or
    twisted element, which shows up as a plausible but wrong stiffness rather
    than as an error.
    """
    for count in (nx, ny, nz):
        if count < 1:
            raise ValueError("subdivisions must be >= 1")
    counts = (nx, ny, nz)
    with _session("structured_box") as gmsh:
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, length_m, height_m, width_m)
        gmsh.model.occ.synchronize()
        for dim, tag in gmsh.model.getEntities(1):
            box = gmsh.model.getBoundingBox(dim, tag)
            extent = np.array([box[3] - box[0], box[4] - box[1],
                               box[5] - box[2]])
            gmsh.model.mesh.setTransfiniteCurve(
                tag, counts[int(np.argmax(extent))] + 1)
        for dim, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(dim, tag)
        for _, tag in gmsh.model.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.recombine()

        tags, flat_coords, _ = gmsh.model.mesh.getNodes()
        types, _, node_tags = gmsh.model.mesh.getElements(3)
        types = list(types)
        if _HEX8 not in types:
            raise RuntimeError(
                "Gmsh did not produce hexahedra; this project's solver has no "
                "other element and cannot consume the result")
        if len(types) > 1:
            raise RuntimeError(
                f"Gmsh produced mixed element types {types}; only a pure "
                f"hexahedral mesh can be consumed")
        coords = np.asarray(flat_coords, dtype=np.float64).reshape(-1, 3)
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        raw = np.asarray(node_tags[types.index(_HEX8)]).reshape(-1, 8)
        connectivity = np.vectorize(lookup.get)(raw)

    cell = np.array([length_m / nx, height_m / ny, width_m / nz])
    ordered = np.zeros_like(connectivity)
    for e in range(connectivity.shape[0]):
        points = coords[connectivity[e]]
        corner = points.min(axis=0)
        for slot, offset in enumerate(NODE_OFFSETS):
            distance = np.abs(points - (corner + offset * cell)).max(axis=1)
            nearest = int(np.argmin(distance))
            if distance[nearest] > 1e-9 * float(cell.min()):
                raise RuntimeError(
                    f"element {e} is not an axis-aligned box of the expected "
                    f"size; this project's solver cannot consume it")
            ordered[e, slot] = connectivity[e][nearest]

    return Mesh(nx=nx, ny=ny, nz=nz, dx=float(cell[0]), dy=float(cell[1]),
                dz=float(cell[2]), connectivity=ordered, node_coords=coords,
                origin=np.zeros(3))


@dataclass(frozen=True)
class VolumeCheck:
    """An independently measured volume against the one this project used."""

    exact_m3: float
    meshed_m3: float

    @property
    def relative_error(self) -> float:
        return (self.meshed_m3 - self.exact_m3) / self.exact_m3


def hollow_rectangle_volume(length_m: float, outer_height_m: float,
                            outer_width_m: float, wall_thickness_m: float
                            ) -> float:
    """Volume of a hollow rectangular prism, from the OpenCASCADE kernel."""
    with _session("hollow_rectangle") as gmsh:
        outer = gmsh.model.occ.addBox(0.0, 0.0, 0.0, length_m, outer_height_m,
                                      outer_width_m)
        cavity = gmsh.model.occ.addBox(
            0.0, wall_thickness_m, wall_thickness_m, length_m,
            outer_height_m - 2.0 * wall_thickness_m,
            outer_width_m - 2.0 * wall_thickness_m)
        gmsh.model.occ.cut([(3, outer)], [(3, cavity)])
        gmsh.model.occ.synchronize()
        return float(sum(gmsh.model.occ.getMass(3, tag)
                         for _, tag in gmsh.model.getEntities(3)))


def l_bracket_volume(size_m: float, thickness_m: float, width_m: float
                     ) -> float:
    """Volume of an L bracket of the GIVEN arm thickness, from OpenCASCADE.

    Takes the thickness in metres rather than as a fraction, deliberately.
    `l_bracket_mesh` takes a fraction and then rounds it to a whole number of
    cells, so asking this function for the fraction would reproduce the very
    rounding it exists to measure.
    """
    with _session("l_bracket") as gmsh:
        outer = gmsh.model.occ.addBox(0.0, 0.0, 0.0, size_m, size_m, width_m)
        cut = gmsh.model.occ.addBox(thickness_m, thickness_m, 0.0,
                                    size_m - thickness_m,
                                    size_m - thickness_m, width_m)
        gmsh.model.occ.cut([(3, outer)], [(3, cut)])
        gmsh.model.occ.synchronize()
        return float(sum(gmsh.model.occ.getMass(3, tag)
                         for _, tag in gmsh.model.getEntities(3)))


def realised_l_bracket_thickness(size_m: float, thickness_fraction: float,
                                 n: int) -> float:
    """The arm thickness `l_bracket_mesh` will actually produce.

    Delegates rather than reimplementing the rounding. Two copies of this rule
    would eventually disagree, and the disagreement would look exactly like
    the meshing error this node exists to detect.
    """
    from physics.fem.mesh import realised_arm_thickness

    return realised_arm_thickness(size_m, thickness_fraction, n)


def check_mesh_volume(mesh: Mesh, exact_m3: float) -> VolumeCheck:
    """Compare a mesh's material volume against an independent exact figure."""
    return VolumeCheck(exact_m3=exact_m3,
                       meshed_m3=mesh.n_elements * mesh.element_volume)


#: Gmsh's element type numbers for the two tetrahedra worth using.
_TET4 = 4
_TET10 = 11

#: Gmsh lists a ten node tetrahedron's mid-edge nodes in a different order
#: from CalculiX. Gmsh gives corners then edges (0,1) (1,2) (0,2) (0,3) (2,3)
#: (1,3); CalculiX C3D10 wants mid(1,2) mid(2,3) mid(3,1) mid(1,4) mid(2,4)
#: mid(3,4). Written out rather than assumed, and checked by a patch test,
#: because a wrong permutation yields a twisted element that still solves.
GMSH_TET10_TO_CALCULIX = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)


@dataclass(frozen=True)
class TetMesh:
    """An unstructured tetrahedral mesh.

    THIS PROJECT'S SOLVER CANNOT RUN THIS. It is defined here, beside the
    thing that produces it, so that fact stays attached: the Warp solver is an
    eight node hexahedron on a structured uniform grid and has no tetrahedral
    element at all. A mesh of this kind exists to be handed to CalculiX, which
    is the whole point of the general shape route.
    """

    node_coords: np.ndarray            # (n_nodes, 3) in metres
    connectivity: np.ndarray           # (n_elements, 4) or (n_elements, 10)

    @property
    def n_nodes(self) -> int:
        return int(self.node_coords.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.connectivity.shape[0])

    @property
    def nodes_per_element(self) -> int:
        return int(self.connectivity.shape[1])

    @property
    def is_quadratic(self) -> bool:
        return self.nodes_per_element == 10

    def nodes_at_x(self, x: float, tol: float = 1e-9) -> np.ndarray:
        return np.flatnonzero(np.abs(self.node_coords[:, 0] - x) <= tol)

    def nodes_at_extreme(self, axis: int = 0, side: str = "min",
                         tol: float = 1e-9) -> np.ndarray:
        """Nodes on the extreme face along an axis, wherever the part sits.

        A CAD file places its part wherever the author left it. This project's
        own meshes start at the origin, so code written against them tends to
        assume x=0 is the root; a STEP solid centred on the origin then yields
        an empty selection and a boundary condition that was never applied.
        Selecting by extent rather than by coordinate removes the assumption.
        """
        if side not in ("min", "max"):
            raise ValueError(f"side must be min or max, got {side!r}")
        column = self.node_coords[:, axis]
        target = column.min() if side == "min" else column.max()
        return np.flatnonzero(np.abs(column - target) <= tol)

    def volume_m3(self) -> float:
        """Summed tetrahedron volume, from the corner nodes only.

        For a quadratic mesh this uses the four corners and therefore ignores
        curved faces, which is exact for the straight sided elements Gmsh
        produces on a polyhedral domain and an approximation on a curved one.
        """
        corners = self.node_coords[self.connectivity[:, :4]]
        a = corners[:, 1] - corners[:, 0]
        b = corners[:, 2] - corners[:, 0]
        c = corners[:, 3] - corners[:, 0]
        return float(np.abs(np.einsum("ij,ij->i",
                                      np.cross(a, b), c)).sum() / 6.0)


#: Points Gmsh places around a full circle of curvature when curvature
#: sizing is asked for. Twelve gives elements about half a radius long on a
#: cylinder. Off by default: measured on a two hole plate it doubled the node
#: count and moved the deflection by 0.6 percent, and on that same plate it
#: produced an inverted element that the global size alone did not.
DEFAULT_POINTS_PER_CIRCLE = 12


def tetrahedral_mesh_from_step(step_path: str, target_size_m: float,
                               order: int = 2,
                               points_per_circle: int | None = None) -> TetMesh:
    """Mesh a STEP solid with tetrahedra, for a shape this project cannot build.

    This is the entry point that makes CAD analysable here: Gmsh imports the
    B-rep through its own OpenCASCADE, meshes it, and the result goes to the
    CalculiX general shape capability. There is no route through the Warp
    solver, which has no tetrahedral element.

    A caveat that matters for anything curved: Gmsh reads the file in ITS
    units, and STEP is conventionally millimetres. The mesh therefore comes
    back in the file's units and is scaled to metres here using the same
    declaration the analyzer reads, rather than a guess.

    Second order tetrahedra on a curved face can INVERT: the mid-edge nodes
    are pushed onto the surface and an element folds over, CalculiX reports
    a nonpositive Jacobian and writes no result. Measured on a stepped shaft
    of radius 12.8 mm at a 9.9 mm target, which solved at 6.7 mm. Two
    remedies were measured and neither is in this function. Gmsh's high
    order optimiser fixed both known cases and then, on another part,
    threw a C++ exception that terminated the Python process, which no
    caller can catch; a mesher that can kill its host is not a default.
    Curvature sizing (`points_per_circle`) fixed the shaft, doubled a plate's
    node count for a 0.6 percent change, and inverted an element on that
    plate that the global size alone had not. So this function meshes at
    the size it is given, the caller checks whether the solver accepted it,
    and the labeller retries finer when it did not. The floor is a tenth of
    the target so that curvature sizing, when asked for, can refine.
    """
    from .step_analyzer import read_length_unit_m

    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if target_size_m <= 0.0:
        raise ValueError("target element size must be positive")

    unit = read_length_unit_m(step_path)
    size_in_file_units = target_size_m / unit

    with _session("step_tetrahedra") as gmsh:
        imported = gmsh.model.occ.importShapes(str(step_path))
        if not imported:
            raise RuntimeError(f"Gmsh imported no shapes from {step_path}")
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_in_file_units)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_in_file_units * 0.1)
        if points_per_circle is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",
                                  float(points_per_circle))
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        tags, flat, _ = gmsh.model.mesh.getNodes()
        types, _, node_tags = gmsh.model.mesh.getElements(3)
        types = list(types)
        wanted = _TET10 if order == 2 else _TET4
        if wanted not in types:
            raise RuntimeError(
                f"Gmsh produced element types {types} for {step_path}, not "
                f"the tetrahedra asked for")
        coords = np.asarray(flat, dtype=np.float64).reshape(-1, 3) * unit
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        per = 10 if order == 2 else 4
        raw = np.asarray(node_tags[types.index(wanted)]).reshape(-1, per)
        connectivity = np.vectorize(lookup.get)(raw)

    if order == 2:
        connectivity = connectivity[:, list(GMSH_TET10_TO_CALCULIX)]
    return TetMesh(node_coords=coords, connectivity=connectivity)


def tetrahedral_box_mesh(length_m: float, height_m: float, width_m: float,
                         target_size_m: float, order: int = 2) -> TetMesh:
    """Mesh a box with tetrahedra, for the route this project cannot take.

    A box is chosen for the FIRST general shape deliberately: it is the only
    domain both meshers can cover, so the tetrahedral answer can be checked
    against the structured hexahedral one instead of against nothing.

    `order=2` gives ten node tetrahedra and is the default for a reason.
    Linear tetrahedra are notoriously stiff in bending, and using them on a
    cantilever would produce a confidently wrong deflection.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if target_size_m <= 0.0:
        raise ValueError("target element size must be positive")

    with _session("tetrahedral_box") as gmsh:
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, length_m, height_m, width_m)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size_m * 0.25)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        tags, flat, _ = gmsh.model.mesh.getNodes()
        types, _, node_tags = gmsh.model.mesh.getElements(3)
        types = list(types)
        wanted = _TET10 if order == 2 else _TET4
        if wanted not in types:
            raise RuntimeError(
                f"Gmsh produced element types {types}, not the tetrahedra "
                f"asked for")
        coords = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        per = 10 if order == 2 else 4
        raw = np.asarray(node_tags[types.index(wanted)]).reshape(-1, per)
        connectivity = np.vectorize(lookup.get)(raw)

    if order == 2:
        connectivity = connectivity[:, list(GMSH_TET10_TO_CALCULIX)]
    return TetMesh(node_coords=coords, connectivity=connectivity)
