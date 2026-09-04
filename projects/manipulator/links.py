"""Generating each link as a shape, not as a box with a section.

The first version of this arm sized a hollow rectangle per link. That is a
section, not a structure: it has no place to bolt an actuator, no hole for a
shaft, and no reason for its material to be where it is. This module runs the
free form path instead, with three things the earlier topology runs did not
have.

THE INTERFACES ARE PART OF THE PROBLEM
======================================
Each link gets a slab of passive solid at both ends, thick enough for the
fastening the features stage demands: a 6.4 mm counterbore has to fit and a
tapped hole in aluminium wants 1.5 diameters, so 9 mm. The optimiser cannot
carve those away and has to route load into them, which is the difference
between a shape and a part.

THE ACTUATOR IS A HOLE, WHERE ITS SIZE IS PUBLISHED
===================================================
A drive occupies space that the structure may not. Where the manufacturer
prints an outline, that volume becomes passive void. Where it does not, no
void can be placed, and the link is reported as ungeneratable rather than
generated around a guess.

THE LOADS ARE THE ARM'S, AND THERE IS MORE THAN ONE
===================================================
A single load case design is between 1.90 and 9.30 times worse under a case it
never saw, measured in docs/topology_design.md. Each link is optimised for the
weighted set of cases its own joint actually sees: the transverse load in both
directions and the torsion its drive applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.materials import get_material
from optimization.topology import SimpProblem
from optimization.topology.manufacturing import (support_projection_with_gradient,
                                                 unsupported_fraction)
from optimization.topology.multiload import (LoadCase, couple_force_vector,
                                             cross_evaluation, optimize_multiload)
from optimization.topology.smooth import marching_surface, write_stl
from optimization.topology.verify import elements_touching
from physics.fem.mesh import solid_box_mesh

from .spec import SPEC, ManipulatorSpec

#: Iso level for the extraction. 0.5 is the middle of the three the smoothing
#: study measured (+24.6, -7.2 and -36.0 percent volume error against the
#: density field at 0.3, 0.5 and 0.7), and the one whose error is smallest.
ISO_LEVEL = 0.5

#: The process these parts would be made by, and why. They are organic bodies
#: from a density field: a mill cannot reach inside them, and casting needs a
#: draft this shape does not have. Laser powder bed fusion can make them, so
#: the optimisation carries ITS constraint, the support filter, rather than
#: scoring the result against a process it was not designed for.
PROCESS = "slm"


@dataclass
class LinkDesign:
    name: str
    generated: bool
    reason: str = ""
    mass_kg: float | None = None
    volume_m3: float | None = None
    compliance_j: float | None = None
    grey_fraction: float | None = None
    unsupported_fraction: float | None = None
    volume_error_vs_field: float | None = None
    watertight: bool | None = None
    triangles: int | None = None
    stl_path: str | None = None
    step_path: str | None = None
    cross_evaluation: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def row(self) -> dict:
        return {"link": self.name, "generated": self.generated,
                "mass_kg": self.mass_kg, "compliance_j": self.compliance_j,
                "grey": self.grey_fraction,
                "unsupported": self.unsupported_fraction,
                "volume_error_vs_field": self.volume_error_vs_field,
                "watertight": self.watertight, "triangles": self.triangles,
                "reason": self.reason}


def actuator_for(joint_name: str, drives: dict[str, str]):
    from drivetrain.sourced import sourced_motor

    part = drives.get(joint_name)
    if not part or "+" in str(part):
        return None
    try:
        return sourced_motor(part)
    except Exception:
        return None


def link_domain(spec: ManipulatorSpec, link_index: int, drives: dict[str, str],
                divisions=(28, 12, 12)):
    """The box a link may occupy, with its interfaces and actuator pockets.

    Returns (mesh, passive_solid, passive_void, extents) or a reason it cannot
    be built.
    """
    joints = spec.joints()
    links = spec.links()
    link = links[link_index]
    joint = joints[link_index]
    following = joints[link_index + 1] if link_index + 1 < len(joints) else None

    # The span is whichever offset the next joint carries. The base column
    # runs UP, not along the arm, so its span is the next joint's y offset;
    # taking the x offset alone reported it as having no room to design in.
    if following is None:
        span = link.length_m
    else:
        span = max(following.origin_x_m, following.origin_y_m)
    if span <= 0.0:
        return None, (f"{link.name}: its joints are coincident in every "
                      f"direction, so there is no space to design in")

    actuator = actuator_for(joint.name, drives)
    across = spec.minimum_section_m
    if actuator is not None and actuator.outer_diameter_m:
        across = max(across, actuator.outer_diameter_m)
    mesh = solid_box_mesh(span, across, across, *divisions)

    centroids = mesh.element_centroids()
    flange = spec.flange_thickness_m
    passive_solid = ((centroids[:, 0] <= flange)
                     | (centroids[:, 0] >= span - flange))

    passive_void = np.zeros(mesh.n_elements, dtype=bool)
    void_note = "no actuator outline is printed, so no pocket was cut"
    if actuator is not None and actuator.outer_diameter_m and actuator.axial_length_m:
        axis = np.asarray(joint.axis, dtype=float)
        radius = 0.5 * actuator.outer_diameter_m
        half_length = 0.5 * actuator.axial_length_m
        centre = np.array([min(flange + half_length, span - flange),
                           0.5 * across, 0.5 * across])
        offset = centroids - centre
        if abs(float(axis[0])) > 0.5:              # the drive lies along the arm
            along = offset[:, 0]
            radial = np.linalg.norm(offset[:, 1:], axis=1)
        else:                                      # across the arm
            along = offset[:, 2]
            radial = np.linalg.norm(offset[:, :2], axis=1)
        passive_void = (np.abs(along) <= half_length) & (radial <= radius)
        void_note = (f"a {actuator.outer_diameter_m * 1000:.0f} by "
                     f"{actuator.axial_length_m * 1000:.1f} mm pocket for the "
                     f"{actuator.part_number}")

    free = ~(passive_solid | passive_void)
    if free.sum() < 0.15 * mesh.n_elements:
        return None, (f"{link.name}: the interfaces and the actuator pocket "
                      f"leave {free.sum()} of {mesh.n_elements} elements free, "
                      f"which is no room to design in. The drive fills the link")
    return (mesh, passive_solid, passive_void, span, across, void_note), ""


def link_load_cases(mesh, torque_nm: float, transverse_n: float) -> list[LoadCase]:
    """The cases this link's joint actually sees, weighted equally.

    Two transverse directions and the torsion its own drive applies. A single
    case design is between 1.90 and 9.30 times worse under a case it never
    saw, which is why there are three.
    """
    tip = mesh.nodes_at_x(float(mesh.nx * mesh.dx))
    return [
        LoadCase("bending in the load plane", tip, total_load_n=-transverse_n,
                 load_direction=1),
        LoadCase("bending across it", tip, total_load_n=-transverse_n,
                 load_direction=2),
        LoadCase("torsion from the drive", tip,
                 force_vector=couple_force_vector(mesh, tip, torque_nm)),
    ]


def generate_link(spec: ManipulatorSpec, link_index: int,
                  drives: dict[str, str], torques: dict[str, float],
                  out_dir: Path, iterations: int = 60,
                  volume_fraction: float = 0.3) -> LinkDesign:
    """One link, from a design domain to a watertight body."""
    links = spec.links()
    joints = spec.joints()
    link = links[link_index]
    joint = joints[link_index]

    built, reason = link_domain(spec, link_index, drives)
    if built is None:
        return LinkDesign(name=link.name, generated=False, reason=reason)
    mesh, passive_solid, passive_void, span, across, void_note = built

    material = get_material(spec.materials["link"])
    torque = abs(torques.get(joint.name, 1.0)) or 1.0
    transverse = max(torque / max(span, 1e-6), 10.0)

    projection, vjp = support_projection_with_gradient(mesh, build_axis=1)
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        fixed_nodes=mesh.nodes_at_x(0.0),
        load_nodes=mesh.nodes_at_x(float(mesh.nx * mesh.dx)),
        total_load_n=-transverse, load_direction=1,
        volume_fraction=volume_fraction, filter_radius_elements=2.0,
        passive_solid=passive_solid, passive_void=passive_void,
        density_projection=projection, projection_vjp=vjp)

    cases = link_load_cases(mesh, torque, transverse)
    result = optimize_multiload(problem, cases, max_iterations=iterations)

    from optimization.topology.export import largest_connected_component

    kept = largest_connected_component(mesh, result.density, ISO_LEVEL)
    surface = marching_surface(mesh, kept, ISO_LEVEL, smoothing_iterations=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = write_stl(surface, out_dir / f"{link.name}.stl")
    design = LinkDesign(
        name=link.name, generated=True,
        mass_kg=surface.volume_m3 * material.density_kg_m3,
        volume_m3=surface.volume_m3,
        compliance_j=float(result.final_compliance),
        grey_fraction=float(np.mean((result.density > 0.1)
                                    & (result.density < 0.9))),
        unsupported_fraction=unsupported_fraction(mesh, result.density),
        volume_error_vs_field=surface.volume_error_vs_field,
        watertight=surface.watertight,
        triangles=int(surface.triangles.shape[0]),
        stl_path=str(stl))
    design.notes.append(void_note)
    design.notes.append(
        f"design domain {span * 1000:.0f} by {across * 1000:.0f} by "
        f"{across * 1000:.0f} mm, {mesh.n_elements} elements, "
        f"{int(passive_solid.sum())} held solid for the interfaces and "
        f"{int(passive_void.sum())} held empty for the drive")
    design.cross_evaluation = cross_evaluation(problem, {link.name: result.density},
                                               cases)
    return design


def faceted_step(stl_path: Path, step_path: Path) -> tuple[Path, int]:
    """Sew a triangle body into a solid and write it as STEP.

    THIS IS NOT A CLEAN B-REP, and geometry/cad_export/mesh_fallback.py says
    why at length: a STEP file made of thousands of planar facets is an STL
    wearing another extension. It is written anyway because a CAD system can
    stand it up and measure it, which is what it is for, and the face count is
    returned so nobody mistakes it for a surfaced model.

    build123d's `import_stl` returns ONE face carrying a triangulation, which
    sews into nothing; the facets have to be built as real planar faces first.
    """
    import numpy as np
    import trimesh
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeSolid,
                                    BRepBuilderAPI_Sewing)
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS

    mesh = trimesh.load(str(stl_path))
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.faces, dtype=int)

    sewing = BRepBuilderAPI_Sewing(1e-6)
    for a, b_, c in triangles:
        polygon = BRepBuilderAPI_MakePolygon(
            gp_Pnt(*vertices[a]), gp_Pnt(*vertices[b_]), gp_Pnt(*vertices[c]),
            True)
        face = BRepBuilderAPI_MakeFace(polygon.Wire())
        if face.IsDone():
            sewing.Add(face.Face())
    sewing.Perform()
    sewn = sewing.SewedShape()
    shape = sewn
    try:
        maker = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sewn))
        if maker.IsDone():
            shape = maker.Solid()
    except Exception:
        pass

    step_path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(step_path))
    return step_path, int(triangles.shape[0])
