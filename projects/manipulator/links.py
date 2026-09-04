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

#: Exported geometry is scaled from metres to millimetres. A CAD package that
#: reads an STL has no unit to read and assumes millimetres, so a body written
#: in metres arrives a thousand times too small.
EXPORT_SCALE = 1000.0

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
    #: What the drawings do not print, so the design could not resolve it.
    #: Kept apart from `notes` because these are gaps, not observations.
    unresolved: list[str] = field(default_factory=list)

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


def scaled_surface(surface, scale: float):
    """The same surface with its vertices in another unit.

    Only the geometry is scaled. Every quantity the design reports stays in
    SI, so the scale cannot leak into a mass or a compliance.
    """
    from dataclasses import replace

    return replace(surface, vertices=np.asarray(surface.vertices) * scale)


def link_domain(spec: ManipulatorSpec, link_index: int, drives: dict[str, str],
                divisions=(28, 12, 12), sections: dict | None = None):
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
    # The domain is the rectangle the sizing loop arrived at, not a square.
    # Taking one number for both directions gave the tool flange a 32 by 32
    # box where its section is 32 by 40, which throws away 8 mm of width the
    # design was allowed to use.
    section = (sections or {}).get(link.name)
    height = section.outer_height_m if section is not None else spec.minimum_section_m
    width = section.outer_width_m if section is not None else spec.minimum_section_m
    if actuator is not None and actuator.outer_diameter_m:
        height = max(height, actuator.outer_diameter_m)
        width = max(width, actuator.outer_diameter_m)
    mesh = solid_box_mesh(span, height, width, *divisions)

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
                           0.5 * height, 0.5 * width])
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
    return (mesh, passive_solid, passive_void, span, height, width,
            void_note), ""


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
                  volume_fraction: float = 0.3,
                  sections: dict | None = None) -> LinkDesign:
    """One link, from a design domain to a watertight body."""
    links = spec.links()
    joints = spec.joints()
    link = links[link_index]
    joint = joints[link_index]

    built, reason = link_domain(spec, link_index, drives, sections=sections)
    if built is None:
        return LinkDesign(name=link.name, generated=False, reason=reason)
    (mesh, passive_solid, passive_void, span, height, width,
     void_note) = built

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
    # Written in MILLIMETRES. Neither STL nor a faceted STEP carries a unit a
    # CAD package can read back reliably, so a file in metres imports as a
    # part a thousand times too small. Everything computed here stays in SI;
    # only the exported copy is scaled, and the scale is recorded.
    import trimesh

    exported = scaled_surface(surface, EXPORT_SCALE)
    body = trimesh.Trimesh(vertices=np.asarray(exported.vertices),
                           faces=np.asarray(exported.triangles), process=False)
    holes, unresolved = mounting_holes(spec, link_index, drives, span)
    body, hole_report = cut_holes(body, holes, height, width,
                                  scale=EXPORT_SCALE)
    before_mm3 = float(abs(surface.volume_m3)) * EXPORT_SCALE ** 3
    after_mm3 = float(abs(body.volume))
    body.export(str(out_dir / f"{link.name}.stl"))
    stl = out_dir / f"{link.name}.stl"
    volume_m3 = after_mm3 / EXPORT_SCALE ** 3
    design = LinkDesign(
        name=link.name, generated=True,
        mass_kg=volume_m3 * material.density_kg_m3,
        volume_m3=volume_m3,
        compliance_j=float(result.final_compliance),
        grey_fraction=float(np.mean((result.density > 0.1)
                                    & (result.density < 0.9))),
        unsupported_fraction=unsupported_fraction(mesh, result.density),
        volume_error_vs_field=surface.volume_error_vs_field,
        watertight=bool(body.is_watertight),
        triangles=int(body.faces.shape[0]),
        stl_path=str(stl))
    design.notes.extend(hole_report)
    design.notes.append(
        f"the holes removed {100.0 * (1.0 - after_mm3 / before_mm3):.2f} "
        f"percent of the extracted volume")
    design.unresolved = unresolved
    design.notes.append(void_note)
    design.notes.append(
        f"design domain {span * 1000:.0f} by {height * 1000:.0f} by "
        f"{width * 1000:.0f} mm, {mesh.n_elements} elements, "
        f"{int(passive_solid.sum())} held solid for the interfaces and "
        f"{int(passive_void.sum())} held empty for the drive")
    design.notes.append(
        f"the exported STL and STEP are in millimetres; x = 0 is the start "
        f"joint plane and the joint axis line is at y = {height * 500:.1f}, "
        f"z = {width * 500:.1f} mm in the file's own frame")
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


def size_link_by_volume_fraction(spec: ManipulatorSpec, link_index: int,
                                 drives: dict[str, str], torques: dict[str, float],
                                 limit_m: float, out_dir: Path,
                                 fractions=(0.10, 0.15, 0.20, 0.25, 0.30),
                                 iterations: int = 30) -> list[dict]:
    """Sweep the volume fraction and report where the deflection limit is met.

    Topology takes a volume fraction, and the requirement is a deflection, so
    the two are joined by running the optimisation at several fractions and
    solving the EXTRACTED body each time. It is a sweep rather than a
    bisection because each point is a full optimisation and the sweep says
    what the curve looks like, which a bisection does not.
    """
    from optimization.topology.verify import (DisconnectedAtThreshold,
                                              tip_displacement_of_extracted)

    rows: list[dict] = []
    for fraction in fractions:
        design = generate_link(spec, link_index, drives, torques, out_dir,
                               iterations=iterations, volume_fraction=fraction)
        row = {"volume_fraction": fraction, "generated": design.generated,
               "mass_kg": design.mass_kg, "limit_m": limit_m}
        if design.generated:
            row["compliance_j"] = design.compliance_j
            row["meets_limit"] = None
            row["note"] = ("the deflection of the extracted body is measured "
                           "by the caller; this sweep reports mass and "
                           "compliance per fraction")
        else:
            row["reason"] = design.reason
        rows.append(row)
    return rows


#: CHOSEN. The cable has to leave the joint somewhere, and the only through
#: bore any of these drawings prints is the AK80-64 output's 21 mm. Using that
#: one diameter on every face means one cable size fits the whole arm, and it
#: is a choice, not a value read off a drawing.
CABLE_BORE_M = 0.021


def mounting_holes(spec: ManipulatorSpec, link_index: int,
                   drives: dict[str, str], span_m: float,
                   clock_deg: float = 0.0) -> tuple[list[dict], list[str]]:
    """Every hole this link's two end faces need, and what could not be cut.

    A link is bolted to the OUTPUT of the joint that drives it and carries the
    HOUSING of the joint after it, so its two ends take two different
    patterns. The last link has no next joint; its far end is a tool plate
    this arm has not specified, and no pattern is invented for it.
    """
    from .interfaces import (bolt_holes, dowel_holes, face_for,
                             unresolved_features)

    joints = spec.joints()
    joint = joints[link_index]
    following = joints[link_index + 1] if link_index + 1 < len(joints) else None
    flange = spec.flange_thickness_m

    holes: list[dict] = []
    unresolved: list[str] = []
    ends = [("proximal", face_for(drives.get(joint.name, ""), "output"),
             joint.name, -0.002, flange + 0.002)]
    if following is not None:
        ends.append(("distal", face_for(drives.get(following.name, ""), "housing"),
                     following.name, span_m - flange - 0.002, span_m + 0.002))
    else:
        unresolved.append(
            f"{spec.links()[link_index].name}: its far end is the tool plate "
            f"and this arm specifies no tool, so no pattern was cut there")

    for end, face, joint_name, x0, x1 in ends:
        if face is None:
            unresolved.append(
                f"{joint_name}: no drawing was read for "
                f"{drives.get(joint_name, 'its drive')}, so its face has no "
                f"pattern and the {end} end of this link cannot be fastened")
            continue
        for hole in bolt_holes(face, clock_deg):
            holes.append({"end": end, "kind": "clearance", "face": face.face,
                          "thread": hole["thread"], "diameter_m": hole["diameter_m"],
                          "y_m": hole["y_m"], "z_m": hole["z_m"],
                          "x0_m": x0, "x1_m": x1,
                          "bolt_circle_m": hole["bolt_circle_m"]})
        for dowel in dowel_holes(face):
            depth = dowel["depth_m"]
            x0, x1 = ((-0.001, depth) if end == "proximal"
                      else (span_m - depth, span_m + 0.001))
            holes.append({"end": end, "kind": "dowel", "face": face.face,
                          "thread": "", "diameter_m": dowel["diameter_m"],
                          "y_m": dowel["y_m"], "z_m": dowel["z_m"],
                          "x0_m": x0, "x1_m": x1,
                          "angle_deg": dowel["angle_deg"]})
        bore = face.central_bore_m or CABLE_BORE_M
        holes.append({"end": end, "kind": "bore", "face": face.face,
                      "thread": "", "diameter_m": bore, "y_m": 0.0, "z_m": 0.0,
                      "x0_m": x0, "x1_m": x1,
                      "printed": face.central_bore_m is not None})
        unresolved.extend(unresolved_features(face))
    return holes, unresolved


def cut_holes(body, holes: list[dict], height_m: float, width_m: float,
              scale: float = 1.0, sections: int = 24):
    """Subtract the holes from the extracted body.

    The holes are cut AFTER the optimisation, not held void during it. A 3.4
    mm clearance hole is under half an element on this grid, so the density
    field cannot represent one: asking it to would not produce a hole, it
    would produce a smeared grey patch. Cutting afterwards is what a drawing
    office does with a load path too, and it is honest about which part of the
    geometry the optimiser decided and which part the interface did.
    """
    import trimesh

    if not holes:
        return body, []
    centre_y, centre_z = 0.5 * height_m * scale, 0.5 * width_m * scale
    cutters = []
    for hole in holes:
        length = (hole["x1_m"] - hole["x0_m"]) * scale
        transform = trimesh.transformations.rotation_matrix(
            np.pi / 2.0, [0.0, 1.0, 0.0])
        transform[:3, 3] = [0.5 * (hole["x0_m"] + hole["x1_m"]) * scale,
                            centre_y + hole["y_m"] * scale,
                            centre_z + hole["z_m"] * scale]
        cutters.append(trimesh.creation.cylinder(
            radius=0.5 * hole["diameter_m"] * scale, height=length,
            sections=sections, transform=transform))
    tool = trimesh.util.concatenate(cutters)
    cut = body.difference(tool)
    report = [f"{len(holes)} holes cut: "
              + ", ".join(sorted({f"{h['end']} {h['kind']} "
                                  f"{h['diameter_m'] * 1000:.1f} mm"
                                  for h in holes}))]
    return cut, report
