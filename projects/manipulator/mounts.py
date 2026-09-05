"""The two parts that hold the arm to the world, which it did not have.

An arm is not six links. Its first joint's HOUSING has to be held by
something bolted to the floor, and its last joint's output has to hold a
tool. Neither existed here, and both were listed as gaps for several
revisions. A gap that is listed is honest; a gap that stays listed is
unfinished. These are designed the same way the links are, from the same
topology path, against loads that are computed rather than assumed.

WHAT IS SOURCED AND WHAT IS CHOSEN
==================================
The actuator side of each part is sourced: the base mount bolts to the
AK80-64's housing pattern and the tool plate to the AK60-6's output pattern,
both read off manufacturer drawings. The WORLD side of each is a choice,
because nothing in this specification says what the arm is bolted to or what
tool it holds. Those choices are marked CHOSEN and are the first thing a real
installation would replace.

THE LOADS ARE THE ARM'S WORST CASE, NOT A GUESS
===============================================
The base mount carries the whole arm stretched out: the weight of everything
above it and the overturning moment that weight makes at 600 mm of reach,
plus the base yaw's own reaction torque. The tool plate carries the payload
at the end of its own extent and the torque needed to spin it. Both come from
the same dynamics the drives were chosen against.
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
from physics.fem.mesh import solid_box_mesh

from .interfaces import ISO_273_MEDIUM_M, bolt_holes, face_for
from .links import (EXPORT_SCALE, ISO_LEVEL, clip_to_domain,
                    cut_holes, scaled_surface)
from .spec import SPEC, ManipulatorSpec

GRAVITY = 9.80665

#: CHOSEN. The pattern the base mount presents to the floor. Nothing in this
#: specification says what the arm stands on, so this is a plain four bolt
#: square of a size that suits an optical table or a welded frame. It is the
#: first thing a real installation replaces.
FLOOR_BOLT_SQUARE_M = 0.120
FLOOR_BOLT_THREAD = "M8"
FLOOR_CLEARANCE_M = 0.009        # ISO 273 medium for M8

#: CHOSEN. The pattern the tool plate presents to a tool, on the same basis.
TOOL_BOLT_SQUARE_M = 0.050
TOOL_BOLT_THREAD = "M5"
TOOL_CLEARANCE_M = 0.0055        # ISO 273 medium for M5


@dataclass
class MountDesign:
    name: str
    generated: bool
    reason: str = ""
    mass_kg: float | None = None
    volume_m3: float | None = None
    compliance_j: float | None = None
    unsupported_fraction: float | None = None
    watertight: bool | None = None
    triangles: int | None = None
    stl_path: str | None = None
    cross_evaluation: list = field(default_factory=list)
    loads: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def row(self) -> dict:
        return {"part": self.name, "generated": self.generated,
                "mass_kg": self.mass_kg, "compliance_j": self.compliance_j,
                "unsupported": self.unsupported_fraction,
                "watertight": self.watertight, "triangles": self.triangles,
                "reason": self.reason}


def base_mount_loads(arm_mass_kg: float, payload_kg: float,
                     base_yaw_peak_nm: float,
                     spec: ManipulatorSpec = SPEC) -> dict:
    """What the floor has to take, with the arm straight out.

    The overturning moment is the case that sizes this part, and it is much
    larger than the weight: the payload alone at 600 mm makes 17.6 N m, and
    the arm's own mass acting at roughly half its reach adds its share. The
    base yaw reaction is the torque the drive applies to turn everything,
    which the mount has to hold against.
    """
    weight_n = (arm_mass_kg + payload_kg) * GRAVITY
    payload_moment = payload_kg * GRAVITY * spec.reach_m
    arm_moment = arm_mass_kg * GRAVITY * 0.5 * spec.reach_m
    return {"vertical_n": weight_n,
            "overturning_nm": payload_moment + arm_moment,
            "yaw_reaction_nm": base_yaw_peak_nm,
            "basis": (f"the arm stretched to {spec.reach_m * 1000:.0f} mm, "
                      f"payload at the tip and the arm's own mass at half "
                      f"the reach, plus the base yaw's peak torque as a "
                      f"reaction the mount holds against")}


def tool_plate_loads(payload_kg: float, spin_nm: float,
                     spec: ManipulatorSpec = SPEC) -> dict:
    """What the tool plate takes: the payload hung off its own extent."""
    weight_n = payload_kg * GRAVITY
    return {"vertical_n": weight_n,
            "overturning_nm": weight_n * 0.5 * spec.payload_extent_m,
            "yaw_reaction_nm": spin_nm,
            "basis": (f"{payload_kg} kg acting at half of its stated "
                      f"{spec.payload_extent_m * 1000:.0f} mm extent, plus "
                      f"the torque that turns it")}


def _plate_domain(length_m: float, height_m: float, width_m: float,
                  plate_m: float, divisions=(20, 20, 20)):
    """A box whose two large faces are held solid because bolts go through
    them, and whose middle the optimiser may shape."""
    mesh = solid_box_mesh(length_m, height_m, width_m, *divisions)
    centroids = mesh.element_centroids()
    passive_solid = ((centroids[:, 0] <= plate_m)
                     | (centroids[:, 0] >= length_m - plate_m))
    return mesh, passive_solid


def _mount_cases(mesh, loads: dict, height_m: float, width_m: float
                 ) -> list[LoadCase]:
    """Three cases: the weight, the overturning moment, and the reaction
    torque about the mount's own axis. A part designed for one of these is
    between two and nine times worse under another, measured in
    docs/topology_design.md, so all three are carried."""
    tip = mesh.nodes_at_x(float(mesh.nx * mesh.dx))
    return [
        LoadCase("the weight it holds up", tip,
                 total_load_n=-loads["vertical_n"], load_direction=1,
                 weight=1.0),
        # The overturning moment is a couple about an axis ACROSS the mount,
        # which is the case that sizes it: the payload alone at full reach
        # makes seventeen newton metres against seventy six newtons of weight.
        LoadCase("the overturning moment", tip,
                 force_vector=couple_force_vector(
                     mesh, tip, loads["overturning_nm"], axis=2),
                 weight=1.0),
        # And the reaction torque about the mount's own axis, which is what
        # the drive pushes against when it turns the arm.
        LoadCase("the reaction torque about its own axis", tip,
                 force_vector=couple_force_vector(
                     mesh, tip, max(loads["yaw_reaction_nm"], 1e-3), axis=0),
                 weight=1.0)]


def _world_holes(kind: str, at_x: float, thickness_m: float,
                 height_m: float, width_m: float) -> list[dict]:
    """The four bolts on the world side, on a square, CHOSEN not sourced.

    The endpoints are in the PART's frame, which runs from 0 to its length,
    height and width, so the pattern is centred by adding half of each. The
    holes used to be described by a y, a z and two x values and the cutter no
    longer reads those.
    """
    if kind == "floor":
        half, diameter, thread = (0.5 * FLOOR_BOLT_SQUARE_M,
                                  FLOOR_CLEARANCE_M, FLOOR_BOLT_THREAD)
    else:
        half, diameter, thread = (0.5 * TOOL_BOLT_SQUARE_M,
                                  TOOL_CLEARANCE_M, TOOL_BOLT_THREAD)
    holes = []
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            centre_y = 0.5 * height_m + sy * half
            centre_z = 0.5 * width_m + sz * half
            holes.append({
                "end": kind, "kind": "clearance", "face": kind,
                "thread": thread, "diameter_m": diameter,
                "start_m": [at_x - thickness_m - 0.002, centre_y, centre_z],
                "end_m": [at_x + 0.002, centre_y, centre_z],
                "y_m": sy * half, "z_m": sz * half,
                "bolt_circle_m": None})
    return holes


def generate_mount(name: str, actuator_id: str, face_name: str,
                   world_side: str, loads: dict, length_m: float,
                   height_m: float, width_m: float, out_dir: Path,
                   spec: ManipulatorSpec = SPEC, iterations: int = 60,
                   volume_fraction: float = 0.3) -> MountDesign:
    """One mount, from its two interfaces and its loads to a watertight body.

    x = 0 is the ACTUATOR face and x = length is the WORLD face, so the part
    reads the same way a link does and the same exporter writes it.
    """
    face = face_for(actuator_id, face_name)
    if face is None:
        return MountDesign(
            name=name, generated=False,
            reason=(f"{name}: no drawing was read for {actuator_id}, so the "
                    f"pattern it bolts to is unknown and this part cannot be "
                    f"designed"))

    plate = spec.flange_thickness_m
    mesh, passive_solid = _plate_domain(length_m, height_m, width_m, plate)
    material = get_material(spec.materials["link"])
    projection, vjp = support_projection_with_gradient(mesh, build_axis=1)
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        fixed_nodes=mesh.nodes_at_x(0.0),
        load_nodes=mesh.nodes_at_x(float(mesh.nx * mesh.dx)),
        total_load_n=-loads["vertical_n"], load_direction=1,
        volume_fraction=volume_fraction, filter_radius_elements=2.0,
        passive_solid=passive_solid,
        density_projection=projection, projection_vjp=vjp)
    cases = _mount_cases(mesh, loads, height_m, width_m)
    result = optimize_multiload(problem, cases, max_iterations=iterations)

    from optimization.topology.export import largest_connected_component

    kept = largest_connected_component(mesh, result.density, ISO_LEVEL)
    lost = int((passive_solid & (kept < ISO_LEVEL)).sum())
    if lost:
        return MountDesign(
            name=name, generated=False,
            reason=(f"{name}: the extraction dropped {lost} of "
                    f"{int(passive_solid.sum())} elements held solid for its "
                    f"two bolted faces, so one of its interfaces is missing"))

    surface = marching_surface(mesh, kept, ISO_LEVEL, smoothing_iterations=10)
    out_dir.mkdir(parents=True, exist_ok=True)

    import trimesh

    exported = scaled_surface(surface, EXPORT_SCALE)
    body = trimesh.Trimesh(vertices=np.asarray(exported.vertices),
                           faces=np.asarray(exported.triangles), process=False)
    body, clipped = clip_to_domain(body, length_m, height_m, width_m,
                                   EXPORT_SCALE)
    holes = [{"end": "actuator", "kind": "clearance", "face": face.face,
              "thread": h["thread"], "diameter_m": h["diameter_m"],
              "start_m": [-0.002, 0.5 * height_m + h["y_m"],
                          0.5 * width_m + h["z_m"]],
              "end_m": [plate + 0.002, 0.5 * height_m + h["y_m"],
                        0.5 * width_m + h["z_m"]],
              "y_m": h["y_m"], "z_m": h["z_m"],
              "bolt_circle_m": h["bolt_circle_m"]}
             for h in bolt_holes(face)]
    holes += _world_holes(world_side, length_m, plate, height_m, width_m)
    body, report = cut_holes(body, holes, scale=EXPORT_SCALE)
    body.export(str(out_dir / f"{name}.stl"))

    volume_m3 = float(abs(body.volume)) / EXPORT_SCALE ** 3
    design = MountDesign(
        name=name, generated=True, mass_kg=volume_m3 * material.density_kg_m3,
        volume_m3=volume_m3, compliance_j=float(result.final_compliance),
        unsupported_fraction=unsupported_fraction(mesh, result.density),
        watertight=bool(body.is_watertight),
        triangles=int(body.faces.shape[0]),
        stl_path=str(out_dir / f"{name}.stl"), loads=dict(loads))
    design.notes.extend(report)
    design.notes.append(clipped)
    design.notes.append(
        f"x = 0 is the {face.actuator} {face.face} face, "
        f"{face.patterns[0].count} by {face.patterns[0].thread} on a "
        f"{face.largest_bolt_circle_m() * 1000:.0f} mm circle clocked "
        f"{face.patterns[0].clock_deg} degrees")
    world_thread = (FLOOR_BOLT_THREAD if world_side == "floor"
                    else TOOL_BOLT_THREAD)
    world_square = (FLOOR_BOLT_SQUARE_M if world_side == "floor"
                    else TOOL_BOLT_SQUARE_M)
    design.notes.append(
        f"x = {length_m * 1000:.0f} mm is the {world_side} face, CHOSEN as "
        f"four {world_thread} clearance holes on a "
        f"{world_square * 1000:.0f} mm square. Nothing in this specification "
        f"says what the arm is bolted to, so this pattern is a choice and "
        f"not a source")
    design.notes.append(loads["basis"])
    design.unresolved.append(
        f"{name}: the {world_side} side pattern is CHOSEN. Replace it with "
        f"the real one before anything is made")
    design.cross_evaluation = cross_evaluation(problem, {name: result.density},
                                               cases)
    return design
