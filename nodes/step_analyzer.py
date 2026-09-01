"""Read a STEP file and describe what is in it, in this project's terms.

The entry point for CAD that this project did not author. It produces the same
Part Dataset record a parametric design produces, which is what lets a model
trained on generated parts be evaluated on real ones.

WHAT IT READS AND WHAT IT REFUSES TO GUESS
==========================================
The unit is READ FROM THE FILE, not assumed. STEP declares it, as
`SI_UNIT(.MILLI.,.METRE.)` for the millimetres almost everything writes, and
guessing wrong scales a volume by a factor of a billion while leaving every
number plausible. A file that declares no unit this module recognises is
refused rather than defaulted, because the failure it prevents is silent.

VALIDITY DOMAIN
===============
Stated before implementing.

Reads
    Solid geometry: volume, surface area, bounding box, centre of mass, and
    the topological counts. These come from OpenCASCADE's own evaluators, so
    they are as exact as the B-rep is.

Does not read
    Assembly structure, mates, materials, tolerances, GD and T, threads as
    anything other than the cylinders they are drawn as, or design intent of
    any kind. A cylindrical face is a cylindrical face; whether it is a
    clearance hole, a bearing seat or a lightening pocket is not recoverable
    from the geometry, and this module does not pretend to recover it.

A multi-solid file
    The Part Dataset schema describes ONE part. A STEP file holding several
    solids is therefore split into one record per solid, and the caller is
    told how many there were rather than silently handed the first.

Curved geometry
    Volume and area come from OCCT's exact evaluators on the B-rep, so a
    cylinder is a cylinder rather than a facetted approximation. This is not
    the same as the tessellation a mesher will later produce, and the two are
    expected to differ slightly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.part_dataset import (GeometrySummary, PartRecord, Provenance,
                               TopologySummary)

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

STEP_ANALYZER_NODE_NAME = "step.analyzer"
STEP_ANALYZER_CAPABILITY = "analysis.cad.step"

#: STEP spells its length unit in an SI_UNIT entity. Only the ones that mean a
#: length are listed; a file using anything else is refused rather than scaled
#: by a guess.
_UNIT_TO_METRES = {
    ("", "METRE"): 1.0,
    ("MILLI", "METRE"): 1.0e-3,
    ("CENTI", "METRE"): 1.0e-2,
    ("KILO", "METRE"): 1.0e3,
    ("MICRO", "METRE"): 1.0e-6,
}

#: The prefix is either .MILLI. or a bare $ meaning none, so both spellings
#: have to be read. Missing the second one makes a file in metres look
#: unreadable rather than wrong, which is survivable but needless.
_SI_UNIT = re.compile(
    r"SI_UNIT\s*\(\s*(?:\.([A-Z]+)\.|\$)\s*,\s*\.([A-Z]+)\.\s*\)")


def _occ():
    try:
        import OCP  # noqa: F401
    except ImportError:
        return None
    return True


def is_available() -> bool:
    return _occ() is not None


def version() -> str | None:
    if not is_available():
        return None
    import OCP

    return f"OpenCASCADE via OCP {getattr(OCP, '__version__', 'unknown')}"


def step_analyzer_descriptor(available: bool | None = None) -> NodeDescriptor:
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=STEP_ANALYZER_NODE_NAME, transport=Transport.IN_PROCESS,
        address="OCP", available=present,
        unavailable_reason="" if present else
        "unavailable: OpenCASCADE bindings (OCP) are not installed")


def step_analyzer_capability_method():
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=STEP_ANALYZER_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Read a STEP file into this project's Part Dataset record.",
        inputs=("step_file",),
        outputs=("geometry", "topology", "part_record"),
        fidelity=Fidelity.ANALYTICAL,
        cost=Cost.CHEAP,
        conditions=(
            Condition("the input is a CAD file rather than parameters",
                      lambda c: c.require("has_cad_input")),
        ),
        implementation="nodes.step_analyzer.analyse_step",
        evidence="SIMULATED",
        notes="Geometry and topology only. The length unit is read from the "
              "file's SI_UNIT declaration and a file declaring none is "
              "refused, because guessing millimetres for metres scales a "
              "volume by a billion while every number still looks plausible. "
              "It reads no assembly structure, no materials, no tolerances "
              "and no design intent: a cylindrical face is a cylindrical "
              "face, and whether it is a clearance hole or a bearing seat is "
              "not in the geometry.")


def read_length_unit_m(path: Path) -> float:
    """The file's length unit in metres, read from its own declaration."""
    text = Path(path).read_text(errors="ignore")
    for prefix, name in _SI_UNIT.findall(text):
        key = (prefix, name)          # an unmatched prefix group is ""
        if key in _UNIT_TO_METRES:
            return _UNIT_TO_METRES[key]
    raise ValueError(
        f"{Path(path).name} declares no length unit this module recognises. "
        f"Refusing rather than assuming millimetres: the wrong guess scales "
        f"every volume by a billion and leaves the numbers looking sensible")


@dataclass(frozen=True)
class StepContents:
    """What a STEP file turned out to hold."""

    path: str
    unit_to_metres: float
    solid_count: int
    shapes: tuple


def _explore(shape, kind) -> list:
    from OCP.TopExp import TopExp_Explorer

    found = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        found.append(explorer.Current())
        explorer.Next()
    return found


def read_step(path: str | Path) -> StepContents:
    """Load a STEP file and split it into its solids."""
    if not is_available():
        raise CapabilityUnavailable(
            STEP_ANALYZER_CAPABILITY, STEP_ANALYZER_NODE_NAME,
            "OpenCASCADE bindings (OCP) are not installed")
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no STEP file at {path}")
    unit = read_length_unit_m(path)

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"OpenCASCADE could not read {path.name}")
    reader.TransferRoots()
    shape = reader.OneShape()
    solids = _explore(shape, TopAbs_SOLID)
    if not solids:
        raise ValueError(
            f"{path.name} contains no solids. A surface or wireframe model "
            f"has no volume, so nothing downstream here applies to it")
    return StepContents(path=str(path), unit_to_metres=unit,
                        solid_count=len(solids), shapes=tuple(solids))


def _geometry_of(shape, unit: float) -> GeometrySummary:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    volume = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume)
    area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area)
    centre = volume.CentreOfMass()

    # OpenCASCADE pads a Bnd_Box with a safety gap, so a 300 mm part measures
    # 300.0000002. That is deliberate on their side and wrong for reporting a
    # dimension, so the gap is removed rather than tolerated downstream.
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    box.SetGap(0.0)
    x0, y0, z0, x1, y1, z1 = box.Get()

    return GeometrySummary(
        volume_m3=volume.Mass() * unit ** 3,
        surface_area_m2=area.Mass() * unit ** 2,
        bounding_box_m=((x1 - x0) * unit, (y1 - y0) * unit, (z1 - z0) * unit),
        centre_of_mass_m=(centre.X() * unit, centre.Y() * unit,
                          centre.Z() * unit))


def _topology_of(shape) -> TopologySummary:
    from OCP.TopAbs import (TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL,
                            TopAbs_SOLID, TopAbs_VERTEX)

    return TopologySummary(
        solids=max(1, len(_explore(shape, TopAbs_SOLID))),
        shells=max(1, len(_explore(shape, TopAbs_SHELL))),
        faces=len(_explore(shape, TopAbs_FACE)),
        edges=len(_explore(shape, TopAbs_EDGE)),
        vertices=len(_explore(shape, TopAbs_VERTEX)))


def analyse_step(path: str | Path, provenance: Provenance,
                 part_id: str | None = None) -> list[PartRecord]:
    """One record per solid in the file.

    `provenance` is required rather than defaulted, for the same reason the
    schema requires it: a part whose origin nobody recorded is a part that
    cannot be published, and a bulk import is exactly where that would be
    filled in with something convenient.
    """
    contents = read_step(path)
    stem = part_id or Path(contents.path).stem
    records = []
    for index, shape in enumerate(contents.shapes):
        suffix = "" if contents.solid_count == 1 else f"-solid{index + 1}"
        records.append(PartRecord(
            part_id=f"{stem}{suffix}",
            provenance=provenance,
            geometry=_geometry_of(shape, contents.unit_to_metres),
            topology=_topology_of(shape),
            notes=(f"read from {Path(contents.path).name}, "
                   f"unit {contents.unit_to_metres:g} m, "
                   f"solid {index + 1} of {contents.solid_count}")))
    return records
