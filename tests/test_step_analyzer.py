"""Reading STEP, checked against closed forms rather than against itself.

This project exports its own parts, so a STEP file whose exact volume and area
are known already exists. That makes the analyzer checkable against arithmetic
instead of against another reading of the same file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.part_dataset import Licence, Provenance, ProvenanceKind
from nodes import step_analyzer as sa

requires_occ = pytest.mark.skipif(not sa.is_available(),
                                  reason="OCP is not installed")

ARM_STEP = Path("runs/cad/arm_assembly.step")
has_arm = ARM_STEP.exists()
requires_arm = pytest.mark.skipif(not has_arm,
                                  reason="the arm STEP export is not present")

# The two links this project exports, and their exact section dimensions.
LINK1 = dict(length=0.30, height=0.040, width=0.020, wall=0.002)
LINK2 = dict(length=0.25, height=0.032, width=0.016, wall=0.002)


def section_area(height, width, wall):
    return height * width - (height - 2 * wall) * (width - 2 * wall)


def exact_volume(length, height, width, wall):
    return section_area(height, width, wall) * length


def exact_area(length, height, width, wall):
    outer_perimeter = 2.0 * (height + width)
    inner_perimeter = 2.0 * (height - 2 * wall + width - 2 * wall)
    return ((outer_perimeter + inner_perimeter) * length
            + 2.0 * section_area(height, width, wall))


def provenance() -> Provenance:
    return Provenance(kind=ProvenanceKind.SYNTHETIC_PARAMETRIC,
                      source="daedalus cad export", generator="hollow_rect",
                      licence=Licence(identifier="Apache-2.0",
                                      redistributable=True))


# ------------------------------------------------------- the unit is read

@requires_arm
def test_the_length_unit_is_read_from_the_file():
    """STEP declares it. Guessing scales a volume by a billion."""
    assert sa.read_length_unit_m(ARM_STEP) == pytest.approx(1e-3)


def test_a_file_declaring_no_unit_is_refused(tmp_path):
    """The failure it prevents is silent, so the refusal has to be loud."""
    fake = tmp_path / "no_unit.step"
    fake.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    with pytest.raises(ValueError, match="declares no length unit"):
        sa.read_length_unit_m(fake)


@pytest.mark.parametrize("declaration,expected", [
    ("SI_UNIT(.MILLI.,.METRE.);", 1e-3),
    ("SI_UNIT($,.METRE.);", 1.0),
    ("SI_UNIT(.CENTI.,.METRE.);", 1e-2),
    ("SI_UNIT(.MICRO.,.METRE.);", 1e-6),
])
def test_each_recognised_unit_is_converted(tmp_path, declaration, expected):
    path = tmp_path / "unit.step"
    path.write_text(f"DATA;\n#1 = {declaration}\nENDSEC;\n")
    assert sa.read_length_unit_m(path) == pytest.approx(expected)


# ---------------------------------------------- geometry against arithmetic

@requires_occ
@requires_arm
def test_the_solids_are_found_and_counted():
    contents = sa.read_step(ARM_STEP)
    assert contents.solid_count == 2
    assert contents.unit_to_metres == pytest.approx(1e-3)


@requires_occ
@requires_arm
@pytest.mark.parametrize("index,link", [(0, LINK1), (1, LINK2)])
def test_volume_and_area_match_the_closed_form(index, link):
    """The whole file checked against pen and paper, not against a rerun."""
    record = sa.analyse_step(ARM_STEP, provenance())[index]
    assert record.geometry.volume_m3 == pytest.approx(
        exact_volume(**link), rel=1e-9)
    assert record.geometry.surface_area_m2 == pytest.approx(
        exact_area(**link), rel=1e-9)


@requires_occ
@requires_arm
@pytest.mark.parametrize("index,link", [(0, LINK1), (1, LINK2)])
def test_the_bounding_box_is_the_section_and_the_length(index, link):
    record = sa.analyse_step(ARM_STEP, provenance())[index]
    x, y, z = record.geometry.bounding_box_m
    assert x == pytest.approx(link["length"], rel=1e-9)
    assert y == pytest.approx(link["height"], rel=1e-9)
    assert z == pytest.approx(link["width"], rel=1e-9)


@requires_occ
@requires_arm
def test_a_hollow_rectangular_prism_has_ten_faces():
    """Four outside, four inside, two ends. Countable by hand."""
    for record in sa.analyse_step(ARM_STEP, provenance()):
        assert record.topology.faces == 10
        assert record.topology.solids == 1


# --------------------------------------------------- one record per solid

@requires_occ
@requires_arm
def test_a_multi_solid_file_becomes_one_record_per_solid():
    """The schema describes one part, so the caller is not handed the first."""
    records = sa.analyse_step(ARM_STEP, provenance())
    assert len(records) == 2
    assert {r.part_id for r in records} == {"arm_assembly-solid1",
                                            "arm_assembly-solid2"}
    for record in records:
        assert "solid" in record.notes


@requires_occ
@requires_arm
def test_provenance_is_required_and_travels_into_every_record():
    records = sa.analyse_step(ARM_STEP, provenance())
    for record in records:
        assert record.provenance.kind is ProvenanceKind.SYNTHETIC_PARAMETRIC
        assert record.is_publishable


@requires_occ
def test_a_missing_file_is_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sa.read_step(tmp_path / "nothing.step")


# ------------------------------------------------------------- the node

def test_availability_is_read_from_the_import_system():
    descriptor = sa.step_analyzer_descriptor()
    assert descriptor.available is sa.is_available()


def test_the_capability_admits_what_it_does_not_read():
    """A cylindrical face is a face. What it is FOR is not in the geometry."""
    method = sa.step_analyzer_capability_method()
    assert "no design intent" in method.notes
    assert "refused" in method.notes
    assert method.evidence == "SIMULATED"


# ------------------------------------ STEP through to a solver, end to end

DESIGN_STEP = Path("runs/cad/design.step")
requires_design = pytest.mark.skipif(
    not DESIGN_STEP.exists(), reason="the design STEP export is not present")


@requires_occ
@requires_design
def test_a_step_solid_meshes_to_the_volume_the_analyzer_reported():
    """The mesher and the reader must describe the same solid."""
    from nodes.gmsh_node import is_available, tetrahedral_mesh_from_step

    if not is_available():
        pytest.skip("gmsh is not installed")
    record = sa.analyse_step(DESIGN_STEP, provenance())[0]
    mesh = tetrahedral_mesh_from_step(str(DESIGN_STEP), 0.006, order=2)
    assert mesh.volume_m3() == pytest.approx(record.geometry.volume_m3,
                                             rel=1e-9)


@requires_occ
@requires_design
def test_a_cad_part_need_not_sit_at_the_origin():
    """The assumption that quietly applies no boundary condition at all.

    This project's own meshes start at x=0, so code written against them
    reaches for x=0 as the root. A STEP solid centred on the origin then
    yields an empty node selection, and a fixed face that was never fixed.
    """
    from nodes.gmsh_node import is_available, tetrahedral_mesh_from_step

    if not is_available():
        pytest.skip("gmsh is not installed")
    mesh = tetrahedral_mesh_from_step(str(DESIGN_STEP), 0.008, order=2)
    x = mesh.node_coords[:, 0]
    assert x.min() < 0.0 < x.max(), "this fixture is meant to straddle zero"

    # Selecting by extent finds the face; selecting by coordinate does not.
    assert len(mesh.nodes_at_extreme(0, "min")) > 0
    assert len(mesh.nodes_at_extreme(0, "max")) > 0
    assert len(mesh.nodes_at_extreme(0, "min")) != len(mesh.nodes_at_x(0.0))


@requires_occ
@requires_design
def test_a_bad_side_argument_is_refused():
    from nodes.gmsh_node import is_available, tetrahedral_mesh_from_step

    if not is_available():
        pytest.skip("gmsh is not installed")
    mesh = tetrahedral_mesh_from_step(str(DESIGN_STEP), 0.01, order=1)
    with pytest.raises(ValueError, match="side must be"):
        mesh.nodes_at_extreme(0, "middle")


@requires_occ
@requires_design
def test_step_reaches_a_solver_and_returns_a_deflection():
    """The whole point of the route, exercised once."""
    from nodes.calculix import ElementType
    from nodes.calculix import is_available as ccx_available
    from nodes.calculix import solve as ccx_solve
    from nodes.gmsh_node import is_available, tetrahedral_mesh_from_step

    if not (is_available() and ccx_available()):
        pytest.skip("needs both gmsh and CalculiX")
    mesh = tetrahedral_mesh_from_step(str(DESIGN_STEP), 0.006, order=2)
    result = ccx_solve(mesh, 71.7e9, 0.33,
                       mesh.nodes_at_extreme(0, "min"),
                       mesh.nodes_at_extreme(0, "max"),
                       total_load_n=-100.0, load_direction=1,
                       element_type=ElementType.C3D10)
    assert result.converged
    tip = result.displacements[mesh.nodes_at_extreme(0, "max"), 1].mean()
    assert tip < 0.0, "a downward load must deflect downward"
    assert abs(tip) < 0.01, "a metre scale deflection would be nonsense here"
