"""nodes.code_aster - Code_Aster, for material and contact nonlinearity.

CalculiX already covers linear elasticity here and is verified, so this node
is NOT justified by solving the same problems again. What Code_Aster brings is
nonlinear material behaviour, contact and plasticity.

None of that is what gets verified first. Plasticity has almost no closed form
answers, so a wrong plastic result cannot be told apart from a wrong install,
a wrong material input, or a misreading of the model. The cases here are
linear and exact, and exist to establish that the install and the plumbing are
right. Only after that does a nonlinear result carry information.

See docs/scoping_code_aster.md, written before this module, including a
measured boundary condition trap that cost real time and is recorded rather
than quietly fixed.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASTER_HOME = Path(os.environ.get("ASTER_HOME",
                                 Path.home() / "opt" / "code_aster"))

#: Results are tagged in the study's stdout with this prefix, because
#: Code_Aster's own output is verbose and its result files are a heavier
#: dependency than one printed line.
RESULT = "ASTER_RESULT"
_RESULT_LINE = re.compile(rf"^{RESULT}\s+(\S+)\s+(\S+)\s*$", re.MULTILINE)


class AsterUnavailable(RuntimeError):
    """Code_Aster is not installed where this module expects it."""


def runner_path() -> Path:
    exe = ASTER_HOME / "bin" / "run_aster"
    if not exe.exists():
        raise AsterUnavailable(
            f"run_aster not found at {exe}. This module expects the local "
            f"install under {ASTER_HOME}; set ASTER_HOME to point elsewhere.")
    return exe


def is_available() -> bool:
    return (ASTER_HOME / "bin" / "run_aster").exists()


def run_study(directory, study: str, mesh_file: Path) -> dict:
    """Run a Code_Aster study and return the values it tagged.

    Raises if the run fails or if it produced no tagged values. An empty
    result treated as a default is the failure mode that makes a study that
    never solved look like one that agreed.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "study.py").write_text(study)
    (directory / "study.export").write_text(
        "P actions make_etude\n"
        "P version stable\n"
        "P time_limit 1800\n"
        "P memory_limit 4096\n"
        f"F comm {directory / 'study.py'} D 1\n"
        f"F mmed {Path(mesh_file).resolve()} D 20\n"
        f"F mess {directory / 'study.mess'} R 6\n")

    proc = subprocess.run(
        [str(runner_path()), "study.export"], cwd=directory,
        capture_output=True, text=True, timeout=3600)
    values = {name: float(value)
              for name, value in _RESULT_LINE.findall(proc.stdout)}
    if not values:
        raise RuntimeError(
            "Code_Aster produced no tagged results. The study did not reach "
            "its output, so there is nothing to compare against a closed "
            f"form.\n{proc.stdout[-3000:]}\n{proc.stderr[-1500:]}")
    return values


# ------------------------------------------------------------------ meshing

def _write_mesh(path: Path, build) -> Path:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        build(gmsh)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def bar_mesh(path: Path, length_m: float, width_m: float, height_m: float,
             element_size_m: float) -> Path:
    def build(gmsh):
        gmsh.model.add("bar")
        box = gmsh.model.occ.addBox(0, 0, 0, length_m, width_m, height_m)
        gmsh.model.occ.synchronize()
        faces = {}
        for _, tag in gmsh.model.getEntities(2):
            b = gmsh.model.getBoundingBox(2, tag)
            centre = ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2)
            if abs(centre[0]) < 1e-9:
                faces.setdefault("FIXX", []).append(tag)
            elif abs(centre[0] - length_m) < 1e-9:
                faces.setdefault("LOAD", []).append(tag)
        for i, (name, tags) in enumerate(sorted(faces.items()), start=1):
            gmsh.model.addPhysicalGroup(2, tags, i)
            gmsh.model.setPhysicalName(2, i, name)
        gmsh.model.addPhysicalGroup(3, [box], 100)
        gmsh.model.setPhysicalName(3, 100, "VOL")
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), element_size_m)
        gmsh.model.mesh.generate(3)

    return _write_mesh(path, build)


# ------------------------------------------------- a bar in uniform tension

@dataclass(frozen=True)
class BarTension:
    """Uniform tension, a state linear elements reproduce EXACTLY.

    The expected error is round off, not a percentage, which is what makes
    this the right first case: a discrepancy is a bug and cannot be excused
    as a coarse mesh.
    """

    length_m: float
    youngs_modulus_pa: float
    applied_stress_pa: float
    max_displacement_m: float
    max_stress_pa: float
    end_displacement_spread_m: float
    end_node_count: int = 0

    @property
    def exact_displacement_m(self) -> float:
        return self.applied_stress_pa * self.length_m / self.youngs_modulus_pa

    @property
    def displacement_error(self) -> float:
        return abs(self.max_displacement_m - self.exact_displacement_m) \
            / self.exact_displacement_m

    @property
    def stress_error(self) -> float:
        return abs(self.max_stress_pa - self.applied_stress_pa) \
            / self.applied_stress_pa


def bar_tension(directory, length_m: float = 0.2, width_m: float = 0.02,
                height_m: float = 0.02, youngs_modulus_pa: float = 210.0e9,
                poisson_ratio: float = 0.3,
                applied_stress_pa: float = 50.0e6,
                element_size_m: float = 0.005) -> BarTension:
    """Pull a bar and compare against sigma L / E.

    The lateral faces are deliberately NOT held. Holding whole lateral planes
    perturbs the solution measurably, for reasons recorded and left open in
    docs/scoping_code_aster.md; three point constraints remove the remaining
    rigid body motions instead, and that configuration is exact to 2e-13.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mesh = bar_mesh(directory / "bar.msh", length_m, width_m, height_m,
                    element_size_m)

    study = f"""
from code_aster.Commands import *
from code_aster import CA
import numpy as np

CA.init("--test")
L, W, H = {length_m!r}, {width_m!r}, {height_m!r}
E, NU, SIGMA = {youngs_modulus_pa!r}, {poisson_ratio!r}, {applied_stress_pa!r}

mesh = LIRE_MAILLAGE(UNITE=20, FORMAT="GMSH")
mesh = DEFI_GROUP(reuse=mesh, MAILLAGE=mesh, CREA_GROUP_NO=(
    _F(NOM="P0", OPTION="ENV_SPHERE", POINT=(0., 0., 0.), RAYON=1e-4,
       PRECISION=1e-4),
    _F(NOM="PY", OPTION="ENV_SPHERE", POINT=(0., 0., H), RAYON=1e-4,
       PRECISION=1e-4),
    _F(NOM="PZ", OPTION="ENV_SPHERE", POINT=(0., W, 0.), RAYON=1e-4,
       PRECISION=1e-4)))
model = AFFE_MODELE(MAILLAGE=mesh,
                    AFFE=_F(TOUT="OUI", PHENOMENE="MECANIQUE",
                            MODELISATION="3D"))
steel = DEFI_MATERIAU(ELAS=_F(E=E, NU=NU))
mat = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=_F(TOUT="OUI", MATER=steel))

bcs = AFFE_CHAR_MECA(
    MODELE=model,
    DDL_IMPO=(_F(GROUP_MA="FIXX", DX=0.0),
              _F(GROUP_NO="P0", DY=0.0, DZ=0.0),
              _F(GROUP_NO="PY", DY=0.0),
              _F(GROUP_NO="PZ", DZ=0.0)),
    PRES_REP=_F(GROUP_MA="LOAD", PRES=-SIGMA))

res = MECA_STATIQUE(MODELE=model, CHAM_MATER=mat, EXCIT=_F(CHARGE=bcs))
res = CALC_CHAMP(reuse=res, RESULTAT=res, CONTRAINTE=("SIGM_NOEU",))

field = res.getField("DEPL", 1)
dx, desc = field.getValuesWithDescription("DX")
dx = np.array(dx)
node_ids = np.array(desc[0])
coords = np.array(mesh.getCoordinates().getValues()).reshape(-1, 3)
# The spread must be taken over the nodes that ARE on the loaded end, not
# over the largest values, which on a fine mesh include interior nodes and
# would report a spread that means nothing.
on_end = np.abs(coords[node_ids, 0] - L) < 1e-9
sxx = np.array(res.getField("SIGM_NOEU", 1).getValuesWithDescription("SIXX")[0])

print("{RESULT}", "max_dx", repr(float(dx.max())))
print("{RESULT}", "max_sixx", repr(float(np.abs(sxx).max())))
print("{RESULT}", "n_end_nodes", repr(float(on_end.sum())))
print("{RESULT}", "end_spread",
      repr(float(dx[on_end].max() - dx[on_end].min())))
CA.close()
"""
    values = run_study(directory, study, mesh)
    return BarTension(
        length_m=length_m, youngs_modulus_pa=youngs_modulus_pa,
        applied_stress_pa=applied_stress_pa,
        max_displacement_m=values["max_dx"],
        max_stress_pa=values["max_sixx"],
        end_displacement_spread_m=values["end_spread"],
        end_node_count=int(values["n_end_nodes"]))


# ------------------------------------ a thick cylinder, the Lame solution

@dataclass(frozen=True)
class ThickCylinder:
    """Internal pressure on a thick walled cylinder, plane strain.

    NOT VERIFIED. This case does not reproduce the Lame solution and is not
    used by anything. It is kept because the investigation is worth more than
    a deleted file, and because a test marked as a known failure will announce
    itself if it ever starts passing.

    What was measured, at three mesh densities:

        the bore displacement is about 0.20 of the closed form
        the bore hoop stress is about 0.13 of it
        neither improves with refinement, so it is not discretisation

    What was ruled out: the edge groups are correct, the bore arc has the
    right length, the loaded boundary is oriented, and the resultant of the
    applied pressure in x is exactly right at 5e5 N per metre.

    The cause was then found: the named groups Code_Aster reads back from the
    GMSH file do NOT correspond to the curves they were assigned to. BORE came
    back spanning x from 0 to the outer radius rather than the bore arc, so
    the pressure and the symmetry conditions were being applied to the wrong
    edges. The radial displacement on the bore was exactly zero at zero
    degrees and rose to a maximum at forty five, which is not axisymmetric and
    is what gave it away.

    The study now checks each group against the geometry it is supposed to
    name and refuses if they disagree, so this returns an error rather than a
    plausible wrong field. Writing the mesh in a form whose groups survive the
    round trip is the remaining work.

    Unlike beam theory, the Lame solution IS the exact answer to the
    elasticity problem, so when this is fixed the ORDER of convergence can be
    measured: displacement at second order and stress at first.
    """

    inner_radius_m: float
    outer_radius_m: float
    pressure_pa: float
    youngs_modulus_pa: float
    poisson_ratio: float
    elements: int
    bore_displacement_m: float
    bore_hoop_stress_pa: float

    @property
    def exact_bore_displacement_m(self) -> float:
        a, b = self.inner_radius_m, self.outer_radius_m
        nu, e, p = self.poisson_ratio, self.youngs_modulus_pa, self.pressure_pa
        return ((1.0 + nu) * a ** 2 * p) / (e * (b ** 2 - a ** 2)) \
            * ((1.0 - 2.0 * nu) * a + b ** 2 / a)

    @property
    def exact_bore_hoop_stress_pa(self) -> float:
        a, b = self.inner_radius_m, self.outer_radius_m
        return self.pressure_pa * (b ** 2 + a ** 2) / (b ** 2 - a ** 2)

    @property
    def displacement_error(self) -> float:
        return abs(self.bore_displacement_m - self.exact_bore_displacement_m) \
            / self.exact_bore_displacement_m

    @property
    def hoop_stress_error(self) -> float:
        return abs(self.bore_hoop_stress_pa - self.exact_bore_hoop_stress_pa) \
            / self.exact_bore_hoop_stress_pa


def _quarter_annulus_mesh(path: Path, a: float, b: float,
                          element_size: float) -> Path:
    """A quarter annulus as a TWO dimensional mesh, for plane strain.

    Code_Aster has a plane strain element, D_PLAN, which is the right tool
    here rather than a thin 3D slab with the out of plane displacement held on
    both faces.
    """
    def build(gmsh):
        gmsh.model.add("cyl")
        outer = gmsh.model.occ.addDisk(0, 0, 0, b, b)
        inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
        ring, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
        box = gmsh.model.occ.addRectangle(0, 0, 0, b, b)
        quarter, _ = gmsh.model.occ.intersect(ring, [(2, box)])
        gmsh.model.occ.synchronize()
        surface = quarter[0][1]

        # gmsh pads the bounding boxes it returns by its geometry tolerance,
        # measured as exactly 1e-7. A comparison tolerance of 1e-12 rejects
        # every edge, and so does one of exactly 1e-7, because the test then
        # reads 1e-7 < 1e-7. It must be comfortably larger than the padding
        # and far below any real feature, and relative to the geometry so it
        # is not a length unit test in disguise.
        tol = 1e-4 * b
        groups = {}
        for _, tag in gmsh.model.getEntities(1):
            bb = gmsh.model.getBoundingBox(1, tag)
            reach = max(bb[3], bb[4])
            if abs(bb[0]) < tol and abs(bb[3]) < tol:
                groups.setdefault("SYMX", []).append(tag)
            elif abs(bb[1]) < tol and abs(bb[4]) < tol:
                groups.setdefault("SYMY", []).append(tag)
            elif abs(reach - a) < tol:
                groups.setdefault("BORE", []).append(tag)
        missing = {"SYMX", "SYMY", "BORE"} - set(groups)
        if missing:
            raise RuntimeError(
                f"the quarter annulus did not produce the edges "
                f"{sorted(missing)}; found {sorted(groups)}. Solving with an "
                f"edge group missing would silently drop a boundary "
                f"condition.")
        for i, (name, tags) in enumerate(sorted(groups.items()), start=1):
            gmsh.model.addPhysicalGroup(1, tags, i)
            gmsh.model.setPhysicalName(1, i, name)
        gmsh.model.addPhysicalGroup(2, [surface], 100)
        gmsh.model.setPhysicalName(2, 100, "VOL")
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), element_size)
        gmsh.model.mesh.generate(2)

    return _write_mesh(path, build)


def thick_cylinder(directory, inner_radius_m: float = 0.05,
                   outer_radius_m: float = 0.1,
                   pressure_pa: float = 10.0e6,
                   youngs_modulus_pa: float = 210.0e9,
                   poisson_ratio: float = 0.3,
                   element_size_m: float = 0.01) -> ThickCylinder:
    """A quarter of a pressurised cylinder, held in plane strain.

    The symmetry planes here are physically required rather than convenient:
    the exact axisymmetric solution has no displacement across them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # MED, not GMSH. Code_Aster's GMSH reader was measured to mismap the
    # named edge groups on this mesh: the file holds 14 bore edges spanning
    # x from 0 to the bore radius, and the reader returned 23 cells spanning
    # x to the OUTER radius, having pulled in triangles as well. The file is
    # correct, so the fault is in the reading rather than the writing, and
    # MED is Code_Aster's own format.
    mesh = _quarter_annulus_mesh(directory / "cyl.med", inner_radius_m,
                                 outer_radius_m, element_size_m)

    study = f"""
from code_aster.Commands import *
from code_aster import CA
import numpy as np

CA.init("--test")
A, B, P = {inner_radius_m!r}, {outer_radius_m!r}, {pressure_pa!r}
E, NU = {youngs_modulus_pa!r}, {poisson_ratio!r}

mesh = LIRE_MAILLAGE(UNITE=20, FORMAT="MED")

# The groups Code_Aster reads from a GMSH file are checked against the
# geometry they are supposed to name. They have been observed NOT to match:
# the arc named BORE came back spanning the full outer radius. A pressure
# applied to the wrong edge, and symmetry held on the wrong edge, still
# produce a field, and that field looks plausible until it is compared with a
# closed form. Refusing is better than returning it.
coords = np.array(mesh.getCoordinates().getValues()).reshape(-1, 3)
conn = mesh.getConnectivity()
expected = {{"BORE": ((0.0, A), (0.0, A)),
            "SYMX": ((0.0, 0.0), (A, B)),
            "SYMY": ((A, B), (0.0, 0.0))}}
tol = 1e-4 * B
for name, ((xlo, xhi), (ylo, yhi)) in expected.items():
    nodes = sorted({{n for c in mesh.getCells(name) for n in conn[c]}})
    pts = coords[nodes]
    seen = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    want = (xlo, xhi, ylo, yhi)
    if any(abs(s - w) > tol for s, w in zip(seen, want)):
        raise RuntimeError(
            "group " + name + " spans "
            + repr([round(float(v), 5) for v in seen])
            + " but should span " + repr([round(float(v), 5) for v in want])
            + ". The named groups do not correspond to the curves they were "
            "assigned to, so the load and the symmetry conditions would be "
            "applied to the wrong edges.")

# A pressure load needs the loaded boundary's normals to point consistently
# out of the material. gmsh does not guarantee that, and Code_Aster refuses
# the load rather than applying it with mixed signs, which is the right
# refusal: a pressure pushing inward on some faces and outward on others
# would still produce a field, and that field would look plausible.
mesh = MODI_MAILLAGE(reuse=mesh, MAILLAGE=mesh,
                     ORIE_PEAU=_F(GROUP_MA_PEAU="BORE"))
model = AFFE_MODELE(MAILLAGE=mesh,
                    AFFE=_F(TOUT="OUI", PHENOMENE="MECANIQUE",
                            MODELISATION="D_PLAN"))
steel = DEFI_MATERIAU(ELAS=_F(E=E, NU=NU))
mat = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=_F(TOUT="OUI", MATER=steel))

bcs = AFFE_CHAR_MECA(
    MODELE=model,
    DDL_IMPO=(_F(GROUP_MA="SYMX", DX=0.0),
              _F(GROUP_MA="SYMY", DY=0.0)),
    PRES_REP=_F(GROUP_MA="BORE", PRES=P))

res = MECA_STATIQUE(MODELE=model, CHAM_MATER=mat, EXCIT=_F(CHARGE=bcs))
res = CALC_CHAMP(reuse=res, RESULTAT=res, CONTRAINTE=("SIGM_NOEU",))

coords = np.array(mesh.getCoordinates().getValues()).reshape(-1, 3)
field = res.getField("DEPL", 1)
dx, desc = field.getValuesWithDescription("DX")
ids = np.array(desc[0])
dx = np.array(dx)
dy = np.array(field.getValuesWithDescription("DY")[0])
r = np.hypot(coords[ids, 0], coords[ids, 1])
radial = (dx * coords[ids, 0] + dy * coords[ids, 1]) / np.maximum(r, 1e-30)
bore = np.abs(r - A) < 1e-6

sig = res.getField("SIGM_NOEU", 1)
sxx = np.array(sig.getValuesWithDescription("SIXX")[0])
syy = np.array(sig.getValuesWithDescription("SIYY")[0])
sxy = np.array(sig.getValuesWithDescription("SIXY")[0])
sids = np.array(sig.getValuesWithDescription("SIXX")[1][0])
rs = np.hypot(coords[sids, 0], coords[sids, 1])
cos = coords[sids, 0] / np.maximum(rs, 1e-30)
sin = coords[sids, 1] / np.maximum(rs, 1e-30)
hoop = sxx * sin ** 2 - 2.0 * sxy * sin * cos + syy * cos ** 2
sbore = np.abs(rs - A) < 1e-6

print("{RESULT}", "n_bore_nodes", repr(float(bore.sum())))
print("{RESULT}", "bore_ur", repr(float(radial[bore].mean())))
print("{RESULT}", "bore_hoop", repr(float(hoop[sbore].mean())))
print("{RESULT}", "n_elements", repr(float(len(mesh.getCells()))))
CA.close()
"""
    values = run_study(directory, study, mesh)
    if values["n_bore_nodes"] < 3:
        raise RuntimeError(
            f"only {values['n_bore_nodes']:.0f} nodes were found on the bore; "
            f"an average over too few points is not a field measurement")
    return ThickCylinder(
        inner_radius_m=inner_radius_m, outer_radius_m=outer_radius_m,
        pressure_pa=pressure_pa, youngs_modulus_pa=youngs_modulus_pa,
        poisson_ratio=poisson_ratio, elements=int(values["n_elements"]),
        bore_displacement_m=values["bore_ur"],
        bore_hoop_stress_pa=values["bore_hoop"])
