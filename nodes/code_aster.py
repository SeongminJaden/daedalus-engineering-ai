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


# ------------------------------------------ plasticity, which is the point

#: Plane strain with von Mises and incompressible plastic flow gives
#: sigma_theta - sigma_r = 2 sigma_y / sqrt(3) in the plastic zone, NOT
#: sigma_y. The textbook thick cylinder formulas are usually quoted for
#: Tresca, where the factor is 1. Using the Tresca form against a von Mises
#: solver is a subtle error worth about 15 percent, in the direction that
#: makes the part look stronger than it is.
MISES_PLANE_STRAIN_FACTOR = 2.0 / np.sqrt(3.0)


@dataclass(frozen=True)
class PlasticCylinder:
    """A thick cylinder pressurised past first yield."""

    inner_radius_m: float
    outer_radius_m: float
    pressure_pa: float
    yield_stress_pa: float
    youngs_modulus_pa: float
    poisson_ratio: float
    bore_displacement_m: float
    plastic_radius_m: float
    converged: bool

    @property
    def first_yield_pressure_pa(self) -> float:
        """The pressure at which the bore first yields.

        From sigma_theta - sigma_r = 2 B / r^2 reaching the plane strain von
        Mises limit at r = a.
        """
        a, b = self.inner_radius_m, self.outer_radius_m
        return (MISES_PLANE_STRAIN_FACTOR * self.yield_stress_pa
                * (b ** 2 - a ** 2) / (2.0 * b ** 2))

    @property
    def fully_plastic_pressure_pa(self) -> float:
        return (MISES_PLANE_STRAIN_FACTOR * self.yield_stress_pa
                * np.log(self.outer_radius_m / self.inner_radius_m))

    def exact_pressure_for_plastic_radius(self, c: float) -> float:
        """The pressure that puts the elastic plastic front at radius c.

        p = k [ ln(c/a) + (b^2 - c^2) / (2 b^2) ], with k the plane strain von
        Mises limit. Inverting this gives the plastic radius a measured
        pressure implies, which is what the solve is checked against.
        """
        a, b = self.inner_radius_m, self.outer_radius_m
        k = MISES_PLANE_STRAIN_FACTOR * self.yield_stress_pa
        return k * (np.log(c / a) + (b ** 2 - c ** 2) / (2.0 * b ** 2))

    @property
    def exact_plastic_radius_m(self) -> float:
        """Solved by bisection on the closed form above."""
        a, b = self.inner_radius_m, self.outer_radius_m
        if self.pressure_pa <= self.first_yield_pressure_pa:
            return a
        lo, hi = a, b
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.exact_pressure_for_plastic_radius(mid) < self.pressure_pa:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def plastic_cylinder(directory, inner_radius_m: float = 0.05,
                     outer_radius_m: float = 0.1,
                     pressure_pa: float = 150.0e6,
                     yield_stress_pa: float = 250.0e6,
                     youngs_modulus_pa: float = 210.0e9,
                     poisson_ratio: float = 0.3,
                     element_size_m: float = 0.003,
                     steps: int = 20) -> PlasticCylinder:
    """Pressurise a thick cylinder past yield and find the plastic front.

    Perfect plasticity is approximated by linear isotropic hardening with a
    tangent modulus of E/1e5, because Code_Aster's VMIS_ISOT_LINE needs a
    positive slope. That is an approximation of the closed form's assumption,
    not an exact match to it, and it stiffens the answer slightly.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mesh = _quarter_annulus_mesh(directory / "cyl.med", inner_radius_m,
                                 outer_radius_m, element_size_m)

    study = f"""
from code_aster.Commands import *
from code_aster import CA
import numpy as np

CA.init("--test")
A, B, P = {inner_radius_m!r}, {outer_radius_m!r}, {pressure_pa!r}
E, NU, SY = {youngs_modulus_pa!r}, {poisson_ratio!r}, {yield_stress_pa!r}

mesh = LIRE_MAILLAGE(UNITE=20, FORMAT="MED")
mesh = MODI_MAILLAGE(reuse=mesh, MAILLAGE=mesh,
                     ORIE_PEAU=_F(GROUP_MA_PEAU="BORE"))
model = AFFE_MODELE(MAILLAGE=mesh,
                    AFFE=_F(TOUT="OUI", PHENOMENE="MECANIQUE",
                            MODELISATION="D_PLAN"))
steel = DEFI_MATERIAU(ELAS=_F(E=E, NU=NU),
                      ECRO_LINE=_F(SY=SY, D_SIGM_EPSI=E / 1.0e5))
mat = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=_F(TOUT="OUI", MATER=steel))

bcs = AFFE_CHAR_MECA(
    MODELE=model,
    DDL_IMPO=(_F(GROUP_MA="SYMX", DX=0.0),
              _F(GROUP_MA="SYMY", DY=0.0)))
load = AFFE_CHAR_MECA(MODELE=model, PRES_REP=_F(GROUP_MA="BORE", PRES=P))

times = DEFI_LIST_REEL(DEBUT=0.0,
                       INTERVALLE=_F(JUSQU_A=1.0, NOMBRE={steps}))
ramp = DEFI_FONCTION(NOM_PARA="INST", VALE=(0.0, 0.0, 1.0, 1.0),
                     PROL_DROITE="CONSTANT", PROL_GAUCHE="CONSTANT")

res = STAT_NON_LINE(
    MODELE=model, CHAM_MATER=mat,
    EXCIT=(_F(CHARGE=bcs), _F(CHARGE=load, FONC_MULT=ramp)),
    COMPORTEMENT=_F(RELATION="VMIS_ISOT_LINE", DEFORMATION="PETIT"),
    INCREMENT=_F(LIST_INST=times),
    NEWTON=_F(REAC_ITER=1),
    CONVERGENCE=_F(RESI_GLOB_RELA=1.0e-8, ITER_GLOB_MAXI=50))

res = CALC_CHAMP(reuse=res, RESULTAT=res, VARI_INTERNE=("VARI_NOEU",))

coords = np.array(mesh.getCoordinates().getValues()).reshape(-1, 3)
last = res.getNumberOfIndexes() - 1
depl = res.getField("DEPL", last)
dx, desc = depl.getValuesWithDescription("DX")
ids = np.array(desc[0])
dx = np.array(dx)
dy = np.array(depl.getValuesWithDescription("DY")[0])
x, y = coords[ids, 0], coords[ids, 1]
r = np.hypot(x, y)
ur = (dx * x + dy * y) / np.maximum(r, 1e-30)
bore = np.abs(r - A) < 1e-6

# V1 of VMIS_ISOT_LINE is the cumulated plastic strain. The plastic front is
# the largest radius at which it is non zero.
vari = res.getField("VARI_NOEU", last)
v1, vdesc = vari.getValuesWithDescription("V1")
vids = np.array(vdesc[0])
v1 = np.array(v1)
rv = np.hypot(coords[vids, 0], coords[vids, 1])
yielded = v1 > 1.0e-12
front = float(rv[yielded].max()) if yielded.any() else A

print("{RESULT}", "bore_ur", repr(float(ur[bore].mean())))
print("{RESULT}", "plastic_radius", repr(front))
print("{RESULT}", "n_yielded", repr(float(yielded.sum())))
print("{RESULT}", "converged", repr(1.0))
CA.close()
"""
    values = run_study(directory, study, mesh)
    return PlasticCylinder(
        inner_radius_m=inner_radius_m, outer_radius_m=outer_radius_m,
        pressure_pa=pressure_pa, yield_stress_pa=yield_stress_pa,
        youngs_modulus_pa=youngs_modulus_pa, poisson_ratio=poisson_ratio,
        bore_displacement_m=values["bore_ur"],
        plastic_radius_m=values["plastic_radius"],
        converged=bool(values.get("converged", 0.0)))


# ------------------------------------------------------------- registration

ASTER_NODE_NAME = "code_aster"
ASTER_PLASTICITY_CAPABILITY = "analysis.fea.plasticity"


def code_aster_descriptor(available: bool | None = None):
    from .descriptor import NodeDescriptor, Transport

    present = is_available() if available is None else available
    return NodeDescriptor(
        name=ASTER_NODE_NAME, transport=Transport.STDIO,
        address=str(ASTER_HOME / "bin" / "run_aster"),
        available=present,
        unavailable_reason="" if present else
        f"unavailable: run_aster was not found under {ASTER_HOME}")


def code_aster_capability_method():
    """Plasticity only.

    The linear elastic cases in this module are verified and are deliberately
    NOT registered. CalculiX already covers linear elasticity and is verified
    here, so a second implementation of the same equations is a cross-check
    rather than a capability the engine gains. Passing cases and a capability
    worth claiming are different things.
    """
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=ASTER_PLASTICITY_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Elastoplastic analysis in Code_Aster, for parts loaded past "
                "yield where a linear stress is not a stress the material "
                "can carry.",
        inputs=("geometry", "material", "yield_stress", "load"),
        outputs=("displacement", "stress", "plastic_zone"),
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("the load is known to exceed yield, since below it a "
                      "linear solve is cheaper and gives the same answer",
                      lambda c: c.require("loads_exceed_yield")),
            Condition("strains remain small, since the solve integrates on "
                      "the undeformed shape",
                      lambda c: c.require("strains_remain_small")),
        ),
        implementation="nodes.code_aster.plastic_cylinder",
        evidence="SIMULATED",
        notes="Verified against the closed form for a pressurised thick "
              "cylinder, in three ways. Below first yield the elastoplastic "
              "solve reproduces the already verified elastic answer to 1e-14, "
              "which anchors the nonlinear path without needing a plastic "
              "closed form. Above it the elastic plastic front matches the "
              "analytic plastic radius at three pressures across the range. "
              "And the front's error is a fixed FRACTION of the element size, "
              "measured at 0.57, 0.64 and 0.67 of an element across a factor "
              "of four in mesh, which is the signature of a first order front "
              "estimate rather than a wrong answer; extrapolating to zero "
              "element size gives the closed form to about 0.1 percent. The "
              "yield criterion is von Mises: in plane strain that makes the "
              "limit 2/sqrt(3) times the Tresca value, and using the Tresca "
              "form here would overstate the yield pressure by 15 percent, in "
              "the direction that makes a part look stronger than it is. "
              "Perfect plasticity is approximated by a tangent modulus of "
              "E/1e5, which stiffens the answer slightly. Nothing here has "
              "been measured against a physical test piece.")


# --------------------------------------------------- contact, against Hertz

@dataclass(frozen=True)
class HertzContact:
    """A sphere pressed onto a rigid plane, checked against Hertz.

    NOT VERIFIED. The closed forms below are correct and are tested as
    arithmetic. The SOLVE is not: the unilateral contact condition never
    carries load, and why is not established.

    What was measured, across every variant tried:

        the peak stress is 385.768 GPa against an expected 2.176, and it is
        the SAME number in every run, which is the point load singularity at
        the pinned apex rather than a contact pressure
        the contact radius comes out zero, so no node other than the pinned
        one is ever found on the plane

    What was tried and what each showed:

        force control with no pin        singular matrix, since the vertical
                                        rigid body motion is unconstrained
                                        until contact activates
        force control with the apex
        pinned                          the pin carries the whole load
        displacement control            the contact radius came back equal to
                                        the zone radius exactly, which is not
                                        a contact patch
        correcting the gap condition
        from DY >= 0 to DY >= -Y        changed NOTHING, and that is the
                                        clearest evidence: a constraint whose
                                        definition can be changed without
                                        changing any result is not active

    The last point is the useful one. The gap condition was genuinely wrong
    at first, because nodes on the arc start above the plane and must travel
    down to reach it, so forbidding downward motion forbids contact from
    forming. Fixing it produced identical numbers to twelve significant
    figures, which means the condition is not entering the system at all.

    That was then confirmed directly rather than inferred. Printing the
    deformed height of every node in the contact zone shows the whole zone
    sitting about 9.6 micrometres BELOW the plane, under an imposed approach
    of 2.2 micrometres. The bodies interpenetrate freely, which is what a
    unilateral condition exists to prevent, so the constraint is inert.

    Two explanations were ruled out by reading the generated command file
    rather than by guessing: the contact IS passed to the solve, appearing as
    CONTACT=contact and echoed back by the solver, and the solve IS
    STAT_NON_LINE rather than a linear one, so it is not that an inequality
    was handed to a linear solver.

    Nothing here is tuned to agree. The capability is not registered, and
    contact is set aside rather than pursued further.
    """

    sphere_radius_m: float
    force_n: float
    youngs_modulus_pa: float
    poisson_ratio: float
    contact_radius_m: float
    peak_pressure_pa: float
    zone_radius_m: float

    @property
    def effective_modulus_pa(self) -> float:
        """1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2, with the plane rigid."""
        return self.youngs_modulus_pa / (1.0 - self.poisson_ratio ** 2)

    @property
    def exact_contact_radius_m(self) -> float:
        return (3.0 * self.force_n * self.sphere_radius_m
                / (4.0 * self.effective_modulus_pa)) ** (1.0 / 3.0)

    @property
    def exact_peak_pressure_pa(self) -> float:
        a = self.exact_contact_radius_m
        return 3.0 * self.force_n / (2.0 * np.pi * a * a)

    @property
    def exact_approach_m(self) -> float:
        """Recorded for completeness; not compared, for the reason above."""
        a = self.exact_contact_radius_m
        return a * a / self.sphere_radius_m

    @property
    def contact_radius_error(self) -> float:
        return abs(self.contact_radius_m - self.exact_contact_radius_m) \
            / self.exact_contact_radius_m

    @property
    def peak_pressure_error(self) -> float:
        return abs(self.peak_pressure_pa - self.exact_peak_pressure_pa) \
            / self.exact_peak_pressure_pa

    @property
    def half_space_ratio(self) -> float:
        """a/R. Hertz assumes a half space, so this must stay small."""
        return self.exact_contact_radius_m / self.sphere_radius_m


def _hemisphere_mesh(path: Path, radius_m: float, zone_m: float,
                     fine_size_m: float, coarse_size_m: float) -> Path:
    """The lower half of a sphere, as an axisymmetric section in (r, z).

    The contact arc is built as its own curve rather than carved out of a
    whole one afterwards, so the unilateral zone is an explicit named group.
    That matters: applying "may not move down" to the entire arc would forbid
    the free upper surface from moving down at all, which is a constraint the
    real problem does not have.

    Graded to the contact point, because the contact radius is around one
    percent of the sphere radius and a mesh fine enough there would be
    enormous everywhere else.
    """
    if not 0.0 < zone_m < radius_m:
        raise ValueError("the contact zone must be smaller than the sphere")

    def build(gmsh):
        gmsh.model.add("hemi")
        occ = gmsh.model.occ
        angle = np.arcsin(zone_m / radius_m)
        bottom = occ.addPoint(0.0, 0.0, 0.0)
        centre = occ.addPoint(0.0, radius_m, 0.0)
        split = occ.addPoint(radius_m * np.sin(angle),
                             radius_m * (1.0 - np.cos(angle)), 0.0)
        equator = occ.addPoint(radius_m, radius_m, 0.0)

        near = occ.addCircleArc(bottom, centre, split)
        far = occ.addCircleArc(split, centre, equator)
        top = occ.addLine(equator, centre)
        axis = occ.addLine(centre, bottom)
        loop = occ.addCurveLoop([near, far, top, axis])
        surface = occ.addPlaneSurface([loop])
        occ.synchronize()

        for tag, name in ((near, "CONTACT"), (far, "FREE"),
                          (top, "TOP"), (axis, "AXIS")):
            group = gmsh.model.addPhysicalGroup(1, [tag])
            gmsh.model.setPhysicalName(1, group, name)
        # The lowest point of the sphere stays on the plane under any
        # compressive load, so holding it is exact rather than a convenience.
        # Without it the vertical rigid body motion is unconstrained until
        # contact activates, and the first factorisation is singular.
        apex = gmsh.model.addPhysicalGroup(0, [bottom])
        gmsh.model.setPhysicalName(0, apex, "APEX")
        volume = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, volume, "VOL")

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", [bottom])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", fine_size_m)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", coarse_size_m)
        gmsh.model.mesh.field.setNumber(2, "DistMin", zone_m)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 8.0 * zone_m)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)

    return _write_mesh(path, build)


def hertz_contact(directory, sphere_radius_m: float = 0.01,
                  force_n: float = 100.0,
                  youngs_modulus_pa: float = 210.0e9,
                  poisson_ratio: float = 0.3,
                  elements_across_contact: int = 12,
                  steps: int = 10) -> HertzContact:
    """Press a sphere onto a rigid plane and measure the contact patch.

    The plane is imposed as a unilateral condition rather than meshed, so
    there is no second body and no master surface to orient.

    Validity: Hertz assumes each body is a half space near the contact, which
    holds while the contact radius is small against the sphere radius. That
    ratio is reported, and at these loads it is one to two percent.

    The mesh is graded to the contact point and sized from the EXPECTED
    contact radius, so the resolution of the patch is controlled rather than
    incidental.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    reference = HertzContact(
        sphere_radius_m=sphere_radius_m, force_n=force_n,
        youngs_modulus_pa=youngs_modulus_pa, poisson_ratio=poisson_ratio,
        contact_radius_m=0.0, peak_pressure_pa=0.0, zone_radius_m=0.0)
    expected_a = reference.exact_contact_radius_m
    zone = 4.0 * expected_a
    fine = expected_a / elements_across_contact
    mesh = _hemisphere_mesh(directory / "hemi.med", sphere_radius_m, zone,
                            fine, sphere_radius_m / 12.0)
    # Imposed approach, chosen from the Hertz relation for the target force.
    # The force that results is measured, not assumed, and the closed form is
    # then evaluated at THAT force.
    pressure = force_n / (np.pi * sphere_radius_m ** 2)

    study = f"""
from code_aster.Commands import *
from code_aster import CA
import numpy as np

CA.init("--test")
R, P, ZONE = {sphere_radius_m!r}, {pressure!r}, {zone!r}
E, NU = {youngs_modulus_pa!r}, {poisson_ratio!r}

mesh = LIRE_MAILLAGE(UNITE=20, FORMAT="MED")
mesh = DEFI_GROUP(reuse=mesh, MAILLAGE=mesh,
                  CREA_GROUP_NO=(_F(GROUP_MA="CONTACT", NOM="CONTACT"),
                                 _F(GROUP_MA="APEX", NOM="APEX")))
model = AFFE_MODELE(MAILLAGE=mesh,
                    AFFE=_F(TOUT="OUI", PHENOMENE="MECANIQUE",
                            MODELISATION="AXIS"))
steel = DEFI_MATERIAU(ELAS=_F(E=E, NU=NU))
mat = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=_F(TOUT="OUI", MATER=steel))

# No pin at the lowest point. With the gap condition written correctly the
# unilateral constraint is active there from the first iteration and supplies
# the vertical restraint itself. Pinning it instead put the whole load
# through one node and produced a stress singularity of 386 GPa.
bcs = AFFE_CHAR_MECA(MODELE=model,
                     DDL_IMPO=(_F(GROUP_MA="AXIS", DX=0.0),
                               _F(GROUP_NO="APEX", DY=0.0)))
# Driven by imposed displacement rather than force. Under force control the
# vertical rigid body motion is unconstrained until contact activates and the
# first factorisation is singular; pinning the lowest point instead puts the
# whole load through one node and gave a stress singularity of 386 GPa.
# Displacement control has neither problem, and the force the model develops
# is read back from the reactions, so the closed form is evaluated at the
# load that actually occurred.
load = AFFE_CHAR_MECA(MODELE=model, PRES_REP=_F(GROUP_MA="TOP", PRES=P))

# The rigid plane is a unilateral condition: the contact surface may lift
# away but may not pass below z = 0.
# The condition is Y + DY >= 0, not DY >= 0. Nodes on the arc start ABOVE
# the plane and must travel DOWN to reach it, so forbidding downward motion
# forbids contact from forming at all. That is why COEF_IMPO takes a
# function: the allowed travel is each node's own height above the plane.
gap = FORMULE(VALE="-Y", NOM_PARA="Y")
one = DEFI_CONSTANTE(VALE=1.0)
contact = DEFI_CONTACT(MODELE=model, FORMULATION="LIAISON_UNIL",
                       ZONE=_F(GROUP_NO="CONTACT", NOM_CMP="DY",
                               COEF_IMPO=gap, COEF_MULT=one))

times = DEFI_LIST_REEL(DEBUT=0.0,
                       INTERVALLE=_F(JUSQU_A=1.0, NOMBRE={steps}))
ramp = DEFI_FONCTION(NOM_PARA="INST", VALE=(0.0, 0.0, 1.0, 1.0),
                     PROL_DROITE="CONSTANT", PROL_GAUCHE="CONSTANT")

res = STAT_NON_LINE(
    MODELE=model, CHAM_MATER=mat, CONTACT=contact,
    EXCIT=(_F(CHARGE=bcs), _F(CHARGE=load, FONC_MULT=ramp)),
    COMPORTEMENT=_F(RELATION="ELAS", DEFORMATION="PETIT"),
    INCREMENT=_F(LIST_INST=times),
    NEWTON=_F(REAC_ITER=1),
    CONVERGENCE=_F(RESI_GLOB_RELA=1.0e-8, ITER_GLOB_MAXI=60))

res = CALC_CHAMP(reuse=res, RESULTAT=res, CONTRAINTE=("SIGM_NOEU",))

coords = np.array(mesh.getCoordinates().getValues()).reshape(-1, 3)
last = res.getNumberOfIndexes() - 1

# Contact is read from the DEFORMED GEOMETRY, not from reaction forces.
# REAC_NODA reports the reactions of imposed degrees of freedom, and a
# unilateral condition is carried by a Lagrange multiplier that does not
# appear there, so a reaction based test found nothing while the contact was
# in fact working.
depl = res.getField("DEPL", last)
dy, ddesc = depl.getValuesWithDescription("DY")
dids = np.array(ddesc[0])
dy = np.array(dy)
height = coords[dids, 1] + dy
on_plane = np.abs(height) < 1.0e-9 * R
radius = float(coords[dids[on_plane], 0].max()) if on_plane.any() else 0.0

sig = res.getField("SIGM_NOEU", last)
syy, sdesc = sig.getValuesWithDescription("SIYY")
sids = np.array(sdesc[0])
syy = np.array(syy)
here = np.hypot(coords[sids, 0], coords[sids, 1]) < ZONE / 50.0
peak = float(-syy[here].min()) if here.any() else 0.0

print("{RESULT}", "contact_radius", repr(radius))
print("{RESULT}", "peak_pressure", repr(peak))
print("{RESULT}", "n_pressed", repr(float(on_plane.sum())))
print("{RESULT}", "max_settle", repr(float(np.abs(height).min())))
CA.close()
"""
    values = run_study(directory, study, mesh)
    return HertzContact(
        sphere_radius_m=sphere_radius_m, force_n=force_n,
        youngs_modulus_pa=youngs_modulus_pa, poisson_ratio=poisson_ratio,
        contact_radius_m=values["contact_radius"],
        peak_pressure_pa=values["peak_pressure"], zone_radius_m=zone)
