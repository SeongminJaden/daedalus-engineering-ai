"""MuJoCo as a general multibody simulator, including contact.

Not a robot tool. The capability registered here is contact between rigid
bodies in any mechanism: a linkage, a dropped part, a block on a ramp, an
assembly settling into its fixture. Nothing about it is specific to arms, and
the benchmarks below are deliberately not robot problems.

This fills the one gap Pinocchio explicitly leaves. That node cross-checks
kinematics and dynamics for an open chain of ideal frictionless joints, which
is exactly where a real mechanism stops being ideal. Contact and friction are
what this one covers.

WHAT THIS IS NOT
================
MuJoCo's contact is a SOFT CONSTRAINT, a regularised spring and damper rather
than a hard non-penetration condition. Two consequences run through
everything here:

    A resting body sinks slightly into the surface it rests on, and a body
    below the friction limit creeps slowly rather than sticking exactly.
    Displacement therefore CANNOT distinguish sticking from sliding; only a
    growing velocity can. The friction benchmark uses that test, after the
    displacement version reported a block as sliding at 10 degrees when the
    limit is 21.8.

    There is NO coefficient of restitution. A bounce emerges from the contact
    stiffness and damping in solref, so restitution is measured here and never
    set. Agreement with a target bounce is a statement about the tuning, not
    about the material.

VALIDITY DOMAIN
===============
Stated before implementing, per the standing discipline.

Applies
    Rigid bodies, with contact and Coulomb friction, integrated at a timestep
    compatible with the contact stiffness.

Does not apply
    Deformable bodies, wear, adhesion, lubrication, and any material response
    beyond a friction coefficient.

The stability requirement, which is not optional
    solref and the timestep are coupled, and violating the coupling does not
    raise: it INFLATES ENERGY while continuing to run. Measured on a dropped
    ball with dampratio 0.05: at a 2e-4 timestep the total energy went from
    0.158 J to 17.07 J, a factor of 108, and the run looked normal. At 2e-5
    and 2e-6 the same model conserved energy exactly. A contact simulation
    that was never energy checked is not evidence of anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.assembly.frames import GRAVITY_DIRECTION, STANDARD_GRAVITY
from core.assembly.model import Assembly
from core.assembly.urdf import assembly_to_urdf

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

MUJOCO_NODE_NAME = "mujoco.local"
MUJOCO_CAPABILITY = "simulation.multibody_contact"


def _mujoco():
    try:
        import mujoco
    except ImportError:
        return None
    return mujoco


def is_available() -> bool:
    return _mujoco() is not None


def version() -> str | None:
    module = _mujoco()
    return f"MuJoCo {module.__version__}" if module else None


def _require():
    module = _mujoco()
    if module is None:
        raise CapabilityUnavailable(
            MUJOCO_CAPABILITY, MUJOCO_NODE_NAME,
            "the mujoco package is not installed")
    return module


def mujoco_descriptor(available: bool | None = None) -> NodeDescriptor:
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=MUJOCO_NODE_NAME, transport=Transport.STDIO, address="mujoco",
        available=present,
        unavailable_reason="" if present else
        "unavailable: the mujoco package is not installed")


def mujoco_capability_method():
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=MUJOCO_CAPABILITY,
        category=Category.ANALYSIS,
        summary="General rigid multibody simulation with contact and friction, "
                "for any mechanism rather than robots specifically.",
        inputs=("bodies", "contacts", "initial_state"),
        outputs=("trajectory", "contact_forces", "energy"),
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("the problem has bodies that move and can touch",
                      lambda c: c.require("has_contact")),
        ),
        implementation="nodes.mujoco_node.simulate_assembly",
        evidence="SIMULATED",
        notes="General purpose, not robot specific: any rigid multibody system "
              "with contact goes through it. Contact is a SOFT CONSTRAINT, a "
              "regularised spring and damper, so a resting body sinks slightly "
              "and a stuck body creeps; displacement cannot tell sticking from "
              "sliding and only a growing velocity can. There is no "
              "restitution coefficient, so bounce is measured from the contact "
              "parameters rather than specified. solref and the timestep are "
              "coupled and violating that INFLATES ENERGY rather than raising: "
              "a dropped ball went from 0.158 J to 17.07 J at too large a "
              "step, and conserved energy exactly at a smaller one. Agreement "
              "on a known benchmark means the settings are right for that "
              "case, not that arbitrary contact is right.")


def load_assembly(assembly: Assembly, density_kg_m3: float,
                  urdf_path: Path | None = None):
    """Load one of this project's assemblies into MuJoCo, through URDF.

    The same exported URDF the Pinocchio node reads, so the three
    implementations compare the same model rather than three transcriptions.

    Two conventions are corrected here, and neither is cosmetic. MuJoCo
    defaults to gravity along -z at 9.81; this project is y-up at 9.80665.
    And MuJoCo would enforce the joint limits carried in the URDF as
    constraints, which this project records but does not enforce, so they are
    switched off to keep the comparison like for like.
    """
    import tempfile

    mujoco = _require()
    if urdf_path is None:
        urdf_path = Path(tempfile.mkdtemp()) / f"{assembly.name}.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(assembly_to_urdf(assembly, density_kg_m3))

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    model.opt.gravity[:] = np.asarray(GRAVITY_DIRECTION,
                                      dtype=np.float64) * STANDARD_GRAVITY
    model.jnt_limited[:] = 0
    return model, mujoco.MjData(model), urdf_path


def accelerations(assembly: Assembly, q, qd, tau, density_kg_m3: float
                  ) -> np.ndarray:
    """Joint accelerations from MuJoCo, for comparison with this project's."""
    mujoco = _require()
    model, data, _ = load_assembly(assembly, density_kg_m3)
    data.qpos[:] = np.asarray(q, dtype=np.float64)
    data.qvel[:] = np.asarray(qd, dtype=np.float64)
    data.qfrc_applied[:] = np.asarray(tau, dtype=np.float64)
    mujoco.mj_forward(model, data)
    return np.array(data.qacc, dtype=np.float64)


@dataclass(frozen=True)
class ContactRun:
    """A contact simulation, with the energy history that validates it."""

    solver: str
    solver_version: str
    time_s: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    energy_j: np.ndarray

    def rebound_apex_m(self, radius_m: float = 0.02) -> float:
        """Highest point reached AFTER the first contact.

        The raw maximum is the drop height, which is the same whatever the
        contact does, so it measures nothing about the bounce.
        """
        touched = np.flatnonzero(self.position <= radius_m * 1.02)
        if touched.size == 0:
            return float("nan")
        after = self.position[touched[0]:]
        return float(after.max()) if after.size else float("nan")

    @property
    def energy_grew(self) -> bool:
        """True when the contact model added energy, which is never physical.

        The check that catches an unstable timestep. A run that fails this is
        not a soft result, it is not a result.
        """
        return bool(self.energy_j.max() > self.energy_j[0] * 1.001 + 1e-12)


def _model_from_xml(xml: str):
    return _require().MjModel.from_xml_string(xml)


def block_on_incline_xml(friction: float, angle_deg: float,
                         timestep_s: float = 5e-4, size_m: float = 0.05,
                         density_kg_m3: float = 1000.0) -> str:
    """A block on a slope, posed by tilting GRAVITY rather than the floor.

    Identical mechanics and a much simpler model: the normal direction stays
    along z, so the contact geometry is unchanged and only the load rotates.
    """
    theta = math.radians(angle_deg)
    gx = STANDARD_GRAVITY * math.sin(theta)
    gz = -STANDARD_GRAVITY * math.cos(theta)
    return f"""
<mujoco>
  <option timestep="{timestep_s}" gravity="{gx} 0 {gz}"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"
          friction="{friction} 0 0"/>
    <body name="block" pos="0 0 {size_m + 1e-4}">
      <freejoint/>
      <geom name="block" type="box" size="{size_m} {size_m} {size_m}"
            density="{density_kg_m3}" friction="{friction} 0 0"/>
    </body>
  </worldbody>
</mujoco>"""


def block_slides(friction: float, angle_deg: float, settle_s: float = 1.0,
                 timestep_s: float = 5e-4) -> bool:
    """Whether the block is SLIDING, judged by velocity and not by position.

    Soft contact lets a held block creep, so displacement says nothing: the
    first version of this test called a block sliding at 10 degrees when the
    limit is 21.8. A held block creeps at a steady crawl; a sliding one
    accelerates, so comparing the velocity across two equal intervals is what
    separates them.
    """
    mujoco = _require()
    model = _model_from_xml(block_on_incline_xml(friction, angle_deg,
                                                 timestep_s))
    data = mujoco.MjData(model)
    steps = int(round(settle_s / timestep_s))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    first = float(data.qvel[0])
    for _ in range(steps):
        mujoco.mj_step(model, data)
    second = float(data.qvel[0])
    return second > first + 1e-4 and second > 1e-3


def critical_angle_deg(friction: float, tolerance_deg: float = 0.02) -> float:
    """The measured slope at which sliding begins, found by bisection.

    Compare against degrees(atan(friction)), which is the exact Coulomb
    result. They do not match perfectly and should not: see the tests.
    """
    low, high = 1.0, 60.0
    while high - low > tolerance_deg:
        middle = 0.5 * (low + high)
        if block_slides(friction, middle):
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def resting_contact_force_n(size_m: float = 0.05,
                            density_kg_m3: float = 1000.0,
                            settle_s: float = 2.0,
                            timestep_s: float = 5e-4) -> tuple[float, float]:
    """Summed normal force under a resting block, and its weight.

    The simplest statement contact can make: what holds a body up must equal
    what pulls it down.
    """
    mujoco = _require()
    model = _model_from_xml(block_on_incline_xml(1.0, 0.0, timestep_s, size_m,
                                                 density_kg_m3))
    data = mujoco.MjData(model)
    for _ in range(int(round(settle_s / timestep_s))):
        mujoco.mj_step(model, data)
    total = np.zeros(6)
    force = np.zeros(6)
    for index in range(data.ncon):
        mujoco.mj_contactForce(model, data, index, force)
        total += force
    weight = float(mujoco.mj_getTotalmass(model)) * STANDARD_GRAVITY
    return float(total[0]), weight


def drop_ball(damping_ratio: float, timestep_s: float = 2e-5,
              height_m: float = 0.5, radius_m: float = 0.02,
              duration_s: float = 4.0, time_constant_s: float = 0.002
              ) -> ContactRun:
    """Drop a ball and record its energy, which is the point of the exercise.

    `damping_ratio` is the second solref number. There is no restitution to
    set: how much bounce comes back is a consequence of this and of the
    timestep, and is measured afterwards.
    """
    mujoco = _require()
    xml = f"""
<mujoco>
  <option timestep="{timestep_s}" gravity="0 0 {-STANDARD_GRAVITY}"/>
  <worldbody>
    <geom type="plane" size="10 10 0.1"
          solref="{time_constant_s} {damping_ratio}"/>
    <body pos="0 0 {height_m}">
      <freejoint/>
      <geom type="sphere" size="{radius_m}" density="1000"
            solref="{time_constant_s} {damping_ratio}"/>
    </body>
  </worldbody>
</mujoco>"""
    model = _model_from_xml(xml)
    data = mujoco.MjData(model)
    mass = float(mujoco.mj_getTotalmass(model))
    steps = int(round(duration_s / timestep_s))

    times = np.zeros(steps)
    heights = np.zeros(steps)
    speeds = np.zeros(steps)
    energies = np.zeros(steps)
    for step in range(steps):
        mujoco.mj_step(model, data)
        z, v = float(data.qpos[2]), float(data.qvel[2])
        times[step] = data.time
        heights[step] = z
        speeds[step] = v
        energies[step] = (0.5 * mass * v * v
                          + mass * STANDARD_GRAVITY * (z - radius_m))
    return ContactRun(solver="MuJoCo", solver_version=version() or "unknown",
                      time_s=times, position=heights, velocity=speeds,
                      energy_j=energies)


def simulate_assembly(assembly: Assembly, q0, qd0, density_kg_m3: float,
                      duration_s: float, timestep_s: float = 1e-3):
    """Integrate one of this project's assemblies in MuJoCo, unforced."""
    mujoco = _require()
    model, data, _ = load_assembly(assembly, density_kg_m3)
    model.opt.timestep = timestep_s
    data.qpos[:] = np.asarray(q0, dtype=np.float64)
    data.qvel[:] = np.asarray(qd0, dtype=np.float64)

    steps = int(round(duration_s / timestep_s))
    times = np.zeros(steps + 1)
    positions = np.zeros((steps + 1, model.nq))
    velocities = np.zeros((steps + 1, model.nv))
    positions[0], velocities[0] = data.qpos, data.qvel
    for step in range(steps):
        mujoco.mj_step(model, data)
        times[step + 1] = data.time
        positions[step + 1] = data.qpos
        velocities[step + 1] = data.qvel
    return times, positions, velocities
