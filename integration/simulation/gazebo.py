"""Gazebo as a cross-check on the assembly statics, not as a new capability.

The machine this was written on has Gazebo Fortress (ign gazebo 6.18.0) with
ROS 2 Humble's ros_gz, and no Isaac Sim; Isaac Sim's published requirements
(16 GB of GPU memory, 32 GB of RAM) are above this machine's 4 GB and 15 GB,
so it is not installed and nothing here assumes it. Everything goes through
the `ign` command line: URDF to SDF conversion, a headless server run, and a
topic echo for the joint states. Fortress ships no Python bindings.

WHAT IS COMPARED, AND HOW
=========================
The assembly is exported as URDF with envelope boxes, converted to SDF, and
placed in a world whose gravity is -y (this project's frame; SDF is told).
The pose q is folded into the joint origins so every joint starts at zero,
and each actuated joint gets a torsional SPRING whose rest angle is
tau / k, so that at the start the spring applies exactly the torque tau the
caller supplies. If tau is the torque that holds the pose, the joints stay
at zero; if it is not, they settle where the spring balances the shortfall,
and the settled angle is the measurement. At rest the spring torque equals
whatever gravity torque the physics engine computes at that pose, so
comparing it with `core.assembly.statics.joint_torques` at the SETTLED pose
compares two independent statics: this project's Jacobian and DART's
constraint solver. Agreement grades SIMULATED; both are simulations.

MEASURED on the planar two link arm (300 and 250 mm hollow links, 2810
kg/m3), spring 2 N m/rad, damping 0.5, four seconds of simulated time:

    torque applied            settled angle (rad)        spring vs statics
    the statics torque        -0.0002, -0.0000           0.03 percent
    zero                      -0.4399, -0.0730           0.03 percent
    half the statics torque   -0.2315, -0.0387           0.04 percent

The zero row is the control: it shows the test can fail (the arm sags by
0.44 rad) and that the engine's equilibrium still matches the statics at the
pose it sagged to.

WHAT WAS TRIED AND IS NOT HERE
==============================
A JointPositionController hold, reading the torque off the proportional
error, sat 0.3 rad from its target with a torque estimate sixteen times the
statics; the plugin's semantics in this build were not established and the
route was removed rather than left as an option. An ApplyJointForce hold
fed by one-shot `ign topic -p` publishes never received a message (the
drift was the same for the statics torque, zero and its negative). Two
capture faults were found on the way and are in the code as comments: the
first N messages are not the last N, and a pipe fills at 64 KB.

INTERFERENCE is checked geometrically, not by the engine: the envelope boxes
are placed by this project's forward kinematics and their pairwise overlap
volumes are computed with the CAD kernel. An overlap is an overlap of
ENVELOPES, which the parts inside may or may not share.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.assembly.model import Assembly, JointType
from core.assembly.statics import joint_torques
from core.assembly.urdf import assembly_to_urdf

IGN = "ign"


class GazeboUnavailable(RuntimeError):
    """The ign command line is not on this machine."""


def gazebo_available() -> bool:
    return shutil.which(IGN) is not None


def gazebo_version() -> str | None:
    if not gazebo_available():
        return None
    out = subprocess.run([IGN, "gazebo", "--version"], capture_output=True, text=True,
                         timeout=30)
    match = re.search(r"version\s+([\d.]+)", out.stdout)
    return match.group(1) if match else out.stdout.strip()[:40]


def urdf_to_sdf(urdf_path: Path) -> str:
    """`ign sdf -p` converts a URDF file to an SDF model document."""
    if not gazebo_available():
        raise GazeboUnavailable("ign is not on PATH")
    out = subprocess.run([IGN, "sdf", "-p", str(urdf_path)], capture_output=True,
                         text=True, timeout=60)
    if out.returncode != 0 or "<model" not in out.stdout:
        raise RuntimeError(f"ign sdf -p failed: {out.stderr[-500:]}")
    return out.stdout


def posed_copy(assembly: Assembly, q) -> Assembly:
    """The same assembly with q folded into the joint origins, so that the
    zero configuration of the copy is the pose q of the original."""
    from core.assembly.frames import compose, rotation_about_axis, translation_along_axis

    q = np.asarray(q, dtype=float).reshape(-1)
    values = dict(zip([j.name for j in assembly.actuated_joints()], q))
    joints = []
    for joint in assembly.joints:
        if joint.name in values:
            origin = np.asarray(joint.origin_transform(), dtype=float)
            motion = (rotation_about_axis(joint.axis, values[joint.name])
                      if joint.type is JointType.REVOLUTE
                      else translation_along_axis(joint.axis, values[joint.name]))
            joints.append(joint.model_copy(update={"origin": compose(origin, motion).tolist()}))
        else:
            joints.append(joint)
    return assembly.model_copy(update={"joints": joints})


def _parse_joint_state(text: str) -> dict[str, float]:
    """Joint positions from the LAST ignition.msgs.Model in an echo."""
    messages = [m for m in re.split(r"\n(?=name: )", text) if "joint {" in m]
    if not messages:
        return {}
    positions: dict[str, float] = {}
    for block in re.finditer(r"joint \{(.*?)\n\}", messages[-1], flags=re.S):
        body = block.group(1)
        name = re.search(r'name: "([^"]+)"', body)
        pos = re.search(r"position: ([-\d.eE+]+)", body)
        if name:
            positions[name.group(1)] = float(pos.group(1)) if pos else 0.0
    return positions


@dataclass
class GazeboRun:
    world: Path
    iterations: int
    seconds: float
    joint_positions: dict[str, float] = field(default_factory=dict)
    messages: int = 0
    server_log_tail: str = ""


def run_headless(world_path: Path, world_name: str, model_name: str,
                 iterations: int, timeout_s: float = 180.0) -> GazeboRun:
    """Run the server headless for `iterations` steps and keep the LAST joint
    state it published.

    Two things this does on purpose, both measured: the echo streams for the
    whole run and is stopped after the server exits, because asking it for a
    fixed count returned the first messages at time zero; and it writes to a
    FILE, because a pipe read at the end blocked at 64 KB, about 65 messages,
    so every earlier "last message" was really the 65th.
    """
    if not gazebo_available():
        raise GazeboUnavailable("ign is not on PATH")
    topic = f"/world/{world_name}/model/{model_name}/joint_state"
    capture = world_path.with_suffix(".joint_state.txt")
    started = time.perf_counter()
    with capture.open("w") as sink:
        echo = subprocess.Popen([IGN, "topic", "-e", "-t", topic],
                                stdout=sink, stderr=subprocess.DEVNULL, text=True)
        time.sleep(1.0)
        server = subprocess.run([IGN, "gazebo", "-s", "-r", "--iterations",
                                 str(iterations), "-v", "1", str(world_path)],
                                capture_output=True, text=True, timeout=timeout_s)
        time.sleep(0.5)
        echo.terminate()
        try:
            echo.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            echo.kill()
    out = capture.read_text()
    positions = _parse_joint_state(out)
    return GazeboRun(world=world_path, iterations=iterations,
                     seconds=time.perf_counter() - started, joint_positions=positions,
                     messages=out.count("joint {") // max(len(positions), 1),
                     server_log_tail=(server.stdout + server.stderr)[-800:])


def write_spring_world(posed: Assembly, density_kg_m3: float, torques_nm,
                       directory: str | Path, stiffness_nm_rad: float = 2.0,
                       damping_nm_s_rad: float = 0.5) -> tuple[Path, str, str]:
    """URDF, SDF model and a real-time world with a preloaded spring on
    every actuated joint. Returns (world path, world name, model name)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    urdf_path = directory / f"{posed.name}.urdf"
    urdf_path.write_text(assembly_to_urdf(posed, density_kg_m3, envelopes=True))
    model = re.search(r"<model\b.*</model>", urdf_to_sdf(urdf_path), flags=re.S).group(0)
    model_name = re.search(r"""<model\s+name=['"]([^'"]+)['"]""", model).group(1)
    torques = np.asarray(torques_nm, dtype=float).reshape(-1)
    for joint, tau in zip(posed.actuated_joints(), torques):
        block = re.search(rf"<joint name='{joint.name}'.*?</joint>", model, flags=re.S)
        if block is None:
            raise RuntimeError(f"joint {joint.name} not found in the converted SDF")
        dynamics = (f"<dynamics><spring_reference>{float(tau / stiffness_nm_rad)!r}"
                    f"</spring_reference><spring_stiffness>{float(stiffness_nm_rad)!r}"
                    f"</spring_stiffness><damping>{float(damping_nm_s_rad)!r}"
                    f"</damping></dynamics>")
        model = model.replace(block.group(0), re.sub(r"<dynamics>.*?</dynamics>", dynamics,
                                                     block.group(0), flags=re.S), 1)
    model = model.replace("</model>",
                          '<joint name="world_fixed" type="fixed"><parent>world</parent>'
                          '<child>base</child></joint><self_collide>false</self_collide>'
                          '<plugin filename="ignition-gazebo-joint-state-publisher-system" '
                          'name="ignition::gazebo::systems::JointStatePublisher"></plugin>'
                          "</model>", 1)
    world_name = f"{posed.name}_spring"
    world = f'''<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="{world_name}">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor></physics>
    <gravity>0 -9.81 0</gravity>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"></plugin>
    {model}
  </world>
</sdf>
'''
    path = directory / f"{world_name}.sdf"
    path.write_text(world)
    return path, world_name, model_name


@dataclass
class SpringHold:
    joint_names: list[str]
    applied_nm: np.ndarray          # torque the spring applies at the start pose
    settled_rad: np.ndarray         # joint angles at the end, relative to the pose
    stiffness_nm_rad: float
    statics_at_settled_nm: np.ndarray
    seconds_simulated: float
    messages: int

    @property
    def spring_torque_at_settled_nm(self) -> np.ndarray:
        return self.applied_nm - self.stiffness_nm_rad * self.settled_rad

    @property
    def relative_errors(self) -> np.ndarray:
        """Spring torque against this project's statics at the settled pose."""
        scale = np.maximum(np.abs(self.statics_at_settled_nm), 1e-9)
        return np.abs(self.spring_torque_at_settled_nm - self.statics_at_settled_nm) / scale

    @property
    def max_drift_rad(self) -> float:
        return float(np.nanmax(np.abs(self.settled_rad)))

    @property
    def evidence(self) -> str:
        return "simulated"

    def summary(self) -> str:
        rows = [f"{n}: applied {a:+.4f} N m, settled {s:+.4f} rad, spring {g:+.4f} N m, "
                f"statics {t:+.4f} N m, error {e:.2%}"
                for n, a, s, g, t, e in zip(self.joint_names, self.applied_nm, self.settled_rad,
                                            self.spring_torque_at_settled_nm,
                                            self.statics_at_settled_nm, self.relative_errors)]
        return "\n".join(rows) + f"\n(two simulations compared, {self.evidence})"


def hold_with_springs(assembly: Assembly, density_kg_m3: float, q, torques_nm,
                      directory: str | Path | None = None, seconds: float = 4.0,
                      stiffness_nm_rad: float = 2.0, damping_nm_s_rad: float = 0.5
                      ) -> SpringHold:
    """Start at q with springs preloaded to `torques_nm`, run in real time,
    report where the joints settled and the statics torque there."""
    if not gazebo_available():
        raise GazeboUnavailable("ign is not on PATH")
    directory = Path(directory) if directory else Path(tempfile.mkdtemp())
    posed = posed_copy(assembly, q)
    world, world_name, model_name = write_spring_world(
        posed, density_kg_m3, torques_nm, directory, stiffness_nm_rad, damping_nm_s_rad)
    run = run_headless(world, world_name, model_name, int(round(seconds / 0.001)),
                       timeout_s=seconds + 60.0)
    if not run.joint_positions:
        raise RuntimeError("no joint state captured; server log: " + run.server_log_tail)
    names = [j.name for j in assembly.actuated_joints()]
    settled = np.array([run.joint_positions.get(n, np.nan) for n in names])
    q = np.asarray(q, dtype=float).reshape(-1)
    return SpringHold(joint_names=names, applied_nm=np.asarray(torques_nm, dtype=float),
                      settled_rad=settled, stiffness_nm_rad=stiffness_nm_rad,
                      statics_at_settled_nm=joint_torques(assembly, q + settled, density_kg_m3),
                      seconds_simulated=seconds, messages=run.messages)


def statics_cross_check(assembly: Assembly, density_kg_m3: float, q,
                        directory: str | Path | None = None, seconds: float = 4.0) -> SpringHold:
    """Apply this project's statics torque through springs and see whether
    the engine agrees that it holds the pose."""
    tau = joint_torques(assembly, np.asarray(q, dtype=float), density_kg_m3)
    return hold_with_springs(assembly, density_kg_m3, q, tau, directory, seconds)


# --------------------------------------------------------------------------- #
# envelope interference, by this project's kinematics and the CAD kernel
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EnvelopeClash:
    first: str
    second: str
    overlap_m3: float


@dataclass
class EnvelopeInterference:
    q: np.ndarray
    clashes: list[EnvelopeClash]
    checked_pairs: int

    @property
    def clear(self) -> bool:
        return not self.clashes

    def summary(self) -> str:
        if self.clear:
            return f"{self.checked_pairs} envelope pairs checked, none overlap"
        return "; ".join(f"{c.first} and {c.second} overlap by {c.overlap_m3:.3e} m3"
                         for c in self.clashes) + " (envelopes, not parts)"


def envelope_interference(assembly: Assembly, q, tolerance_m3: float = 1e-12,
                          skip_adjacent: bool = True) -> EnvelopeInterference:
    """Place each link's envelope box by forward kinematics and intersect
    every pair. Adjacent links share a joint and touch there by construction;
    they are skipped unless asked for, and a touching face is zero volume
    anyway. Envelopes, not parts."""
    from core.assembly.kinematics import forward_kinematics
    from geometry.cad_export.hollow_rect import METRES_TO_MM
    from geometry.cad_export.kernel import require_kernel

    kernel = require_kernel()
    b = kernel.module
    pose = forward_kinematics(assembly, np.asarray(q, dtype=float))
    solids = {}
    for link in assembly.links:
        section = link.genome.section
        box = b.Pos(link.length_m / 2 * METRES_TO_MM, 0, 0) * b.Box(
            link.length_m * METRES_TO_MM, section.outer_height_m * METRES_TO_MM,
            section.outer_width_m * METRES_TO_MM)
        T = np.asarray(pose.link_transforms[link.name], dtype=float)
        # Placed through OpenCASCADE directly: gp_Trsf takes the twelve
        # numbers of a rigid transform, translation in the kernel's mm.
        from OCP.gp import gp_Trsf
        from OCP.TopLoc import TopLoc_Location
        trsf = gp_Trsf()
        trsf.SetValues(*[float(v) for v in T[0, :3]], float(T[0, 3]) * METRES_TO_MM,
                       *[float(v) for v in T[1, :3]], float(T[1, 3]) * METRES_TO_MM,
                       *[float(v) for v in T[2, :3]], float(T[2, 3]) * METRES_TO_MM)
        solids[link.name] = b.Solid(box.wrapped.Moved(TopLoc_Location(trsf)))
    parent_of = {j.child: j.parent for j in assembly.joints}
    names = [l.name for l in assembly.links]
    clashes, pairs = [], 0
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            adjacent = parent_of.get(first) == second or parent_of.get(second) == first
            if skip_adjacent and adjacent:
                continue
            pairs += 1
            # the & operator returns the common Solid; .intersect returns a
            # ShapeList with no volume, which read as "no overlap" until the
            # folded arm test said otherwise
            overlap = solids[first] & solids[second]
            volume = float(getattr(overlap, "volume", 0.0) or 0.0) / METRES_TO_MM ** 3
            if volume > tolerance_m3:
                clashes.append(EnvelopeClash(first, second, volume))
    return EnvelopeInterference(q=np.asarray(q, dtype=float), clashes=clashes,
                                checked_pairs=pairs)
