# core.assembly: links, joints, kinematics and statics

Extends the project from a single part to a multi-body assembly.

| module | role |
|---|---|
| `frames.py` | coordinate conventions and homogeneous transforms |
| `model.py` | Link, Joint, Assembly; open-chain tree validation |
| `kinematics.py` | forward kinematics, geometric Jacobian, damped least squares IK |
| `statics.py` | holding torques and per-link load cases |

## Conventions, fixed so nothing has to guess

Right-handed frames, SI units. World frame is x forward, **y up**, z out of the
plane, and gravity acts along **-y**. That matches `configs/default.yaml` and
the beam model, where the section height and the tip load both run along y.
Choosing z-up here would have silently flipped the load direction relative to
every structural result already in the project.

A revolute joint rotates about its own axis, positive by the right-hand rule.
`joint_torques` returns the **actuator torque required to hold the pose**: a
horizontal arm with a downward payload needs positive torque to hold it up.

## What is checked, and against what

- **Forward kinematics** against the two-link closed form. The implementation is
  axis-and-origin based and knows nothing about planar arms, so the closed form
  is an outside check rather than a restatement. Agreement is to 1e-12.
- **Jacobian** against central differences of the FK, to better than 1e-6.
- **IK** by round trip: the solution goes back through FK and must reproduce the
  target. Unreachable targets are reported as failures rather than returned as
  confident wrong answers.
- **Static torques** against hand-computed moment sums, including each link's
  own weight at mid-span. Agreement is to 1e-12.

### One bug worth recording

The geometric Jacobian formula `Jv = z x (p - p_joint)` is only valid for a
point that the joint actually moves. Applied to every joint regardless, an
inboard link's centre of mass picks up a sensitivity to an outboard joint that
cannot move it. The resulting torque error was under 1%, which is small enough
to look plausible in a result table; the hand-computed moment sum is what caught
it. `supporting_joints` now restricts the columns, and a test pins it.

## Statics only

There is no inertia, no Coriolis term, no acceleration torque, no friction, no
backlash and no joint compliance. Everything is a rigid body on an ideal joint.

That covers "hold this pose against gravity and a payload", which is what sizes
a **link**. It does **not** size a **motor or a gearbox**: those need the dynamic
terms, and that is a later phase.

## Per-link load cases

Each link is checked as a cantilever carrying the `equivalent_tip_load_n` that
reproduces its **root bending moment**. Root moment drives bending stress and is
where the structure is critical, so matching it is the right equivalence for
sizing. It deliberately does not reproduce the distributed shape of self-weight
along the span, so a link whose own weight dominates its loading would need a
distributed-load model instead. For the payload-dominated cases here the tip
term dominates.
