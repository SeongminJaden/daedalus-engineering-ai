# physics.dynamics: rigid-body dynamics and duty cycles

Statics answers "hold this pose". Dynamics answers "make this motion", which is
what an actuator is actually selected against.

```
tau = M(q) qdd + C(q, qd) qd + G(q) + F_friction
```

| module | role |
|---|---|
| `inertia.py` | link inertia tensors derived from the section geometry |
| `equations.py` | M, C, G, inverse dynamics, power |
| `load_cases.py` | the duty cycle, and peak vs continuous ratings |

## How each piece is verified

- **Inertia**: a hollow prism is a solid box minus a concentric cavity, so its
  tensor is the difference of two closed forms. Checked against the CAD kernel,
  which integrates the actual B-rep: agreement to **7.6e-16** relative.
  `is_valid_inertia` also enforces the triangle inequality on the principal
  moments, which is what separates a real rigid body from an arbitrary positive
  definite matrix.
- **M, C, G**: checked against the textbook two-link planar closed form. The
  implementation is general (Jacobians and Christoffel symbols, no planar
  assumption), so those formulas are an outside check. Measured worst errors:
  M **3.5e-18**, C **3.4e-12** (limited by the central differences used for
  `dM/dq`), G **5.6e-17**.
- **Statics bridge**: at zero velocity and acceleration the torque must equal
  the Phase 10 holding torque. `G` reuses the statics routine, so the limit is
  exact by construction and cannot drift. Measured difference: **0**.
- **Passivity**: `M_dot - 2C` is skew-symmetric. This is the property that
  separates a correct `C` from one that merely produces the right torque, and
  building `C` from Christoffel symbols makes it hold identically.

## Peak is not continuous

A motor has a peak rating it can hold briefly and a continuous rating it can
hold indefinitely, and they differ by a factor of two or three. Sizing to the
peak alone over-specifies the drive; sizing to the continuous value alone gives
one that overheats on every acceleration. Both are reported.

The continuous figure is the **RMS torque weighted by duty fraction**, because
motor heating goes as current squared and current tracks torque. A case with
zero duty fraction (a momentary worst-case combination) drives the peak but
deliberately does not raise the continuous rating.

Measured on the two-link arm: peak/continuous is **2.44x** at the shoulder and
**2.50x** at the elbow.

## A result worth reading carefully

The "dynamic share" column can go **negative**, as it does in the combined worst
case. That is not an error: at the worst-gravity pose the commanded acceleration
partly opposes gravity, so the total torque is smaller than the holding torque.
Acceleration does not always add.

## What this does not model

Friction, backlash and joint compliance are all **zero**. The terms exist so the
interface does not change when data arrives, but real values need breakaway
torque, a viscous coefficient and gearbox efficiency per joint, none of which
this project has. Inventing them would put fabricated numbers into a torque an
actuator gets selected from.

There is also no joint flexibility and no control dynamics. Everything is a
rigid body on an ideal joint.

**This phase produces the required torque and power. It does not select a motor
or a gearbox**: that needs efficiency, thermal and inertia-matching data and is
a later phase. Still SIMULATED.
