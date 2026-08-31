# drivetrain: motor and gearbox selection

Takes the Phase 11 duty cycle and selects a drive for each joint.

| package | role |
|---|---|
| `motors/` | BLDC archetypes |
| `gearboxes/` | planetary and harmonic archetypes |
| `selection/` | gear relations, margin checks, ranking, alternatives |

## The catalogue is not real parts

Every entry is a generic archetype tagged `illustrative`, with
`source = "representative archetype, replace with vendor datasheet"`. **None of
them corresponds to a vendor part number, and none was invented to look like
one.** A fabricated catalogue would be read later as if it had been sourced,
which is the same failure the material database avoids by tagging everything
`reference_typical`.

**The selection logic is the deliverable.** Replace the catalogue with datasheet
values before ordering anything.

The archetypes are at least internally consistent, and tests enforce it:
continuous torque times rated speed reproduces the nominal power of each size
class, peak is about 3x continuous, and planetary efficiency falls as ratio
rises because that means more stages.

## The relations

```
output torque   = motor torque * ratio * efficiency
motor speed     = joint speed * ratio
reflected load  = J_load / ratio^2              (at the motor shaft)
output inertia  = (J_rotor + J_gearbox) * ratio^2 + J_load
```

A high ratio buys torque and costs speed. That trade is what makes the choice
non-obvious, and it is why the speed check can fail a pairing that easily meets
the torque.

## Both ratings, always

A drive that meets the **peak** can still overheat holding the **continuous**
load, and one sized to the continuous value can stall on the first acceleration.
Two tests construct exactly those cases and confirm each is rejected, because
checking one alone would have accepted it.

The gearbox has its own ratings, independent of the motor. A larger motor cannot
rescue an undersized gearbox, and there is a test for that too.

## Output format

Every check reports Required against Available with a margin, and the
`limiting_check` names the one with the least headroom, which is what actually
sizes the drive. Alternatives are ranked and each carries a stated reason, so
the trade-off is a record rather than an assertion.

Ranking is by total mass, lightest first: on a serial arm every kilogram at a
joint is carried by every joint inboard of it, so mass is the cost that
compounds.

**When nothing is feasible the selector returns nothing** and reports which
check failed, in how many pairings, and by what factor. It does not hand back
the least bad option dressed up as a selection.

## What this does not do

- The **thermal check is a continuous-torque proxy**. A real one needs the duty
  profile, ambient temperature and thermal resistance. Results are labelled
  "subject to thermal validation".
- Efficiency and backlash are single representative values, not curves over
  speed, load and temperature.
- Friction and joint compliance are still zero, inherited from Phase 11.
- No cost, no lead time, no mounting or shaft interface, no controller matching.

This is **first-pass screening, not a final component decision**. Still
SIMULATED.
