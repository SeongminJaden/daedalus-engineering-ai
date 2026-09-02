# Measurement and evaluation guideline

This document is a procedure, not evidence. Nothing in it raises the grade of
anything in the repository. The grade rises when measured data enters the
Brain as `PHYSICAL_TEST` evidence through the format below and the checks
pass, and not before. Until then every result here is SIMULATED or lower, and
the code says so.

## What is being compared

A part designed here carries labels from a solver (deflection, twist,
elongation, stress, mass) and geometry from the analyzer (volume, bounding
box, holes, fillets). The comparison is between those numbers and what a
manufactured part does on a bench. The prediction is fixed before the test
and written into the test record; a prediction adjusted after seeing the
measurement is not a prediction.

## Quantities, instruments, and what agreement means

| quantity | how it is measured | instrument, resolution | tolerance for agreement | note |
|---|---|---|---|---|
| dimensions | outer dimensions, hole diameters, wall thickness at three places | calipers or micrometer, 0.01 mm; CMM where available | manufacturing tolerance of the process, stated per part (machined aluminium about 0.05 mm, FDM about 0.3 mm) | checks the part IS the part that was analysed; a mismatch stops the test |
| mass | scale | 0.1 g or 0.1 percent of mass, whichever is finer | 2 percent | density is a datasheet value; a 2 percent miss is normal for a typical density |
| tip deflection | dial indicator or laser displacement sensor at the loaded face, load applied by calibrated weights or a load cell | 0.01 mm; load 0.5 percent | 10 percent, after the clamp compliance test below | the model clamps a face perfectly; a real clamp does not, and that dominates small deflections |
| twist | angle of the free face under a known torque, from two displacement gauges across a known lever | 0.01 mm at a 50 mm lever, about 0.2 mrad | 10 percent | the torque arm and its friction are part of the measurement and must be written down |
| natural frequency | accelerometer and impact hammer, or a microphone and a tap for a first estimate | 1 Hz below 1 kHz | 5 percent | the model does not compute this for the CAD families yet; where it does (beam evaluator), clamped-free assumption applies |
| strain or stress | strain gauge at a stated location, NOT at the clamped edge | gauge factor as certified | 15 percent at a gauge site away from the clamp | the model's peak stress sits at a singularity and is not a measurable quantity; compare at a gauge location the model can be read at |
| motor torque | current and torque constant, or an inline torque transducer | 2 percent | 10 percent | for assemblies; the drivetrain catalogues are archetypes until vendor data replaces them |
| temperature | thermocouple or thermistor at stated locations, ambient recorded | 0.5 K | 2 K or 10 percent of the rise | for the thermal cases; the expansion coefficient is a datasheet value |

"Agreement" is the model within the tolerance of the measurement at the
stated location. Agreement at one location for one load does not validate
the model; it validates that one prediction.

## Order of operations

1. **Measure the part before loading it.** Dimensions and mass first. If the
   part is not within the process tolerance of the STEP file, record that and
   stop: a test on a different part validates nothing about the design.
2. **Measure the fixture.** Apply the load with no specimen where the
   fixture allows it, or with a stiff dummy, and record the fixture's own
   displacement. Clamp compliance is the largest systematic error in a
   cantilever test and must be subtracted or, if it is above 20 percent of
   the expected deflection, the fixture is redesigned before any comparison.
3. **Load in steps.** At least five load levels up and down. Linearity and
   hysteresis are results too: a nonlinear curve says the model's small
   strain assumption or the clamp is wrong before any single number does.
4. **Repeat.** Three mountings of the same part. The scatter between them is
   the measurement's own uncertainty and goes into the record. One reading
   is an anecdote.
5. **Write the record before comparing.** The format below, with the
   prediction that was made before the test.

## The record format

One JSON object per measurement, validated by
`brain.semantic.physical_test.validate_measurement` (to be added when the
first record exists; the schema here is the contract):

```json
{
  "record_version": "0.1.0",
  "part_id": "hollow_rect-08c59fe35f",
  "step_sha256": "...",
  "as_built": {
    "dimensions_m": {"length": 0.2001, "height": 0.0400, "width": 0.0299,
                     "wall_at_three_places": [0.00301, 0.00298, 0.00302]},
    "mass_kg": 0.2559,
    "process": "cnc_milling",
    "material_lot": "6061-T6 bar, supplier and heat lot as stamped",
    "within_process_tolerance": true
  },
  "fixture": {"description": "vise, 40 mm jaw, 30 mm insertion",
              "compliance_m_per_n": 1.2e-7,
              "load_application": "hanging weights on a hook 5 mm from the free face"},
  "prediction": {"quantity": "tip_deflection_m", "value": -1.447e-4,
                 "load_n": -100.0, "source": "records.jsonl, labelled 2026-09-03",
                 "evidence": "simulated"},
  "measurement": {"quantity": "tip_deflection_m",
                  "loads_n": [-20, -40, -60, -80, -100, -80, -60, -40, -20],
                  "values_m": [...], "mountings": 3,
                  "fixture_corrected": true,
                  "instrument": "dial indicator 0.01 mm",
                  "ambient_k": 295.2, "date": "2026-..-..", "operator": "initials"},
  "result": {"measured_at_100n_m": -1.52e-4, "scatter_m": 4e-6,
             "relative_error": 0.05, "tolerance": 0.10, "agrees": true}
}
```

## Validation rules the loader applies

- `record_version` is the one the code writes; an unknown version is refused.
- `step_sha256` matches a STEP file the repository can reproduce from the
  part id; a record about a part that cannot be rebuilt is refused.
- `as_built.within_process_tolerance` must be true for the record to count
  as a test of the design; otherwise it is stored as a test of a different
  part and enters the Brain as a counterexample only if the design is
  blamed for it explicitly.
- `prediction.evidence` must be `simulated`, and the prediction value must
  equal the stored label for the part id; a prediction typed by hand is
  refused.
- `measurement.mountings` at least 3 and `loads_n` at least 5 distinct
  levels including an unloading branch, or the record is stored as
  preliminary and does not create evidence.
- `fixture_corrected` must be true when the fixture compliance exceeds 5
  percent of the predicted deflection.
- `result.agrees` is computed by the loader from the tolerance table, never
  read from the record.

## What the Brain does with it

A record that passes creates one `Evidence(kind=PHYSICAL_TEST)` on the
statement "the labelled deflection of part X under load Y agrees with
measurement within Z", with the record's identifier as `ref` and the test
campaign as `run_id`. That is the only key to `EXPERIMENTALLY_VALIDATED`,
and by the ladder's own rule it opens for that statement alone. A record
that fails creates a `Counterexample` on the same statement, which caps the
statement below HIGH_CONFIDENCE until it is resolved by an explanation that
is itself written down.

Neither outcome changes the grade of any other part, any other load case, or
the solver in general. Ten agreeing parts make ten validated statements and
one strong reason to trust the eleventh; they do not make the eleventh
validated.

## What this document does not do

It does not describe how to build the part, choose the material lot, or run
the machine. It does not set safety procedures for loading; a 100 N hanging
weight and a 5 N m torque arm are small, and larger tests need their own
assessment. It does not exist to be satisfied by a single successful test:
the purpose of the procedure is to make a disagreement between the model and
the world visible and recorded, because that is the only way the model gets
better.
