"""The specification for the six axis arm, and every number's origin.

This is the input to the whole pipeline, written once so that every stage
reads the same numbers and a change is a change in one place. Values marked
GIVEN came with the task. Values marked CHOSEN were picked here because the
task did not state them, and each says why; a measurement that contradicts one
is a reason to change it, and the design document records both the choice and
what happened to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JointSpec:
    """One axis of the arm."""

    name: str
    axis: tuple[float, float, float]
    origin_x_m: float = 0.0
    #: Along the vertical. This project's gravity is -y (core.assembly.frames),
    #: so the column rises in y and the pitch axes are z.
    origin_y_m: float = 0.0
    lower_limit_rad: float = -3.14
    upper_limit_rad: float = 3.14
    description: str = ""


@dataclass(frozen=True)
class LinkSpec:
    """One structural member, before any design has been done to it."""

    name: str
    length_m: float
    #: The starting section. The design stages replace it; it exists so the
    #: first dynamics pass has a mass to work with, and the iteration loop
    #: reports how far the final section moved from it.
    outer_height_m: float
    outer_width_m: float
    wall_thickness_m: float
    material_id: str
    role: str = ""


@dataclass(frozen=True)
class ManipulatorSpec:
    # --- GIVEN ---
    payload_kg: float = 3.0
    reach_m: float = 0.600
    degrees_of_freedom: int = 6
    move_angle_rad: float = 1.5707963267948966      # 90 degrees per joint
    move_time_s: float = 2.0
    static_safety_factor_metal: float = 1.5
    base: str = "fixed table mount"

    # --- CHOSEN, with the reason in the field name's comment ---
    #: A 3 kg payload on a 600 mm reach is a small industrial arm, and the
    #: proportions below are the common ones: a short base yaw column, an
    #: upper arm and a forearm of similar length, and a compact wrist. The
    #: reach is the sum of the three moving lengths measured from the
    #: shoulder, which is what "maximum reach" means for this layout.
    base_height_m: float = 0.150

    #: Room between two drives for the structure that joins them. CHOSEN at
    #: 10 mm, and the arithmetic is stated rather than implied: a 3 mm housing
    #: wall on each drive plus 4 mm to get a fastener head and a wire past.
    #: It is a design allowance and not a measurement. Without it the envelope
    #: check passes a 0.25 mm gap, which is two actuators touching in mid air
    #: with nothing holding them together, and a Fusion model of exactly that
    #: is what put this constant here.
    assembly_clearance_m: float = 0.010

    #: The room each wrist joint gets along the arm. This number has been
    #: wrong twice and the arithmetic is now written out.
    #:
    #: A drive occupies its own axial length when its axis runs along the arm
    #: and its DIAMETER when the axis is across it. The wrist roll is a 38.5 mm
    #: long drive on an along-arm axis; the wrist pitch is the same drive on a
    #: cross-arm axis, so it occupies 98 mm. Two neighbours need half of each,
    #: which is 19.25 + 49.0 = 68.25 mm, plus the 10 mm assembly clearance,
    #: rounded up to 78.5.
    #:
    #: All three wrist gaps take that number because this specification
    #: carries one wrist spacing, so the binding pair sets them all. The three
    #: wrist joints therefore take 235.5 mm of the 600 mm reach, which is the
    #: real price of hanging 98 mm drives off a serial wrist: one wrist joint
    #: is now longer than the forearm.
    wrist_spacing_m: float = 0.0785

    #: What is left of the reach after the wrist, split in the same ratio the
    #: first version used. These are derived rather than stated, so a change
    #: to the wrist spacing moves them instead of breaking the reach.
    @property
    def wrist_m(self) -> float:
        return 3.0 * self.wrist_spacing_m

    @property
    def upper_arm_m(self) -> float:
        return (self.reach_m - self.wrist_m) * (280.0 / 520.0)

    @property
    def forearm_m(self) -> float:
        return (self.reach_m - self.wrist_m) * (240.0 / 520.0)

    #: Jerk limit for the S curve. Chosen so that the jerk phase is about a
    #: fifth of the move: a smaller value makes the move longer than the two
    #: seconds the task states, and a larger one is indistinguishable from a
    #: trapezoid.
    jerk_limit_rad_s3: float = 40.0

    #: Deflection limit at the tool under the payload at full reach. Chosen at
    #: 1 mm, which is 1/600 of the reach; the task states a tip deflection
    #: constraint without a number.
    tip_deflection_limit_m: float = 1.0e-3

    #: Torque margin a joint must keep against its actuator's continuous
    #: rating. Chosen at 1.3, so a drive sized here has 30 percent in hand for
    #: friction, which this project refuses to model without data.
    torque_margin: float = 1.3

    #: The largest reflected load inertia, as a multiple of the rotor
    #: inertia, that this design will accept. CHOSEN at 10: a joint whose
    #: load inertia is hundreds of times its rotor inertia is hard to control
    #: and its drivetrain is compliant against the load, which is the standard
    #: argument for a reduction. The number is a design choice, not a
    #: measurement, and the drive comparison table prints the ratio for every
    #: candidate so a reader can apply a different one.
    max_inertia_ratio: float = 10.0

    #: The arm runs one bus. A module whose printed performance belongs to a
    #: different voltage is not comparable at this one, and the selection
    #: refuses it rather than assuming the numbers carry over. CHOSEN at 48 V
    #: because the joints that need the most torque are only offered there.
    bus_voltage_v: float = 48.0

    #: The smallest outer dimension a link may have, in metres. CHOSEN, and
    #: the arithmetic is stated: a joint interface carries four M6 socket head
    #: screws whose counterbores are 10.4 mm across (ISO 4762 head plus the
    #: clearance in geometry/cad_export/manufacturing_features.py). Four of
    #: them around a central bore with an edge margin needs about three
    #: counterbore diameters across the face, which is 31.2 mm, rounded to 32.
    #: Without this the section optimiser returns a 10 mm wide tube: feasible
    #: arithmetic, and a link that cannot bolt to anything.
    minimum_section_m: float = 0.032

    #: Thickness of the flange at each end of a link, where the bolts go.
    #: CHOSEN as the larger of what the two failed checks demand: a 6.4 mm
    #: counterbore has to fit, and a tapped hole in aluminium wants 1.5
    #: diameters, which is 9 mm for M6. So 9 mm, and the mass of these
    #: flanges is counted rather than ignored: two per link, a solid plate of
    #: the section's outer size less the bolt holes.
    flange_thickness_m: float = 0.009

    #: The thinnest wall the milling rules in geometry/manufacturability
    #: accept for aluminium. Below this the optimiser buys mass by asking for
    #: a wall no shop will cut.
    minimum_wall_m: float = 0.001

    materials: dict[str, str] = field(default_factory=lambda: {
        "link": "al_6061_t6",
        "link_alternative": "al_7075_t6",
        "cover": "pa12",
    })

    def reach_check_m(self) -> float:
        return self.upper_arm_m + self.forearm_m + self.wrist_m

    def joints(self) -> list[JointSpec]:
        """A classic six revolute layout: yaw, two pitches, then a roll pitch
        roll wrist."""
        return [
            # j1 sits at the floor and the column is its child, so the column
            # spins about its own vertical axis and its inertia belongs on
            # this joint. The first version put j1 at the top of the column
            # with the column still hanging from it, which is the same
            # inertia on a joint the column did not rotate about.
            JointSpec("j1_base_yaw", (0.0, 1.0, 0.0),
                      description="base rotation about the vertical, at the "
                                  "floor, carrying the column above it"),
            JointSpec("j2_shoulder", (0.0, 0.0, 1.0),
                      origin_y_m=self.base_height_m,
                      description="shoulder pitch at the top of the column"),
            JointSpec("j3_elbow", (0.0, 0.0, 1.0), origin_x_m=self.upper_arm_m,
                      description="elbow pitch"),
            JointSpec("j4_wrist_roll", (1.0, 0.0, 0.0), origin_x_m=self.forearm_m,
                      description="forearm roll"),
            # Every joint origin sits at the end of its parent link, so the
            # wrist body lengths appear here as offsets. A link's own length
            # moves nothing in the forward kinematics except for the tip link,
            # where it becomes the tool offset: the first build left these at
            # zero and the tool came out 55 mm short.
            JointSpec("j5_wrist_pitch", (0.0, 0.0, 1.0),
                      origin_x_m=self.wrist_spacing_m,
                      description="wrist pitch"),
            JointSpec("j6_tool_roll", (1.0, 0.0, 0.0),
                      origin_x_m=self.wrist_spacing_m,
                      description="tool roll"),
        ]

    def links(self) -> list[LinkSpec]:
        """Starting sections, deliberately generous.

        Every one is a hollow rectangle because that is the section this
        project's genome, statics and CAD export all share. The design stages
        replace them.
        """
        return [
            LinkSpec("base_column", self.base_height_m, 0.090, 0.090, 0.005,
                     self.materials["link"], role="base yaw column"),
            LinkSpec("upper_arm", self.upper_arm_m, 0.080, 0.060, 0.004,
                     self.materials["link"], role="shoulder to elbow"),
            LinkSpec("forearm", self.forearm_m, 0.070, 0.050, 0.004,
                     self.materials["link"], role="elbow to wrist"),
            # The wrist bodies carry no joint origin offset, so their own
            # lengths are what separate the wrist axes. They add to the reach,
            # and the tool flange takes what is left of the stated wrist
            # length so that the three sum to it exactly.
            LinkSpec("wrist_roll_body", self.wrist_spacing_m, 0.050, 0.050,
                     0.004, self.materials["link"], role="wrist roll housing"),
            LinkSpec("wrist_pitch_body", self.wrist_spacing_m, 0.045, 0.045,
                     0.004, self.materials["link"], role="wrist pitch housing"),
            LinkSpec("tool_flange", self.wrist_spacing_m, 0.040, 0.040, 0.004,
                     self.materials["link"], role="tool interface"),
        ]

    def goal_sentence(self) -> str:
        """The task as a sentence, for the policy layer to read.

        It states the safety factor explicitly, because the policy refuses a
        problem that bounds neither stress nor a safety factor, and picking
        one on the caller's behalf is exactly what it will not do.
        """
        return (f"a {self.upper_arm_m * 1000:.0f} mm long robot arm link "
                f"carrying {self.payload_kg * 9.81:.0f} N at the tip, "
                f"deflection under {self.tip_deflection_limit_m * 1000:.0f} mm, "
                f"safety factor {self.static_safety_factor_metal}, "
                f"al_6061_t6, 80 mm tall and 60 mm wide")


SPEC = ManipulatorSpec()
