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
    upper_arm_m: float = 0.280
    forearm_m: float = 0.240
    wrist_m: float = 0.080

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
            JointSpec("j1_base_yaw", (0.0, 1.0, 0.0), origin_y_m=self.base_height_m,
                      description="base rotation about the vertical"),
            JointSpec("j2_shoulder", (0.0, 0.0, 1.0),
                      description="shoulder pitch, carries the whole arm"),
            JointSpec("j3_elbow", (0.0, 0.0, 1.0), origin_x_m=self.upper_arm_m,
                      description="elbow pitch"),
            JointSpec("j4_wrist_roll", (1.0, 0.0, 0.0), origin_x_m=self.forearm_m,
                      description="forearm roll"),
            # Every joint origin sits at the end of its parent link, so the
            # wrist body lengths appear here as offsets. A link's own length
            # moves nothing in the forward kinematics except for the tip link,
            # where it becomes the tool offset: the first build left these at
            # zero and the tool came out 55 mm short.
            JointSpec("j5_wrist_pitch", (0.0, 0.0, 1.0), origin_x_m=0.030,
                      description="wrist pitch"),
            JointSpec("j6_tool_roll", (1.0, 0.0, 0.0), origin_x_m=0.025,
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
            LinkSpec("wrist_roll_body", 0.030, 0.050, 0.050, 0.004,
                     self.materials["link"], role="wrist roll housing"),
            LinkSpec("wrist_pitch_body", 0.025, 0.045, 0.045, 0.004,
                     self.materials["link"], role="wrist pitch housing"),
            LinkSpec("tool_flange", self.wrist_m - 0.055, 0.040, 0.040, 0.004,
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
