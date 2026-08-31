"""Registration of the methods this project has actually implemented.

**Only implemented and verified methods appear here.** A registry entry is a
claim that the method exists and is usable, and a selector will route to
whatever it finds. Listing a method that is a stub, or one whose applicability
range nobody measured, is the same failure as a fabricated part number: it
reads as a record and it is not one. `physics/thermal.py`,
`physics/collision.py`, `physics/rigid_body.py` and `optimization/bayesian.py`
are one-line stubs in this tree and are deliberately absent. MMA is absent for
the same reason, despite being the natural optimizer for the stress-constrained
problem: it is not implemented here.

The applicability numbers are not textbook folklore. The slenderness
thresholds come from this project's own 3D FEM comparison, recorded in
DESIGN.md Phase 7.5.
"""

from __future__ import annotations

from .context import ProblemContext
from .method import Category, Condition, Cost, Fidelity, Method
from .registry import MethodRegistry

# --- reusable conditions -----------------------------------------------------

# Measured against 3D FEM on a 10x40x1 mm section (DESIGN.md Phase 7.5):
# Euler-Bernoulli deflection over FEM is 0.9382 at L/h = 4, 0.9750 at 6,
# 0.9881 at 8, 0.9975 at 12, 1.0022 at 20. So the model costs 6.2% at L/h = 4
# and 2.5% at 6, falling under 0.5% by 12. The threshold is set at 12.
#
# This is the condition that Phase 7 needed and did not have. The MVP link had
# a slenderness of about 6, the cheap model omitted shear, and the optimizer
# found a design sitting exactly on that blind spot; the 3D FEM gate rejected
# it. With this declared, the selector refuses the model before it runs.
EULER_BERNOULLI_SLENDERNESS = 12.0

# Timoshenko over FEM stays within 0.5% across the whole measured range, 4 to
# 20, and is 0.9995 at L/h = 4. Below 4 there is no measurement, so the method
# does not claim to apply there.
TIMOSHENKO_SLENDERNESS = 4.0

_prismatic = Condition(
    "the problem can be posed as a prismatic beam",
    lambda c: c.supports("prismatic_beam"))
_voxel = Condition(
    "the problem can be posed as a voxel grid",
    lambda c: c.supports("voxel_domain"))
_assembly = Condition(
    "the problem can be posed as a jointed assembly",
    lambda c: c.supports("assembly"))
_no_stress_field = Condition(
    "a full stress field is not required (a 1D model has no field to give)",
    lambda c: not c.require("needs_stress_field"))


def build_default_registry() -> MethodRegistry:
    """The methods implemented in this tree, with their declared ranges."""
    registry = MethodRegistry()

    # --- design generation ---------------------------------------------------
    registry.register(Method(
        name="parametric_section",
        category=Category.DESIGN_GENERATION,
        summary="Vary the section parameters of a fixed topology.",
        inputs=("design_genome", "engineering_ir"),
        outputs=("design_genome",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_prismatic,),
        implementation="core.design_genome.genome",
        evidence="SIMULATED",
        notes="Cannot change topology. It moves within one shape family."))

    registry.register(Method(
        name="topology_compliance",
        category=Category.DESIGN_GENERATION,
        summary="SIMP density optimisation minimising compliance at fixed volume.",
        inputs=("voxel_mesh", "volume_fraction", "load_case"),
        outputs=("density_field",),
        fidelity=Fidelity.FEM3D, cost=Cost.HEAVY,
        conditions=(_voxel,),
        implementation="optimization.topology.simp.optimize",
        evidence="SIMULATED",
        notes="Says nothing about stress. Adjoint sensitivity checked against "
              "finite differences to 2.6e-05."))

    registry.register(Method(
        name="topology_stress",
        category=Category.DESIGN_GENERATION,
        summary="SIMP compliance minimisation carrying an aggregated stress "
                "constraint.",
        inputs=("voxel_mesh", "volume_fraction", "load_case", "stress_limit"),
        outputs=("density_field",),
        fidelity=Fidelity.FEM3D, cost=Cost.HEAVY,
        conditions=(
            _voxel,
            Condition("the problem states a stress constraint",
                      lambda c: c.require("has_stress_constraint")),
        ),
        implementation="optimization.topology.stress.optimize_constrained",
        evidence="SIMULATED",
        notes="Lowers the peak stress and pulls material off a re-entrant "
              "corner, at a compliance cost. Does not produce a clean fillet: "
              "the result is greyer than the compliance design. Adjoint "
              "checked to 3.9e-08."))

    # --- analysis ------------------------------------------------------------
    registry.register(Method(
        name="beam_eb",
        category=Category.ANALYSIS,
        summary="Euler-Bernoulli cantilever, differentiable, on the GPU.",
        inputs=("design_genome", "load_case"),
        outputs=("tip_deflection", "root_stress", "mass"),
        fidelity=Fidelity.BEAM, cost=Cost.TRIVIAL,
        conditions=(
            _prismatic, _no_stress_field,
            Condition(
                f"slenderness L/h is at least {EULER_BERNOULLI_SLENDERNESS:g} "
                f"(shear is omitted; measured error is 2.5% at L/h 6 and 6.2% "
                f"at L/h 4)",
                lambda c: c.require("slenderness") >= EULER_BERNOULLI_SLENDERNESS),
        ),
        implementation="physics.structural.beam (shear_deformation=False)",
        evidence="SIMULATED",
        notes="This is the method Phase 7 used out of range."))

    registry.register(Method(
        name="beam_timoshenko",
        category=Category.ANALYSIS,
        summary="Timoshenko cantilever with a shear term, differentiable.",
        inputs=("design_genome", "load_case"),
        outputs=("tip_deflection", "root_stress", "mass"),
        fidelity=Fidelity.TIMOSHENKO, cost=Cost.TRIVIAL,
        conditions=(
            _prismatic, _no_stress_field,
            Condition(
                f"slenderness L/h is at least {TIMOSHENKO_SLENDERNESS:g} "
                f"(the range where the shear term was measured)",
                lambda c: c.require("slenderness") >= TIMOSHENKO_SLENDERNESS),
        ),
        implementation="physics.structural.beam (shear_deformation=True)",
        evidence="SIMULATED",
        notes="A_s = 2th is an assumed thin-wall factor, validated against 3D "
              "FEM to a 0.350% mean error over L/h 4 to 20."))

    registry.register(Method(
        name="fem3d",
        category=Category.ANALYSIS,
        summary="Matrix-free 3D linear elasticity on a hex grid, GPU CG.",
        inputs=("mesh", "material", "boundary_conditions", "load"),
        outputs=("displacements", "stress_field", "compliance"),
        fidelity=Fidelity.FEM3D, cost=Cost.HEAVY,
        conditions=(
            Condition("the problem can be posed on a structured hex grid",
                      lambda c: c.supports("prismatic_beam")
                      or c.supports("voxel_domain")),
        ),
        implementation="physics.fem.solver.solve_linear_elasticity",
        evidence="SIMULATED",
        notes="Linear elastic and small strain. Reports stress at a clamped "
              "boundary that is singular under mesh refinement, so peak values "
              "there are mesh-dependent by nature."))

    registry.register(Method(
        name="statics",
        category=Category.ANALYSIS,
        summary="Joint torques holding an assembly against gravity.",
        inputs=("assembly", "joint_positions"),
        outputs=("joint_torques",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_assembly,),
        implementation="core.assembly.statics",
        evidence="SIMULATED",
        notes="Quasi-static. No inertial terms."))

    registry.register(Method(
        name="dynamics",
        category=Category.ANALYSIS,
        summary="Rigid-body inverse dynamics, M(q) q'' + C(q,q') q' + g(q).",
        inputs=("assembly", "trajectory"),
        outputs=("joint_torques", "joint_power"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.CHEAP,
        conditions=(_assembly,),
        implementation="physics.dynamics.equations.inverse_dynamics",
        evidence="SIMULATED",
        notes="Rigid links. Link flexibility is not modelled."))

    registry.register(Method(
        name="surrogate_screen",
        category=Category.ANALYSIS,
        summary="Learned MLP predicting beam metrics, for screening only.",
        inputs=("design_genome",),
        outputs=("tip_deflection", "root_stress"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_prismatic, _no_stress_field),
        implementation="surrogate.inference.screening",
        evidence="SIMULATED",
        notes="Screening only. A final decision may never rest on it. It is "
              "also not faster than the batched solver it approximates "
              "(measured 0.38x), so it earns its place by shape, not speed."))

    registry.register(Method(
        name="fatigue_sn",
        category=Category.ANALYSIS,
        summary="High-cycle stress-life check with a mean-stress correction.",
        inputs=("stress_cycle", "material"),
        outputs=("fatigue_safety_factor",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the duty involves repeated loading (a static single "
                      "load has no fatigue to check)",
                      lambda c: c.require("has_cyclic_load")),
        ),
        implementation="physics.fatigue.sn.fatigue_safety_factor",
        evidence="SIMULATED",
        notes="Stress-life, so high-cycle only; low-cycle fatigue needs a "
              "strain-life model that is not implemented. No notch, surface, "
              "size or temperature factors, each of which lowers a real "
              "endurance limit, so the result is optimistic. Fatigue data is "
              "reference_typical and scatters widely."))

    registry.register(Method(
        name="buckling_euler",
        category=Category.ANALYSIS,
        summary="Elastic column buckling, with the Euler validity check.",
        inputs=("section_properties", "length", "compressive_load",
                "end_condition"),
        outputs=("critical_load", "buckling_safety_factor", "slenderness"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("a member carries compression (a member in tension "
                      "cannot buckle)",
                      lambda c: c.require("has_compressive_load")),
        ),
        implementation="physics.buckling.euler.analyze_column",
        evidence="SIMULATED",
        notes="An ideal column: perfectly straight, centrally loaded, no "
              "residual stress. Real members collapse below this and no "
              "knock-down factor is applied. Below the critical slenderness "
              "the derivation does not hold and the method reports that "
              "rather than returning a number that looks like a margin. 3D "
              "FEM linear buckling would cover shapes this cannot and is NOT "
              "registered, because it is not implemented."))

    registry.register(Method(
        name="shaft_combined",
        category=Category.ANALYSIS,
        summary="Shaft sizing under combined bending, torsion and axial load, "
                "by DE-Goodman.",
        inputs=("shaft_loads", "material", "diameter", "stress_concentration"),
        outputs=("static_safety_factor", "fatigue_safety_factor",
                 "critical_speed"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("a member transmits torque while rotating (a tension rod "
                      "is not a shaft)",
                      lambda c: c.require("transmits_torque")),
        ),
        implementation="physics.shaft.design.analyze_shaft",
        evidence="SIMULATED",
        notes="Solid round section. Stress-concentration factors default to "
              "assumed mid-range values for a shoulder and keyway and should "
              "be measured for a real design. The critical speed is a bare "
              "uniform shaft on simple supports, so it ignores the attached "
              "masses that lower it. Fatigue is high-cycle, inherited from "
              "the stress-life module."))

    registry.register(Method(
        name="bearing_l10",
        category=Category.SELECTION,
        summary="Basic rating life of a rolling bearing from its equivalent load.",
        inputs=("radial_load", "axial_load", "speed", "bearing"),
        outputs=("l10_hours", "static_safety_factor"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("a rotating support carries load",
                      lambda c: c.require("has_rotating_support")),
        ),
        implementation="drivetrain.bearings.life.rate_bearing",
        evidence="SIMULATED",
        notes="L10 is a statistic: one bearing in ten is expected to fail "
              "before it. The ISO 281 reliability, lubrication, contamination "
              "and temperature factors are not applied, and each can move the "
              "answer by more than an order of magnitude. The catalogue holds "
              "standard ISO boundary dimensions with representative ratings "
              "tagged illustrative, not a manufacturer catalogue."))

    registry.register(Method(
        name="motor_thermal",
        category=Category.ANALYSIS,
        summary="Steady-state winding temperature of a motor under a duty cycle.",
        inputs=("motor", "duty_cycle", "ambient_temperature"),
        outputs=("winding_temperature", "temperature_margin"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the problem states a duty cycle to heat the motor with",
                      lambda c: c.require("has_duty_cycle")),
        ),
        implementation="physics.thermal.motor.check_motor_thermal",
        evidence="SIMULATED",
        notes="Replaces the continuous-torque proxy Phase 12 left as 'subject "
              "to thermal validation'. Steady state only: a brief overload "
              "that this passes may be fine, and a long one it passes on "
              "average may not be. The thermal resistance is a single lumped "
              "number and the mounting dominates it, so it can be out by a "
              "factor of two either way. Iron loss is linear in speed, which "
              "understates it at high speed. Coefficients are illustrative."))

    registry.register(Method(
        name="thermal_stress",
        category=Category.ANALYSIS,
        summary="Stress from restrained thermal expansion, combined with the "
                "mechanical stress.",
        inputs=("material", "temperature_change", "constraint",
                "mechanical_stress"),
        outputs=("thermal_stress", "combined_safety_factor"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the part sees a temperature change",
                      lambda c: c.require("has_temperature_change")),
        ),
        implementation="physics.thermal.stress.check_thermal_stress",
        evidence="SIMULATED",
        notes="A uniform temperature change on a uniformly restrained member. "
              "The constraint factor is the caller's judgement and the answer "
              "is proportional to it. Real parts have temperature gradients "
              "that this cannot represent, and a gradient produces stress even "
              "in a completely unrestrained body. Expansion coefficients are "
              "reference_typical at room temperature and are themselves "
              "temperature dependent."))

    registry.register(Method(
        name="bolted_joint",
        category=Category.ANALYSIS,
        summary="Preloaded bolted joint: load sharing, separation and bolt "
                "fatigue.",
        inputs=("bolt_size", "property_class", "grip_length", "external_load"),
        outputs=("preload", "load_factor", "separation_margin",
                 "fatigue_safety_factor"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the assembly has a preloaded bolted connection",
                      lambda c: c.require("has_bolted_joint")),
        ),
        implementation="physics.joints.bolted.analyze_joint",
        evidence="SIMULATED",
        notes="A single bolt loaded along its axis. The nut factor relating "
              "torque to preload scatters by about 30 percent, so a preload "
              "derived from a torque figure carries that uncertainty; angle "
              "control is better and is not modelled. Bolt groups, eccentric "
              "loading and shear-loaded joints are not covered. ISO 898-1 "
              "property classes and thread stress areas are published "
              "standards, not a vendor catalogue."))

    registry.register(Method(
        name="gear_tooth",
        category=Category.ANALYSIS,
        summary="Gear tooth capacity: Lewis bending and Hertzian contact.",
        inputs=("module", "tooth_counts", "face_width", "torque",
                "allowable_stresses"),
        outputs=("bending_safety_factor", "contact_safety_factor"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("a gear mesh transmits torque",
                      lambda c: c.require("has_gear_mesh")),
        ),
        implementation="physics.gears.tooth.analyze_mesh",
        evidence="SIMULATED",
        notes="Lewis and elementary Hertz, NOT full AGMA. The dynamic, load "
              "distribution, application and size factors are exposed as "
              "corrections defaulting to 1.0, which is the optimistic choice, "
              "so an uncorrected result runs high against a real gear. Surface "
              "treatment is expressed only through the allowable stresses the "
              "caller supplies. Planetary and harmonic internals are not "
              "covered."))

    registry.register(Method(
        name="laminate_clt",
        category=Category.ANALYSIS,
        summary="Classical laminate theory: ABD, ply stresses and first-ply "
                "failure.",
        inputs=("lamina_properties", "stacking_sequence", "load_resultants"),
        outputs=("abd_matrices", "ply_stresses", "first_ply_failure"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the part is a laminate with a stated layup (an "
                      "isotropic or single-ply part has no stack to design)",
                      lambda c: c.require("has_layup")),
        ),
        implementation="physics.composite.clt.abd_matrices",
        evidence="SIMULATED",
        notes="Thin plate with Kirchhoff kinematics, plane stress in every "
              "ply, perfect bonding and linear elasticity. The plane-stress "
              "assumption fails hardest AT A FREE EDGE, which is where real "
              "laminates delaminate, and nothing here will warn about it. "
              "Reports FIRST-ply failure, which is conservative as an ultimate "
              "strength except when the first failure is a fibre failure in "
              "the load direction. No progressive damage, no hygrothermal "
              "terms. The Tsai-Wu interaction term F12 is assumed."))

    _shaft_hub = Condition(
        "the design transfers torque from a shaft to a hub",
        lambda c: c.require("has_shaft_hub_connection"))

    registry.register(Method(
        name="key_joint",
        category=Category.ANALYSIS,
        summary="Parallel key in shear and bearing, with standard sections.",
        inputs=("shaft_diameter", "key_length", "torque", "allowables"),
        outputs=("shear_stress", "bearing_stress", "safety_factor"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_shaft_hub,),
        implementation="physics.elements.keys.analyze_key",
        evidence="SIMULATED",
        notes="Uniform load along the key is assumed and is optimistic: the "
              "load concentrates at the ends, so length beyond about 1.5 shaft "
              "diameters is capped rather than credited. Only half the key "
              "height bears. The keyway's effect on the SHAFT is not included "
              "here and belongs to the shaft check. Sections are the published "
              "DIN 6885 / ISO 773 series."))

    registry.register(Method(
        name="press_fit",
        category=Category.ANALYSIS,
        summary="Interference fit: contact pressure, torque capacity and hub "
                "hoop stress.",
        inputs=("interference", "diameters", "engagement", "materials"),
        outputs=("contact_pressure", "torque_capacity", "hoop_stress"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_shaft_hub,),
        implementation="physics.elements.press_fit.analyze_press_fit",
        evidence="SIMULATED",
        notes="Thick-wall elastic Lame. Surface roughness is NOT deducted and "
              "always reduces the effective interference, so this is "
              "optimistic. The friction coefficient is the weakest number in "
              "the chain and torque capacity is directly proportional to it. "
              "Static holding only: no fretting, no thermal loss of "
              "interference in service."))

    registry.register(Method(
        name="fillet_weld",
        category=Category.ANALYSIS,
        summary="Nominal throat stress of a fillet weld.",
        inputs=("force", "weld_leg", "weld_length", "allowable"),
        outputs=("throat_stress", "safety_factor"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the design has a welded joint",
                      lambda c: c.require("has_welded_joint")),
        ),
        implementation="physics.elements.welds.analyze_fillet_weld",
        evidence="SIMULATED",
        notes="Nominal throat stress with no concentration at the root or "
              "toe, which is where weld cracks start. STATIC ONLY: a welded "
              "joint's endurance strength is a fraction of the parent metal's "
              "and largely independent of the steel's strength, so a "
              "parent-metal fatigue check badly overstates a weld. Full "
              "penetration to the throat is assumed and residual stress is "
              "ignored."))

    registry.register(Method(
        name="iso_fit",
        category=Category.SELECTION,
        summary="ISO 286 limits and fits from the tolerance-unit formula.",
        inputs=("nominal_size", "hole_grade", "shaft_letter_and_grade"),
        outputs=("limits", "clearance_range", "fit_type"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the design needs dimensional tolerances",
                      lambda c: c.require("requires_tolerances")),
        ),
        implementation="physics.elements.fits.fit",
        evidence="SIMULATED",
        notes="Computed from the ISO 286 tolerance-unit expression rather than "
              "a transcribed table, and it does NOT reproduce the published "
              "values exactly because the standard rounds to preferred "
              "numbers: measured agreement is 1.2% mean and 8.4% worst over "
              "IT6 to IT9 above 3 mm. Close enough to compare fits, not close "
              "enough to put on a drawing. Sizes at or below 3 mm are refused. "
              "Only deviations with a published closed form (H, g, h, k, n) "
              "are implemented; p, r, s and u need tabulated increments and "
              "are refused by name."))

    registry.register(Method(
        name="stress_transformation",
        category=Category.ANALYSIS,
        summary="Principal stresses and both in-plane and absolute maximum "
                "shear.",
        inputs=("stress_components",),
        outputs=("principal_stresses", "max_shear", "von_mises"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the stress state is multiaxial",
                      lambda c: c.require("has_multiaxial_stress")),
        ),
        implementation="physics.mechanics.stress_state.principal_stress_2d",
        evidence="SIMULATED",
        notes="The in-plane maximum shear is NOT always the absolute maximum. "
              "Plane stress carries a third principal of zero, and when the "
              "two in-plane principals share a sign the absolute shear "
              "involves that zero and is larger, by up to a factor of two in "
              "equibiaxial tension. Both are returned. This is a stress state, "
              "not a failure criterion."))

    registry.register(Method(
        name="noncircular_torsion",
        category=Category.ANALYSIS,
        summary="Torsion of solid rectangles and thin open or closed sections.",
        inputs=("torque", "section_geometry", "shear_modulus"),
        outputs=("max_shear_stress", "torsion_constant", "twist_rate"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("a non-circular section carries torque",
                      lambda c: c.require("has_noncircular_torsion")),
        ),
        implementation="physics.mechanics.torsion.solid_rectangle",
        evidence="SIMULATED",
        notes="Free warping is assumed; restraining it adds axial stresses "
              "that are not modelled and can exceed the torsional ones. Open "
              "and closed sections differ by ORDERS OF MAGNITUDE and are not "
              "interchangeable: slitting a square tube measured 432 times "
              "less stiff. Forgetting that a seam or slit makes a section "
              "open is a large unsafe error."))

    registry.register(Method(
        name="pressure_vessel",
        category=Category.ANALYSIS,
        summary="Thin and thick walled cylinder stresses under internal "
                "pressure.",
        inputs=("pressure", "radii", "wall_thickness"),
        outputs=("hoop_stress", "longitudinal_stress", "radial_stress"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the part carries internal pressure",
                      lambda c: c.require("has_internal_pressure")),
        ),
        implementation="physics.mechanics.vessels.thick_wall",
        evidence="SIMULATED",
        notes="Internal pressure with closed ends only. EXTERNAL pressure is a "
              "different problem: such a vessel buckles far below its material "
              "strength and nothing here checks that. Away from nozzles, heads "
              "and supports, which is where real vessel codes concentrate "
              "because those dominate the local stress."))

    registry.register(Method(
        name="hertz_contact",
        category=Category.ANALYSIS,
        summary="Elastic contact between curved bodies, with the subsurface "
                "shear.",
        inputs=("force", "radii", "materials"),
        outputs=("contact_patch", "peak_pressure", "subsurface_shear"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("curved bodies bear against each other",
                      lambda c: c.require("has_concentrated_contact")),
        ),
        implementation="physics.mechanics.contact.sphere_contact",
        evidence="SIMULATED",
        notes="Frictionless, elastic, and valid while the contact is small "
              "against the radii forming it. The peak shear is BELOW the "
              "surface, at about 0.48 contact radii deep, which is why rolling "
              "contact fatigue starts subsurface and why surface hardness "
              "alone does not prevent pitting. Concentration factors returned "
              "here are ELASTIC Kt, not the smaller fatigue Kf that a fatigue "
              "check needs."))

    registry.register(Method(
        name="thermal_network",
        category=Category.ANALYSIS,
        summary="Conduction, convection and radiation resistances in a "
                "series-parallel path.",
        inputs=("geometry", "conductivities", "coefficients", "emissivity"),
        outputs=("total_resistance", "dominant_resistance",
                 "temperature_rise"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("there is a heat path to build",
                      lambda c: c.require("has_heat_path")),
        ),
        implementation="physics.thermal.network.ThermalPath",
        evidence="SIMULATED",
        notes="One-dimensional flow through each branch, so lateral spreading "
              "is not represented and a pure series network is conservative "
              "for a concentrated source. Contact resistance between parts is "
              "NOT included and is often comparable to the conduction it "
              "joins, which under-predicts temperature. The convection "
              "coefficient is the dominant uncertainty and is not a material "
              "property: natural air spans 5 to 25 W/m2K. Emissivity is a "
              "surface property, and polished against anodised aluminium "
              "differ by a factor of sixteen."))

    registry.register(Method(
        name="lumped_transient",
        category=Category.ANALYSIS,
        summary="First-order thermal response with the Biot validity check.",
        inputs=("mass", "specific_heat", "resistance", "geometry"),
        outputs=("time_constant", "biot_number", "temperature_history"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            Condition("the question is transient rather than steady state",
                      lambda c: c.require("has_thermal_transient")),
        ),
        implementation="physics.thermal.network.lumped_response",
        evidence="SIMULATED",
        notes="Valid only for a Biot number below about 0.1, which is a hard "
              "condition and not a guideline: above it the body has real "
              "internal gradients and a single temperature does not describe "
              "it. The check is computed and reported rather than assumed, "
              "because a lumped model applied to a thick or poorly conducting "
              "body looks entirely reasonable and is not. The time constant "
              "inherits the convection resistance's factor-of-two uncertainty "
              "and should be read as an order of magnitude."))

    # --- optimization --------------------------------------------------------
    registry.register(Method(
        name="slsqp",
        category=Category.OPTIMIZATION,
        summary="Gradient-based constrained local optimisation.",
        inputs=("objective", "constraints", "start_point"),
        outputs=("design_vector",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.CHEAP,
        conditions=(
            Condition("gradients are available",
                      lambda c: c.require("needs_gradients")),
        ),
        implementation="optimization.gradient.slsqp.optimize_slsqp",
        evidence="SIMULATED",
        notes="Local. Finds a local optimum and does not know it is local, "
              "which is why it is cross-checked against differential evolution."))

    registry.register(Method(
        name="differential_evolution",
        category=Category.OPTIMIZATION,
        summary="Population-based global search, no gradients required.",
        inputs=("objective", "constraints", "bounds"),
        outputs=("design_vector",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.MODERATE,
        conditions=(),
        implementation="optimization.evolutionary.differential"
                       ".optimize_differential_evolution",
        evidence="SIMULATED",
        notes="Polish is off: the local step crosses the validity "
              "discontinuity at invalid geometries and fails."))

    registry.register(Method(
        name="optimality_criteria",
        category=Category.OPTIMIZATION,
        summary="OC update with a volume Lagrange multiplier bisection.",
        inputs=("density_field", "sensitivity", "volume_fraction"),
        outputs=("density_field",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(
            _voxel,
            Condition("the sensitivity is negative everywhere, as compliance "
                      "is (the OC exponent has no meaning for a positive entry)",
                      lambda c: not c.require("has_stress_constraint")),
        ),
        implementation="optimization.topology.simp.oc_update",
        evidence="SIMULATED",
        notes="Holds the volume exactly at every iteration."))

    registry.register(Method(
        name="penalty_projection",
        category=Category.OPTIMIZATION,
        summary="Move-limited projected gradient with an exterior penalty, for "
                "sensitivities of either sign.",
        inputs=("density_field", "sensitivity", "volume_fraction", "constraint"),
        outputs=("density_field",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_voxel,),
        implementation="optimization.topology.stress._project_to_volume",
        evidence="SIMULATED",
        notes="Oscillates about the constraint boundary, so the best feasible "
              "iterate is kept rather than the last one. MMA would converge in "
              "fewer iterations and is not implemented."))

    registry.register(Method(
        name="nsga2",
        category=Category.OPTIMIZATION,
        summary="Multi-objective search returning an approximated Pareto front.",
        inputs=("objectives", "bounds", "constraints"),
        outputs=("pareto_front", "front_designs"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.MODERATE,
        conditions=(
            Condition("the problem states more than one objective (a single "
                      "objective has no trade-off to map)",
                      lambda c: c.require("n_objectives") > 1),
            Condition("the design variables are continuous (the crossover and "
                      "mutation operators interpolate, and nothing lies "
                      "between two materials)",
                      lambda c: not c.require("has_discrete_variables")),
        ),
        implementation="optimization.multi_objective.nsga2.nsga2",
        evidence="SIMULATED",
        notes="The returned front is a finite-population approximation, not "
              "the true front, which is generally a continuum. Non-dominated "
              "with respect to what was evaluated only; no global optimality "
              "is proven. Deterministic for a given seed."))

    registry.register(Method(
        name="pareto_front",
        category=Category.OPTIMIZATION,
        summary="Non-dominated filtering for competing objectives.",
        inputs=("objective_values",),
        outputs=("non_dominated_mask",),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(),
        implementation="optimization.multi_objective.pareto",
        evidence="SIMULATED"))

    # --- selection -----------------------------------------------------------
    registry.register(Method(
        name="drivetrain_selection",
        category=Category.SELECTION,
        summary="Match motor and gearbox archetypes to a torque and speed duty.",
        inputs=("joint_torques", "joint_speeds"),
        outputs=("motor", "gearbox"),
        fidelity=Fidelity.ANALYTICAL, cost=Cost.TRIVIAL,
        conditions=(_assembly,),
        implementation="drivetrain.selection.select",
        evidence="SIMULATED",
        notes="The catalogue holds representative archetypes, not vendor part "
              "numbers. Every entry is tagged illustrative and must be "
              "replaced with a datasheet before it means anything."))

    return registry


DEFAULT_REGISTRY = build_default_registry()
