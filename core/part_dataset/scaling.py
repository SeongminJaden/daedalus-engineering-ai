"""Exact material scaling of solver labels, and the one thing it is not exact in.

A linear elastic, isotropic, homogeneous solve is a map from load to response
that the material enters in two places: Young's modulus scales every
displacement as 1/E, and Poisson's ratio shapes the three dimensional field.
Density enters only through mass. So a part labelled once with a reference
material can be labelled for another by arithmetic, with one residual that
arithmetic does not remove: the Poisson difference.

MEASURED before this was written, hollow rectangle cantilever, 100 N:

    E doubled, same Poisson         deflection x 0.500000   stress x 1.000000
    Poisson 0.33 to 0.22            deflection +0.23 percent, stress +0.66
    Poisson 0.33 to 0.29            deflection +0.13 percent, stress +0.23
    Poisson 0.33 to 0.40            deflection -0.44 percent, stress -0.30

The E scaling is exact to the printed digit. The Poisson residual across the
whole database (0.22 to 0.40) is under one percent for bending, and it is
recorded on every scaled label as a bound rather than hidden. It is measured
per load case, because torsion and thermal cases need not behave like
bending, and a case without a measured residual cannot be scaled.

Each label carries a `scaling` tag written by the labeller:

    density                  mass: times rho / rho_ref
    inverse_modulus          displacements: times E_ref / E
    inverse_shear_modulus    twists: times G_ref / G, G = E / 2 (1 + nu)
    expansion                thermal deflections: times alpha / alpha_ref
    modulus_times_expansion  thermal stresses: times (E alpha) / (E_ref alpha_ref)
    none                     mechanical stresses: unchanged

A scaled label is graded DERIVED from a SIMULATED solve, names the reference
material and the residual bound, and never claims to be a solve of the target
material. cfrp_ud is orthotropic, and this module refuses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.semantic.evidence import EvidenceKind
from core.materials import MaterialClass, MaterialSpec, MissingMaterialValue

from .labeller import LoadKind
from .schema import PartRecord, label

#: Upper bound on the relative change of a label across the database's Poisson
#: range (0.22 alumina to 0.40 PEEK), per load case, as (displacements,
#: stresses). MEASURED on 2026-09-03 by scaling an al_7075_t6 solve to
#: alumina, PEEK and 304 and solving each directly:
#:
#:     bending           displacements 0.35 percent   peak stress 1.05 percent
#:     axial                           0.32                        1.66
#:     torsion                         0.20                        0.87
#:     combined                        0.58                        0.30
#:     thermal gradient                0.81                        5.85
#:
#: The bounds below sit above those. The thermal stress residual is large
#: because the peak sits at the clamp and the clamp's constraint of lateral
#: expansion depends on Poisson's ratio directly; it is the one label where
#: scaling costs several percent, and its bound says so. A case absent here
#: has not been measured and cannot be scaled.
POISSON_RESIDUAL_BOUND: dict[LoadKind, tuple[float, float]] = {
    LoadKind.BENDING: (0.005, 0.015),
    LoadKind.AXIAL: (0.005, 0.02),
    LoadKind.TORSION: (0.005, 0.01),
    LoadKind.COMBINED: (0.01, 0.01),
    LoadKind.THERMAL_GRADIENT: (0.01, 0.07),
}
STRESS_TAGS = ("none", "modulus_times_expansion")


class UnscalableLabel(ValueError):
    """A label whose material dependence is not known well enough to scale."""


def shear_modulus(material: MaterialSpec) -> float:
    return material.youngs_modulus_pa / (2.0 * (1.0 + material.poisson_ratio))


def scale_factor(tag: str, reference: MaterialSpec, target: MaterialSpec) -> float:
    if tag == "density":
        return target.density_kg_m3 / reference.density_kg_m3
    if tag == "inverse_modulus":
        return reference.youngs_modulus_pa / target.youngs_modulus_pa
    if tag == "inverse_shear_modulus":
        return shear_modulus(reference) / shear_modulus(target)
    if tag == "expansion":
        return _alpha(target) / _alpha(reference)
    if tag == "modulus_times_expansion":
        return (target.youngs_modulus_pa * _alpha(target)) / (
            reference.youngs_modulus_pa * _alpha(reference))
    if tag == "none":
        return 1.0
    raise UnscalableLabel(f"unknown scaling tag {tag!r}")


def _alpha(material: MaterialSpec) -> float:
    if material.thermal_expansion_1_k is None:
        raise MissingMaterialValue(
            f"{material.id} has no sourced thermal expansion coefficient, so a "
            f"thermal label cannot be scaled to it")
    return material.thermal_expansion_1_k


def scale_record(record: PartRecord, reference: MaterialSpec,
                 target: MaterialSpec) -> PartRecord:
    """The same part, its labels rewritten for `target`, graded derived.

    Refuses when either material is not isotropic, when the record's load
    case has no measured Poisson residual, or when a label carries no
    scaling tag: a label the labeller did not tag is one whose material
    dependence nobody has stated.
    """
    for material in (reference, target):
        if material.material_class is not MaterialClass.ISOTROPIC:
            raise UnscalableLabel(
                f"{material.id} is {material.material_class.value}; scaling "
                f"assumes isotropy and does not hold for it")
    if record.material_id != reference.id:
        raise UnscalableLabel(
            f"record {record.part_id} was labelled with {record.material_id!r}, "
            f"not the reference {reference.id!r}")
    case = record.labels.get("load_case")
    if not case or "load_kind" not in case:
        raise UnscalableLabel(f"record {record.part_id} carries no load case")
    kind = LoadKind(case["load_kind"])
    if kind not in POISSON_RESIDUAL_BOUND:
        raise UnscalableLabel(
            f"the Poisson residual for {kind.value} has not been measured; "
            f"scaling it would state a bound nobody checked")
    displacement_bound, stress_bound = POISSON_RESIDUAL_BOUND[kind]

    scaled: dict[str, Any] = {}
    for name, item in record.labels.items():
        if name in ("load_case", "parameters"):
            scaled[name] = dict(item)
            continue
        if not isinstance(item, dict) or "value" not in item:
            scaled[name] = item
            continue
        tag = item.get("scaling")
        if tag is None:
            raise UnscalableLabel(
                f"label {name!r} on {record.part_id} has no scaling tag")
        factor = scale_factor(tag, reference, target)
        residual = (0.0 if tag == "density"
                    else stress_bound if tag in STRESS_TAGS
                    else displacement_bound)
        scaled[name] = label(
            item["value"] * factor, item.get("unit", ""),
            EvidenceKind.ANALYTICAL if tag == "density" else EvidenceKind.SIMULATION,
            f"scaled_from_{reference.id}",
            note=(f"{tag} scaling of a {reference.id} solve by {factor:.6g}; "
                  f"Poisson residual bound {residual:.1%} for {kind.value}; "
                  + item.get("note", "")),
            scaling=tag, scaled_from=reference.id,
            poisson_residual_bound=residual,
            **{k: v for k, v in item.items()
               if k == "mesh_sensitivity"})
        scaled[name]["derived"] = True
    scaled["load_case"]["material_id"] = target.id
    return record.model_copy(update={"material_id": target.id,
                                     "labels": scaled,
                                     "part_id": f"{record.part_id}-{target.id}"})
