"""core.materials.inference: filling gaps, without pretending.

Two categories, kept strictly apart because conflating them is how a guess ends
up being treated as a measurement:

  DERIVED   an exact identity from values already present. G = E/(2(1+nu)) for
            an isotropic material is not an estimate, it is the definition.
            Recorded in `derived_fields` and carries full confidence.

  ESTIMATED a correlation or rule of thumb. Always returns an uncertainty and a
            stated basis, and forces the material's status to ASSUMED. It is
            never allowed to masquerade as `reference_typical` or
            `supplier_datasheet`.

No ML predictor here on purpose. A learned property model would need training
data this project does not have, and its output would be much harder to audit
than a rule with a written-down basis. Rules first; if a predictor is added
later it must produce the same Estimate shape, uncertainty included.
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import MaterialClass, MaterialSpec, MaterialStatus


@dataclass(frozen=True)
class Estimate:
    """An inferred value that is NOT a measurement."""

    value: float
    relative_uncertainty: float      # e.g. 0.30 for +/-30%
    basis: str                       # the rule, stated
    is_estimate: bool = True

    def interval(self) -> tuple[float, float]:
        spread = self.value * self.relative_uncertainty
        return self.value - spread, self.value + spread

    def describe(self) -> str:
        lo, hi = self.interval()
        return (f"{self.value:.4g} [ESTIMATED +/-{self.relative_uncertainty:.0%}, "
                f"range {lo:.4g} to {hi:.4g}] basis: {self.basis}")


# --------------------------------------------------------------------------- #
# derived: exact
# --------------------------------------------------------------------------- #
def shear_modulus_from_isotropic(youngs_modulus: float,
                                 poisson_ratio: float) -> float:
    """G = E / (2 (1 + nu)). Exact for an isotropic material."""
    if youngs_modulus <= 0:
        raise ValueError("youngs_modulus must be > 0")
    if not 0.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be in (0, 0.5)")
    return youngs_modulus / (2.0 * (1.0 + poisson_ratio))


def bulk_modulus_from_isotropic(youngs_modulus: float,
                                poisson_ratio: float) -> float:
    """K = E / (3 (1 - 2 nu)). Exact for an isotropic material."""
    return youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))


def reciprocal_poisson(nu_ij: float, e_i: float, e_j: float) -> float:
    """nu_ji = nu_ij * Ej / Ei. Exact, from Maxwell-Betti reciprocity."""
    return nu_ij * e_j / e_i


def check_derived_fields(material: MaterialSpec, tolerance: float = 1e-6) -> dict:
    """Confirm every value the material claims is derived actually is.

    Guards against a data file where `derived_fields` says one thing and the
    number says another.
    """
    problems = {}
    if ("shear_modulus_pa" in material.derived_fields
            and material.material_class is MaterialClass.ISOTROPIC):
        expected = shear_modulus_from_isotropic(material.youngs_modulus_pa,
                                                material.poisson_ratio)
        rel = abs(material.shear_modulus_pa - expected) / expected
        if rel > tolerance:
            problems["shear_modulus_pa"] = (
                f"claimed derived but {material.shear_modulus_pa:.6g} != "
                f"E/(2(1+nu)) = {expected:.6g} (relative {rel:.2e})")
    return problems


# --------------------------------------------------------------------------- #
# estimated: rules of thumb, always flagged
# --------------------------------------------------------------------------- #
def estimate_fatigue_strength(material: MaterialSpec) -> Estimate:
    """Endurance strength from ultimate strength.

    Wrought steels below roughly 1400 MPa ultimate follow Se ~ 0.5 Su as a
    first approximation. Aluminium has no true endurance limit, so the same
    ratio is far weaker evidence and the uncertainty reflects that.
    """
    ultimate = material.ultimate_strength_pa
    ident = material.id.lower()
    if "steel" in ident or "ss_" in ident:
        return Estimate(0.5 * ultimate, 0.25,
                        "wrought steel rule of thumb Se ~ 0.5 * Su, valid "
                        "below about 1400 MPa ultimate")
    if ident.startswith("al_") or ident.startswith("mg_"):
        return Estimate(0.35 * ultimate, 0.40,
                        "aluminium and magnesium alloys have no true endurance "
                        "limit; Se ~ 0.35 * Su at 5e8 cycles is a weak "
                        "reference point, not a limit")
    return Estimate(0.4 * ultimate, 0.50,
                    "generic Se ~ 0.4 * Su; no material-specific basis, treat "
                    "as an order-of-magnitude placeholder only")


def estimate_yield_from_ultimate(ultimate_strength_pa: float,
                                 material_id: str = "") -> Estimate:
    ident = material_id.lower()
    if "steel" in ident:
        return Estimate(0.65 * ultimate_strength_pa, 0.20,
                        "wrought steel: yield is typically 0.6 to 0.7 of ultimate")
    return Estimate(0.7 * ultimate_strength_pa, 0.35,
                    "generic metal ratio; strongly temper and process dependent")


def apply_estimate(material: MaterialSpec, field: str,
                   estimate: Estimate) -> MaterialSpec:
    """Return a copy carrying an estimated value, downgraded to ASSUMED.

    The status downgrade is the point. A material with an estimated property is
    no longer a reference-typical entry, and the evidence rules elsewhere in
    the system must be able to see that from the material alone.
    """
    if not estimate.is_estimate:
        raise ValueError("apply_estimate is for estimates; derived values "
                         "should be written directly and listed in derived_fields")
    data = material.model_dump()
    data[field] = estimate.value
    data["status"] = MaterialStatus.ASSUMED
    note = (f"{field} is ESTIMATED (+/-{estimate.relative_uncertainty:.0%}): "
            f"{estimate.basis}. NOT a datasheet value.")
    data["notes"] = (material.notes + " " + note).strip()
    return MaterialSpec.model_validate(data)
