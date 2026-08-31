"""Rolling bearing archetypes.

THE DIMENSIONS ARE STANDARD; THE RATINGS ARE NOT A CATALOGUE. The designations
(608, 6000, 6004, 6206 and so on) are ISO boundary-dimension codes, not vendor
part numbers: a 6206 is 30 mm bore, 62 mm outside, 16 mm wide from any
manufacturer, and quoting those dimensions states a standard rather than
inventing a part.

The load ratings are a different matter. C and C0 vary between manufacturers
and between grades within a manufacturer, easily by ten percent, so the values
here are representative magnitudes for the size class and are tagged
`illustrative`. Replace them with a manufacturer catalogue before anything is
ordered. This is the same rule the motor and gearbox archetypes follow and the
same rule the material database follows.

The deliverable is the life calculation, not the table.

SI throughout: N, m, rad/s.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drivetrain.motors.catalog import PartStatus


class BearingType(str, Enum):
    """The rolling element, which sets the life exponent."""

    DEEP_GROOVE_BALL = "deep_groove_ball"
    CYLINDRICAL_ROLLER = "cylindrical_roller"


# ISO 281 life exponent: 3 for point contact (balls), 10/3 for line contact
# (rollers). A roller bearing gains life faster as the load falls, which is the
# whole reason to reach for one.
LIFE_EXPONENT: dict[BearingType, float] = {
    BearingType.DEEP_GROOVE_BALL: 3.0,
    BearingType.CYLINDRICAL_ROLLER: 10.0 / 3.0,
}


class BearingSpec(BaseModel):
    """A bearing archetype: standard dimensions, representative ratings."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    designation: str = Field(min_length=1)
    bearing_type: BearingType
    bore_m: float = Field(gt=0.0)
    outer_diameter_m: float = Field(gt=0.0)
    width_m: float = Field(gt=0.0)
    dynamic_rating_n: float = Field(gt=0.0)     # C
    static_rating_n: float = Field(gt=0.0)      # C0
    limiting_speed_rad_s: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    status: PartStatus = PartStatus.ILLUSTRATIVE
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _geometry_is_consistent(self) -> "BearingSpec":
        if self.outer_diameter_m <= self.bore_m:
            raise ValueError(
                f"{self.id}: outer diameter ({self.outer_diameter_m}) must "
                f"exceed the bore ({self.bore_m})")
        return self

    @model_validator(mode="after")
    def _ratings_are_plausibly_paired(self) -> "BearingSpec":
        """Catch a transposed or mis-scaled rating without over-constraining.

        C and C0 are not the same kind of quantity: C is a load for a million
        revolutions, C0 a limit on permanent indentation. C exceeds C0 for the
        small and medium bearings here, but **that is not a general rule** and
        asserting it would reject legitimate entries: for large roller bearings
        C0 is routinely the larger of the two. So the check is on the ratio
        being in a sane band, which still catches a value entered in kN against
        one in N, or the two fields swapped by an order of magnitude.
        """
        ratio = self.dynamic_rating_n / self.static_rating_n
        if not 0.1 <= ratio <= 20.0:
            raise ValueError(
                f"{self.id}: C/C0 = {ratio:.3g} is outside the plausible band "
                f"[0.1, 20]; check for a units error or swapped fields")
        return self

    @property
    def life_exponent(self) -> float:
        return LIFE_EXPONENT[self.bearing_type]


_SOURCE = ("representative deep groove ball bearing, verify vs manufacturer "
           "catalog")
_ROLLER_SOURCE = ("representative cylindrical roller bearing, verify vs "
                  "manufacturer catalog")

BEARINGS: list[BearingSpec] = [
    BearingSpec(
        id="bearing_608", designation="608",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        bore_m=0.008, outer_diameter_m=0.022, width_m=0.007,
        dynamic_rating_n=3350.0, static_rating_n=1370.0,
        limiting_speed_rad_s=3665.0, mass_kg=0.012, source=_SOURCE),
    BearingSpec(
        id="bearing_6000", designation="6000",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        bore_m=0.010, outer_diameter_m=0.026, width_m=0.008,
        dynamic_rating_n=4550.0, static_rating_n=1960.0,
        limiting_speed_rad_s=3350.0, mass_kg=0.019, source=_SOURCE),
    BearingSpec(
        id="bearing_6002", designation="6002",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        bore_m=0.015, outer_diameter_m=0.032, width_m=0.009,
        dynamic_rating_n=5850.0, static_rating_n=2850.0,
        limiting_speed_rad_s=2620.0, mass_kg=0.030, source=_SOURCE),
    BearingSpec(
        id="bearing_6004", designation="6004",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        bore_m=0.020, outer_diameter_m=0.042, width_m=0.012,
        dynamic_rating_n=9950.0, static_rating_n=5000.0,
        limiting_speed_rad_s=2000.0, mass_kg=0.069, source=_SOURCE),
    BearingSpec(
        id="bearing_6206", designation="6206",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        bore_m=0.030, outer_diameter_m=0.062, width_m=0.016,
        dynamic_rating_n=20300.0, static_rating_n=11200.0,
        limiting_speed_rad_s=1466.0, mass_kg=0.200, source=_SOURCE),
    BearingSpec(
        id="bearing_nu206", designation="NU 206",
        bearing_type=BearingType.CYLINDRICAL_ROLLER,
        bore_m=0.030, outer_diameter_m=0.062, width_m=0.016,
        dynamic_rating_n=44000.0, static_rating_n=36500.0,
        limiting_speed_rad_s=1570.0, mass_kg=0.210, source=_ROLLER_SOURCE),
]

_BY_ID = {spec.id: spec for spec in BEARINGS}


def get_bearing(bearing_id: str) -> BearingSpec:
    try:
        return _BY_ID[bearing_id]
    except KeyError:
        raise KeyError(
            f"unknown bearing {bearing_id!r}. Known: "
            f"{', '.join(sorted(_BY_ID))}") from None


def all_bearings() -> list[BearingSpec]:
    """Every archetype, ordered by id so iteration is deterministic."""
    return [_BY_ID[key] for key in sorted(_BY_ID)]
