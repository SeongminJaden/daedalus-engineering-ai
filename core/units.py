"""core.units - SI conversion constants and helpers.

**Everything stored in an IR or a genome is SI base units**: metres, pascals,
newtons, kilograms, radians. These constants exist to convert *at the edges* -
when parsing human input or formatting output - never to store a value in some
other unit and remember to convert later.

    length_m = 50 * MM          # human writes 50 mm, we store 0.05 m
    print(to_mm(length_m))      # 50.0, only at the display boundary
"""

from __future__ import annotations

import math

# --- length ---
MM: float = 1e-3
CM: float = 1e-2
M: float = 1.0

# --- pressure / stress / modulus ---
PA: float = 1.0
KPA: float = 1e3
MPA: float = 1e6
GPA: float = 1e9

# --- force ---
N: float = 1.0
KN: float = 1e3

# --- angle ---
DEG: float = math.pi / 180.0
RAD: float = 1.0

# --- misc ---
KGF: float = 9.80665          # 1 kgf in newtons; also standard gravity in m/s^2
G0: float = 9.80665           # standard gravity, m/s^2


def to_mm(value_m: float) -> float:
    """Metres -> millimetres. Display boundary only."""
    return value_m / MM


def to_mpa(value_pa: float) -> float:
    """Pascals -> megapascals. Display boundary only."""
    return value_pa / MPA


def to_gpa(value_pa: float) -> float:
    """Pascals -> gigapascals. Display boundary only."""
    return value_pa / GPA


def to_kn(value_n: float) -> float:
    """Newtons -> kilonewtons. Display boundary only."""
    return value_n / KN


def to_deg(value_rad: float) -> float:
    """Radians -> degrees. Display boundary only."""
    return value_rad / DEG


def kgf_to_n(value_kgf: float) -> float:
    """Kilogram-force -> newtons (a mass hanging under standard gravity)."""
    return value_kgf * KGF
