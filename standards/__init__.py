"""standards: conformance to published dimensional standards, and snapping.

Conformance is NOT correctness. A design can use only standard threads, grades
and preferred dimensions and still be unfit for its duty. This layer says
whether a supplier would recognise the numbers, not whether the design works.
"""

from .compliance import (ComplianceReport, Conformance, DimensionCheck,
                         check_bolt_grade, check_fit_size, check_key_shaft,
                         check_preferred, check_thread)
from .renard import (Series, SnapDirection, SnapReport, rounding_deviation,
                     series_values, snap, snap_with_report, theoretical_step)

__all__ = [
    "ComplianceReport", "Conformance", "DimensionCheck", "Series",
    "SnapDirection", "SnapReport", "check_bolt_grade", "check_fit_size",
    "check_key_shaft", "check_preferred", "check_thread",
    "rounding_deviation", "series_values", "snap", "snap_with_report",
    "theoretical_step",
]
