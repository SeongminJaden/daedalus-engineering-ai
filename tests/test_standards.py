"""Preferred numbers, snapping and standards conformance.

Two tests carry the phase. `test_the_snap_direction_is_required_because_neither_way_is_safe`
pins that rounding a dimension is a design change with no universally safe
direction. `test_an_unverifiable_dimension_is_not_a_pass` pins the three-state
report: "we have no table for this" is a different statement from "this is in
the standard", and collapsing them would manufacture conformance.
"""

import pytest

from standards import (ComplianceReport, Conformance, Series, SnapDirection,
                       check_bolt_grade, check_fit_size, check_key_shaft,
                       check_preferred, check_thread, rounding_deviation,
                       series_values, snap, snap_with_report,
                       theoretical_step)


# --- preferred numbers -------------------------------------------------------

def test_each_series_has_the_number_of_steps_its_name_says():
    for series, count in ((Series.R5, 5), (Series.R10, 10), (Series.R20, 20),
                          (Series.R40, 40)):
        assert len(series_values(series, decades=1, start_decade=0)) == count


def test_the_series_close_the_decade():
    """Each series starts at 1 and its last step lands back on 10.

    The last value differs per series and is NOT 8 for all of them: R5 ends at
    6.30, because with five steps a decade the final value is 10^(4/5). The
    check that holds for every series is that one more step closes the decade,
    which is what makes them repeat cleanly.
    """
    for series in Series:
        values = series_values(series, decades=1, start_decade=0)
        assert values == sorted(values)
        assert values[0] == pytest.approx(1.0)
        assert values[-1] < 10.0
        assert values[-1] * theoretical_step(series) == pytest.approx(
            10.0, rel=0.02)
    assert series_values(Series.R5, 1, 0)[-1] == pytest.approx(6.30)
    assert series_values(Series.R10, 1, 0)[-1] == pytest.approx(8.00)


def test_the_published_values_are_rounded_from_the_progression():
    """Measured, because the published series is what a drawing must use.

    R10 nominally steps by the tenth root of ten, 1.2589, and the standard says
    1.25. Computing the progression and using it would put unstandard numbers
    on a drawing while appearing to follow the standard.
    """
    assert theoretical_step(Series.R10) == pytest.approx(10.0 ** 0.1)
    for series in Series:
        deviation = rounding_deviation(series)
        assert deviation > 0.001, f"{series.value} is expected to be rounded"
        assert deviation < 0.02, f"{series.value} deviates by {deviation:.2%}"


def test_finer_series_contain_the_coarser_ones():
    """R10 contains R5, R20 contains R10, which is what makes them a family."""
    for coarse, fine in ((Series.R5, Series.R10), (Series.R10, Series.R20),
                         (Series.R20, Series.R40)):
        coarse_values = set(series_values(coarse, 1, 0))
        fine_values = set(series_values(fine, 1, 0))
        assert coarse_values <= fine_values


def test_the_snap_direction_is_required_because_neither_way_is_safe():
    """Rounding a dimension is a design change.

    37.4 mm on R20 goes UP to 40.00 and DOWN to 35.50, and NEAREST happens to
    go down. For a load-bearing thickness the downward move removes material
    and can invalidate the check that sized it, so the direction cannot have a
    universally correct default.
    """
    up = snap_with_report(37.4, Series.R20, SnapDirection.UP)
    down = snap_with_report(37.4, Series.R20, SnapDirection.DOWN)
    nearest = snap_with_report(37.4, Series.R20, SnapDirection.NEAREST)

    assert up.snapped == pytest.approx(40.0)
    assert down.snapped == pytest.approx(35.5)
    assert nearest.snapped == pytest.approx(35.5)

    assert up.is_conservative_for_strength
    assert not down.is_conservative_for_strength
    assert not nearest.is_conservative_for_strength
    assert up.changed and down.changed


def test_a_value_already_on_the_series_does_not_move():
    for direction in SnapDirection:
        report = snap_with_report(40.0, Series.R20, direction)
        assert report.snapped == pytest.approx(40.0)
        assert not report.changed


def test_snapping_outside_the_expanded_range_is_refused():
    """Rather than silently clamping to an endpoint."""
    with pytest.raises(ValueError, match="outside the expanded series"):
        snap(1e9, Series.R10, SnapDirection.UP)
    with pytest.raises(ValueError, match="positive"):
        snap(-1.0, Series.R10, SnapDirection.UP)


# --- conformance -------------------------------------------------------------

def test_a_known_thread_and_grade_are_reported_standard():
    assert check_thread("bolt", "M6").conformance is Conformance.STANDARD
    assert check_bolt_grade("bolt", "8.8").conformance is Conformance.STANDARD
    assert check_bolt_grade("bolt", "12.9").conformance is Conformance.STANDARD


def test_an_unverifiable_dimension_is_not_a_pass():
    """The three-state distinction, and the reason it exists.

    M7 is not in this project's table. That is a statement about the table, not
    about ISO, and reporting it as non-standard would be a claim this code
    cannot support while reporting it as standard would manufacture
    conformance. It is NOT_CHECKABLE, and a report containing one is not fully
    conformant.
    """
    unknown = check_thread("odd bolt", "M7")
    assert unknown.conformance is Conformance.NOT_CHECKABLE
    assert "may still be a standard size" in unknown.detail

    report = ComplianceReport()
    report.add(check_thread("good", "M6"))
    report.add(unknown)
    assert not report.fully_conformant
    assert report.not_checkable == [unknown]
    assert not report.non_standard


def test_a_fully_standard_set_is_conformant():
    report = ComplianceReport()
    report.add(check_thread("bolt", "M8"))
    report.add(check_bolt_grade("grade", "10.9"))
    report.add(check_fit_size("seat", 30.0))
    report.add(check_key_shaft("shaft", 30.0))
    report.add(check_preferred("width", 40.0))
    assert report.fully_conformant
    assert not report.non_standard and not report.not_checkable


def test_sizes_outside_the_tolerance_expression_are_not_checkable():
    """Which is what the Phase 23 domain restriction implies here."""
    assert check_fit_size("pin", 2.0).conformance is Conformance.NOT_CHECKABLE
    assert check_fit_size("shaft", 30.0).conformance is Conformance.STANDARD
    assert check_fit_size("huge", 900.0).conformance is Conformance.NOT_CHECKABLE


def test_a_non_preferred_dimension_gets_a_proposal_not_an_edit():
    """Snapping is a suggestion: applying it can invalidate the analysis."""
    check = check_preferred("wall", 3.7, Series.R20, SnapDirection.UP)
    assert check.conformance is Conformance.NON_STANDARD
    assert check.suggested == pytest.approx(4.0)
    assert check.relative_change > 0.0
    # The original value is unchanged in the report.
    assert check.value == pytest.approx(3.7)


def test_the_default_snap_direction_favours_strength():
    """Most free dimensions here are load-bearing, where growing is safe."""
    check = check_preferred("thickness", 3.7)
    assert check.suggested > 3.7


def test_a_clearance_can_be_snapped_the_other_way():
    check = check_preferred("gap", 3.7, Series.R20, SnapDirection.DOWN)
    assert check.suggested == pytest.approx(3.55)
    assert check.relative_change < 0.0


def test_the_report_summarises_all_three_states():
    report = ComplianceReport()
    report.add(check_thread("a", "M6"))
    report.add(check_thread("b", "M7"))
    report.add(check_preferred("c", 3.7))
    summary = report.summary()
    assert "1 standard" in summary
    assert "1 non-standard" in summary
    assert "1 not checkable" in summary


def test_conformance_is_not_correctness():
    """The limit worth stating, expressed as a property of the API.

    Nothing in a compliance report says whether the design works. A fully
    conformant report carries no safety factor, no verdict and no analysis.
    """
    report = ComplianceReport()
    report.add(check_thread("bolt", "M8"))
    assert report.fully_conformant
    assert not hasattr(report, "safety_factor")
    assert not hasattr(report, "passes")
