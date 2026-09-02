"""Gazebo Fortress as an independent statics engine, measured before it was pinned.

Every run here is real time (the state stream keeps up only then), so the
file costs about half a minute. The control run with zero torque is the test
that shows the check can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.assembly import Assembly, Joint, JointType, Link, forward_kinematics
from core.assembly.statics import joint_torques
from core.assembly.urdf import assembly_to_urdf
from core.design_genome import DesignGenome, HollowRectangleSection
from geometry.cad_export.kernel import kernel_available
from integration.simulation import (envelope_interference, gazebo_available,
                                    hold_with_springs, posed_copy, statics_cross_check)

pytestmark = pytest.mark.slow
requires_gazebo = pytest.mark.skipif(not gazebo_available(), reason="ign gazebo is not installed")
requires_cad = pytest.mark.skipif(not kernel_available(), reason="build123d is required")
DENSITY = 2810.0


def _link(name, length, height=0.040, width=0.020):
    return Link(name=name, length_m=length, genome=DesignGenome(
        section=HollowRectangleSection(outer_width_m=width, outer_height_m=height,
                                       wall_thickness_m=0.002), material_id="al_7075_t6"))


def _origin(x):
    m = np.eye(4)
    m[0, 3] = x
    return m.tolist()


def arm() -> Assembly:
    limits = dict(lower_limit=-3.0, upper_limit=3.0)
    return Assembly(name="planar2", material_id="al_7075_t6",
                    links=[_link("link1", 0.30), _link("link2", 0.25)],
                    joints=[Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                                  child="link1", axis=[0, 0, 1], **limits),
                            Joint(name="j2", type=JointType.REVOLUTE, parent="link1",
                                  child="link2", axis=[0, 0, 1], origin=_origin(0.30),
                                  **limits)])


def test_the_posed_copy_puts_the_zero_configuration_at_q():
    a = arm()
    q = np.array([0.3, -0.5])
    posed = posed_copy(a, q)
    original = forward_kinematics(a, q)
    copy = forward_kinematics(posed, np.zeros(2))
    for name in ("link1", "link2"):
        assert np.allclose(original.link_transforms[name], copy.link_transforms[name], atol=1e-12)
    assert np.allclose(original.tool_position(), copy.tool_position(), atol=1e-12)


def test_envelopes_are_written_and_labelled_as_envelopes():
    text = assembly_to_urdf(arm(), DENSITY, envelopes=True)
    assert text.count("<collision") == 2 and text.count("<visual") == 2
    assert "ENVELOPE" in text and 'size="0.3 0.04 0.02"' in text
    assert "nominal inertia" in text                      # the base link keeps the tree
    bare = assembly_to_urdf(arm(), DENSITY)
    assert "<collision" not in bare


@requires_gazebo
def test_the_statics_torque_holds_the_pose_and_gazebo_agrees(tmp_path):
    """Measured: settled within 0.0002 rad, spring torque within 0.03 percent
    of the statics at the settled pose."""
    hold = statics_cross_check(arm(), DENSITY, np.array([0.3, -0.5]), tmp_path, seconds=3.0)
    assert hold.messages > 1000
    assert hold.max_drift_rad < 0.005
    assert np.all(hold.relative_errors < 0.01)
    assert hold.evidence == "simulated"
    print("\n" + hold.summary())


@requires_gazebo
def test_zero_torque_sags_and_the_engine_still_matches_the_statics_there(tmp_path):
    """The control. Measured: the shoulder sags 0.44 rad, and the spring
    torque where it stops matches the statics at that pose to 0.03 percent.
    A check that could not fail would prove nothing."""
    hold = hold_with_springs(arm(), DENSITY, np.array([0.3, -0.5]), np.zeros(2), tmp_path,
                             seconds=3.0)
    assert hold.max_drift_rad > 0.2
    assert np.all(hold.relative_errors < 0.01)


@requires_cad
def test_envelope_interference_is_geometric_and_names_the_pair():
    a = arm()
    clear = envelope_interference(a, np.array([0.3, -0.5]))
    assert clear.clear and clear.checked_pairs == 0        # only adjacent links, skipped
    folded = envelope_interference(a, np.array([0.0, np.pi - 0.05]), skip_adjacent=False)
    # link2 folded back over link1: the envelopes overlap along most of link2
    assert not folded.clear
    assert folded.clashes[0].first == "link1" and folded.clashes[0].second == "link2"
    assert folded.clashes[0].overlap_m3 > 0.5 * 0.25 * 0.04 * 0.02
    assert "envelopes, not parts" in folded.summary()
