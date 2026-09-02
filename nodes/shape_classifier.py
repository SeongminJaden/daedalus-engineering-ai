"""The shape classifier as a capability: rules that decide, a model that suggests.

Registered on its own in-process node so that the router can say whether a
part can be classified at all, which depends on OpenCASCADE being present,
and so that a caller asking "what kind of part is this" is answered by a
method with a stated domain rather than by whatever code happens to be near.

The capability is the RULE classifier. The nearest-neighbour classifier is
reachable through the same module but is not what this capability promises,
because a SURROGATE answer is not something a capability may hand back as a
result. It is exposed for comparison, and a caller who wants it says so.
"""

from __future__ import annotations

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

CLASSIFIER_NODE_NAME = "shape.classifier"
CLASSIFY_CAPABILITY = "analysis.cad.classify"


def is_available() -> bool:
    try:
        import OCP  # noqa: F401
    except ImportError:
        return False
    return True


def shape_classifier_descriptor(available: bool | None = None
                                ) -> NodeDescriptor:
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=CLASSIFIER_NODE_NAME, transport=Transport.IN_PROCESS,
        address="OCP", available=present,
        unavailable_reason="" if present else
        "unavailable: OpenCASCADE bindings (OCP) are not installed")


def shape_classifier_capability_method():
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=CLASSIFY_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Which of the five synthetic families a B-rep belongs to, by "
                "topology rules, or UNKNOWN.",
        inputs=("solid",),
        outputs=("family", "descriptor"),
        fidelity=Fidelity.ANALYTICAL,
        cost=Cost.CHEAP,
        conditions=(
            Condition("the input is a CAD solid rather than parameters",
                      lambda c: c.require("has_cad_input")),
        ),
        implementation="core.part_dataset.classify.rule_classify",
        evidence="SIMULATED",
        notes="Five families: box, hollow_rect, l_bracket, plate_with_holes, "
              "stepped_shaft. Everything else is UNKNOWN with the reasons "
              "listed, and UNKNOWN is the correct answer for most real parts. "
              "The genus comes from V - E + F minus inner face loops; the "
              "version without the loops called a hollow tube solid, and was "
              "refuted by measurement. Two Fusion-authored plates classify as "
              "plate_with_holes, which is a check across kernels. The "
              "nearest-neighbour classifier beside this one grades SURROGATE, "
              "rejects what lies beyond the training set, and suggests only.")


def classify_step(path):
    """Rule classification of every solid in a STEP file."""
    from core.part_dataset.classify import rule_classify
    from core.part_dataset.descriptors import describe_step

    if not is_available():
        raise CapabilityUnavailable(
            CLASSIFY_CAPABILITY, CLASSIFIER_NODE_NAME,
            "OpenCASCADE bindings (OCP) are not installed")
    return [rule_classify(d) for d in describe_step(path)]
