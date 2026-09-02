"""The node roster: every node this system knows about, available or not.

Unavailable nodes are in the roster. That is deliberate. A router that only
knows about reachable nodes cannot distinguish a capability nobody has built
from one that exists and is blocked, and those lead to different decisions:
the first needs implementing, the second needs an entitlement.
"""

from __future__ import annotations

from core.registry import DEFAULT_REGISTRY, MethodRegistry

from .calculix import (calculix_capability_method,
                       calculix_descriptor,
                       calculix_general_capability_method)
from .feature_recognizer import (feature_recognizer_capability_method,
                                 feature_recognizer_descriptor)
from .code_aster import (code_aster_capability_method,
                         code_aster_descriptor)
from .elmer import elmer_capability_method, elmer_descriptor
from .gmsh_node import gmsh_capability_method, gmsh_descriptor
from .mujoco_node import mujoco_capability_method, mujoco_descriptor
from .openfoam import openfoam_capability_method, openfoam_descriptor
from .step_analyzer import (step_analyzer_capability_method,
                            step_analyzer_descriptor)
from .pinocchio_node import (pinocchio_capability_method,
                             pinocchio_descriptor)
from .shape_classifier import (shape_classifier_capability_method,
                               shape_classifier_descriptor)
from .engine_node import engine_descriptor
from .fusion_node import fusion_capability_method, fusion_descriptor
from .reasoning_node import reasoning_capability_method, reasoning_descriptor
from .registry import CapabilityRegistry


def build_roster(methods: MethodRegistry | None = None,
                 fusion_available: bool = False,
                 reasoning_available: bool = False) -> CapabilityRegistry:
    """Assemble the full capability registry across all known nodes.

    `fusion_available` exists so the consuming code can be exercised against
    the available branch without pretending the entitlement is bought. Tests
    use it to check that a real report would be accepted; nothing in the
    shipped path sets it.
    """
    registry = CapabilityRegistry()
    registry.adopt(methods if methods is not None else DEFAULT_REGISTRY,
                   engine_descriptor())
    registry.register(fusion_capability_method(),
                      fusion_descriptor(available=fusion_available))
    # CalculiX reports its own availability from the filesystem rather than
    # from a flag, because whether the binary is there is a fact and not a
    # policy.
    registry.register(calculix_capability_method(), calculix_descriptor())
    registry.register(calculix_general_capability_method(),
                      calculix_descriptor())
    registry.register(feature_recognizer_capability_method(),
                      feature_recognizer_descriptor())
    registry.register(code_aster_capability_method(),
                      code_aster_descriptor())
    registry.register(elmer_capability_method(), elmer_descriptor())
    registry.register(gmsh_capability_method(), gmsh_descriptor())
    registry.register(mujoco_capability_method(), mujoco_descriptor())
    registry.register(openfoam_capability_method(), openfoam_descriptor())
    registry.register(step_analyzer_capability_method(),
                      step_analyzer_descriptor())
    registry.register(pinocchio_capability_method(),
                      pinocchio_descriptor())
    registry.register(shape_classifier_capability_method(),
                      shape_classifier_descriptor())
    registry.register(reasoning_capability_method(),
                      reasoning_descriptor(available=reasoning_available))
    return registry
