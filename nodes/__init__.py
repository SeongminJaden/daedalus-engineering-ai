"""nodes: capabilities routed across process and machine boundaries over MCP.

The Phase 14b method registry decides whether a method may be used on a
problem. This layer adds where it runs, and applies the same rule to a method
in this process, a subprocess over stdio and a service over HTTP.

The MCP SDK is an optional dependency (requirements-nodes.txt). Everything
except `engine_node.build_engine_server` works without it.
"""

from .descriptor import (CapabilityUnavailable, NodeDescriptor, Transport)
from .engine_node import (ENGINE_NODE_NAME, build_engine_server,
                          default_capability_registry, engine_descriptor,
                          evaluate_beam_section, route)
from .fusion_node import (FUSION_ANALYSES, FUSION_CAPABILITY,
                          FUSION_NODE_NAME, FUSION_UNAVAILABLE_REASON,
                          FusionVerificationReport, FusionVerificationRequest,
                          fusion_capability_method, fusion_descriptor, verify)
from .reasoning_node import (REASONING_CAPABILITY, REASONING_NODE_NAME,
                             reasoning_capability_method, reasoning_descriptor)
from .roster import build_roster
from .registry import (IN_PROCESS_NODE, Capability, CapabilityCandidates,
                       CapabilityExclusion, CapabilityRegistry,
                       DuplicateCapability, UnknownCapability)
from .pinocchio_node import (PINOCCHIO_CAPABILITY, PINOCCHIO_NODE_NAME,
                             MultibodyComparison, compare as pinocchio_compare,
                             load_model as pinocchio_load_model,
                             pinocchio_descriptor,
                             version as pinocchio_version)
from .openfoam import (OPENFOAM_CAPABILITY, OPENFOAM_NODE_NAME, ChannelCase,
                       FlowResult, PipeCase, openfoam_descriptor,
                       solve as openfoam_solve,
                       version as openfoam_version)
from .calculix import (CALCULIX_CAPABILITY, CALCULIX_NODE_NAME, CalculixResult,
                       ElementType, calculix_descriptor, is_available,
                       solve as calculix_solve, version as calculix_version,
                       write_deck)
from .verification import (CROSS_VALIDATED, EXTERNALLY_VERIFIED,
                           SELF_FEM_ONLY, STATUS_ORDER, CrossValidation,
                           FlowCrossValidation, VerificationStatus,
                           cross_validated_status,
                           flow_cross_validated_status,
                           request_external_verification)

__all__ = [
    "Capability", "CapabilityCandidates", "CapabilityExclusion",
    "CapabilityRegistry", "CapabilityUnavailable", "DuplicateCapability",
    "CALCULIX_CAPABILITY", "CALCULIX_NODE_NAME", "CROSS_VALIDATED",
    "CalculixResult", "CrossValidation", "ElementType", "ENGINE_NODE_NAME",
    "EXTERNALLY_VERIFIED", "STATUS_ORDER", "calculix_descriptor",
    "OPENFOAM_CAPABILITY", "OPENFOAM_NODE_NAME", "ChannelCase", "PipeCase",
    "FlowResult", "openfoam_descriptor", "openfoam_solve", "openfoam_version",
    "FlowCrossValidation", "flow_cross_validated_status",
    "PINOCCHIO_CAPABILITY", "PINOCCHIO_NODE_NAME", "MultibodyComparison",
    "pinocchio_compare", "pinocchio_load_model", "pinocchio_descriptor",
    "pinocchio_version",
    "calculix_solve", "calculix_version", "cross_validated_status",
    "is_available", "write_deck", "FUSION_ANALYSES",
    "FUSION_CAPABILITY", "FUSION_NODE_NAME", "FUSION_UNAVAILABLE_REASON",
    "FusionVerificationReport", "FusionVerificationRequest", "IN_PROCESS_NODE",
    "NodeDescriptor", "REASONING_CAPABILITY", "REASONING_NODE_NAME",
    "SELF_FEM_ONLY", "Transport", "UnknownCapability", "VerificationStatus",
    "build_engine_server", "build_roster", "default_capability_registry", "engine_descriptor",
    "evaluate_beam_section", "fusion_capability_method", "fusion_descriptor",
    "reasoning_capability_method", "reasoning_descriptor",
    "request_external_verification", "route", "verify",
]
