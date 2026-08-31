"""brain - the engineering Brain: evidence-graded accumulated experience.

WHAT THIS STORE IS, STATED PLAINLY:

  * It is **not** a repository of experimentally validated facts. Everything in
    it that came from a run came from simulation at beam-theory fidelity
    (physics/README.md), and is graded accordingly.
  * Retrieval is **numeric feature similarity**, not semantic/text search.
    There is no embedding model here.
  * `EXPERIMENTALLY_VALIDATED` is reachable only with physical-test evidence.
    No volume of simulation can promote anything to it.

It is a store of *evidence-graded experience*, and every item says how far it
has earned trust.

The Brain is a plain SQLite file and depends on nothing but stdlib sqlite3 and
numpy, so it can be opened and queried with no reasoner, no model and no GPU -
the model/brain separation.
"""

from __future__ import annotations

from pathlib import Path

from .db import DEFAULT_DB_PATH, BrainDB
from .episodic import EpisodicMemory
from .knowledge_graph import KnowledgeGraph
from .retrieval import (
    design_vector,
    nearest_designs,
    retrieve_knowledge,
    retrieve_similar,
    retrieve_strategies,
    warm_start_from_memory,
)
from .semantic import (
    Counterexample,
    Evidence,
    EvidenceKind,
    EvidenceLevel,
    Knowledge,
    SemanticMemory,
    generalize_all,
)
from .skills import Skill, SkillLibrary
from .strategy import (Strategy, StrategyStore, derive_method_strategies,
                       derive_stiffness_strategy)


class Brain:
    """Facade over the stores. Open it, query it, close it."""

    def __init__(self, path: str | Path | None = None):
        self.db = BrainDB(path)
        self.episodic = EpisodicMemory(self.db)
        self.semantic = SemanticMemory(self.db)
        self.graph = KnowledgeGraph(self.db)
        self.strategies = StrategyStore(self.db)
        self.skills = SkillLibrary()

    # --- writing -------------------------------------------------------- #
    def record_loop_result(self, result, problem_name: str,
                           problem_params: dict | None = None) -> str:
        """Ingest a Phase 4 LoopResult: run, designs, episodes.

        This is what the loop's UPDATE_BRAIN phase calls.
        """
        run_id = result.run_id
        self.episodic.record_run(
            run_id=run_id,
            problem_name=problem_name,
            termination=result.termination.value,
            iterations=result.iterations,
            best_mass_kg=result.best_mass_kg,
            meta={"budget": result.budget, "problem_params": problem_params or {}},
        )
        for episode in result.episodes:
            design_id = self.episodic.record_design(
                genome=episode.design_genome,
                metrics=episode.observation,
                feasible=episode.feasible,
                active_constraints=episode.constraint_status.get("active", []),
                run_id=run_id,
                design_id=f"d-{episode.id}",
            )
            self.episodic.record_episode(episode, run_id=run_id, design_id=design_id)
        return run_id

    def record_episode(self, episode, run_id: str, problem_name: str = "unknown") -> str:
        """Store a single episode as it happens (incremental UPDATE_BRAIN)."""
        if self.episodic.get_run(run_id) is None:
            self.episodic.record_run(run_id, problem_name)
        design_id = self.episodic.record_design(
            genome=episode.design_genome,
            metrics=episode.observation,
            feasible=episode.feasible,
            active_constraints=episode.constraint_status.get("active", []),
            run_id=run_id,
            design_id=f"d-{episode.id}",
        )
        return self.episodic.record_episode(episode, run_id=run_id,
                                            design_id=design_id)

    def generalize(self, domain: str = "cantilever_link") -> list[Knowledge]:
        """Promote repeated observations into evidence-graded knowledge."""
        return generalize_all(self.episodic.episodes(), self.semantic, domain)

    # --- reading -------------------------------------------------------- #
    def similar_designs(self, genome=None, metrics=None, problem_params=None,
                        k: int = 5, feasible_only: bool = True):
        return retrieve_similar(self.episodic, genome, metrics, problem_params,
                                k=k, feasible_only=feasible_only)

    def knowledge(self, domain: str | None = None,
                  min_level: EvidenceLevel | None = None):
        return retrieve_knowledge(self.semantic, domain, min_level)

    def applicable_strategies(self, context: dict | None = None,
                              min_confidence: float = 0.0):
        return retrieve_strategies(self.strategies, context, min_confidence)

    def warm_start(self, problem_params: dict | None = None):
        return warm_start_from_memory(self.episodic, problem_params)

    def summary(self) -> dict:
        return {
            "path": str(self.db.path),
            "counts": self.db.counts(),
            "knowledge_by_level": {
                level.value: sum(
                    1 for k in self.semantic.by_domain() if k.evidence_level is level
                )
                for level in EvidenceLevel
            },
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Brain":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


__all__ = [
    "DEFAULT_DB_PATH", "Brain", "BrainDB", "Counterexample", "EpisodicMemory",
    "Evidence", "EvidenceKind", "EvidenceLevel", "Knowledge", "KnowledgeGraph",
    "SemanticMemory", "Skill", "SkillLibrary", "Strategy", "StrategyStore",
    "derive_stiffness_strategy", "design_vector", "generalize_all",
    "nearest_designs", "retrieve_knowledge", "retrieve_similar",
    "retrieve_strategies", "warm_start_from_memory",
]
