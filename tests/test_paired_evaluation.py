from pathlib import Path

import pytest

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.evaluation.paired import (
    EvaluationAgentSpec,
    PairedStats,
    available_agent_kinds,
    confidence_interval_95,
    load_evaluation_agent,
)


def test_paired_stats_delta_is_candidate_minus_reference() -> None:
    stats = PairedStats()

    stats.add(reference_payment=20, candidate_payment=35)
    stats.add(reference_payment=-10, candidate_payment=-30)

    assert stats.reference_payments == [20, -10]
    assert stats.candidate_payments == [35, -30]
    assert stats.deltas == [15, -20]


def test_confidence_interval_single_value_is_exact() -> None:
    assert confidence_interval_95([7]) == (7.0, 7.0)


def test_default_agent_registry_contains_current_training_methods() -> None:
    kinds = available_agent_kinds()

    assert "heuristic" in kinds
    assert "bc" in kinds
    assert "dagger" in kinds
    assert "ppo" in kinds


def test_load_heuristic_agent_without_checkpoint() -> None:
    agent = load_evaluation_agent(
        EvaluationAgentSpec(kind="heuristic", name="Heuristic"),
        device="cpu",
    )

    assert isinstance(agent, HeuristicAgent)


def test_heuristic_agent_rejects_checkpoint() -> None:
    with pytest.raises(ValueError, match="does not use a checkpoint"):
        load_evaluation_agent(
            EvaluationAgentSpec(
                kind="heuristic",
                name="Heuristic",
                checkpoint=Path("unused.pt"),
            ),
            device="cpu",
        )


def test_unknown_agent_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation agent kind"):
        load_evaluation_agent(
            EvaluationAgentSpec(kind="future-method", name="Future"),
            device="cpu",
        )
