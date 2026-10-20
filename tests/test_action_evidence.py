"""Regression tests for evidence-weighted sampling and timed agents."""
import random

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.rollout_card_play_agent import RolloutCardPlayAgent
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.search.action_evidence import observed_action_log_likelihood
from schafkopf_ai.search.hidden_state import HiddenStateSampler


def test_weighted_sampling_keeps_legal_reconstructed_states() -> None:
    agents = [HeuristicAgent() for _ in range(4)]
    result = RoundRunner(
        agents=agents, rules=RAMSCH_RULES, rng=random.Random(51)
    ).run()
    observation = result.game_state.observation_for(0)
    sampler = HiddenStateSampler(rng=random.Random(17))
    deal = sampler.sample(observation)
    assert deal.build_state(observation).is_complete
    assert observed_action_log_likelihood(observation, deal) <= 0.0


def test_rollout_evidence_parameters() -> None:
    agent = RolloutCardPlayAgent(
        samples_per_action=8, continuation_profile="mixed",
        evidence_candidates=4, evidence_agreement=0.7,
    )
    assert agent.evaluator.evidence_candidates == 4
    assert agent.evaluator.evidence_agreement == 0.7
