"""Common-pool sampler comparison: no truth leakage and shared proposals."""
import importlib.util
from pathlib import Path

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.search.hidden_state import HiddenStateSampler


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_sampler_quality.py"
    spec = importlib.util.spec_from_file_location("evaluate_sampler_quality_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_pool_outputs_valid_deals() -> None:
    import random

    module = _module()
    result = RoundRunner(
        agents=[HeuristicAgent() for _ in range(4)],
        rules=RAMSCH_RULES, rng=random.Random(93),
    ).run()
    state = result.game_state
    assert state.is_complete
    # A turn-12 observation contains public history without hidden hands.
    _, _, observation, truth = next(module.snapshots(
        games=1, seed=321, opponents="heuristic", turns=(12,),
    ))
    methods = module.budget_matched_deals(
        observation, samples=4, candidates=2, agreement=0.7,
        seed=123, max_attempts=5000,
    )
    assert len(methods["unweighted_32"][0]) == 4
    assert len(methods["weighted_32"][0]) == 4
    assert len(methods["unweighted_full_pool"][0]) == 8
    control = set(methods["unweighted_full_pool"][0])
    assert set(methods["weighted_32"][0]).issubset(control)
    for deals, _ in methods.values():
        for deal in deals:
            assert deal.build_state(observation).observation_for(
                observation.player_index
            ) == observation
        assert module.score_distribution(observation, truth, deals)["brier"] >= 0
