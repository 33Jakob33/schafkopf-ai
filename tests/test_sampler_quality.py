"""Tests for offline sampler probability metrics, including no truth leakage."""
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.search.hidden_state import Determinization


def test_exact_deal_has_perfect_marginal_calibration() -> None:
    # Import the stand-alone script by file path, not as a 'scripts' package.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_sampler_quality.py"
    spec = importlib.util.spec_from_file_location("sampler_quality_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    deck = Deck()
    hands = deck.deal()
    state = GameState.from_hands(
        hands=hands, contract=GameContract(GameType.WENZ, declarer=0)
    )
    observation = state.observation_for(0)
    truth = tuple(frozenset(p.cards) for p in state.players)
    score = module.score_distribution(
        observation, truth, [Determinization(initial_hands=hands)]
    )
    assert score["brier"] == 0.0
    assert score["true_holder_probability"] == 1.0


def test_snapshot_exposes_only_public_observation_to_sampler() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_sampler_quality.py"
    spec = importlib.util.spec_from_file_location("sampler_quality_script_2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _, _, observation, truth = next(module.snapshots(
        games=1, seed=1234, opponents="random", turns=(4,)
    ))
    assert observation.current_player is not None
    assert len(observation.hand) == 7
    assert len(truth) == 4
    assert all(len(hand) == 7 for hand in truth)
