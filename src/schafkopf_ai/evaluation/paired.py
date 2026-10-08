from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.neural_card_play_agent import NeuralCardPlayAgent
from schafkopf_ai.agents.ppo_card_play_agent import PPOCardPlayAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner

AgentLoader = Callable[[Path | None, str], Agent]


@dataclass(frozen=True, slots=True)
class EvaluationAgentSpec:
    kind: str
    name: str
    checkpoint: Path | None = None


@dataclass(slots=True)
class PairedStats:
    reference_payments: list[int] = field(default_factory=list)
    candidate_payments: list[int] = field(default_factory=list)
    deltas: list[int] = field(default_factory=list)

    def add(self, reference_payment: int, candidate_payment: int) -> None:
        self.reference_payments.append(reference_payment)
        self.candidate_payments.append(candidate_payment)
        self.deltas.append(candidate_payment - reference_payment)


@dataclass(frozen=True, slots=True)
class PairedEvaluationResult:
    games: int
    reference_name: str
    candidate_name: str
    reference_kind: str
    candidate_kind: str
    reference_checkpoint: Path | None
    candidate_checkpoint: Path | None
    overall: PairedStats
    by_contract: dict[GameType, PairedStats]
    reference_wins: int
    candidate_wins: int
    candidate_better: int
    equal: int
    reference_better: int

    @property
    def mean_reference_payment(self) -> float:
        return statistics.fmean(self.overall.reference_payments)

    @property
    def mean_candidate_payment(self) -> float:
        return statistics.fmean(self.overall.candidate_payments)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(self.overall.deltas)

    @property
    def reference_win_rate(self) -> float:
        return self.reference_wins / self.games

    @property
    def candidate_win_rate(self) -> float:
        return self.candidate_wins / self.games


def confidence_interval_95(values: list[int]) -> tuple[float, float]:
    if len(values) < 2:
        mean = float(values[0]) if values else 0.0
        return mean, mean

    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * standard_error
    return mean - margin, mean + margin


def _checkpoint_required(
    checkpoint: Path | None,
    *,
    kind: str,
) -> Path:
    if checkpoint is None:
        raise ValueError(f"Agent kind {kind!r} requires a checkpoint.")
    return checkpoint


def _load_heuristic(checkpoint: Path | None, device: str) -> Agent:
    del device
    if checkpoint is not None:
        raise ValueError("HeuristicAgent does not use a checkpoint.")
    return HeuristicAgent()


def _load_neural(checkpoint: Path | None, device: str) -> Agent:
    path = _checkpoint_required(checkpoint, kind="bc/dagger")
    return NeuralCardPlayAgent.from_checkpoint(path, device=device)


def _load_ppo(checkpoint: Path | None, device: str) -> Agent:
    path = _checkpoint_required(checkpoint, kind="ppo")
    return PPOCardPlayAgent.from_checkpoint(
        path,
        device=device,
        stochastic=False,
    )


# Adding another learned card-play method only requires registering a loader
# here. Pairing, seeding, statistics, and reporting remain unchanged.
AGENT_LOADERS: dict[str, AgentLoader] = {
    "heuristic": _load_heuristic,
    "bc": _load_neural,
    "dagger": _load_neural,
    "neural": _load_neural,
    "ppo": _load_ppo,
}


def available_agent_kinds() -> tuple[str, ...]:
    return tuple(sorted(AGENT_LOADERS))


def load_evaluation_agent(
    spec: EvaluationAgentSpec,
    *,
    device: str,
) -> Agent:
    kind = spec.kind.lower()
    try:
        loader = AGENT_LOADERS[kind]
    except KeyError as exc:
        supported = ", ".join(available_agent_kinds())
        raise ValueError(
            f"Unsupported evaluation agent kind {spec.kind!r}. "
            f"Supported kinds: {supported}."
        ) from exc

    return loader(spec.checkpoint, device)


def validate_round(result: RoundResult) -> None:
    if not result.game_state.is_complete:
        raise RuntimeError("Evaluation round finished with an incomplete GameState.")
    if len(result.game_state.completed_tricks) != 8:
        raise RuntimeError("Completed evaluation round does not contain eight tricks.")
    if sum(result.player_points) != 120:
        raise RuntimeError(f"Augen do not sum to 120: {result.player_points}.")
    if sum(result.payments) != 0:
        raise RuntimeError(f"Payments do not sum to zero: {result.payments}.")


def run_paired_variant(
    *,
    focal_agent: Agent,
    game_index: int,
    master_seed: int,
) -> RoundResult:
    focal_seat = game_index % 4
    starting_player = (game_index // 4) % 4
    deal_seed = master_seed + game_index * 1_009 + 1

    agents: list[Agent] = []
    for seat in range(4):
        if seat == focal_seat:
            agents.append(focal_agent)
        else:
            opponent_seed = master_seed + game_index * 10_007 + seat * 101 + 50_000
            agents.append(RandomAgent(rng=random.Random(opponent_seed)))

    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=starting_player,
        rng=random.Random(deal_seed),
    ).run()
    validate_round(result)
    return result


def evaluate_paired_agents(
    *,
    reference_spec: EvaluationAgentSpec,
    candidate_spec: EvaluationAgentSpec,
    games: int,
    seed: int,
    device: str,
    progress_every: int = 0,
    progress_callback: Callable[[int, PairedStats], None] | None = None,
) -> PairedEvaluationResult:
    if games <= 0:
        raise ValueError("games must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")

    reference = load_evaluation_agent(reference_spec, device=device)
    candidate = load_evaluation_agent(candidate_spec, device=device)

    overall = PairedStats()
    by_contract: dict[GameType, PairedStats] = defaultdict(PairedStats)
    reference_wins = 0
    candidate_wins = 0
    candidate_better = 0
    equal = 0
    reference_better = 0

    for game_index in range(games):
        focal_seat = game_index % 4
        reference_result = run_paired_variant(
            focal_agent=reference,
            game_index=game_index,
            master_seed=seed,
        )
        candidate_result = run_paired_variant(
            focal_agent=candidate,
            game_index=game_index,
            master_seed=seed,
        )

        if reference_result.contract != candidate_result.contract:
            raise RuntimeError(
                "Paired runs produced different contracts. The generic evaluator "
                "currently assumes both focal agents use the same bidding policy."
            )

        reference_payment = reference_result.payments[focal_seat]
        candidate_payment = candidate_result.payments[focal_seat]
        overall.add(reference_payment, candidate_payment)
        by_contract[reference_result.contract.game_type].add(
            reference_payment,
            candidate_payment,
        )

        reference_wins += int(
            focal_seat in reference_result.game_result.winner_players
        )
        candidate_wins += int(
            focal_seat in candidate_result.game_result.winner_players
        )

        if candidate_payment > reference_payment:
            candidate_better += 1
        elif candidate_payment == reference_payment:
            equal += 1
        else:
            reference_better += 1

        completed = game_index + 1
        if (
            progress_callback is not None
            and progress_every
            and completed % progress_every == 0
        ):
            progress_callback(completed, overall)

    return PairedEvaluationResult(
        games=games,
        reference_name=reference_spec.name,
        candidate_name=candidate_spec.name,
        reference_kind=reference_spec.kind,
        candidate_kind=candidate_spec.kind,
        reference_checkpoint=reference_spec.checkpoint,
        candidate_checkpoint=candidate_spec.checkpoint,
        overall=overall,
        by_contract=dict(by_contract),
        reference_wins=reference_wins,
        candidate_wins=candidate_wins,
        candidate_better=candidate_better,
        equal=equal,
        reference_better=reference_better,
    )
