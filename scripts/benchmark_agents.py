"""Run existing and future Schafkopf card-play agents on a frozen test suite."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.neural_card_play_agent import NeuralCardPlayAgent
from schafkopf_ai.agents.ppo_card_play_agent import PPOCardPlayAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.evaluation.benchmark import BenchmarkCase, load_manifest, save_manifest
from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner

FIELDS = (
    "benchmark_sha256", "agent_id", "case_id", "contract", "role",
    "focal_seat", "starting_player", "payment", "win", "schneider", "schwarz",
)


class FixedBiddingAgent(Agent):
    """Keep bidding identical across policies; evaluate *only* card play."""

    def __init__(self, card_policy: Agent) -> None:
        self.card_policy = card_policy
        self.bidder = HeuristicAgent()

    def choose_bidding_action(
        self, observation: BiddingObservation, legal_actions: tuple[BiddingAction, ...]
    ) -> BiddingAction:
        return self.bidder.choose_bidding_action(observation, legal_actions)

    def choose_card(
        self, observation: PlayerObservation, legal_cards: tuple[Card, ...]
    ) -> Card:
        return self.card_policy.choose_card(observation, legal_cards)


def load_agent(kind: str, *, checkpoint: Path | None, factory: str | None) -> Agent:
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "random":
        return RandomAgent(rng=random.Random(42))
    if kind in {"bc", "dagger"}:
        if checkpoint is None:
            raise ValueError(f"--checkpoint required for {kind}")
        return NeuralCardPlayAgent.from_checkpoint(checkpoint)
    if kind == "ppo":
        if checkpoint is None:
            raise ValueError("--checkpoint required for ppo")
        return PPOCardPlayAgent.from_checkpoint(checkpoint, stochastic=False)
    if kind == "custom":
        if not factory or ":" not in factory:
            raise ValueError("--factory must be module:Class for custom agents")
        module_name, class_name = factory.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        instance = cls()
        if not isinstance(instance, Agent):
            raise TypeError("Custom agent must extend Agent")
        return instance
    raise ValueError(f"Unknown agent type: {kind}")


def play_case(case: BenchmarkCase, focal_agent: Agent, benchmark_sha256: str, agent_id: str) -> dict:
    agents: list[Agent] = [
        RandomAgent(rng=random.Random(case.opponent_seeds[seat])) for seat in range(4)
    ]
    agents[case.focal_seat] = FixedBiddingAgent(focal_agent)
    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=case.starting_player,
        rng=random.Random(case.deal_seed),
    ).run()

    state = result.game_state
    played = [play.card for trick in state.completed_tricks for play in trick.plays]
    if (
        not state.is_complete
        or len(state.completed_tricks) != 8
        or len(played) != 32
        or len(set(played)) != 32
        or sum(result.player_points) != 120
        or sum(result.payments) != 0
        or result.redeals != 0
    ):
        raise RuntimeError(f"Invalid game in benchmark case {case.case_id}")

    if result.contract.game_type.value == "ramsch":
        role = "ramsch"
    elif result.contract.declarer == case.focal_seat:
        role = "declarer"
    else:
        role = "defender"

    won = case.focal_seat in result.game_result.winner_players
    return {
        "benchmark_sha256": benchmark_sha256,
        "agent_id": agent_id,
        "case_id": case.case_id,
        "contract": result.contract.game_type.value,
        "role": role,
        "focal_seat": case.focal_seat,
        "starting_player": case.starting_player,
        "payment": result.payments[case.focal_seat],
        "win": int(won),
        "schneider": int(won and result.game_result.schneider),
        "schwarz": int(won and result.game_result.schwarz),
    }


def summarise(rows: list[dict]) -> dict:
    payments = [int(row["payment"]) for row in rows]
    n = len(payments)
    mean = statistics.fmean(payments)
    sem = statistics.stdev(payments) / math.sqrt(n) if n > 1 else 0.0
    return {
        "games": n,
        "mean_payment": mean,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "total_payment": sum(payments),
        "win_rate": sum(int(row["win"]) for row in rows) / n,
        "schneider_rate": sum(int(row["schneider"]) for row in rows) / n,
        "schwarz_rate": sum(int(row["schwarz"]) for row in rows) / n,
    }


def run(args: argparse.Namespace) -> None:
    if args.command == "generate":
        print("Created benchmark SHA256:", save_manifest(
            args.manifest, count=args.games, seed=args.seed
        ))
        return

    cases, digest = load_manifest(args.manifest)
    if args.command == "compare":
        grouped: dict[str, list[dict]] = defaultdict(list)
        for path in args.results:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["benchmark_sha256"] != digest:
                        raise ValueError(f"Incompatible benchmark manifest in {path}")
                    grouped[row["agent_id"]].append(row)
        for agent_id, rows in sorted(grouped.items()):
            if len(rows) != len(cases) or {int(r["case_id"]) for r in rows} != set(range(len(cases))):
                raise ValueError(f"Incomplete or duplicate cases for {agent_id}")
            print(agent_id, json.dumps(summarise(rows), indent=2))
        if len(grouped) > 1:
            baseline_id = args.baseline
            if baseline_id not in grouped:
                raise ValueError(f"Missing baseline: {baseline_id}")
            baseline = {int(r["case_id"]): int(r["payment"]) for r in grouped[baseline_id]}
            for agent_id, rows in sorted(grouped.items()):
                if agent_id == baseline_id:
                    continue
                differences = [
                    int(row["payment"]) - baseline[int(row["case_id"])] for row in rows
                ]
                avg = statistics.fmean(differences)
                sem = (statistics.stdev(differences) / math.sqrt(len(differences))
                       if len(differences) > 1 else 0.0)
                print(f"{agent_id} minus {baseline_id}: {avg:+.3f} "
                      f"(paired 95% CI [{avg - 1.96*sem:+.3f}, {avg + 1.96*sem:+.3f}])")
        return

    if not args.agent_id.strip():
        raise ValueError("--agent-id cannot be empty")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {args.output}")
    agent = load_agent(args.agent, checkpoint=args.checkpoint, factory=args.factory)
    rows = [play_case(case, agent, digest, args.agent_id) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "benchmark_sha256": digest, "agent_id": args.agent_id, "agent_type": args.agent,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "factory": args.factory, "games": len(rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": str(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summarise(rows), indent=2))
    print("Saved", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Create a frozen, seed-based test manifest")
    gen.add_argument("--manifest", type=Path, required=True)
    gen.add_argument("--games", type=int, default=10_000)
    gen.add_argument("--seed", type=int, default=20261020)
    evaluate = sub.add_parser("evaluate", help="Evaluate one agent on all manifest cases")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--agent", choices=["heuristic", "random", "bc", "dagger", "ppo", "custom"], required=True)
    evaluate.add_argument("--agent-id", required=True, help="Unique model/version label")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--factory", help="module:Class (custom agents, e.g. rollouts)")
    evaluate.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare", help="Compare complete matching benchmark results")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--results", type=Path, nargs="+", required=True)
    compare.add_argument("--baseline", default="heuristic")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
