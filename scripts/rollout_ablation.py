"""Paired rollout ablations with per-decision latency and persisted results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import random
from pathlib import Path

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.rollout_card_play_agent import RolloutCardPlayAgent
from schafkopf_ai.evaluation.paired import run_paired_variant
from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation


class TimedAgent(Agent):
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.decision_seconds: list[float] = []

    def choose_bidding_action(
        self, observation: BiddingObservation, legal_actions: tuple[BiddingAction, ...]
    ) -> BiddingAction:
        return self.agent.choose_bidding_action(observation, legal_actions)

    def choose_card(
        self, observation: PlayerObservation, legal_cards: tuple[Card, ...]
    ) -> Card:
        start = time.perf_counter()
        result = self.agent.choose_card(observation, legal_cards)
        self.decision_seconds.append(time.perf_counter() - start)
        return result


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def run(args: argparse.Namespace) -> None:
    if args.games < 1:
        raise ValueError("--games must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows: list[dict] = []
    summaries: list[dict] = []
    for mode in args.opponents:
        for profile in args.profiles:
            for samples in args.samples:
                for evidence in args.evidence_candidates:
                    differences: list[int] = []
                    baseline_times: list[float] = []
                    rollout_times: list[float] = []
                    wins = 0
                    for index in range(args.games):
                        # Each configuration repeats precisely the same deals,
                        # seat rotations and opponent seeds.
                        baseline = TimedAgent(HeuristicAgent())
                        rollout = TimedAgent(RolloutCardPlayAgent(
                            samples_per_action=samples,
                            seed=args.seed + index * 103 + 17,
                            continuation_profile=profile,
                            evidence_candidates=evidence,
                            evidence_agreement=args.evidence_agreement,
                        ))
                        ref = run_paired_variant(
                            focal_agent=baseline, game_index=index,
                            master_seed=args.seed, opponent_mode=mode,
                        )
                        candidate = run_paired_variant(
                            focal_agent=rollout, game_index=index,
                            master_seed=args.seed, opponent_mode=mode,
                        )
                        if ref.contract != candidate.contract:
                            raise RuntimeError("Bidding diverged between paired games")
                        seat = index % 4
                        delta = candidate.payments[seat] - ref.payments[seat]
                        differences.append(delta)
                        wins += seat in candidate.game_result.winner_players
                        baseline_times.extend(baseline.decision_seconds)
                        rollout_times.extend(rollout.decision_seconds)
                        rows.append({
                            "seed": args.seed, "case_id": index, "opponents": mode,
                            "profile": profile, "samples": samples,
                            "evidence_candidates": evidence,
                            "evidence_agreement": args.evidence_agreement,
                            "baseline_payment": ref.payments[seat],
                            "rollout_payment": candidate.payments[seat],
                            "delta": delta,
                            "baseline_decisions": len(baseline.decision_seconds),
                            "rollout_decisions": len(rollout.decision_seconds),
                            "baseline_seconds": sum(baseline.decision_seconds),
                            "rollout_seconds": sum(rollout.decision_seconds),
                        })
                    mean = statistics.fmean(differences)
                    se = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences)>1 else 0.0
                    result = {
                        "opponents": mode, "profile": profile,
                        "samples": samples, "evidence_candidates": evidence,
                        "games": args.games, "delta_mean": mean,
                        "delta_ci95": [mean - 1.96 * se, mean + 1.96 * se],
                        "rollout_win_rate": wins / args.games,
                        "baseline_decision_latency_seconds": summarize(baseline_times),
                        "rollout_decision_latency_seconds": summarize(rollout_times),
                        "baseline_decisions": len(baseline_times),
                        "rollout_decisions": len(rollout_times),
                    }
                    summaries.append(result)
                    print(json.dumps(result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps({"seed": args.seed, "results": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Wrote", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--opponents", nargs="+", choices=["random", "heuristic", "heuristic-bid-random-card"], default=["random", "heuristic"])
    parser.add_argument("--samples", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--profiles", nargs="+", choices=["heuristic", "mixed"], default=["heuristic", "mixed"])
    parser.add_argument("--evidence-candidates", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--evidence-agreement", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=Path("results/rollout_ablation.csv"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
