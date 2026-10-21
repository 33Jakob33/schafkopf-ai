"""Pre-registered independent holdout for the three selected rollout policies.

Each (case, variant) is appended immediately. The baseline is evaluated once
per case. Resume is idempotent, refuses incompatible configurations, and
does not repeat already completed runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
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
        chosen = self.agent.choose_card(observation, legal_cards)
        self.decision_seconds.append(time.perf_counter() - start)
        return chosen


VARIANTS = {
    "fast-h8-e1": (8, "heuristic", 1),
    "balanced-m32-e4": (32, "mixed", 4),
    "best-m128-e4": (128, "mixed", 4),
}
FIELDS = (
    "case_id",
    "variant",
    "baseline_payment",
    "rollout_payment",
    "delta",
    "baseline_win",
    "rollout_win",
    "baseline_seconds",
    "rollout_seconds",
    "baseline_decisions",
    "rollout_decisions",
)


def summary(rows: list[dict[str, str]]) -> dict[str, object]:
    deltas = [int(r["delta"]) for r in rows]
    mean = statistics.fmean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    seconds = sum(float(r["rollout_seconds"]) for r in rows)
    decisions = sum(int(r["rollout_decisions"]) for r in rows)
    return {
        "games": len(rows),
        "mean_delta": mean,
        "paired_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "baseline_win_rate": sum(int(r["baseline_win"]) for r in rows) / len(rows),
        "rollout_win_rate": sum(int(r["rollout_win"]) for r in rows) / len(rows),
        "rollout_seconds_total": seconds,
        "mean_seconds_per_card_decision": seconds / decisions,
    }


def run(args: argparse.Namespace) -> None:
    if args.games < 1:
        raise ValueError("--games must be positive")
    if args.seed == 20261020:
        raise ValueError("Use an independent seed, not the exploratory seed 20261020")
    chosen = list(VARIANTS) if args.variants == ["all"] else args.variants
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate variant requested")
    if any(name not in VARIANTS for name in chosen):
        raise ValueError("Unknown variant")
    # A fixed protocol is part of the immutable benchmark identity.
    protocol = {
        "version": 1,
        "seed": args.seed,
        "games": args.games,
        "opponents": "heuristic",
        "bidding_profile": "baseline",
        "selected_variants": chosen,
        "evidence_agreement": 0.7,
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    metadata_path = args.output.with_suffix(".protocol.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored.get("sha256") != digest:
            raise ValueError("Existing result protocol differs; choose another output")
    elif args.output.exists():
        raise ValueError("Results exist without protocol metadata: refusing to append")
    else:
        metadata_path.write_text(
            json.dumps({"sha256": digest, **protocol}, indent=2) + "\n",
            encoding="utf-8",
        )

    completed: dict[tuple[int, str], dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Unexpected results CSV schema")
            for row in reader:
                key = (int(row["case_id"]), row["variant"])
                if (
                    key in completed
                    or key[0] not in range(args.games)
                    or key[1] not in chosen
                ):
                    raise ValueError(f"Invalid/duplicate completed case {key}")
                completed[key] = row

    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not completed:
            writer.writeheader()
            handle.flush()
        for case_id in range(args.games):
            pending = [v for v in chosen if (case_id, v) not in completed]
            if not pending:
                continue
            baseline = TimedAgent(HeuristicAgent())
            baseline_result = run_paired_variant(
                focal_agent=baseline,
                game_index=case_id,
                master_seed=args.seed,
                opponent_mode="heuristic",
            )
            seat = case_id % 4
            for name in pending:
                samples, profile, evidence = VARIANTS[name]
                candidate = TimedAgent(
                    RolloutCardPlayAgent(
                        samples_per_action=samples,
                        continuation_profile=profile,
                        evidence_candidates=evidence,
                        evidence_agreement=0.7,
                        seed=args.seed + case_id * 103 + 17,
                    )
                )
                result = run_paired_variant(
                    focal_agent=candidate,
                    game_index=case_id,
                    master_seed=args.seed,
                    opponent_mode="heuristic",
                )
                if baseline_result.contract != result.contract:
                    raise RuntimeError("Different contracts in paired games")
                row = {
                    "case_id": case_id,
                    "variant": name,
                    "baseline_payment": baseline_result.payments[seat],
                    "rollout_payment": result.payments[seat],
                    "delta": result.payments[seat] - baseline_result.payments[seat],
                    "baseline_win": int(
                        seat in baseline_result.game_result.winner_players
                    ),
                    "rollout_win": int(seat in result.game_result.winner_players),
                    "baseline_seconds": sum(baseline.decision_seconds),
                    "rollout_seconds": sum(candidate.decision_seconds),
                    "baseline_decisions": len(baseline.decision_seconds),
                    "rollout_decisions": len(candidate.decision_seconds),
                }
                writer.writerow(row)
                handle.flush()
                completed[(case_id, name)] = {k: str(v) for k, v in row.items()}
            if (case_id + 1) % args.progress_every == 0:
                print(
                    f"Completed {case_id + 1}/{args.games} deals, "
                    f"{len(completed)} of {args.games * len(chosen)} results",
                    flush=True,
                )

    report = {
        name: summary([completed[(i, name)] for i in range(args.games)])
        for name in chosen
    }
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(
            {
                "protocol_sha256": digest,
                "seed": args.seed,
                "results": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help="all or selected variants: " + ", ".join(VARIANTS),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/holdout_20261119.csv")
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
