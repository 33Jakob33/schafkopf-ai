"""Offline hidden-card calibration: unweighted vs action-evidence sampling.

Truth is used only in score_distribution(), never in sampling or weighting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.evaluation.paired import run_paired_variant
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.search.rollout import MonteCarloActionEvaluator
from schafkopf_ai.search.action_evidence import observed_action_log_likelihood

METHODS = ("unweighted", "weighted")
BUDGET_METHODS = ("unweighted_32", "weighted_32", "unweighted_full_pool")


def snapshots(*, games: int, seed: int, opponents: str, turns: tuple[int, ...]):
    """Reconstruct observable positions from completed legal simulation games."""
    for game_index in range(games):
        finished = run_paired_variant(
            focal_agent=HeuristicAgent(), game_index=game_index,
            master_seed=seed, opponent_mode=opponents,
        )
        completed = finished.game_state
        state = GameState.from_hands(
            hands=completed.initial_hands, contract=completed.contract,
            starting_player=(game_index // 4) % 4,
        )
        plays = [play for trick in completed.completed_tricks for play in trick.plays]
        for turn, play in enumerate(plays):
            if turn in turns:
                player = state.current_player
                assert player is not None
                # Export only the observation to the sampler; truth remains in scorer.
                yield game_index, turn, state.observation_for(player), tuple(
                    frozenset(p.cards) for p in state.players
                )
            state.play_card(play.player, play.card)


def score_distribution(observation, truth, deals):
    """Proper marginal scores for every unseen (card, holder) probability."""
    excluded = set(observation.hand) | set(observation.cards_played)
    unseen = [card for card in Deck().cards if card not in excluded]
    opponents = [p for p in range(4) if p != observation.player_index]
    squared = 0.0
    logloss = 0.0
    true_mass = 0.0
    n = len(unseen)
    if n == 0:
        raise ValueError("No hidden cards remain")
    for card in unseen:
        actual = next(p for p in opponents if card in truth[p])
        counts = {p: 0 for p in opponents}
        for deal in deals:
            holder = next(p for p in opponents if card in deal.initial_hands[p])
            counts[holder] += 1
        for p in opponents:
            probability = counts[p] / len(deals)
            target = float(p == actual)
            squared += (probability - target) ** 2
        observed = counts[actual] / len(deals)
        true_mass += observed
        # Laplace smoothing prevents infinite log loss from a finite sample.
        smooth = (counts[actual] + 1) / (len(deals) + len(opponents))
        logloss -= math.log(smooth)
    return {
        "cards": n,
        "brier": squared / (n * len(opponents)),
        "logloss": logloss / n,
        "true_holder_probability": true_mass / n,
    }



def budget_matched_deals(observation, *, samples, candidates, agreement, seed, max_attempts):
    """Draw one common proposal pool to isolate action evidence from RNG luck.

    Unweighted and weighted methods each retain one deal per block of
    `candidates`. The full-pool baseline consumes all proposals, matching
    the weighted method's proposal-generation budget (but not scoring cost).
    """
    evaluator = MonteCarloActionEvaluator(
        samples_per_action=samples, seed=seed,
        evidence_candidates=1, max_sampling_attempts=max_attempts,
    )
    t0 = time.perf_counter()
    pool = evaluator.sampler.sample_many(observation, samples * candidates)
    proposal_time = time.perf_counter() - t0
    blocks = [pool[i:i + candidates] for i in range(0, len(pool), candidates)]
    control = tuple(block[0] for block in blocks)
    t1 = time.perf_counter()
    weighted = []
    for block in blocks:
        scores = [
            observed_action_log_likelihood(
                observation, deal, agreement_probability=agreement,
            )
            for deal in block
        ]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        weighted.append(evaluator.sampler.rng.choices(block, weights=weights, k=1)[0])
    scoring_time = time.perf_counter() - t1
    return {
        "unweighted_32": (control, proposal_time),
        "weighted_32": (tuple(weighted), proposal_time + scoring_time),
        "unweighted_full_pool": (pool, proposal_time),
    }


def evaluate_budget_matched(args):
    if args.games < 1 or args.samples < 1 or args.candidates < 2:
        raise ValueError("Require games >=1, samples >=1, candidates >=2")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("game", "turn", "contract", "opponents", "method", "samples",
              "proposals", "hidden_cards", "brier", "logloss",
              "true_holder_probability", "seconds")
    rows = []
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for game_index, turn, observation, truth in snapshots(
            games=args.games, seed=args.seed, opponents=args.opponents,
            turns=tuple(args.turns),
        ):
            pools = budget_matched_deals(
                observation, samples=args.samples, candidates=args.candidates,
                agreement=args.agreement,
                seed=args.seed + 1_000_003 * game_index + 1_009 * turn,
                max_attempts=args.max_attempts,
            )
            for method, (deals, elapsed) in pools.items():
                measures = score_distribution(observation, truth, deals)
                row = {
                    "game": game_index, "turn": turn,
                    "contract": observation.contract.game_type.value,
                    "opponents": args.opponents, "method": method,
                    "samples": len(deals),
                    "proposals": args.samples * args.candidates,
                    "hidden_cards": measures["cards"],
                    "brier": measures["brier"],
                    "logloss": measures["logloss"],
                    "true_holder_probability": measures["true_holder_probability"],
                    "seconds": elapsed,
                }
                rows.append(row)
                writer.writerow(row)
                output.flush()
            if turn == args.turns[-1] and (game_index + 1) % args.progress_every == 0:
                print(f"{game_index+1}/{args.games} games", flush=True)

    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["game"], row["turn"])][row["method"]] = row
    report = {
        "protocol": "common_pool_v1", "seed": args.seed,
        "games": args.games, "samples": args.samples,
        "candidates": args.candidates, "agreement": args.agreement,
        "opponents": args.opponents, "turns": args.turns,
        "results": {},
        "paired_brier_improvement": {},
    }
    for method in BUDGET_METHODS:
        subset = [r for r in rows if r["method"] == method]
        report["results"][method] = {
            field: statistics.fmean(float(r[field]) for r in subset)
            for field in ("brier", "logloss", "true_holder_probability", "seconds")
        }
    for baseline in ("unweighted_32", "unweighted_full_pool"):
        differences = [
            float(group[baseline]["brier"]) - float(group["weighted_32"]["brier"])
            for group in by_key.values()
        ]
        report["paired_brier_improvement"][baseline] = statistics.fmean(differences)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))

def evaluate(args):
    if args.games < 1 or args.samples < 1 or args.candidates < 2:
        raise ValueError("Require games >=1, samples >=1, candidates >=2")
    if len(set(args.turns)) != len(args.turns) or any(t < 0 or t >= 32 for t in args.turns):
        raise ValueError("turns must be unique integers from 0 to 31")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as out:
        columns = [
            "game", "turn", "contract", "opponents", "method", "samples",
            "candidates", "hidden_cards", "brier", "logloss",
            "true_holder_probability", "seconds",
        ]
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for game_index, turn, observation, truth in snapshots(
            games=args.games, seed=args.seed, opponents=args.opponents,
            turns=tuple(args.turns),
        ):
            for method in METHODS:
                # Same seed for each method at the same position. Weighted
                # method uses more proposals and is more expensive by design.
                rng_seed = args.seed + 1_000_003 * game_index + 1_009 * turn
                evaluator = MonteCarloActionEvaluator(
                    samples_per_action=args.samples,
                    seed=rng_seed,
                    evidence_candidates=args.candidates if method == "weighted" else 1,
                    evidence_agreement=args.agreement,
                    max_sampling_attempts=args.max_attempts,
                )
                tic = time.perf_counter()
                deals = evaluator._sample_deals(observation)
                elapsed = time.perf_counter() - tic
                metrics = score_distribution(observation, truth, deals)
                row = {
                    "game": game_index, "turn": turn,
                    "contract": observation.contract.game_type.value,
                    "opponents": args.opponents, "method": method,
                    "samples": args.samples,
                    "candidates": args.candidates if method == "weighted" else 1,
                    "hidden_cards": metrics["cards"],
                    "brier": metrics["brier"], "logloss": metrics["logloss"],
                    "true_holder_probability": metrics["true_holder_probability"],
                    "seconds": elapsed,
                }
                rows.append(row)
                writer.writerow(row)
                out.flush()
            if (game_index + 1) % args.progress_every == 0 and turn == args.turns[-1]:
                print(f"{game_index+1}/{args.games} games ({time.perf_counter()-start:.1f}s)", flush=True)
    report = {"seed": args.seed, "games": args.games, "samples": args.samples,
              "candidates": args.candidates, "agreement": args.agreement,
              "opponents": args.opponents, "turns": args.turns, "results": {}}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        report["results"][method] = {
            key: statistics.fmean(float(r[key]) for r in selected)
            for key in ("brier", "logloss", "true_holder_probability", "seconds")
        }
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["game"], row["turn"])][row["method"]] = row
    report["paired_improvement"] = {
        key: statistics.fmean(
            float(item["unweighted"][key]) - float(item["weighted"][key])
            for item in paired.values()
        ) for key in ("brier", "logloss")
    }
    report["paired_improvement"]["true_holder_probability"] = statistics.fmean(
        float(item["weighted"]["true_holder_probability"]) -
        float(item["unweighted"]["true_holder_probability"])
        for item in paired.values()
    )
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261205)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--agreement", type=float, default=0.7)
    parser.add_argument("--turns", type=int, nargs="+", default=[4, 12, 20, 28])
    parser.add_argument("--opponents", choices=["heuristic", "random", "heuristic-bid-random-card"], default="heuristic")
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/sampler_quality.csv"))
    parser.add_argument("--budget-matched", action="store_true",
                        help="Compare methods using one shared proposal pool")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if args.budget_matched:
        evaluate_budget_matched(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
