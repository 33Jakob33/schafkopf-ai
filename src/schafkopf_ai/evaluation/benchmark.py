"""Reproducible, versioned evaluation cases shared by all card-play agents.

A case stores RNG seeds rather than hidden cards. RoundRunner deterministically
recreates the deal. The manifest must be frozen before model evaluation.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

BENCHMARK_VERSION = 1


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: int
    deal_seed: int
    starting_player: int
    focal_seat: int
    opponent_seeds: tuple[int, int, int, int]


def generate_cases(*, count: int, seed: int) -> list[BenchmarkCase]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    result: list[BenchmarkCase] = []
    for index in range(count):
        result.append(
            BenchmarkCase(
                case_id=index,
                deal_seed=rng.randrange(2**63),
                starting_player=(index // 4) % 4,
                focal_seat=index % 4,
                opponent_seeds=tuple(rng.randrange(2**63) for _ in range(4)),  # type: ignore[arg-type]
            )
        )
    return result


def _payload(cases: list[BenchmarkCase], *, seed: int) -> dict:
    return {
        "version": BENCHMARK_VERSION,
        "seed": seed,
        "rules": "RAMSCH_RULES",
        "bidding": "HeuristicAgent",
        "opponents": "RandomAgent",
        "cases": [asdict(case) for case in cases],
    }


def manifest_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_manifest(path: Path, *, count: int, seed: int) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen benchmark: {path}")
    payload = _payload(generate_cases(count=count, seed=seed), seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_hash(payload)


def load_manifest(path: Path) -> tuple[list[BenchmarkCase], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != BENCHMARK_VERSION:
        raise ValueError("Unsupported benchmark version")
    if (payload.get("rules"), payload.get("bidding"), payload.get("opponents")) != (
        "RAMSCH_RULES", "HeuristicAgent", "RandomAgent"
    ):
        raise ValueError("Unexpected benchmark protocol")
    cases = [
        BenchmarkCase(
            case_id=int(item["case_id"]),
            deal_seed=int(item["deal_seed"]),
            starting_player=int(item["starting_player"]),
            focal_seat=int(item["focal_seat"]),
            opponent_seeds=tuple(int(v) for v in item["opponent_seeds"]),  # type: ignore[arg-type]
        )
        for item in payload["cases"]
    ]
    if not cases or [case.case_id for case in cases] != list(range(len(cases))):
        raise ValueError("Invalid benchmark case IDs")
    if any(
        not 0 <= case.focal_seat < 4
        or not 0 <= case.starting_player < 4
        or len(case.opponent_seeds) != 4
        for case in cases
    ):
        raise ValueError("Invalid benchmark seats or seeds")
    return cases, manifest_hash(payload)
