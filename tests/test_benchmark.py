"""Tests for frozen benchmark cases and evaluation invariants."""
from pathlib import Path

import pytest

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.evaluation.benchmark import generate_cases, load_manifest, save_manifest


def test_manifest_is_reproducible(tmp_path: Path) -> None:
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    assert save_manifest(one, count=8, seed=42) == save_manifest(two, count=8, seed=42)
    cases, digest = load_manifest(one)
    assert len(cases) == 8
    assert len(digest) == 64
    assert {c.focal_seat for c in cases} == {0, 1, 2, 3}
    with pytest.raises(FileExistsError):
        save_manifest(one, count=8, seed=42)


def test_case_replay_is_deterministic() -> None:
    from schafkopf_ai.evaluation.benchmark_runner import play_case

    case = generate_cases(count=1, seed=7)[0]
    first = play_case(case, HeuristicAgent(), "digest", "heuristic")
    second = play_case(case, HeuristicAgent(), "digest", "heuristic")
    assert first == second
    assert first["case_id"] == 0
