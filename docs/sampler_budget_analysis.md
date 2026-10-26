# Sampler quality: results by phase and contract

The initial 100-game CSVs on this branch contain 400 positions per opponent
mode, each evaluated by 32 draws using one candidate (unweighted) or four
candidate proposals (weighted). **Brier improvement = unweighted minus weighted**
(positive is better); these are descriptive values, not independent confidence
intervals. Results are from the tracked CSVs.

## By public play phase

| Cards played | Heuristic opponents | Random opponents |
| ---: | ---: | ---: |
| 4 | +0.0069 | -0.0005 |
| 12 | +0.0195 | +0.0047 |
| 20 | +0.0325 | +0.0099 |
| 28 | +0.1062 | +0.0198 |

The weighted advantage is concentrated in later-game observations, especially
against heuristic opponents. At the end of the game, the inference task also
contains fewer hidden cards; compare phases separately.

## By contract

| Contract | Heuristic opponents | Random opponents |
| --- | ---: | ---: |
| Sauspiel | +0.0408 | +0.0156 |
| Wenz | +0.0446 | +0.0007 |
| Solo | +0.0315 | +0.0091 |
| Ramsch | not observed | +0.0060 |

Different contract frequencies and varying sample sizes mean that these are
not controlled comparisons between contract types.

## Common-pool budget experiment

The previous test spent **32** proposal draws on unweighted sampling but
**128** on weighted sampling. The new `--budget-matched` mode generates
a single shared pool of 128 valid determinizations and compares:

- `unweighted_32`: first deal from each consecutive 4-deal block
- `weighted_32`: one deal sampled from each block using soft evidence weights
- `unweighted_full_pool`: all 128 unweighted deals (same number of generated
  proposals as weighted, but 4x as many scored outputs)

All three see **exactly the same pool of legal proposals** on each observation.
Thus the first two isolate the value of action-evidence selection given the
same proposals and same output count; the third checks whether simply keeping
all proposals yields better estimates. Weighted still pays extra CPU for
likelihood calculation, so this is **proposal-budget matched, not compute-time
matched**.

```powershell
git switch feature/unified-agent-benchmark
git pull origin feature/unified-agent-benchmark
uv run pytest tests/test_sampler_budget.py tests/test_sampler_quality.py
uv run python scripts/evaluate_sampler_quality.py --budget-matched --games 3 --samples 8 --candidates 4 --opponents heuristic --seed 20270103 --output results/sampler_budget_smoke.csv
uv run python scripts/evaluate_sampler_quality.py --budget-matched --games 100 --samples 32 --candidates 4 --opponents heuristic --seed 20270104 --output results/sampler_budget_heuristic.csv
uv run python scripts/evaluate_sampler_quality.py --budget-matched --games 100 --samples 32 --candidates 4 --opponents random --seed 20270105 --output results/sampler_budget_random.csv
```

Note that the new runs use new seeds, not the earlier exploratory data. The
result CSV contains a row per method and position; the accompanying
`*.summary.json` includes paired mean Brier improvements.

**Further design cautions:** candidate backtracking is not a uniform proposal
distribution. The evidence score is an imperfect behavioral model. Repeated
snapshots in a game are correlated. Estimate uncertainty with game-level
resampling in subsequent analysis, and reserve further unseen seeds for
confirmatory testing. Actual hidden hands are only accessed by the scorer.
