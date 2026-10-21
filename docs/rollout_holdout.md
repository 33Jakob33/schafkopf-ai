# Independent rollout holdout

This is a **confirmatory** experiment with a seed different from the earlier
exploratory 100-game ablation (seed 20261020). Do not use these results to tune
the three selected policies, otherwise this run ceases to be independent.

Three frozen candidates from exploratory results:

| ID | Samples | Continuation | Evidence candidates |
| --- | ---: | --- | ---: |
| fast-h8-e1 | 8 | heuristic | 1 |
| balanced-m32-e4 | 32 | mixed | 4 |
| best-m128-e4 | 128 | mixed | 4 |

All play the same 1,000 **new initial deals**, separately against three
HeuristicAgent opponents, with heuristic bidding. Comparisons use paired
payout differences, not individual win percentages alone.

```powershell
git switch feature/unified-agent-benchmark
git pull origin feature/unified-agent-benchmark
uv run pytest
uv run python scripts/evaluate_rollout_holdout.py --games 1000 --seed 20261119 --output results/holdout_20261119.csv
```

If interrupted, **run exactly the same command again**. Already completed
(case, variant) rows are skipped. Protocol arguments must match the JSON
sidecar; use a new output path for a new seed, test size, or subset of variants.

Three files are written:
- `holdout_20261119.csv`: individual paired case results, flushed after
  each candidate game.
- `holdout_20261119.protocol.json`: frozen seed, selected variants and
  protocol hash.
- `holdout_20261119.summary.json`: only created once the run is complete,
  with paired 95% confidence intervals and measured card-decision times.

The full run can be slow, especially with 128 samples and behavioral weighting.
For a separate smoke test use `--games 2 --seed 20261201 --output
results/holdout_smoke.csv`. Never recycle a smoke/test seed into the final
confirmation protocol.

Cautions: the current protocol varies the initial deals but keeps the same
agent and opponent implementations and a fixed rule set. A paired confidence
interval for a selected policy is informative, but the best-of-three selection
still deserves a cautious interpretation; measure performance over roles and
contracts in subsequent exploratory analyses. This protocol measures time
inside `choose_card`, including forced decisions, not total wall-clock time.
