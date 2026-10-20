# Rollout experiments

The experimental suite compares **HeuristicAgent vs RolloutCardPlayAgent** on
identical *starting deals* (same seed, seat rotation, starting player) against
both random opponents and heuristic opponents. Rollout candidates and the
baseline use the same heuristic bidding logic. It measures **per-card-decision
wall-clock latency** (including forced moves) and stores every paired payment
as CSV plus a JSON summary.

## Smoke test (small enough to troubleshoot)

```powershell
uv run pytest tests/test_action_evidence.py tests/test_rollout_card_play_agent.py
uv run python scripts/rollout_ablation.py --games 2 --samples 8 --profiles heuristic --evidence-candidates 1 --output results/rollout-smoke.csv
```

## Full ablation

```powershell
uv run python scripts/rollout_ablation.py --games 100 --samples 8 32 128 --profiles heuristic mixed --opponents random heuristic --evidence-candidates 1 4 --output results/rollout-ablation.csv
```

This is **24 configurations**, 100 paired deals each (4,800 full games,
plus many rollout simulations). Start with a smoke test. The 128-sample
conditions and four-candidate evidence sampler can be *very slow*.

- `heuristic`: all simulated continuations use heuristic card play.
- `mixed`: half heuristic, a quarter epsilon-heuristic and a quarter random.
- `evidence-candidates 1`: existing constraint-based sampler (control).
- `evidence-candidates 4`: draw 4 legal hands and probabilistically choose
  one based on how well heuristic-plus-random behavior explains prior opponent
  card choices. `--evidence-agreement 0.7` sets the assumed probability of
  heuristic-following behavior. This is **soft evidence**, not a proof that an
  opponent held a card; all legal alternatives retain nonzero likelihood.

**Caveats:** selection among several proposed legal deals is not exact
posterior sampling; the original backtracking sampler is not necessarily
uniform either. Different choices lead to different later game trajectories.
The observed opponents may not actually follow our heuristic, so weighting
can worsen results. Avoid making final claims from a repeatedly tuned fixed
test seed: use a fresh, withheld seed for final confirmation.

Output files:
- `results/rollout-ablation.csv`: every paired deal, payments, number of card
  decisions and total time spent in those decisions.
- `results/rollout-ablation.summary.json`: configuration-level paired
  mean payment deltas, 95% normal-approximation confidence intervals, rollout
  win rate and mean/p95 decision time.

No measurements have been recorded in the repository; run the scripts locally.
