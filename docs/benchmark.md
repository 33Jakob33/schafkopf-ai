# Unified agent benchmark (v1)

Use **one frozen manifest** for all agents. It stores deal and opponent random
seeds, starting seat and focal seat, not complete game trajectories. Each agent
gets the same initial deal, bidder and seeded RandomAgent opponents.

```powershell
uv run python scripts/benchmark_agents.py generate --manifest data/benchmarks/v1.json --games 10000 --seed 20261020
uv run python scripts/benchmark_agents.py evaluate --manifest data/benchmarks/v1.json --agent heuristic --agent-id heuristic --output results/heuristic.csv
uv run python scripts/benchmark_agents.py evaluate --manifest data/benchmarks/v1.json --agent bc --agent-id bc-v1 --checkpoint models/behavior_cloning/card_play_mlp.pt --output results/bc-v1.csv
uv run python scripts/benchmark_agents.py evaluate --manifest data/benchmarks/v1.json --agent dagger --agent-id dagger-v1 --checkpoint models/behavior_cloning/card_play_dagger.pt --output results/dagger-v1.csv
uv run python scripts/benchmark_agents.py evaluate --manifest data/benchmarks/v1.json --agent ppo --agent-id ppo-v1 --checkpoint models/ppo/card_play_ppo_best.pt --output results/ppo-v1.csv
uv run python scripts/benchmark_agents.py compare --manifest data/benchmarks/v1.json --results results/heuristic.csv results/bc-v1.csv results/dagger-v1.csv results/ppo-v1.csv
```

For a custom agent such as a rollout agent (not currently committed), create an
`Agent` subclass with a no-argument constructor and pass
`--agent custom --factory package.module:RolloutAgent --agent-id rollout-v1`.
No custom rollout implementation is included in this branch.

The result CSV contains one row per case and a SHA256 of the manifest. A
sidecar metadata JSON records agent identity, checkpoint path and run time.
The compare command rejects incomplete results or files generated from a
different manifest and reports mean payment, win rates, and paired confidence
intervals against the heuristic.

## Important limitations

- **These are card-play comparisons, not full-agent comparisons.** The focal
  player's bidding is always delegated to the heuristic. All other players are
  seeded RandomAgents.
- **Fixed RNG seeds are not identical gameplay trajectories.** Decisions change
  subsequent states; paired comparisons are paired *initial deals*.
- **Random-only opponents can favor exploitable policies.** A future benchmark
  version should add fixed HeuristicAgent opponents and possibly other tiers,
  with separate results per opponent pool.
- **No train/test leakage.** Generate the final manifest once, before tuning.
  Keep its seeds and resulting games out of all training, DAgger and PPO
  validation loops. Do not repeatedly optimize against these benchmark scores.
- **Version the rules and engine.** The digest validates the manifest, not
  code or model versions. If game mechanics, baseline bidding or opponents
  change, create a new benchmark version rather than comparing scores directly.
- **Statistical precision.** 95% confidence intervals use a normal
  approximation; repeated games and adaptive model selection can invalidate
  naive interpretations. Compare paired differences, and confirm improvements
  against a fresh hold-out before publishing claims.

The manifest is typically small enough to version in Git, whereas per-agent
CSV results and checkpoints can be kept outside the repository.
