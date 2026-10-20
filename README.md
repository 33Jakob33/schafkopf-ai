# Schafkopf AI

[![CI](https://github.com/33Jakob33/schafkopf-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/33Jakob33/schafkopf-ai/actions/workflows/ci.yml)

A Python hobby/research project for playing Bavarian Schafkopf and experimenting with game-playing AI. It combines a deterministic, rule-based game engine with heuristic search, imitation learning and reinforcement learning. Learning experiments currently focus on **card play**; bidding is generally delegated to configurable heuristics.

> **Status:** Experimental. Models and data sets are generated locally and are not included in Git. A training script or agent implementation does not imply that the model has been trained or outperforms the heuristic baseline.

## Game engine

- Complete 32-card deck and four-player rounds with eight tricks and 120 Augen.
- Sauspiel, Solo, Wenz, Geier and Ramsch.
- Bidding and contract selection, including the all-pass Ramsch/redeal alternatives.
- Contract-specific trump ordering, legal-move enforcement, Sauspiel called-Ace obligations and Davonlaufen.
- Results, Schneider/Schwarz, Laufende, settlement and zero-sum payments.
- Reproducible simulations with seeded deals and opponent behavior.
- Player observations designed to respect hidden information.

## Agents and experiments

| Agent / approach | Implementation | Purpose |
| --- | --- | --- |
| Random | `RandomAgent` | Baseline; random bidding/card play |
| Heuristic | `HeuristicAgent` | Contract-aware rule-based card play and bidding |
| Strategic heuristic | `StrategicHeuristicAgent` | Experimental role-dependent tactical decisions |
| Heuristic-bid/random-card | `HeuristicBiddingRandomCardAgent` | Isolate card-play performance while keeping bidding meaningful |
| Behavioral cloning (BC) | `NeuralCardPlayAgent` | Imitate heuristic demonstrations with a learned policy |
| DAgger | `NeuralCardPlayAgent` with DAgger checkpoint | Retrain on states visited by the learned policy |
| PPO | `PPOCardPlayAgent` | Actor-critic reinforcement learning |
| Specialist BC | `SpecialistCardPlayAgent` | Separate card-play models for role-specific decisions |
| Multi-head BC | `MultiHeadCardPlayAgent` | Shared representation with role-specific action heads |
| Action-value / Q | `QCardPlayAgent` | Predict rollout-derived values for legal card actions |
| Rollout search | `RolloutCardPlayAgent` | Sample hidden hands, simulate candidate plays and select an action |

These methods are experimental, not a performance ranking. Learned agents load checkpoints from disk. Rollout search samples plausible hidden states and depends on its continuation-policy assumptions and computation budget.

## Repository layout

```text
.github/workflows/ci.yml
scripts/                       Simulation, data generation, training, evaluation
src/schafkopf_ai/
    agents/                     Baselines, neural policies, Q, specialists, rollout
    game/                       Rules, contracts, cards, tricks, bidding, scoring
    runner/                     Bidding, games and full-round execution
    evaluation/                 Shared paired-agent evaluation
    search/                     Hidden-state sampling and rollout continuations
    training/                   Observation encoders and learning algorithms
tests/                         Rule, agent, training and evaluation tests
pyproject.toml
uv.lock
```

## Installation

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/33Jakob33/schafkopf-ai.git
cd schafkopf-ai
uv sync
uv run schafkopf-ai
```

The project uses PyTorch and NumPy. Depending on the machine, model training and inference may run on CPU or an available supported accelerator.

## Simulations and heuristic baselines

```powershell
uv run python scripts/simulate_rounds.py
uv run python scripts/evaluate_heuristic.py --games 10000
uv run python scripts/evaluate_heuristic_ab.py --games 10000
```

The heuristic A/B evaluator compares heuristic variants using paired games.

## Evaluate agents against one another

The **shared paired evaluator already present on `main`** compares reference and candidate agents on seeded initial deals with rotating focal seats. It reports win rates, payment differences, approximate 95% confidence intervals and breakdowns by game contract and role.

```powershell
uv run python scripts/evaluate_agents.py --reference-kind heuristic --candidate-kind rollout --games 1000 --seed 123 --opponent-mode heuristic-bid-random-card
```

Compare a trained PPO checkpoint:

```powershell
uv run python scripts/evaluate_agents.py --reference-kind heuristic --candidate-kind ppo --candidate-checkpoint models/ppo/card_play_ppo_best.pt --games 10000 --seed 123
```

Supported kinds: `heuristic`, `bc`, `dagger`, `neural`, `multihead`, `specialist`, `ppo`, `q` and `rollout`. Use `--reference-checkpoint` or `--candidate-checkpoint` as required for learned agents. Specialist evaluation expects a specialist manifest.

Opponent modes:

- `heuristic-bid-random-card` (default): heuristic bidding, random card play.
- `heuristic`: heuristic opponents for both bidding and card play.
- `random`: random opponents, including bidding.

`--bidding-profile` chooses the bidding-threshold preset; rollout comparisons can additionally configure rollout samples, continuation profile, epsilon and sampling seeds. See `uv run python scripts/evaluate_agents.py --help`.

**Interpretation:** Paired games share initial deals, not necessarily identical later game states. Results depend on opponent strength, contract distributions, role mix and rollout budget. Hold back independent test seeds instead of optimizing repeatedly on the final evaluation sample.

> The separate, frozen-manifest multi-agent benchmark currently being developed on `feature/unified-agent-benchmark` is **not part of `main`**. This README only documents features already on `main`.

## Imitation learning: BC and DAgger

Generate heuristic demonstrations:

```powershell
uv run python scripts/generate_bc_dataset.py --games 2000 --observation-version v2 --output data/behavior_cloning/heuristic_card_play_v2.npz
```

Train a neural card-play policy:

```powershell
uv run python scripts/train_behavior_clone.py --dataset data/behavior_cloning/heuristic_card_play_v2.npz --output models/behavior_cloning/card_play_mlp.pt --epochs 10
```

Then evaluate it:

```powershell
uv run python scripts/evaluate_agents.py --reference-kind heuristic --candidate-kind bc --candidate-checkpoint models/behavior_cloning/card_play_mlp.pt --games 10000
```

For **DAgger**, collect states reached by the learner and add heuristic action labels, then continue BC training on the aggregated dataset:

```powershell
uv run python scripts/generate_dagger_dataset.py --games 2000
uv run python scripts/train_behavior_clone.py --dataset data/behavior_cloning/dagger_card_play.npz --init-checkpoint models/behavior_cloning/card_play_mlp.pt --output models/behavior_cloning/card_play_dagger.pt --epochs 5 --learning-rate 0.0003
```

The data generator also supports other observation versions and configurable heuristic bidding/opponent setups; inspect `--help` before larger runs. A BC/DAgger checkpoint must match the input encoding and model architecture it was trained with.

### Role-aware imitation learning

Two additional training approaches are available:

- `scripts/train_specialists.py`: trains separate policies for defined card-play roles and writes a manifest consumed by `SpecialistCardPlayAgent`.
- `scripts/train_multihead_behavior_clone.py`: trains a shared model with specialized action heads, consumed by `MultiHeadCardPlayAgent`.

Example (using a previously generated compatible dataset):

```powershell
uv run python scripts/train_specialists.py --dataset data/behavior_cloning/heuristic_card_play_v2.npz
uv run python scripts/train_multihead_behavior_clone.py --dataset data/behavior_cloning/heuristic_card_play_v2.npz
```

## Reinforcement learning: PPO

PPO uses an actor-critic card-play policy, legal-action masking, rollout collection, policy updates and periodic validation. It can initialize from a compatible BC/DAgger checkpoint, or resume from a compatible PPO checkpoint. Its training loop supports a mixture of heuristic, random and frozen policy-snapshot opponents.

```powershell
uv run python scripts/train_ppo.py --init-checkpoint models/behavior_cloning/card_play_dagger.pt --iterations 30 --rollout-games 2000 --learning-rate 0.0001
```

```powershell
uv run python scripts/evaluate_ppo_agent.py --checkpoint models/ppo/card_play_ppo_best.pt --games 10000
```

## Action-value learning and rollout search

The `search/` package provides **hidden-state sampling** and rollout continuations for imperfect-information decision making. `RolloutCardPlayAgent` evaluates legal moves using sampled possible hidden hands and simulated continuations.

The Q-learning experiment takes a different route: generate rollout-based action-value targets, train a neural value predictor and then choose legal actions using the learned scores.

```powershell
uv run python scripts/generate_action_value_dataset.py --states 20000
uv run python scripts/train_action_value.py --dataset data/action_values/action_values_v2_mixed_20k.npz
uv run python scripts/evaluate_agents.py --reference-kind heuristic --candidate-kind q --candidate-checkpoint models/action_values/card_play_q_v2.pt --games 10000
```

These are experimental estimates rather than ground-truth optimal card values. Sampling and continuation-policy quality matter.

## Observation encoding

Card-play encodings are explicitly versioned so older experiments and checkpoints can coexist.

- **V1:** original 1,254-feature encoding with the hand and slot-by-slot public trick history.
- **V2:** compact 222-feature encoding summarizing current trick, points, unseen trumps/suit cards, publicly inferred voids and contract/role context.
- **V3:** newer observation format used by further model experiments, with version-aware checkpoint loading and a dedicated recurrent-policy path. See `training/observation_encoding.py`, `training/card_play_encoding_v3.py` and the V3 tests for exact feature definitions and compatibility requirements.

Policies select from a fixed set of 32 physical cards, with illegal actions masked. Encoders should use only the current player's hand and information observable or inferable from public play; never feed hidden opponent hands to the policy.

## Tests and code quality

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

GitHub Actions runs the project checks. The suite covers core card/game invariants, bidding and scoring, learned-policy components, hidden-state sampling, rollout search, action-value learning and paired evaluation.

## Reproducibility and limitations

- Use explicit random seeds and keep training/validation/test games separate.
- Compare multiple opponent modes and enough games for meaningful uncertainty estimates; a favorable win rate against random opponents is not proof of strong play.
- Keep model, encoding, bidding profile and rollout configurations alongside evaluation results.
- Generated datasets and model checkpoints are ignored by Git and must be produced locally.
- This is an evolving hobby project, not a production-ready or verified optimal Schafkopf AI.
