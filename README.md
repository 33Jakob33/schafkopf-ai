# Schafkopf AI

[![CI](https://github.com/33Jakob33/schafkopf-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/33Jakob33/schafkopf-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-orange)

A typed Python implementation of **Bavarian Schafkopf** combined with an experimental game-AI stack.

The project started as a rules-correct simulation engine and evolved into a small research environment for comparing handcrafted strategies, imitation learning, DAgger, and reinforcement learning with PPO. The current learning focus is **card play**; bidding is intentionally kept heuristic-controlled so card-play improvements can be measured independently.

## Why this project?

Schafkopf is a useful AI environment because it combines:

- partial observability,
- changing team structure,
- contract-dependent card rankings,
- sequential bidding,
- long-horizon card-play decisions,
- sparse final rewards,
- and imperfect information about opponents.

That makes it substantially more interesting than a perfect-information toy game while still being compact enough to simulate and test locally.

The repository demonstrates work across **software engineering, game modelling, ML pipelines, PyTorch, imitation learning, reinforcement learning, reproducible evaluation, and automated testing**.

## Highlights

- Complete 32-card Schafkopf engine with deterministic seeded simulation.
- Supported contracts: **Sauspiel, Solo, Wenz, Geier, and Ramsch**.
- Sequential bidding with configurable enabled games and all-pass behaviour.
- Rule-aware trump ordering, follow-suit logic, Sauspiel called-Ace handling, and Davonlaufen.
- Scoring and settlement including Schneider, Schwarz, Laufende, and Ramsch/Jungfrau handling.
- Pluggable agent interface with Random, heuristic, neural imitation, and PPO agents.
- PyTorch card-play policy with legal-action masking.
- Behavioral cloning from heuristic demonstrations.
- DAgger dataset aggregation on learner-visited states.
- Actor-critic PPO initialized from the imitation-learning policy.
- Fixed paired evaluation with confidence intervals.
- Validation-based PPO checkpoint selection and frozen snapshot-opponent training.
- Type checking, linting, formatting checks, and automated tests.

## Current results

The main benchmark uses **10,000 paired deals** with seed 42. The focal learned agent and the heuristic receive the same deal, starting position, bidding setup, and RandomAgent seeds. Bidding is controlled by the same heuristic policy; only card play differs.

| Card-play policy | Mean payment | Gap vs. heuristic | Win rate |
| --- | ---: | ---: | ---: |
| HeuristicAgent | +37.307 | 0.000 | 84.63% |
| Behavioral cloning | +32.171 | -5.136 | 78.70% |
| DAgger | +32.522 | -4.785 | 79.23% |
| PPO, 100k training games | **+33.016** | **-4.291** | 79.85% |
| PPO + snapshot self-play | +32.865 | -4.442 | 79.65% |

The experiments show a clear improvement from pure imitation to DAgger and then PPO. The first snapshot self-play experiment did **not** improve the independent test benchmark, despite improving its fixed validation score. This is being treated as a useful negative result rather than hidden: the next planned experiment focuses on a richer observation representation and stronger validation methodology.

## Learning pipeline

```text
HeuristicAgent
      |
      | expert demonstrations
      v
Behavioral Cloning
      |
      | learner-visited states + expert labels
      v
DAgger
      |
      | initialize actor
      v
Actor-Critic PPO
      |
      | terminal settlement reward
      v
Validation / paired evaluation
      |
      +--> frozen PPO snapshots for opponent-pool experiments
```

### Behavioral cloning

The heuristic generates state/action demonstrations. A feed-forward PyTorch model maps the encoded player observation to 32 card logits:

```text
1,254 observation features
        |
       512
        |
       256
        |
   32 card logits
```

Illegal cards are masked before the loss or action selection is computed.

### DAgger

The neural policy plays the game, but the heuristic labels the states that the learner actually visits. These examples are aggregated with the original behavioral-cloning dataset and used to refine the policy.

This reduces the distribution shift between expert-generated states and states caused by the learner's own mistakes.

### PPO

The PPO model reuses the DAgger policy weights and adds a value head:

```text
                 +--> policy head --> 32 card logits
observation --> shared MLP
                 +--> value head  --> expected return
```

Training uses the player's final settlement as the reward. The settlement is scaled numerically during optimization but the reported evaluation metric remains the original game payment.

The current trainer also supports:

- warm-starting from an existing PPO checkpoint,
- deterministic fixed validation,
- validation-based best-checkpoint selection,
- RandomAgent / HeuristicAgent / frozen-PPO opponent mixtures,
- and a rolling snapshot opponent pool.

## Game engine

The engine is independent of the ML implementation. It handles legal play and game state; agents only choose among legal actions.

Key invariants include:

- 32 unique cards per round,
- 8 tricks,
- 120 Augen in total,
- exactly 8 cards played by each player,
- legal move enforcement,
- and zero-sum settlement.

The separation is intentional:

```text
Game rules and legality
        |
        v
PlayerObservation + legal actions
        |
        v
Agent
        |
        v
chosen legal action
```

A learned model therefore cannot bypass Schafkopf rules by producing an illegal move.

## Project structure

```text
schafkopf-ai/
├── .github/
│   └── workflows/
│       └── ci.yml
├── scripts/
│   ├── simulate_rounds.py
│   ├── evaluate_heuristic.py
│   ├── evaluate_heuristic_ab.py
│   ├── generate_bc_dataset.py
│   ├── train_behavior_clone.py
│   ├── generate_dagger_dataset.py
│   ├── evaluate_bc_agent.py
│   ├── train_ppo.py
│   └── evaluate_ppo_agent.py
├── src/
│   └── schafkopf_ai/
│       ├── agents/
│       ├── game/
│       ├── runner/
│       └── training/
├── tests/
├── pyproject.toml
└── uv.lock
```

### `game/`

Core domain model and rules:

- cards and deck,
- game contracts,
- bidding,
- trump hierarchy,
- tricks,
- legal moves,
- observations,
- scoring,
- game results,
- and settlement.

### `runner/`

Orchestrates bidding, card play, complete rounds, seeded deals, and agent interaction.

### `agents/`

Contains the common agent interface plus:

- `RandomAgent`,
- `HeuristicAgent`,
- experimental strategic heuristic variants,
- `NeuralCardPlayAgent`,
- and `PPOCardPlayAgent`.

### `training/`

Reusable ML components for:

- observation encoding,
- behavioral-cloning datasets,
- demonstration collection,
- DAgger,
- PPO rollouts,
- actor-critic models,
- checkpoint loading/saving,
- and policy updates.

## Setup

The project uses **Python 3.12** and [uv](https://docs.astral.sh/uv/) for environment and dependency management.

Clone the repository and install the locked environment:

```powershell
git clone https://github.com/33Jakob33/schafkopf-ai.git
cd schafkopf-ai
uv sync
```

Run the package entry point:

```powershell
uv run schafkopf-ai
```

## Run a simulation

```powershell
uv run python scripts/simulate_rounds.py
```

Evaluate the heuristic baseline:

```powershell
uv run python scripts/evaluate_heuristic.py --games 10000
```

## Train the imitation-learning policy

Generate heuristic demonstrations:

```powershell
uv run python scripts/generate_bc_dataset.py --games 2000
```

Train the behavioral clone:

```powershell
uv run python scripts/train_behavior_clone.py --epochs 10
```

Evaluate it on paired deals:

```powershell
uv run python scripts/evaluate_bc_agent.py --games 10000
```

## Run DAgger

Collect learner-visited states and aggregate expert labels:

```powershell
uv run python scripts/generate_dagger_dataset.py --games 2000
```

Warm-start training from the behavioral-cloning checkpoint:

```powershell
uv run python scripts/train_behavior_clone.py `
  --dataset data/behavior_cloning/dagger_card_play.npz `
  --init-checkpoint models/behavior_cloning/card_play_mlp.pt `
  --output models/behavior_cloning/card_play_dagger.pt `
  --epochs 5 `
  --learning-rate 0.0003
```

## Train PPO

PPO can be initialized from either a BC/DAgger checkpoint or an existing PPO checkpoint.

Example:

```powershell
uv run python scripts/train_ppo.py `
  --init-checkpoint models/behavior_cloning/card_play_dagger.pt `
  --iterations 30 `
  --rollout-games 2000 `
  --learning-rate 0.0001
```

The default trainer includes fixed validation and a mixed opponent pool.

Evaluate a PPO checkpoint:

```powershell
uv run python scripts/evaluate_ppo_agent.py `
  --checkpoint models/ppo/card_play_ppo_best.pt `
  --games 10000
```

Generated datasets and model checkpoints are intentionally excluded from Git.

## Development checks

Run the same checks used by CI:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

## Observation representation

The current neural card-play observation has 1,254 features. It contains:

- the player's hand,
- public trick history,
- game type,
- trump/called suit,
- declarer,
- relative player information,
- collected points,
- called-Ace release state,
- and trick number.

A current limitation is that most features belong to a sparse raw trick-history representation. A planned **Observation V2** will add explicit public strategic information such as current trick strength, known voids, trump counts, role/team context, and bidding history while avoiding access to hidden opponent cards.

## Reproducibility

Evaluation scripts use deterministic seeds and paired comparisons where possible. For neural-vs-heuristic comparisons, both policies are evaluated on matching deals and opponent seeds, and the scripts report a 95% confidence interval for the paired payment difference.

Model checkpoints and generated training datasets are local artifacts and are excluded through `.gitignore`.

## Current limitations and roadmap

The project is intentionally still evolving.

Current limitations:

- neural learning is limited to card play; bidding remains heuristic,
- the observation encoding is still a first-generation flat representation,
- snapshot self-play has not yet improved the independent benchmark,
- trained model files are not committed to the repository,
- and there is currently no graphical user interface.

Next experiments:

1. richer Observation V2 with explicit public game-state features,
2. multi-seed validation for more reliable checkpoint selection,
3. re-train BC / DAgger / PPO on the new representation,
4. revisit opponent-pool and self-play strategies,
5. later investigate learned bidding separately.

## Tech stack

- Python 3.12
- PyTorch
- NumPy
- uv
- pytest
- Ruff
- mypy

---
