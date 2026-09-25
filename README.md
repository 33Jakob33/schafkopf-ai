# Schafkopf AI

[![CI](https://github.com/33Jakob33/schafkopf-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/33Jakob33/schafkopf-ai/actions/workflows/ci.yml)

A hobby project for implementing Bavarian Schafkopf in Python and experimenting with different kinds of game-playing agents.

The repository contains a rules-based game engine, several baseline agents, and an experimental learning pipeline using PyTorch. The current learning work focuses on card play, while bidding is still handled by heuristic agents.

## Features

The game engine currently supports:

- the 32-card Bavarian Schafkopf deck,
- Sauspiel,
- Solo,
- Wenz,
- Geier,
- Ramsch,
- sequential bidding,
- contract-dependent trump rules,
- legal move enforcement,
- Sauspiel called-Ace rules and Davonlaufen,
- scoring with 120 Augen,
- Schneider and Schwarz,
- Laufende,
- Ramsch settlement,
- seeded simulation for reproducible games.

The agent side currently includes:

- `RandomAgent`,
- `HeuristicAgent`,
- an experimental `StrategicHeuristicAgent`,
- a behavior-cloned neural card-play agent,
- a DAgger-trained neural card-play agent,
- and a PPO actor-critic card-play agent.

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
├── README.md
├── pyproject.toml
└── uv.lock
```

### `game/`

Contains the core Schafkopf model and rules:

- cards and deck,
- game contracts,
- bidding,
- trump handling,
- tricks,
- legal moves,
- player observations,
- scoring,
- game results,
- settlement.

### `runner/`

Contains the logic that connects the game engine with agents and runs:

- bidding,
- card play,
- complete rounds.

### `agents/`

Contains the common agent interface and the different player implementations.

### `training/`

Contains reusable ML components for:

- card-play observation encoding,
- behavior-cloning datasets,
- demonstration collection,
- DAgger,
- PPO rollout storage,
- actor-critic models,
- checkpoint loading and saving,
- PPO updates.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency management.

```powershell
git clone https://github.com/33Jakob33/schafkopf-ai.git
cd schafkopf-ai
uv sync
```

You can run the package entry point with:

```powershell
uv run schafkopf-ai
```

## Running simulations

Run a basic simulation:

```powershell
uv run python scripts/simulate_rounds.py
```

Evaluate the heuristic agent against random opponents:

```powershell
uv run python scripts/evaluate_heuristic.py --games 10000
```

There is also an A/B evaluation script for comparing heuristic variants:

```powershell
uv run python scripts/evaluate_heuristic_ab.py --games 10000
```

## Neural card-play experiments

The neural agents currently use a fixed-size encoding of the player's observable game state and output one score for each of the 32 physical cards.

Illegal actions are masked before a card is selected.

The current training path is:

```text
Heuristic demonstrations
        ↓
Behavioral cloning
        ↓
DAgger
        ↓
PPO
```

Bidding remains heuristic-controlled during these experiments so that card play can be evaluated separately.

### Behavioral cloning

Generate heuristic demonstrations:

```powershell
uv run python scripts/generate_bc_dataset.py --games 2000
```

Train the card-play network:

```powershell
uv run python scripts/train_behavior_clone.py --epochs 10
```

Evaluate the resulting checkpoint:

```powershell
uv run python scripts/evaluate_bc_agent.py --games 10000
```

### DAgger

Generate learner-visited states and heuristic labels:

```powershell
uv run python scripts/generate_dagger_dataset.py --games 2000
```

Train on the aggregated dataset:

```powershell
uv run python scripts/train_behavior_clone.py `
  --dataset data/behavior_cloning/dagger_card_play.npz `
  --init-checkpoint models/behavior_cloning/card_play_mlp.pt `
  --output models/behavior_cloning/card_play_dagger.pt `
  --epochs 5 `
  --learning-rate 0.0003
```

### PPO

PPO can be initialized from either a BC/DAgger checkpoint or an existing PPO checkpoint.

For example:

```powershell
uv run python scripts/train_ppo.py `
  --init-checkpoint models/behavior_cloning/card_play_dagger.pt `
  --iterations 30 `
  --rollout-games 2000 `
  --learning-rate 0.0001
```

The PPO trainer supports:

- fixed validation games,
- validation-based checkpoint selection,
- heuristic and random opponents,
- frozen PPO snapshot opponents,
- configurable opponent mixtures.

Evaluate a PPO checkpoint with:

```powershell
uv run python scripts/evaluate_ppo_agent.py `
  --checkpoint models/ppo/card_play_ppo_best.pt `
  --games 10000
```

Generated datasets and model checkpoints are not committed to the repository.

## Observation encoding

The current card-play model uses a flat observation vector containing information such as:

- the player's hand,
- public trick history,
- game type,
- trump or called suit,
- declarer,
- relative player information,
- collected points,
- called-Ace release state,
- current trick number.

This is still an experimental representation. One of the next planned changes is to add more explicit public game-state information, such as known voids, current trick strength, trump counts, role information, and bidding history.

## Testing and code quality

Run the test suite:

```powershell
uv run pytest
```

Run all local checks:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

The same checks are also run through GitHub Actions.

## Notes

This is an ongoing hobby project. Parts of the ML pipeline and some strategy implementations are experimental and will likely change as the project develops.
