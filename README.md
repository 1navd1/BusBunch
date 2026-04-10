# BusBunch: Map-First AI Bus Anti-Bunching Demo for BMTC

## What This Project Is
This repository contains a `BMTC-inspired hackathon MVP` for visualizing and controlling bus bunching on the `KBS - MG Road - Domlur` corridor.

The demo is designed as a lean product simulation:
- real geographic corridor map
- stop markers and route path
- moving bus replay on the corridor
- side-by-side `static schedule vs AI control`
- one separate `Driver Assist` tablet view

## What Is Implemented
- custom corridor simulator in `src/sim/`
- shared system contracts in `src/models/contracts.py`
- STGNN predictor + inference wrapper in `src/models/stgnn.py` and `src/models/stgnn_infer.py`
- static, heuristic, and SB3 PPO policies in `src/policies/`
- evaluation and artifact generation in `src/eval/compare.py`
- Streamlit visual demo in `app/`

## App Pages
- `Simulation`: the full split-screen corridor replay with shared controls
- `Driver Assist`: the in-bus tablet view with driver responses and AI re-evaluation

## How To Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.train.generate_rollouts
python3 -m src.train.train_stgnn
python3 -m src.train.train_ppo
python3 -m src.eval.compare
streamlit run app/Home.py
```

Open:
```text
http://localhost:8501
```

## Generated Artifacts
Running `python3 -m src.eval.compare` writes:
- `artifacts/kpi_summary.json`
- `artifacts/demo_seed.json`
- `artifacts/comparison_full.json`
- `artifacts/models/ppo_best.zip` (loaded by policy inference)
- `artifacts/models/stgnn_best.pt`
- `artifacts/models/stgnn_norm.json`

These are used for deterministic replay in the Streamlit app.

## Current Hackathon Simplifications
- one corridor instead of the full BMTC network
- synthetic data instead of production AVL/APC feed
- single-corridor modeling for fast iteration
- synthetic but BMTC-shaped passenger and traffic behavior

## Current UX
- the `Simulation` page is the only judge-facing replay page
- baseline and AI views stay synchronized with one slider
- autoplay replays the same scenario on both maps
- `Driver Assist` stays separate as the operational tablet for the driver

## Optional Future Upgrades
- replace the predictor with a real STGNN
- replace the RL fallback with true PPO using PyTorch / Stable-Baselines3
- ingest richer BMTC / GTFS / live traffic sources
- scale from one corridor to a multi-corridor control room
