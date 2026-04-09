# BusBunch: Map-First AI Bus Anti-Bunching Demo for BMTC

## What This Project Is
This repository contains a `BMTC-inspired hackathon MVP` for visualizing and controlling bus bunching on the `KBS - MG Road - Domlur` corridor.

The demo is designed for judges:
- real geographic corridor map
- stop markers and route path
- moving bus replay on the corridor
- side-by-side `static schedule vs AI control`
- a visual control-room story instead of a numbers-heavy dashboard

## What Is Implemented
- custom corridor simulator in `src/sim/`
- shared system contracts in `src/models/contracts.py`
- graph-aware prediction interface in `src/models/predictor.py`
- static, heuristic, and RL-style policies in `src/policies/`
- evaluation and artifact generation in `src/eval/compare.py`
- Streamlit visual demo in `app/`

## Demo Pages
- `Home`: map-first overview of the corridor and demo flow
- `Why Bunching Happens`: visual replay of bunching under static control
- `Baseline vs AI`: side-by-side geographic replay
- `Control Room`: focus bus, intervention, and route situation
- `AI Brain`: what the predictor sees and how the controller acts
- `Outcome Summary`: final judge wrap-up

## How To Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
- `artifacts/ppo_checkpoint.json`

These are used for deterministic replay in the Streamlit demo.

## Current Hackathon Simplifications
- one corridor instead of the full BMTC network
- lightweight graph-aware forecaster instead of a full trained STGNN
- lightweight PPO-style trainer fallback instead of a heavy dependency stack
- synthetic but BMTC-shaped passenger and traffic behavior

## Why The Demo Looks Better Now
- buses have geographic positions on the map
- each stop includes lat/lon and names
- the route is drawn as a corridor path
- focus bus and intervention are visually highlighted
- story cards explain what judges should notice at each replay step

## Optional Future Upgrades
- replace the predictor with a real STGNN
- replace the RL fallback with true PPO using PyTorch / Stable-Baselines3
- ingest richer BMTC / GTFS / live traffic sources
- scale from one corridor to a multi-corridor control room
