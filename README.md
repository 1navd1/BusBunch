# BusBunch: Hybrid DRL-GNN Bus Anti-Bunching System for BMTC

## Problem
BMTC corridors suffer from bus bunching when static schedules meet real-world variability:
- uneven traffic delays
- dwell time spikes from passenger surges
- dispatch perturbations at terminals

Bunching hurts service quality in three ways:
1. more waiting time for passengers
2. overload/underuse imbalance across buses
3. unreliable headways and poor corridor capacity utilization

## What We Built (Hackathon MVP)
This repo implements a **one-corridor, BMTC-inspired MVP** with a **city-scalable architecture**:
- custom Python simulator for corridor dynamics
- graph-aware prediction module behind a GNN-ready interface
- PPO controller for hold/speed/dispatch interventions
- baseline policies (static + heuristic headway)
- Streamlit judge demo with deterministic scenario replay

## Architecture
### Data Layer
- BMTC-style corridor abstraction (stops, segments, demand bands, traffic priors)
- route represented as graph-like stop/segment structure
- peak/off-peak shaping through scenario profiles

### Prediction Layer
- `Predictor.predict(sim_state) -> PredictionBundle`
- current implementation: lightweight graph-temporal forecaster
- interface is preserved for future STGNN replacement without app/policy contract changes

### Control Layer
- compact `ControlState` for decision making
- action model: `hold_sec`, `speed_delta_pct`, `dispatch_offset_sec`
- PPO-based controller trained on simplified corridor simulator
- baseline policies:
  - static schedule (no intervention)
  - heuristic headway control

### Simulation + Evaluation Layer
- deterministic seeded scenarios
- `ScenarioRunner.run(policy, scenario) -> EpisodeReport`
- KPI aggregation for baseline vs AI comparison

## Contracts
Core types are in `src/models/contracts.py`:
- `BusState`
- `StopState`
- `SegmentState`
- `SystemState`
- `ControlState`
- `PredictionBundle`
- `ControlAction`
- `EpisodeMetrics`
- `EpisodeReport`

## Repository Structure
- `src/models/`: contracts, predictor, PPO trainer/checkpoint logic
- `src/policies/`: static, headway heuristic, PPO policy wrapper
- `src/eval/`: simulator wrapper, scenario generator/runner, comparison pipeline
- `app/`: Streamlit demo pages
- `artifacts/`: checkpoints, seeded replay traces, KPI summaries, pitch assets

## How It Works
1. Create a scenario (`ScenarioGenerator`).
2. Build system state each step.
3. Predictor outputs congestion and bunching-risk bundle.
4. Policy consumes `ControlState + PredictionBundle` and produces control action.
5. Simulator applies action and advances environment.
6. Runner records trace + metrics for replay and evaluation.

## Run Instructions
### 0) Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Generate comparison artifacts
```bash
python3 -m src.eval.compare
```

This writes:
- `artifacts/kpi_summary.json`
- `artifacts/demo_seed.json`
- `artifacts/comparison_full.json`
- `artifacts/ppo_checkpoint.json`

### 2) Launch Streamlit demo
```bash
streamlit run app/Home.py
```

## Streamlit Demo Flow (2-3 minutes)
### Page 1: Why Bunching Happens
Shows headway compression under static scheduling.

### Page 2: Baseline vs AI
Side-by-side seeded replay and KPI table.

### Page 3: Control Room
Operational alerts, intervention logs, per-bus status.

### Page 4: AI Brain
Prediction bundle, route graph context, decision explanation.

### Page 5: Outcome Summary
Judge-facing KPI wins and corridor-to-city scale narrative.

## Current Results (From `artifacts/kpi_summary.json`)
- Bunching reduction vs static: **100.0%**
- Wait-time reduction vs static: **31.22%**
- Occupancy variance reduction vs static: **88.09%**
- Headline KPI wins: **3/3**

## Test Scenarios Covered
- normal corridor variability
- delayed bus cascade behavior
- peak/off-peak profile handling
- terminal intervention effects
- visible control actions in replay trace

## Known Hackathon Simplifications
- single-corridor MVP (not full BMTC network runtime)
- lightweight graph forecaster instead of full STGNN training
- simplified simulator tuned for quick RL iteration and deterministic replay

## Pitch Assets
- timed judge script: `artifacts/pitch_script.md`
- backup static demo assets: `artifacts/backup_assets.json`

## Impact Story
The MVP validates a hybrid predictive-control loop on one corridor. The same contracts (`Predictor`, `Policy`, `ScenarioRunner`) are designed to scale corridor-by-corridor into a city-level BMTC control platform.
