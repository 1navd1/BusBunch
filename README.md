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
### 0) Clone and enter project
```bash
git clone <your-repo-url>
cd BusBunch
```

### 1) Create virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate comparison artifacts (train/evaluate + export JSON)
```bash
python3 -m src.eval.compare
```

This writes:
- `artifacts/kpi_summary.json`
- `artifacts/demo_seed.json`
- `artifacts/comparison_full.json`
- `artifacts/ppo_checkpoint.json`

### 3) Launch Streamlit demo
```bash
streamlit run app/Home.py
```

### 4) Open in browser
```text
http://localhost:8501
```

### Quick rerun commands (after first setup)
```bash
cd /home/bharath/code/BusBunch
source .venv/bin/activate
python3 -m src.eval.compare
streamlit run app/Home.py
```

### Run using explicit venv binaries (without activating shell)
```bash
cd /home/bharath/code/BusBunch
.venv/bin/python -m src.eval.compare
.venv/bin/streamlit run app/Home.py
```

### Optional: run Streamlit on a custom port
```bash
streamlit run app/Home.py --server.port 8502
```

### Optional: clean and regenerate artifacts
```bash
rm -f artifacts/kpi_summary.json artifacts/demo_seed.json artifacts/comparison_full.json artifacts/ppo_checkpoint.json
python3 -m src.eval.compare
```

## APIs and Configuration
This project runs end-to-end **without any API keys** in its current MVP form.

### Required APIs
- None

### Optional APIs (future upgrade path)
- Google Maps Distance Matrix / Routes API for real travel-time priors
- OpenStreetMap-based routing stack (self-hosted or external service)
- BMTC/GTFS/live feed sources if available

### If you add APIs, store keys safely
Create a local `.env` file (do not commit secrets):
```bash
cp .env.example .env
```

Suggested keys:
```bash
GOOGLE_MAPS_API_KEY=your_key_here
TRAFFIC_PROVIDER=osm
BMTC_FEED_URL=
```

### Important
- Never hardcode API keys in Python files.
- Add `.env` to `.gitignore` if not already ignored.
- Current code paths do not require these values; they are only for optional extensions.

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
