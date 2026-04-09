# Judge Pitch Script (2:45)

## 0:00-0:30 Problem (30s)
BMTC buses often bunch because static schedules cannot absorb real-world disturbances. Once one bus is delayed, it collects more passengers, dwell time rises, and the next bus catches up. That creates long waits followed by bus convoys.

## 0:30-1:15 Technical Novelty (45s)
We built a hybrid DRL-GNN-style system:
- a graph-aware predictor estimates downstream congestion and bunching risk
- a PPO controller chooses interventions in real time
- actions include hold-at-stop, speed adjustment, and terminal dispatch offset

For hackathon reliability, the predictor is lightweight but contract-compatible with a future STGNN.

## 1:15-2:15 Live Demo Narration (60s)
1. Page 1: show why bunching emerges under static control.
2. Page 2: same seeded scenario, static vs AI side-by-side.
3. Page 3: control room alerts and action logs.
4. Page 4: AI brain view with prediction bundle + decision explanation.
5. Page 5: outcome summary with KPI deltas.

## 2:15-2:45 Impact + Scale (30s)
We demonstrate one corridor end-to-end, but architecture is city-scalable: each BMTC corridor can plug into the same predictor-policy contracts, then aggregate in a unified control room.

## One-Line Close
From static timetables to predictive, corridor-aware interventions: fewer bunching events, lower wait times, and more balanced utilization.
