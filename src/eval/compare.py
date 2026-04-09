from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

from src.eval.runner import ScenarioConfig, ScenarioGenerator, ScenarioRunner
from src.models.ppo_policy import PPOConfig, PPOTrainer
from src.policies.headway_policy import HeadwayPolicy
from src.policies.rl_policy import RLPolicy
from src.policies.static_policy import StaticPolicy


def evaluate_policy(policy, seeds: List[int], scenario: ScenarioConfig) -> Dict:
    reports = []
    rewards = []
    best = None

    for seed in seeds:
        report = ScenarioRunner.run(policy, scenario, seed=seed)
        m = asdict(report.metrics)
        m["seed"] = seed
        m["total_reward"] = round(report.total_reward, 4)
        reports.append(m)
        rewards.append(report.total_reward)

        if best is None or report.total_reward > best["total_reward"]:
            best = {
                "seed": seed,
                "trace": report.trace,
                "metrics": m,
                "total_reward": report.total_reward,
            }

    summary = {
        "policy": policy.name,
        "mean_reward": round(mean(rewards), 4),
        "mean_metrics": {
            "bunching_count": round(mean([x["bunching_count"] for x in reports]), 4),
            "headway_std": round(mean([x["headway_std"] for x in reports]), 4),
            "avg_wait_time": round(mean([x["avg_wait_time"] for x in reports]), 4),
            "occupancy_std": round(mean([x["occupancy_std"] for x in reports]), 4),
            "fuel_proxy": round(mean([x["fuel_proxy"] for x in reports]), 4),
            "total_delay": round(mean([x["total_delay"] for x in reports]), 4),
        },
        "seeds": reports,
        "best": best,
    }
    return summary


def run_comparison(out_dir: str = "artifacts") -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenario = ScenarioGenerator.create(day_type="weekday", peak_profile="peak")

    # The compact control-state observation is fixed to 8, action dims fixed to 3 in runner.
    obs_dim, action_dim = 8, 3

    trainer = PPOTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=PPOConfig(total_steps=5400, rollout_steps=180, seed=11, population=12, noise_std=0.11),
    )

    # Use an internal env from runner-compatible simplified simulator.
    from src.eval.runner import MiniBusEnv

    train_log = trainer.train(MiniBusEnv(scenario))
    ckpt_path = out / "ppo_checkpoint.json"
    trainer.save(str(ckpt_path))

    seeds = [3, 7, 11, 17, 23]
    static_res = evaluate_policy(StaticPolicy(), seeds, scenario)
    headway_res = evaluate_policy(HeadwayPolicy(), seeds, scenario)
    ppo_res = evaluate_policy(RLPolicy(str(ckpt_path), obs_dim=obs_dim), seeds, scenario)

    def pct_improve(metric: str) -> float:
        base = static_res["mean_metrics"][metric]
        improved = ppo_res["mean_metrics"][metric]
        if base == 0:
            return 0.0
        return round(100.0 * (base - improved) / base, 2)

    headline_win = {
        "bunching_reduction": ppo_res["mean_metrics"]["bunching_count"] < static_res["mean_metrics"]["bunching_count"],
        "wait_time_reduction": ppo_res["mean_metrics"]["avg_wait_time"] < static_res["mean_metrics"]["avg_wait_time"],
        "occupancy_balance": ppo_res["mean_metrics"]["occupancy_std"] < static_res["mean_metrics"]["occupancy_std"],
    }

    kpi = {
        "training": {
            "episodes_seen": len(train_log["episode_reward"]),
            "last_episode_reward": round(train_log["episode_reward"][-1], 4) if train_log["episode_reward"] else None,
            "best_episode_reward": round(max(train_log["episode_reward"]), 4) if train_log["episode_reward"] else None,
        },
        "results": {
            "static": {"mean_reward": static_res["mean_reward"], **static_res["mean_metrics"]},
            "headway": {"mean_reward": headway_res["mean_reward"], **headway_res["mean_metrics"]},
            "ppo": {"mean_reward": ppo_res["mean_reward"], **ppo_res["mean_metrics"]},
        },
        "improvement_vs_static": {
            "bunching_count_pct": pct_improve("bunching_count"),
            "avg_wait_time_pct": pct_improve("avg_wait_time"),
            "occupancy_std_pct": pct_improve("occupancy_std"),
            "headway_std_pct": pct_improve("headway_std"),
        },
        "headline_kpi_win": {
            **headline_win,
            "wins_out_of_3": sum(1 for x in headline_win.values() if x),
        },
    }

    # Best-seed replay for deterministic demo.
    best = ppo_res["best"]
    demo_seed = {
        "policy": "ppo",
        "scenario": {"day_type": scenario.day_type, "peak_profile": scenario.peak_profile},
        "seed": best["seed"],
        "metrics": best["metrics"],
        "trace": best["trace"],
    }

    (out / "kpi_summary.json").write_text(json.dumps(kpi, indent=2), encoding="utf-8")
    (out / "demo_seed.json").write_text(json.dumps(demo_seed, indent=2), encoding="utf-8")

    full = {
        "kpi": kpi,
        "static": static_res,
        "headway": headway_res,
        "ppo": ppo_res,
        "checkpoint": str(ckpt_path),
        "scenario": {"day_type": scenario.day_type, "peak_profile": scenario.peak_profile},
    }
    (out / "comparison_full.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    return full


if __name__ == "__main__":
    summary = run_comparison()
    print(json.dumps(summary["kpi"], indent=2))
