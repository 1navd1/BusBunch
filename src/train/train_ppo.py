from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_checker import check_env
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required for PPO training. Install requirements and retry."
    ) from exc

from src.rl.env import BusBunchEnv, EnvScenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 PPO policy")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/ppo_best.zip")
    parser.add_argument("--tensorboard", type=str, default="artifacts/eval/tb")
    parser.add_argument("--peak-profile", type=str, default="peak", choices=["peak", "off_peak"])
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    scenario = EnvScenario(peak_profile=args.peak_profile)
    train_env = BusBunchEnv(scenario=scenario, seed=args.seed)
    eval_env = BusBunchEnv(scenario=scenario, seed=args.seed + 1)

    check_env(train_env, warn=True)

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_path.parent),
        eval_freq=2000,
        n_eval_episodes=3,
        deterministic=True,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=args.tensorboard,
        seed=args.seed,
        verbose=0,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)

    # SB3 EvalCallback stores best checkpoint as best_model.zip
    best_src = ckpt_path.parent / "best_model.zip"
    if best_src.exists():
        best_src.replace(ckpt_path)
    else:
        model.save(str(ckpt_path.with_suffix("")))

    out = {
        "checkpoint": str(ckpt_path),
        "timesteps": args.timesteps,
        "config": {
            "policy": "MlpPolicy",
            "n_steps": 1024,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
