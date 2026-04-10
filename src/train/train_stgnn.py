from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import load_dataset
from src.models.stgnn import STGNN, LossWeights, stgnn_loss


def _to_loader(x: np.ndarray, y_tt: np.ndarray, y_dem: np.ndarray, y_bun: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x[idx], dtype=torch.float32),
        torch.tensor(y_tt[idx], dtype=torch.float32),
        torch.tensor(y_dem[idx], dtype=torch.float32),
        torch.tensor(y_bun[idx], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model: STGNN, loader: DataLoader, adj: torch.Tensor, optimizer: torch.optim.Optimizer | None, weights: LossWeights) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    agg = {"total": 0.0, "travel_time": 0.0, "demand": 0.0, "bunching": 0.0}
    n_batches = 0

    for x, y_tt, y_dem, y_bun in loader:
        pred = model(x, adj)
        losses = stgnn_loss(pred, y_tt, y_dem, y_bun, weights=weights)

        if train_mode:
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

        n_batches += 1
        for k in agg:
            agg[k] += float(losses[k].detach().cpu().item())

    if n_batches == 0:
        return agg
    return {k: v / n_batches for k, v in agg.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STGNN predictor")
    parser.add_argument("--dataset", type=str, default="artifacts/eval/stgnn_dataset.npz")
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/stgnn_best.pt")
    parser.add_argument("--log", type=str, default="artifacts/eval/stgnn_train_log.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    x = data["x"]
    y_tt = data["y_travel_time"]
    y_dem = data["y_demand"]
    y_bun = data["y_bunching"]

    idx_train = data["idx_train"].astype(np.int64)
    idx_val = data["idx_val"].astype(np.int64)

    n_nodes = int(x.shape[2])
    in_features = int(x.shape[3])

    train_loader = _to_loader(x, y_tt, y_dem, y_bun, idx_train, batch_size=args.batch_size, shuffle=True)
    val_loader = _to_loader(x, y_tt, y_dem, y_bun, idx_val, batch_size=args.batch_size, shuffle=False)

    adj = torch.tensor(data["adj_norm"], dtype=torch.float32)
    model = STGNN(in_features=in_features, hidden_dim=args.hidden_dim, num_nodes=n_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = LossWeights(travel_time=1.0, demand=0.7, bunching=0.9)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    logs = []

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, adj, optimizer=optimizer, weights=weights)
        va = _run_epoch(model, val_loader, adj, optimizer=None, weights=weights)

        row = {
            "epoch": epoch,
            "train_total": round(tr["total"], 6),
            "val_total": round(va["total"], 6),
            "train_tt": round(tr["travel_time"], 6),
            "val_tt": round(va["travel_time"], 6),
            "train_demand": round(tr["demand"], 6),
            "val_demand": round(va["demand"], 6),
            "train_bunch": round(tr["bunching"], 6),
            "val_bunch": round(va["bunching"], 6),
        }
        logs.append(row)

        if va["total"] < best_val:
            best_val = va["total"]
            patience_left = args.patience
            best_state = {
                "state_dict": model.state_dict(),
                "in_features": in_features,
                "hidden_dim": args.hidden_dim,
                "num_nodes": n_nodes,
                "best_val": best_val,
                "epoch": epoch,
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("STGNN training did not produce a checkpoint")

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"best_val": best_val, "epochs": logs}, indent=2), encoding="utf-8")

    print(json.dumps({"checkpoint": str(ckpt), "best_val": best_val, "epochs_ran": len(logs)}, indent=2))


if __name__ == "__main__":
    main()
