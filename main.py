#!/usr/bin/env python3
import configargparse

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import dataset
import model as m
import poison
import train
import utils

def run_attack(cfg: dict[str, Any]) -> None:
    utils.set_seed(cfg["seed"])
    device = utils.resolve_device(cfg["device"])

    run_dir = utils.create_rundir(cfg['run_dir'], cfg['run_name'])

    clean_data, in_dim, out_dim = dataset.load_dataset(**cfg['dataset'])
    poisoned_data, attack_info, adaptive = poison.apply_attack(data=clean_data, **cfg["attack"])

    clean_data = clean_data.to(device)
    poisoned_data = poisoned_data.to(device)

    model = m.build_model(cfg['model'], in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    utils.write_json(run_dir / "config.json", cfg)
    utils.write_json(run_dir / "model" / "config.json", cfg)

    best = {"train_acc_clean": -1.0, "epoch": 0}
    start = time.time()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train.train_step(model, poisoned_data, optimizer)
        metrics = {"epoch": epoch, "loss": loss, "time_sec": time.time() - start}
        
        if epoch % cfg["train"]["eval_every"] == 0:
            metrics.update(train.eval_all(model, clean_data, poisoned_data))

        new_dataset = adaptive(model, poisoned_data)

        print(metrics)
        utils.write_metrics(run_dir, metrics)

        if cfg["logging"]["save_best"]:
            if metrics["train_acc_clean"] > best["train_acc_clean"]:
                best = {
                    "epoch": epoch,
                    "train_acc_clean": metrics["train_acc_clean"],
                    "train_acc_poisoned": metrics["train_acc_poisoned"],
                    "val_acc": metrics["val_acc"],
                    "test_acc": metrics["test_acc"],
                }
                torch.save(model.state_dict(), run_dir / "model" / "model.pt")

    utils.write_summary(run_dir, {"best": best, "final_epoch": cfg["train"]["epochs"]})


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", is_config_file=True, help="INI config file")

    # core
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # dataset
    parser.add_argument("--dataset-name", type=str, default="planetoid")
    parser.add_argument("--dataset-variant", type=str, default="Cora")
    parser.add_argument("--dataset-root", type=str, default="data/planetoid")
    parser.add_argument("--dataset-normalize-features", action="store_true", default=True)

    # model
    parser.add_argument("--model-name", type=str, default="gcn")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=8)

    # attack
    parser.add_argument("--attack-name", type=str, default="label_flipping")
    parser.add_argument("--attack-flip-frac", type=float, default=0.1)
    parser.add_argument("--attack-target-label", type=int, default=None)
    # parser.add_argument("--attack-apply-to", type=str, default="train")
    parser.add_argument("--attack-seed", type=int, default=42)

    # train
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=1)

    # logging
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-best", action="store_true", default=True)

    args = parser.parse_args()
    cfg: dict = {
        "seed": args.seed,
        "device": args.device,
        "dataset": {
            "name": args.dataset_name,
            "variant": args.dataset_variant,
            "root": args.dataset_root,
            "normalize_features": args.dataset_normalize_features,
        },
        "model": {
            "name": args.model_name,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "heads": args.heads,
        },
        "attack": {
            "name": args.attack_name,
            "flip_frac": args.attack_flip_frac,
            "target_label": args.attack_target_label,
            # "apply_to": args.attack_apply_to,
            "seed": args.attack_seed,
        },
        "train": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
        },
        "logging": {
            "save_best": args.save_best,
        },
        "run_dir": args.run_dir,
        "run_name": args.run_name
    }

    run_attack(cfg)
