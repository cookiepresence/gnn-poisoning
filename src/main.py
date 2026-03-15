#!/usr/bin/env python3
import configargparse

import time
from typing import Any

import torch

from src import dataset
from src import model as m
from src import poison
from src import train
from src import utils

def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _expand_per_dataset(values: str, num_datasets: int, field: str) -> list[str]:
    items = _split_csv(values)
    if len(items) == 1:
        return items * num_datasets
    if len(items) == num_datasets:
        return items
    raise ValueError(
        f"{field} expects either 1 value or {num_datasets} comma-separated values; got {len(items)}"
    )


def _parse_target_label(value: str) -> int | tuple[int, int] | None:
    v = value.strip().lower()
    if v in {"", "none", "null"}:
        return None
    if "," in v:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("attack-target-label tuple form must have exactly 2 integers, e.g. 1,2")
        return int(parts[0]), int(parts[1])
    return int(v)


def run_attack(cfg: dict[str, Any], dataset_cfg: dict[str, Any], run_name: str) -> None:
    utils.set_seed(cfg["seed"])
    device = utils.resolve_device(cfg["device"])

    run_dir = utils.create_rundir(cfg["run_dir"], run_name)

    clean_data, in_dim, out_dim = dataset.load_dataset(**dataset_cfg)
    poisoned_data, attack_info, adaptive = poison.apply_attack(data=clean_data, **cfg["attack"])

    clean_data = clean_data.to(device)
    poisoned_data = poisoned_data.to(device)

    model = m.build_model(cfg["model"], in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    run_cfg = dict(cfg)
    run_cfg["dataset"] = dataset_cfg

    utils.write_json(run_dir / "config.json", run_cfg)
    utils.write_json(run_dir / "model" / "config.json", run_cfg)

    best = {"train_acc_clean": -1.0, "epoch": 0}
    start = time.time()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train.train_step(model, poisoned_data, optimizer)
        metrics = {"epoch": epoch, "loss": loss, "time_sec": time.time() - start}

        if epoch % cfg["train"]["eval_every"] == 0:
            metrics.update(train.eval_all(model, clean_data, poisoned_data))

        poisoned_data = adaptive(model, poisoned_data)

        print({"run": run_name, "dataset": dataset_cfg["name"], "variant": dataset_cfg["variant"], **metrics})
        utils.write_metrics(run_dir, metrics)

        if cfg["logging"]["save_best"] and "train_acc_clean" in metrics:
            if metrics["train_acc_clean"] > best["train_acc_clean"]:
                best = {
                    "epoch": epoch,
                    "train_acc_clean": metrics["train_acc_clean"],
                    "train_acc_poisoned": metrics["train_acc_poisoned"],
                    "val_acc": metrics["val_acc"],
                    "test_acc": metrics["test_acc"],
                }
                torch.save(model.state_dict(), run_dir / "model" / "model.pt")

    utils.write_summary(
        run_dir,
        {
            "best": best,
            "final_epoch": cfg["train"]["epochs"],
            "dataset": dataset_cfg,
            "attack": attack_info,
        },
    )


def main() -> None:
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
    parser.add_argument("--attack-target-label", type=str, default="none")
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--lafak-atk-epochs", type=int, default=200)
    parser.add_argument("--lafak-gcn-l2", type=float, default=5e-4)
    parser.add_argument("--lafak-lr", type=float, default=1e-4)
    parser.add_argument("--mg-n-iter", type=int, default=2)
    parser.add_argument("--mg-attack-prop", type=str, default="SK")
    parser.add_argument("--mg-pred-prop", type=str, default="SK")
    parser.add_argument("--mg-gamma", type=float, default=1.0)
    parser.add_argument("--mg-pagerank-alpha", type=float, default=0.1)
    parser.add_argument("--mg-prop-k", type=int, default=2)

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
    target_label = _parse_target_label(args.attack_target_label)
    dataset_names = _split_csv(args.dataset_name)
    dataset_variants = _expand_per_dataset(args.dataset_variant, len(dataset_names), "dataset-variant")
    dataset_roots = _expand_per_dataset(args.dataset_root, len(dataset_names), "dataset-root")
    dataset_cfgs = []
    for i in range(len(dataset_names)):
        dataset_cfgs.append(
            {
                "name": dataset_names[i],
                "variant": dataset_variants[i],
                "root": dataset_roots[i],
                "normalize_features": args.dataset_normalize_features,
            }
        )

    cfg: dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
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
            "target_label": target_label,
            "seed": args.attack_seed,
            "lafak_atk_epochs": args.lafak_atk_epochs,
            "lafak_gcn_l2": args.lafak_gcn_l2,
            "lafak_lr": args.lafak_lr,
            "mg_n_iter": args.mg_n_iter,
            "mg_attack_prop": args.mg_attack_prop,
            "mg_pred_prop": args.mg_pred_prop,
            "mg_gamma": args.mg_gamma,
            "mg_pagerank_alpha": args.mg_pagerank_alpha,
            "mg_prop_k": args.mg_prop_k,
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

    for ds_cfg in dataset_cfgs:
        ds_tag = f"{ds_cfg['name']}-{ds_cfg['variant']}".replace("/", "_").replace(" ", "_")
        base_run_name = args.run_name if args.run_name else f"{args.attack_name}"
        run_name = base_run_name if len(dataset_cfgs) == 1 else f"{base_run_name}__{ds_tag}"
        run_attack(cfg, ds_cfg, run_name)


if __name__ == "__main__":
    main()
