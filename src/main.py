#!/usr/bin/env python3
import configargparse

import time
from typing import Any

import torch
import wandb

from src import dataset
from src import model as m
from src import poison
from src import train
from src import utils


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _init_wandb(
    cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    run_name: str,
    dataset_cfg: dict[str, Any],
):
    if not cfg["logging"]["wandb_enable"]:
        return None

    tags = cfg["logging"]["wandb_tags"]
    if isinstance(tags, str):
        tags = _split_csv(tags)

    return wandb.init(
        project=cfg["logging"]["wandb_project"],
        entity=cfg["logging"]["wandb_entity"] or None,
        group=cfg["logging"]["wandb_group"] or None,
        name=cfg["logging"]["wandb_run_name"] or run_name,
        tags=tags,
        mode=cfg["logging"]["wandb_mode"],
        dir=cfg["run_dir"],
        config=run_cfg,
        reinit=True,
        job_type="train",
        notes=f"dataset={dataset_cfg['name']} variant={dataset_cfg['variant']}",
    )


def run_attack(cfg: dict[str, Any], dataset_cfg: dict[str, Any], run_name: str) -> None:
    utils.set_seed(cfg["seed"])
    device = utils.resolve_device(cfg["device"])

    run_dir = utils.create_rundir(cfg["run_dir"], run_name)

    clean_data, in_dim, out_dim = dataset.load_dataset(**dataset_cfg)
    attack_cfg = cfg["attack"]
    attack_name = str(attack_cfg["name"]).strip().lower()

    match attack_name:
        case "no_attack":
            attack = poison.NoAttack(seed=attack_cfg["seed"], flip_frac=0.0)
        case "label_flipping":
            attack = poison.RandomFlipAttack(seed=attack_cfg["seed"], flip_frac=attack_cfg["flip_frac"])
        case "degree_flipping":
            attack = poison.DegreeFlipAttack(seed=attack_cfg["seed"], flip_frac=attack_cfg["flip_frac"])
        case "lafak":
            attack = poison.LafAKAttack(
                seed=attack_cfg["seed"],
                flip_frac=attack_cfg["flip_frac"],
                target_label=attack_cfg["target_label"],
                atk_epochs=attack_cfg["lafak_atk_epochs"],
                gcn_l2=attack_cfg["lafak_gcn_l2"],
                lr=attack_cfg["lafak_lr"],
            )
        case "mg":
            attack = poison.MGAttack(
                seed=attack_cfg["seed"],
                flip_frac=attack_cfg["flip_frac"],
                n_iter=attack_cfg["mg_n_iter"],
                attack_prop=attack_cfg["mg_attack_prop"],
                pred_prop=attack_cfg["mg_pred_prop"],
                gamma=attack_cfg["mg_gamma"],
                pagerank_alpha=attack_cfg["mg_pagerank_alpha"],
                prop_K=attack_cfg["mg_prop_k"],
            )
        case _:
            raise ValueError(
                "unknown attack name: "
                f"{attack_cfg['name']!r}. "
                "Use one of: no_attack, label_flipping, degree_flipping, lafak, mg."
            )

    poisoned_data = attack.init_attack(clean_data, clean_data.train_mask)

    attack_info = {
        "attack_class": attack.__class__.__name__,
        **cfg["attack"],
    }

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

    wandb_run = _init_wandb(cfg, run_cfg, run_name, dataset_cfg)

    best = {"train_acc_clean": -1.0, "epoch": 0}
    start = time.time()

    try:
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            loss = train.train_step(model, poisoned_data, optimizer)
            metrics = {"epoch": epoch, "loss": loss, "time_sec": time.time() - start}

            if epoch % cfg["train"]["eval_every"] == 0:
                metrics.update(train.eval_all(model, clean_data, poisoned_data))

            metrics_payload = {"run": run_name, "dataset": dataset_cfg["name"], "variant": dataset_cfg["variant"], **metrics}
            print(metrics_payload)
            utils.write_metrics(run_dir, metrics)

            if wandb_run is not None:
                wandb_run.log(metrics_payload, step=epoch)

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

        summary = {
            "best": best,
            "final_epoch": cfg["train"]["epochs"],
            "dataset": dataset_cfg,
            "attack": attack_info,
        }
        utils.write_summary(run_dir, summary)

        if wandb_run is not None:
            for key, value in best.items():
                wandb_run.summary[f"best/{key}"] = value
            wandb_run.summary["final_epoch"] = cfg["train"]["epochs"]
            wandb_run.summary["dataset/name"] = dataset_cfg["name"]
            wandb_run.summary["dataset/variant"] = dataset_cfg["variant"]
    finally:
        if wandb_run is not None:
            wandb_run.finish()


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
    parser.add_argument("--wandb-enable", type=_to_bool, default=False)
    parser.add_argument("--wandb-project", type=str, default="gnn-label-poisoning")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online")

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
            "wandb_enable": args.wandb_enable,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_run_name": args.wandb_run_name,
            "wandb_tags": _split_csv(args.wandb_tags) if args.wandb_tags else [],
            "wandb_mode": args.wandb_mode,
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
