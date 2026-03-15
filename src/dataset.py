from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import (
    Planetoid,
    TUDataset,
    Reddit,
    Amazon,
)

def load_dataset(name: str,
                 root: str,
                 variant: Optional[str] = None,
                 normalize_features: bool = False):

    transform = NormalizeFeatures() if normalize_features else None
    name = name.lower()

    match name:
        case "planetoid":
            dataset = Planetoid(root, variant, transform=transform)
            data = dataset[0]

        case "tu" | "tudataset":
            dataset = TUDataset(root, variant, transform=transform)
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes  # graph-level

        case "reddit":
            dataset = Reddit(root, transform=transform)
            data = dataset[0]

        case "amazon":
            dataset = Amazon(root, variant, transform=transform)
            data = dataset[0]

        case "ogb":
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(name=variant, root=root)
            data = dataset[0]
            if data.y.dim() > 1:
                data.y = data.y.view(-1)
            data.y = data.y.long()

            split = dataset.get_idx_split()
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

            data.train_mask[split["train"]] = True
            data.val_mask[split["valid"]] = True
            data.test_mask[split["test"]] = True

            return data, dataset.num_features, dataset.num_classes

        case _:
            raise ValueError(f"Unsupported dataset: {name}")

    # If masks already exist (Planetoid), keep them
    if not hasattr(data, "train_mask"):
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)

        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:train_end]] = True
        data.val_mask[perm[train_end:val_end]] = True
        data.test_mask[perm[val_end:]] = True

    return data, dataset.num_features, dataset.num_classes

def clone_data(data: Data) -> Data:
    data2 = data.clone()
    for attr in ["x", "edge_index", "y", "train_mask", "val_mask", "test_mask"]:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            setattr(data2, attr, getattr(data, attr).clone())
    return data2
