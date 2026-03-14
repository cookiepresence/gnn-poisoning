from typing import Any

from torch_geometric.nn.models import GCN, GAT, GraphSAGE, MLP

def build_model(cfg: dict[str, Any], in_dim: int, out_dim: int):
    name = cfg["name"]
    h = cfg["hidden_dim"]
    n = cfg["num_layers"]
    d = cfg["dropout"]
    heads = cfg["heads"]

    match name:
        case 'gcn':
            return GCN(
                in_channels=in_dim,
                hidden_channels=h,
                num_layers=n,
                out_channels=out_dim,
                dropout=d
            )
        case "sage":
            return GraphSAGE(
                in_channels=in_dim,
                hidden_channels=h,
                num_layers=n,
                out_channels=out_dim,
                dropout=d
            )
        case "gat":
            return GAT(
                in_channels=in_dim,
                hidden_channels=h,
                num_layers=n,
                out_channels=out_dim,
                heads=heads,
                dropout=d,
            )
        case "mlp":
            return MLP(
                in_channels=in_dim,
                hidden_channels=h,
                out_channels=out_dim,
                num_layers=n,
                dropout=d
            )

    raise ValueError(f"Unsupported model: {name}")
