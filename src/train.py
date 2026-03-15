import torch
import torch.nn.functional as F

from src import utils


def accuracy(logits, ground_truth, mask):
    return (torch.sum(torch.argmax(logits[mask], dim=-1) == ground_truth[mask]) / torch.sum(mask)).item()

def train_step(model, data, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_all(model, clean_data, poisoned_data) -> dict[str, float]:
    model.eval()
    logits = model(clean_data.x, clean_data.edge_index)
    return {
        "train_acc_clean": accuracy(logits, clean_data.y, clean_data.train_mask),
        "train_acc_poisoned": accuracy(logits, poisoned_data.y, poisoned_data.train_mask),
        "val_acc": accuracy(logits, clean_data.y, clean_data.val_mask),
        "test_acc": accuracy(logits, clean_data.y, clean_data.test_mask),
    }
