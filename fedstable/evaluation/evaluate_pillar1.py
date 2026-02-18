"""Evaluate Pillar 1 model on test set. Returns metrics dict."""

from typing import Any, Dict

import torch

from fedstable.evaluation.metrics import auc_roc, auc_pr, brier_score


def evaluate_pillar1(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    horizon_idx: int = 1,
) -> Dict[str, float]:
    """Run model on dataloader; return AUC-ROC, AUC-PR, Brier."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits[:, horizon_idx])
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return {
        "auc_roc": auc_roc(y_true, y_pred),
        "auc_pr": auc_pr(y_true, y_pred),
        "brier_score": brier_score(y_true, y_pred),
    }


import numpy as np  # noqa: E402
