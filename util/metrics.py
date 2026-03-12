import torch
from torchmetrics.classification import (MulticlassAUROC, 
                                         MulticlassF1Score, 
                                         MulticlassAccuracy)
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import balanced_accuracy_score

# TODO: Now only support one gpu, need to extend to multi-gpu later.
class ClassificationMetrics:
    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes

        # torchmetrics on the single device
        self.auroc = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        self.acc = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)

        # store CPU tensors for sklearn balanced accuracy
        self.all_preds = []
        self.all_targets = []

        # simple integer accumulators for top1
        self.top1_correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, outputs, targets):
        # outputs: logits shape N by C, targets: shape N
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=1)

        # update torchmetrics on device
        self.auroc.update(probs, targets)
        self.f1.update(preds, targets)
        self.acc.update(preds, targets)

        # store for sklearn balanced accuracy later
        self.all_preds.append(preds.cpu())
        self.all_targets.append(targets.cpu())

        # count correct for top1
        correct = preds.eq(targets).sum().item()
        self.top1_correct += int(correct)
        self.total += int(targets.size(0))

    def compute(self):
        if self.total == 0:
            return {
                "acc1": 0.0,
                "aucroc": float("nan"),
                "balanced_acc": float("nan"),
                "f1": float("nan")
            }

        # finalize torchmetrics
        try:
            auc_val = self.auroc.compute().item()
        except Exception:
            auc_val = float("nan")
        try:
            f1_val = self.f1.compute().item()
        except Exception:
            f1_val = float("nan")

        # sklearn balanced accuracy on all gathered preds and targets
        preds = torch.cat(self.all_preds, dim=0).numpy()
        targets = torch.cat(self.all_targets, dim=0).numpy()
        try:
            balanced_acc = balanced_accuracy_score(targets, preds)
        except Exception:
            balanced_acc = float("nan")

        # top1
        acc1 = float(self.top1_correct) / float(self.total)

        # reset state for next epoch
        self.auroc.reset()
        self.f1.reset()
        self.acc.reset()
        self.all_preds = []
        self.all_targets = []
        self.top1_correct = 0
        self.total = 0

        return {
            "acc1": acc1*100,
            "aucroc": auc_val*100,
            "balanced_acc": balanced_acc*100,
            "f1": f1_val*100
        }


class RegressionMetrics:
    def __init__(self, device):
        self.device = device

        # torchmetrics on the single device
        self.mae = MeanAbsoluteError().to(device)
        self.mse = MeanSquaredError().to(device)

        # keep CPU copies in case you want to compute other metrics with numpy
        self.all_preds = []
        self.all_targets = []

        # sample counter
        self.total = 0

    @torch.no_grad()
    def update(self, outputs, targets):
        # outputs can be logits or direct predictions
        preds = _to_device(outputs, self.device)
        t = _to_device(targets, self.device)

        # squeeze singleton channel if present
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.view(-1)
        if t.ndim > 1 and t.shape[1] == 1:
            t = t.view(-1)

        # ensure both are 1d tensors of same length
        preds = preds.view(-1)
        t = t.view(-1)

        # update torchmetrics on device
        self.mae.update(preds, t)
        self.mse.update(preds, t)

        # store CPU copies for any offline checks
        self.all_preds.append(preds.cpu())
        self.all_targets.append(t.cpu())

        self.total += int(preds.size(0))

    def compute(self):
        if self.total == 0:
            # nothing seen
            return {"mae": float("nan"), "mse": float("nan")}

        try:
            mae_val = float(self.mae.compute().item())
        except Exception:
            mae_val = float("nan")
        try:
            mse_val = float(self.mse.compute().item())
        except Exception:
            mse_val = float("nan")

        # reset state for next epoch
        self.mae.reset()
        self.mse.reset()
        self.all_preds = []
        self.all_targets = []
        self.total = 0

        return {"mae": mae_val, "mse": mse_val}


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device)
    else:
        # wrap scalar or list/array into tensor
        return torch.tensor(x, device=device, dtype=torch.float32)
 