import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import List, Dict

class OODMetricsEvaluator:
    """
    Evaluates Out-of-Distribution (OOD) generalization metrics.
    
    Mathematical Context:
    The OOD generalization gap is defined as:
        \Delta_{OOD} = \mathbb{E}_{e \sim \mathcal{E}_{train}}[\mathcal{R}_e(f)] - \mathbb{E}_{e' \sim \mathcal{E}_{test}}[\mathcal{R}_{e'}(f)]
    where \mathcal{R}_e = 1 - \text{ROC-AUC}_e for binary tasks, or 1 - \text{Acc}_e.
    """
    
    def __init__(self):
        pass

    def compute_accuracy(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Computes classification accuracy."""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = torch.argmax(y_pred, dim=1).cpu().numpy()
        return accuracy_score(y_true_np, y_pred_np)

    def compute_roc_auc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Computes ROC-AUC for binary classification."""
        y_true_np = y_true.cpu().numpy()
        if y_pred.dim() == 2 and y_pred.size(1) == 2:
            y_score = torch.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
        else:
            y_score = y_pred.cpu().numpy()
        return roc_auc_score(y_true_np, y_score)

    def compute_risk(self, y_true: torch.Tensor, y_pred: torch.Tensor, metric: str = 'auc') -> float:
        """Computes the risk \mathcal{R}_e = 1 - \text{metric}."""
        if metric == 'auc':
            return 1.0 - self.compute_roc_auc(y_true, y_pred)
        elif metric == 'acc':
            return 1.0 - self.compute_accuracy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_ood_gap(self, train_risks: List[float], test_risks: List[float]) -> float:
        """Computes the OOD generalization gap."""
        return np.mean(train_risks) - np.mean(test_risks)

    def evaluate_split(self, model: torch.nn.Module, dataloader, device: torch.device, metric: str = 'auc') -> Dict[str, float]:
        """Evaluates the model on a specific data split."""
        model.eval()
        all_y_true, all_y_pred = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                logits, _, _, _ = model(batch)
                all_y_true.append(batch.y)
                all_y_pred.append(logits)
                
        y_true = torch.cat(all_y_true, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        risk = self.compute_risk(y_true, y_pred, metric)
        
        return {'risk': risk, metric: 1.0 - risk}
