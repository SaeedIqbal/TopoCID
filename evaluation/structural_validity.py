import torch
import torch.nn.functional as F
from typing import Dict

class StructuralValidityEvaluator:
    """
    Evaluates the Structural Validity Rate (SVR) of generated counterfactuals.
    
    Mathematical Context:
    SVR = \frac{1}{N_{cf}} \sum_{k=1}^{N_{cf}} \mathbb{I}[C(G_{cf}^{(k)}) = 1]
    where C(G) enforces chemical valency rules: \sum_j A_{ij} \le v_{type(i)}.
    """
    
    def __init__(self, max_valency: float = 4.0):
        self.max_valency = max_valency

    def check_chemical_valency(self, A_cf: torch.Tensor) -> torch.Tensor:
        """Checks if the continuous adjacency matrix satisfies valency constraints."""
        degrees = A_cf.sum(dim=-1) # (B, N)
        violations = F.relu(degrees - self.max_valency)
        return (violations == 0).all(dim=-1) # (B,) boolean tensor

    def compute_svr(self, A_cf_batch: torch.Tensor) -> float:
        """Computes the Structural Validity Rate for a batch of counterfactuals."""
        valid_mask = self.check_chemical_valency(A_cf_batch)
        return valid_mask.float().mean().item()
