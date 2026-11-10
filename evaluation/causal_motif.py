import torch
from typing import Tuple, Dict

class CausalMotifEvaluator:
    """
    Evaluates causal motif identification using ground-truth masks.
    
    Mathematical Context:
    Causal Precision: Prec_{causal} = \frac{\sum_e M_{gt}(e) \cdot M_{topo}(e)}{\sum_e M_{topo}(e) + \epsilon}
    Causal Recall: Rec_{causal} = \frac{\sum_e M_{gt}(e) \cdot M_{topo}(e)}{\sum_e M_{gt}(e) + \epsilon}
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def extract_mutag_ground_truth(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Extracts the ground-truth causal mask for MUTAG (-NO2 and -NH2 groups)."""
        atom_types = torch.argmax(x, dim=1) if x.dim() == 2 and x.size(1) > 1 else x.squeeze().long()
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.long))
        
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        for i in range(num_nodes):
            if atom_types[i] == 1: # Nitrogen
                neighbors = col[row == i]
                neighbor_types = atom_types[neighbors]
                num_oxygens = (neighbor_types == 2).sum().item()
                if num_oxygens >= 2: # -NO2
                    mask[i] = True
                    mask[neighbors[neighbor_types == 2]] = True
                elif deg[i] == 3 and num_oxygens == 0: # -NH2
                    mask[i] = True
        return mask.float()

    def compute_precision_recall(self, M_gt: torch.Tensor, M_topo: torch.Tensor) -> Tuple[float, float]:
        """Computes Causal Precision and Recall."""
        intersection = (M_gt * M_topo).sum().item()
        precision = intersection / (M_topo.sum().item() + self.epsilon)
        recall = intersection / (M_gt.sum().item() + self.epsilon)
        return precision, recall


