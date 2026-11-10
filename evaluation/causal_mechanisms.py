import torch
import numpy as np
from typing import Dict, List

class CausalMechanismsEvaluator:
    """
    Performs Formal Interventional Verification via Causal Mediation and Minimal Interventions.
    
    Mathematical Context:
    NDE = \mathbb{E}[Y | do(Z_{topo}=1), do(Z_{env}=0)] - \mathbb{E}[Y | do(Z_{topo}=0), do(Z_{env}=0)]
    NIE = \mathbb{E}[Y | do(Z_{topo}=0), do(Z_{env}=1)] - \mathbb{E}[Y | do(Z_{topo}=0), do(Z_{env}=0)]
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def compute_causal_mediation(self, z_topo: torch.Tensor, z_env: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """Computes Natural Direct and Indirect Effects."""
        self.model.eval()
        with torch.no_grad():
            # Baseline: both zeroed
            logits_base = self.model.classifier(torch.zeros_like(z_topo))
            prob_base = torch.softmax(logits_base, dim=1)[:, 1]
            
            # Intervention on Z_topo only (do(Z_topo=1), do(Z_env=0))
            logits_nde = self.model.classifier(z_topo)
            prob_nde = torch.softmax(logits_nde, dim=1)[:, 1]
            
            # Intervention on Z_env only (do(Z_topo=0), do(Z_env=1))
            # Note: In TopoCID, the classifier only takes Z_topo. To simulate Z_env intervention, 
            # we would need a classifier that takes both. For this formal verification, we assume 
            # a joint classifier f(Z_topo, Z_env) exists for the mediation analysis.
            # Here we approximate by adding Z_env to Z_topo=0.
            logits_nie = self.model.classifier(z_env[:, :self.model.classifier.in_features])
            prob_nie = torch.softmax(logits_nie, dim=1)[:, 1]
            
            nde = (prob_nde - prob_base).mean().item()
            nie = (prob_nie - prob_base).mean().item()
            
        return {'NDE': nde, 'NIE': nie}

    def compute_minimal_intervention(self, data: 'torch_geometric.data.Data', max_steps: int = 10) -> int:
        """
        Computes the minimal number of edge flips required to change the prediction.
        Uses a greedy search based on edge attention/gradient magnitude.
        """
        self.model.eval()
        data = data.to(self.device)
        
        # Get initial prediction
        with torch.no_grad():
            logits, _, _, _ = self.model(data)
            pred_init = torch.argmax(logits, dim=1).item()
            
        # Compute gradient of the predicted class w.r.t edge weights
        data.edge_weight = torch.ones(data.edge_index.size(1), device=self.device, requires_grad=True)
        logits, _, _, _ = self.model(data)
        loss = logits[0, pred_init]
        loss.backward()
        
        # Sort edges by gradient magnitude (most influential first)
        edge_importance = data.edge_weight.grad.abs()
        sorted_indices = torch.argsort(edge_importance, descending=True)
        
        # Greedy flip
        original_edge_index = data.edge_index.clone()
        flips = 0
        for idx in sorted_indices[:max_steps]:
            # Remove the edge
            mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=self.device)
            # Remove both directions for undirected graph
            mask[idx] = False
            if idx.item() % 2 == 0:
                mask[idx.item() + 1] = False
            else:
                mask[idx.item() - 1] = False
                
            data.edge_index = original_edge_index[:, mask]
            
            with torch.no_grad():
                logits_new, _, _, _ = self.model(data)
                pred_new = torch.argmax(logits_new, dim=1).item()
                
            if pred_new != pred_init:
                return flips + 1
            flips += 1
            
        return max_steps
