import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path to import models and data loaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloaders import TopoCIDDataModule
from models.topocid.topocid import TopoCID


class SupervisedLoss(nn.Module):
    """
    Supervised Classification Loss.
    
    Mathematical Context (TopoCID - Section 4.4):
    Ensures the predictive accuracy of the topological invariant Z_topo:
        L_sup = - (1/|D|) sum_{(G, Y) in D} Y^T log f_phi(Phi(G))
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the supervised cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (B, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (B,).
            
        Returns:
            torch.Tensor: Scalar supervised loss.
        """
        return self.ce_loss(logits, targets.view(-1).long())


class CounterfactualLoss(nn.Module):
    """
    Counterfactual Intervention Loss.
    
    Mathematical Context (TopoCID - Section 4.4):
    Enforces that the classifier relies exclusively on the invariant topology 
    by evaluating predictions on the structure-preserving counterfactuals G_cf:
        L_cf = - (1/|B|) sum_{i=1}^{|B|} y_i^T log f_phi(Phi(G_cf^{(i,j)}))
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits_cf: torch.Tensor, targets_i: torch.Tensor) -> torch.Tensor:
        """
        Computes the counterfactual cross-entropy loss.
        
        Args:
            logits_cf (torch.Tensor): Predicted logits on counterfactuals, shape (B, num_classes).
            targets_i (torch.Tensor): Ground truth labels of the source graphs, shape (B,).
            
        Returns:
            torch.Tensor: Scalar counterfactual loss.
        """
        return self.ce_loss(logits_cf, targets_i.view(-1).long())


class TopologicalContrastiveLoss(nn.Module):
    """
    Topological Contrastive Disentanglement Loss (InfoNCE).
    
    Mathematical Context (TopoCID - Section 4.4):
    Aligns topological invariants across distinct environments:
        L_TCD = - (1/|P|) sum_{(i,j) in P} log [ exp(kappa(z_i, z_j) / tau) / sum_{k != i} exp(kappa(z_i, z_k) / tau) ]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_topo: torch.Tensor, positive_pairs: torch.Tensor) -> torch.Tensor:
        """
        Computes the InfoNCE contrastive loss.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants, shape (B, D).
            positive_pairs (torch.Tensor): Indices of positive pairs, shape (N_pos, 2).
            
        Returns:
            torch.Tensor: Scalar InfoNCE loss.
        """
        if positive_pairs.size(0) == 0:
            return torch.tensor(0.0, device=z_topo.device, requires_grad=True)
            
        # Normalize for cosine similarity
        z_norm = F.normalize(z_topo, p=2, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        
        # Mask out self-similarity
        B = z_topo.size(0)
        mask = torch.eye(B, dtype=torch.bool, device=z_topo.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Log denominator
        log_denom = torch.logsumexp(sim_matrix, dim=1)
        
        # Extract positive pairs
        i_idx = positive_pairs[:, 0]
        j_idx = positive_pairs[:, 1]
        sim_pos = sim_matrix[i_idx, j_idx]
        
        loss = - (sim_pos - log_denom[i_idx]).mean()
        return loss


class MutualInformationLoss(nn.Module):
    """
    Mutual Information Minimization Loss (CLUB Bound).
    
    Mathematical Context (TopoCID - Section 4.4):
    Enforces the independence constraint between Z_topo and Z_env via the 
    Donsker-Varadhan variational bound:
        L_MI = E_{p(z,e)}[T_psi(z,e)] - log E_{p(z)p(e)}[exp(T_psi(z,e))]
    """
    def __init__(self, club_estimator: nn.Module):
        super().__init__()
        self.club = club_estimator
        
    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the CLUB mutual information upper bound.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants, shape (B, D_topo).
            z_env (torch.Tensor): Environmental representations, shape (B, D_env).
            
        Returns:
            torch.Tensor: Scalar CLUB MI bound.
        """
        return self.club(z_topo, z_env)


class DenoisingScoreMatchingLoss(nn.Module):
    """
    Denoising Score Matching Loss for the Diffusion Model.
    
    Mathematical Context (TopoCID - Section 4.2):
    Trains the score network s_theta to approximate the gradient of the log-probability:
        L_DSM = E_{t, G_0, G_t} [ || s_theta(G_t, t, z^i, e^j) - nabla_{G_t} log p_{0t}(G_t | G_0) ||_2^2 ]
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, score_pred: torch.Tensor, score_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Denoising Score Matching loss.
        
        Args:
            score_pred (torch.Tensor): Predicted score, shape (B, N, N).
            score_target (torch.Tensor): Target score (gradient of log p_{0t}), shape (B, N, N).
            
        Returns:
            torch.Tensor: Scalar DSM loss.
        """
        return self.mse_loss(score_pred, score_target)


class TopoCIDLossAggregator(nn.Module):
    """
    Aggregates all loss components into the unified TopoCID objective.
    
    Mathematical Context (TopoCID - Section 4.4):
        L_total = L_sup + lambda_cf * L_cf + lambda_TCD * L_TCD + lambda_MI * L_MI
    """
    def __init__(self, lambda_cf: float = 1.0, lambda_tcd: float = 0.5, lambda_mi: float = 0.1):
        super().__init__()
        self.lambda_cf = lambda_cf
        self.lambda_tcd = lambda_tcd
        self.lambda_mi = lambda_mi
        
        self.l_sup = SupervisedLoss()
        self.l_cf = CounterfactualLoss()
        self.l_tcd = TopologicalContrastiveLoss()
        self.l_dsm = DenoisingScoreMatchingLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                logits_cf: torch.Tensor, targets_cf: torch.Tensor,
                z_topo: torch.Tensor, positive_pairs: torch.Tensor,
                club_estimator: nn.Module, z_env: torch.Tensor) -> dict:
        """
        Computes all loss components and the total loss.
        
        Args:
            logits (torch.Tensor): Predicted logits on original graphs.
            targets (torch.Tensor): Ground truth labels for original graphs.
            logits_cf (torch.Tensor): Predicted logits on counterfactual graphs.
            targets_cf (torch.Tensor): Ground truth labels for source graphs of counterfactuals.
            z_topo (torch.Tensor): Causal topological invariants.
            positive_pairs (torch.Tensor): Indices of positive pairs for InfoNCE.
            club_estimator (nn.Module): The CLUB critic network.
            z_env (torch.Tensor): Environmental representations.
            
        Returns:
            dict: Dictionary containing all individual losses and the total loss.
        """
        # 1. Supervised Loss
        loss_sup = self.l_sup(logits, targets)
        
        # 2. Counterfactual Loss
        loss_cf = self.l_cf(logits_cf, targets_cf)
        
        # 3. Topological Contrastive Loss
        loss_tcd = self.l_tcd(z_topo, positive_pairs)
        
        # 4. Mutual Information Loss
        mi_loss_module = MutualInformationLoss(club_estimator)
        loss_mi = mi_loss_module(z_topo, z_env)
        
        # 5. Total Loss
        loss_total = loss_sup + self.lambda_cf * loss_cf + self.lambda_tcd * loss_tcd + self.lambda_mi * loss_mi
        
        return {
            'total': loss_total,
            'sup': loss_sup,
            'cf': loss_cf,
            'tcd': loss_tcd,
            'mi': loss_mi
        }


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    DATASET_ROOT = "/home/phd/datasets/"
    device = torch.device('cpu')
    
    print("=" * 60)
    print("Initializing TopoCID Loss Functions")
    print("=" * 60)
    
    # 1. Load original dataset
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    batch = batch.to(device)
    
    num_node_features = batch.x.size(1)
    
    # 2. Initialize Model
    print("\n--- Initializing TopoCID Model ---")
    model = TopoCID(
        num_node_features=num_node_features, 
        hidden_dim=64, 
        topo_dim=64, 
        env_dim=32, 
        num_classes=2
    ).to(device)
    
    # 3. Forward Pass to get representations
    print("\n--- Executing Forward Pass ---")
    logits, z_topo, z_env, node_embs = model(batch)
    
    # Generate counterfactuals
    Z_cf, idx_i, idx_j, _ = model.spcg(z_topo, z_env, node_embs, batch.edge_index, batch.batch)
    logits_cf = model.classifier(Z_cf)
    
    # 4. Compute Losses
    print("\n--- Computing Loss Components ---")
    loss_aggregator = TopoCIDLossAggregator(
        lambda_cf=1.0, 
        lambda_tcd=0.5, 
        lambda_mi=0.1
    ).to(device)
    
    loss_dict = loss_aggregator(
        logits=logits, 
        targets=batch.y,
        logits_cf=logits_cf, 
        targets_cf=batch.y[idx_i],
        z_topo=z_topo, 
        positive_pairs=batch.positive_pairs,
        club_estimator=model.tcd.club, 
        z_env=z_env
    )
    
    print(f"L_sup (Supervised): {loss_dict['sup'].item():.4f}")
    print(f"L_cf (Counterfactual): {loss_dict['cf'].item():.4f}")
    print(f"L_tcd (InfoNCE): {loss_dict['tcd'].item():.4f}")
    print(f"L_mi (CLUB MI Bound): {loss_dict['mi'].item():.4f}")
    print(f"L_total (Total Loss): {loss_dict['total'].item():.4f}")
    
    # 5. Backward Pass Verification
    print("\n--- Verifying Backward Pass ---")
    loss_dict['total'].backward()
    
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Total Gradient Norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("Loss Functions Verification Complete. No synthetic/random data used.")
    print("=" * 60)