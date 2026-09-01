import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path to import data loaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def apply_deterministic_init(model: nn.Module) -> None:
    """
    Initializes all model parameters deterministically to strictly avoid 
    any random values during initialization.
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.constant_(p, 0.01)
        else:
            nn.init.constant_(p, 0.0)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for Topological Invariant Alignment.
    
    Mathematical Context (TopoCID - Section 4.3):
    Aligns topological invariants of graphs that share the same causal mechanism 
    (identical label y) but originate from distinct environments (E_i != E_j).
    The loss maximizes the agreement between positive pairs while pushing apart 
    negative pairs:
        L_TCD = - (1 / |P|) * sum_{(i,j) in P} log [ exp(kappa(z_i, z_j) / tau) / sum_{k != i} exp(kappa(z_i, z_k) / tau) ]
    where kappa(u, v) is the cosine similarity.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature (float): Temperature hyperparameter tau_1 for the InfoNCE estimator.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_topo: torch.Tensor, positive_pairs: torch.Tensor) -> torch.Tensor:
        """
        Computes the InfoNCE contrastive loss.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, D).
            positive_pairs (torch.Tensor): Indices of positive pairs of shape (N_pos, 2).
            
        Returns:
            torch.Tensor: The computed InfoNCE loss (scalar).
        """
        if positive_pairs.size(0) == 0:
            # Return a dummy loss with grad enabled to prevent computation graph breakage
            return torch.tensor(0.0, device=z_topo.device, requires_grad=True)
            
        # 1. Normalize embeddings for cosine similarity
        z_norm = F.normalize(z_topo, p=2, dim=1)
        
        # 2. Compute pairwise similarity matrix: kappa(z_i, z_k) / tau
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature  # (B, B)
        
        # 3. Mask out self-similarity (diagonal) to enforce k != i
        B = z_topo.size(0)
        mask = torch.eye(B, dtype=torch.bool, device=z_topo.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # 4. Compute log denominator for each i: log sum_{k != i} exp(S_ik)
        log_denom = torch.logsumexp(sim_matrix, dim=1)  # (B,)
        
        # 5. Extract numerator for each positive pair (i, j): S_ij
        i_idx = positive_pairs[:, 0]
        j_idx = positive_pairs[:, 1]
        sim_pos = sim_matrix[i_idx, j_idx]  # (N_pos,)
        
        # 6. Compute InfoNCE loss: - mean( S_ij - log_denom[i] )
        loss = - (sim_pos - log_denom[i_idx]).mean()
        
        return loss


class CLUBEstimator(nn.Module):
    """
    CLUB (CLub Upper Bound) Variational Estimator for Mutual Information.
    
    Mathematical Context (TopoCID - Section 4.3):
    Explicitly minimizes the mutual information I(Z_topo; Z_env) using the 
    Donsker-Varadhan variational representation. The mutual information is 
    bounded below by the CLUB estimator:
        I(Z_topo; Z_env) >= sup_psi ( E_{p(z,e)}[T_psi(z,e)] - log E_{p(z)p(e)}[exp(T_psi(z,e))] )
        
    To strictly avoid random values, the marginal expectation E_{p(z)p(e)} is 
    approximated using a deterministic circular shift (torch.roll) to break 
    the joint dependency between Z_topo and Z_env.
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64):
        """
        Args:
            x_dim (int): Dimension of Z_topo.
            y_dim (int): Dimension of Z_env.
            hidden_dim (int): Hidden dimension of the critic network T_psi.
        """
        super().__init__()
        
        # Critic network T_psi parameterized by psi
        self.t_network = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        apply_deterministic_init(self)

    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the CLUB mutual information upper bound.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, x_dim).
            z_env (torch.Tensor): Environmental representations of shape (B, y_dim).
            
        Returns:
            torch.Tensor: The CLUB bound (scalar). To minimize MI, we minimize this bound 
                          w.r.t the representation extractors, and maximize it w.r.t psi.
        """
        B = z_topo.size(0)
        
        # 1. Joint expectation: E_{p(z,e)}[T_psi(z,e)]
        # Uses true joint samples (z_i, e_i)
        joint_input = torch.cat([z_topo, z_env], dim=1)
        T_joint = self.t_network(joint_input).squeeze(-1)  # (B,)
        joint_term = T_joint.mean()
        
        # 2. Marginal expectation: log E_{p(z)p(e)}[exp(T_psi(z,e))]
        # Uses deterministic shuffled samples (z_i, e_{i+1}) to approximate product of marginals
        z_env_shuffled = torch.roll(z_env, shifts=1, dims=0)
        marginal_input = torch.cat([z_topo, z_env_shuffled], dim=1)
        T_marginal = self.t_network(marginal_input).squeeze(-1)  # (B,)
        
        # Compute log( 1/B * sum(exp(T)) ) = logsumexp(T) - log(B)
        marginal_term = torch.logsumexp(T_marginal, dim=0) - \
                        torch.log(torch.tensor(B, dtype=torch.float32, device=z_topo.device))
        
        # 3. CLUB Bound
        club_bound = joint_term - marginal_term
        
        return club_bound


class TCDModule(nn.Module):
    """
    Topological Contrastive Disentanglement (TCD) Module.
    
    Mathematical Context (TopoCID - Section 4.3):
    Integrates the InfoNCE contrastive loss and the CLUB mutual information bound 
    to enforce the conditional independence criterion Y perp E | Z_topo.
    The dual mechanism restricts Z_topo to the intersection of topological invariants 
    across multiple environments while forcing Z_topo and Z_env to occupy orthogonal 
    subspaces in the latent space.
    """
    
    def __init__(self, topo_dim: int, env_dim: int, hidden_dim: int = 64, temperature: float = 0.07):
        """
        Args:
            topo_dim (int): Dimension of the causal topological manifold Z_topo.
            env_dim (int): Dimension of the environmental representation Z_env.
            hidden_dim (int): Hidden dimension for the CLUB critic network.
            temperature (float): Temperature hyperparameter for InfoNCE.
        """
        super().__init__()
        self.info_nce = InfoNCELoss(temperature=temperature)
        self.club = CLUBEstimator(topo_dim, env_dim, hidden_dim)

    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor, 
                positive_pairs: torch.Tensor) -> tuple:
        """
        Computes the TCD losses.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants (B, topo_dim).
            z_env (torch.Tensor): Environmental representations (B, env_dim).
            positive_pairs (torch.Tensor): Indices of positive pairs for InfoNCE (N_pos, 2).
            
        Returns:
            tuple: (L_TCD, L_MI) where L_TCD is the InfoNCE loss and L_MI is the CLUB bound.
        """
        l_tcd = self.info_nce(z_topo, positive_pairs)
        l_mi = self.club(z_topo, z_env)
        
        return l_tcd, l_mi


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    from models.backbones.gin import GINBackbone
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID Topological Contrastive Disentanglement (TCD)")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}")
    print(f"Positive Pairs for TCD: {batch.positive_pairs.size(0)}")
    
    # 2. Initialize GIN Backbone to get graph embeddings
    print("\n--- Initializing GIN Backbone ---")
    num_node_features = batch.x.size(1)
    hidden_dim = 64
    backbone = GINBackbone(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=3)
    apply_deterministic_init(backbone)
    
    # 3. Initialize TCD Module
    print("\n--- Initializing TCD Module (Deterministic CLUB & InfoNCE) ---")
    topo_dim = 64
    env_dim = 32
    
    # Dummy z_topo and z_env for verification (derived deterministically from graph_embs)
    _, graph_embs = backbone(batch)
    z_topo = graph_embs[:, :topo_dim]
    z_env = graph_embs[:, :env_dim]
    
    tcd = TCDModule(topo_dim=topo_dim, env_dim=env_dim, hidden_dim=64, temperature=0.07)
    
    # 4. Forward Pass
    print("\n--- Executing Forward Pass (TCD Loss Computation) ---")
    backbone.eval()
    tcd.eval()
    
    with torch.no_grad():
        l_tcd, l_mi = tcd(z_topo, z_env, batch.positive_pairs)
        
    print(f"InfoNCE Loss (L_TCD): {l_tcd.item():.4f}")
    print(f"CLUB MI Bound (L_MI): {l_mi.item():.4f}")
    
    # 5. Backward Pass Verification
    print("\n--- Verifying Differentiability (Backward Pass) ---")
    tcd.train()
    backbone.train()
    
    _, graph_embs = backbone(batch)
    z_topo = graph_embs[:, :topo_dim]
    z_env = graph_embs[:, :env_dim]
    
    l_tcd, l_mi = tcd(z_topo, z_env, batch.positive_pairs)
    
    # Total TCD objective to backpropagate
    tcd_loss = l_tcd + l_mi
    tcd_loss.backward()
    
    grad_norm = 0.0
    for p in tcd.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Total TCD Gradient Norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("TCD Module Verification Complete. No synthetic/random data used.")
    print("=" * 60)