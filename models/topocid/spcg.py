import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj

# Add parent directory to path to import backbones and data loaders
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


class ManifoldProjector(nn.Module):
    """
    Applies the orthogonal projection matrix Pi_M onto the tangent space of the valid manifold M.
    
    Mathematical Context (TopoCID - Section 4.2):
    The valid data manifold M is defined by the constraint C(G) = 0, where 
    C_i(A) = max(0, sum_j A_ij - v_i). 
    The orthogonal projection matrix is Pi_M = I - J_C^T (J_C J_C^T + delta I)^-1 J_C.
    
    For the specific constraint C_i(A) = sum_j A_ij - v_i, the Jacobian J_C has a 
    highly structured form. The action of the projection on a matrix M simplifies 
    computationally to subtracting the row mean of M for all active constraints 
    (nodes where degree > max_valency), avoiding expensive matrix inversions.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, M: torch.Tensor, A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Projects the matrix M onto the tangent space of M.
        
        Args:
            M (torch.Tensor): The matrix to project (e.g., drift or score), shape (B, N, N).
            A (torch.Tensor): Current continuous adjacency matrix, shape (B, N, N).
            v (torch.Tensor): Maximum allowed valency per node, shape (B, N).
            
        Returns:
            torch.Tensor: The projected matrix M_proj, shape (B, N, N).
        """
        # Compute current degrees
        degree = A.sum(dim=-1) # (B, N)
        
        # Identify active constraints (nodes violating valency)
        active_mask = (degree > v).float() # (B, N)
        
        # Compute row sums of M
        row_sums = M.sum(dim=-1) # (B, N)
        
        # Divide by N (number of columns) to compute the mean
        N = M.size(-1)
        correction = row_sums / (N + 1e-8) # (B, N)
        
        # Apply correction only to active rows
        correction = correction * active_mask # (B, N)
        
        # Broadcast correction to all columns and subtract
        M_proj = M - correction.unsqueeze(-1)
        
        return M_proj


class ScoreNetwork(nn.Module):
    """
    Score network s_theta(G_t, t, z^i, e^j) parameterizing the conditional score function.
    
    Mathematical Context (TopoCID - Section 4.2):
    Approximates the gradient of the log-probability of the continuous graph state:
        nabla_{G_t} log p_t(G_t | z^i, e^j)
    """
    
    def __init__(self, max_nodes: int, topo_dim: int, env_dim: int, hidden_dim: int = 128):
        """
        Args:
            max_nodes (int): Maximum number of nodes in the batch (N).
            topo_dim (int): Dimension of the causal topological invariant z^i.
            env_dim (int): Dimension of the environmental context e^j.
            hidden_dim (int): Hidden dimension of the MLP.
        """
        super().__init__()
        self.max_nodes = max_nodes
        
        # Input: flattened adjacency (N*N) + time (1) + z_topo + z_env
        input_dim = max_nodes * max_nodes + 1 + topo_dim + env_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes)
        )
        apply_deterministic_init(self)
        
    def forward(self, A_t: torch.Tensor, t: torch.Tensor, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_t (torch.Tensor): Continuous adjacency matrix, shape (B, N, N).
            t (torch.Tensor): Diffusion time, shape (B, 1).
            z_topo (torch.Tensor): Causal topological invariant, shape (B, topo_dim).
            z_env (torch.Tensor): Environmental context, shape (B, env_dim).
            
        Returns:
            torch.Tensor: The score matrix, shape (B, N, N).
        """
        B, N, _ = A_t.shape
        A_flat = A_t.view(B, -1)
        
        # Concatenate all conditioning inputs
        x = torch.cat([A_flat, t, z_topo, z_env], dim=1)
        
        # Predict score
        score_flat = self.mlp(x)
        score = score_flat.view(B, N, N)
        
        # Enforce symmetry for undirected graphs (A_t must remain symmetric)
        score = (score + score.transpose(1, 2)) / 2.0
        
        return score


class SPCGModule(nn.Module):
    """
    Structure-Preserving Counterfactual Generation (SPCG) Module.
    
    Mathematical Context (TopoCID - Section 4.2):
    Generates valid counterfactual graphs by solving a manifold-constrained reverse 
    Stochastic Differential Equation (SDE). To strictly avoid random values (the 
    Wiener process dW_t), we employ the deterministic Probability Flow ODE counterpart:
        dG_t = [ -0.5*beta(t)*G_t + beta(t)*score - 0.5*beta(t)*n_corr ] dt
        
    The normal restoring drift n_corr = J_C^T C(G_t) enforces the structural 
    constraint C(G) = 0 (e.g., chemical valency rules) by projecting onto the 
    valid manifold M.
    """
    
    def __init__(self, max_nodes: int, topo_dim: int, env_dim: int, 
                 num_steps: int = 50, beta_min: float = 0.1, beta_max: float = 20.0):
        """
        Args:
            max_nodes (int): Maximum number of nodes in the batch (N).
            topo_dim (int): Dimension of the causal topological invariant.
            env_dim (int): Dimension of the environmental context.
            num_steps (int): Number of Euler-Maruyama integration steps.
            beta_min (float): Minimum noise schedule value.
            beta_max (float): Maximum noise schedule value.
        """
        super().__init__()
        self.max_nodes = max_nodes
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        self.score_net = ScoreNetwork(max_nodes, topo_dim, env_dim)
        self.projector = ManifoldProjector()
        
    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise schedule beta(t)."""
        return self.beta_min + 0.5 * (self.beta_max - self.beta_min) * t
        
    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor, 
                node_embs: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        Generates counterfactual graph embeddings Z_cf via Probability Flow ODE.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants (B, topo_dim).
            z_env (torch.Tensor): Environmental representations (B, env_dim).
            node_embs (torch.Tensor): Node embeddings from backbone (N_total, hidden_dim).
            edge_index (torch.Tensor): Sparse edge indices (2, E_total).
            batch (torch.Tensor): Batch vector mapping nodes to graphs (N_total,).
            
        Returns:
            tuple: (Z_cf, idx_i, idx_j, A_cf) where Z_cf is the counterfactual embedding.
        """
        B = z_topo.size(0)
        device = z_topo.device
        N = self.max_nodes
        
        # Create deterministic counterfactual pairs (i, j) with i != j
        idx_i = torch.arange(B, device=device)
        idx_j = (idx_i + 1) % B
        
        z_topo_i = z_topo[idx_i]
        z_env_j = z_env[idx_j]
        
        # 1. Initialize A_t from the original graph (deterministic, no random noise)
        A_0 = to_dense_adj(edge_index, batch=batch, max_num_nodes=N) # (B, N, N)
        A_t = A_0.clone().float()
        
        # Max valency v (assume 4 for all nodes, standard for organic molecules)
        v = torch.full((B, N), 4.0, device=device)
        
        dt = 1.0 / self.num_steps
        
        # 2. Reverse Probability Flow ODE integration
        for step in range(self.num_steps, 0, -1):
            t_val = step / self.num_steps
            t = torch.full((B, 1), t_val, device=device)
            
            beta = self.get_beta(t) # (B, 1)
            
            # Compute unconstrained score
            score = self.score_net(A_t, t, z_topo_i, z_env_j) # (B, N, N)
            
            # Compute unconstrained drift
            drift = -0.5 * beta.unsqueeze(-1) * A_t + beta.unsqueeze(-1) * score
            
            # Project drift onto tangent space of M
            drift_proj = self.projector(drift, A_t, v)
            
            # Compute normal restoring drift n_corr = J_C^T C(G_t)
            degree = A_t.sum(dim=-1) # (B, N)
            C = F.relu(degree - v) # (B, N)
            n_corr = C.unsqueeze(-1) # (B, N, 1) broadcasted to (B, N, N)
            
            # Update A_t
            A_t = A_t + drift_proj * dt - 0.5 * beta.unsqueeze(-1) * n_corr * dt
            
            # Hard projection to [0, 1] to maintain valid probability matrix
            A_t = torch.clamp(A_t, 0.0, 1.0)
            
        # 3. Extract counterfactual graph embedding Z_cf
        # Use the generated A_cf to re-weight the message passing
        H_dense, _ = to_dense_batch(node_embs, batch, max_num_nodes=N) # (B, N, hidden_dim)
        H_cf = torch.bmm(A_t, H_dense) # (B, N, hidden_dim)
        
        # Global sum pooling for Z_cf
        Z_cf = H_cf.sum(dim=1) # (B, hidden_dim)
        
        return Z_cf, idx_i, idx_j, A_t


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    from models.backbones.gin import GINBackbone
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID Structure-Preserving Counterfactual Generation (SPCG)")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}, Nodes: {batch.x.size(0)}, Edges: {batch.edge_index.size(1)}")
    
    # Compute max nodes in the batch for dense representation
    max_nodes = batch.batch.bincount().max().item()
    print(f"Max nodes in batch (N): {max_nodes}")
    
    # 2. Initialize GIN Backbone to get node embeddings H^{(L)}
    print("\n--- Initializing GIN Backbone ---")
    num_node_features = batch.x.size(1)
    hidden_dim = 64
    backbone = GINBackbone(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=3)
    apply_deterministic_init(backbone)
    
    # 3. Initialize SPCG Module
    print("\n--- Initializing SPCG Module (Deterministic Probability Flow ODE) ---")
    topo_dim = 64
    env_dim = 32
    
    # Dummy z_topo and z_env for verification (derived deterministically from graph_embs)
    _, graph_embs = backbone(batch)
    z_topo = graph_embs[:, :topo_dim]
    z_env = graph_embs[:, :env_dim]
    
    spcg = SPCGModule(max_nodes=max_nodes, topo_dim=topo_dim, env_dim=env_dim, num_steps=10)
    
    # 4. Forward Pass
    print("\n--- Executing Forward Pass (Counterfactual Generation) ---")
    backbone.eval()
    spcg.eval()
    
    with torch.no_grad():
        node_embs, _ = backbone(batch)
        Z_cf, idx_i, idx_j, A_cf = spcg(z_topo, z_env, node_embs, batch.edge_index, batch.batch)
        
    print(f"Original Node Embeddings H^{{(L)}} Shape: {node_embs.shape}")
    print(f"Counterfactual Graph Embedding Z_{{cf}} Shape: {Z_cf.shape}")
    print(f"Generated Continuous Adjacency A_{{cf}} Shape: {A_cf.shape}")
    
    # 5. Verify Structural Validity (C(G) = 0)
    print("\n--- Verifying Structural Validity (Valency Constraints) ---")
    degrees_cf = A_cf.sum(dim=-1)
    max_valency = 4.0
    violations = F.relu(degrees_cf - max_valency)
    max_violation = violations.max().item()
    print(f"Maximum Valency Violation in A_{{cf}}: {max_violation:.4f} (Should be close to 0.0)")
    
    # 6. Backward Pass Verification
    print("\n--- Verifying Differentiability (Backward Pass) ---")
    spcg.train()
    backbone.train()
    
    node_embs, _ = backbone(batch)
    Z_cf, _, _, _ = spcg(z_topo, z_env, node_embs, batch.edge_index, batch.batch)
    
    dummy_loss = Z_cf.sum()
    dummy_loss.backward()
    
    grad_norm = 0.0
    for p in spcg.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Total SPCG Gradient Norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("SPCG Module Verification Complete. No synthetic/random data used.")
    print("=" * 60)