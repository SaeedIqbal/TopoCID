import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# Add parent directory to path to import backbones and data loaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.backbones.gin import GINBackbone


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


class GatingFunction(nn.Module):
    """
    Learnable polynomial gating function f_phi(p) to suppress topological noise.
    
    Mathematical Context (TopoCID - Section 4.1):
    Ensures that features with low persistence (transient environmental noise) 
    are smoothly driven toward zero:
        f_phi(p) = sigmoid( sum_{m=1}^M phi_m p^m )
    """
    
    def __init__(self, max_degree: int = 3):
        super().__init__()
        self.max_degree = max_degree
        # Learnable coefficients phi_m
        self.phi = nn.Parameter(torch.ones(max_degree))
        apply_deterministic_init(self)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p (torch.Tensor): Persistence values of shape (N, 1).
        Returns:
            torch.Tensor: Gated values of shape (N, 1).
        """
        # Compute powers p^m for m = 1 to M
        p_powers = torch.cat([p**m for m in range(1, self.max_degree + 1)], dim=1) # (N, M)
        # Polynomial sum
        poly_sum = torch.sum(self.phi * p_powers, dim=1, keepdim=True) # (N, 1)
        return torch.sigmoid(poly_sum)


class AnisotropicKernel(nn.Module):
    """
    Learnable anisotropic Gaussian kernel kappa_theta(b, p).
    
    Mathematical Context (TopoCID - Section 4.1):
    Maps the persistence domain into a fixed-dimensional Euclidean space R^D:
        kappa_theta(b, p) = sum_{j=1}^D alpha_j exp( -0.5 * [b-mu1, p-mu2]^T Sigma_j^-1 [b-mu1, p-mu2] )
    """
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Learnable spatial centers mu_j in R^2
        self.mu = nn.Parameter(torch.linspace(-1.0, 1.0, dim * 2).view(dim, 2))
        
        # Learnable covariance matrices Sigma_j in S_{++}^2
        # Parameterized via Cholesky decomposition L_j to ensure positive definiteness
        self.L = nn.Parameter(torch.stack([torch.eye(2) for _ in range(dim)])) # (D, 2, 2)
        
        # Learnable mixture weights alpha_j
        self.alpha = nn.Parameter(torch.ones(dim) / dim)
        
        apply_deterministic_init(self)

    def forward(self, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            b (torch.Tensor): Birth times of shape (N, 1).
            p (torch.Tensor): Persistence values of shape (N, 1).
        Returns:
            torch.Tensor: Kernel values of shape (N, D).
        """
        # Concatenate birth and persistence: [b, p]
        bp = torch.cat([b, p], dim=1) # (N, 2)
        
        # Ensure Cholesky factors have positive diagonal elements
        L = self.L.clone()
        L[:, 0, 0] = F.softplus(L[:, 0, 0])
        L[:, 1, 1] = F.softplus(L[:, 1, 1])
        
        # Compute Sigma_j = L_j L_j^T + epsilon * I for numerical stability
        Sigma = torch.bmm(L, L.transpose(1, 2)) + 1e-5 * torch.eye(2, device=L.device).unsqueeze(0)
        Sigma_inv = torch.inverse(Sigma) # (D, 2, 2)
        
        # Compute difference vectors: [b - mu1, p - mu2]
        diff = bp.unsqueeze(1) - self.mu.unsqueeze(0) # (N, D, 2)
        
        # Compute Mahalanobis distance: diff^T Sigma^-1 diff
        dist = torch.einsum('ndi,nij,ndj->nd', diff, Sigma_inv, diff) # (N, D)
        
        # Compute Gaussian exponentials
        exp_term = torch.exp(-0.5 * dist) # (N, D)
        
        # Weight by mixture coefficients alpha_j
        out = self.alpha.unsqueeze(0) * exp_term # (N, D)
        return out


class DifferentiablePersistenceDiagram(nn.Module):
    """
    Computes a differentiable proxy for the persistence diagram Dgm_k(G).
    
    Mathematical Context (TopoCID - Section 4.1):
    Given the node-weight function w(v), we construct a nested sublevel filtration.
    For k=0 (connected components), birth times are node weights, and death times 
    are approximated by the minimum connected edge weight.
    For k=1 (cycles), birth times are approximated by edge weights.
    The diagram is reparameterized in birth-persistence coordinates: p_i = d_i - b_i.
    """
    
    def __init__(self, max_persistence: float = 5.0):
        super().__init__()
        self.max_persistence = max_persistence

    def forward(self, w: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> tuple:
        """
        Args:
            w (torch.Tensor): Node filtration weights of shape (N, 1).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            num_nodes (int): Total number of nodes.
        Returns:
            tuple: (b_0, p_0, b_1, p_1) for k=0 and k=1.
        """
        device = w.device
        
        if edge_index.size(1) == 0:
            b_0 = w
            p_0 = torch.zeros_like(w)
            b_1 = torch.zeros((0, 1), device=device)
            p_1 = torch.zeros((0, 1), device=device)
            return b_0, p_0, b_1, p_1

        row, col = edge_index
        # Edge weights in upper-star filtration: w_{uv} = max(w_u, w_v)
        edge_weights = torch.maximum(w[row], w[col]) # (E, 1)
        
        # --- k=0: Connected Components ---
        # Birth at node weight, death at min connected edge weight
        min_edge_weight = torch.full((num_nodes, 1), self.max_persistence, device=device)
        min_edge_weight.scatter_reduce_(0, row.unsqueeze(1).expand_as(edge_weights), edge_weights, reduce='amin', include_self=False)
        min_edge_weight.scatter_reduce_(0, col.unsqueeze(1).expand_as(edge_weights), edge_weights, reduce='amin', include_self=False)
        
        b_0 = w # (N, 1)
        d_0 = min_edge_weight # (N, 1)
        p_0 = F.relu(d_0 - b_0) # (N, 1)
        
        # --- k=1: Cycles ---
        # Proxy: birth at edge weight, persistence decays as weight increases
        b_1 = edge_weights # (E, 1)
        p_1 = F.relu(self.max_persistence - b_1) # (E, 1)
        
        return b_0, p_0, b_1, p_1


class TCPModule(nn.Module):
    """
    Differentiable Topological Causal Projection (TCP) Module.
    
    Mathematical Context (TopoCID - Section 4.1):
    Extracts global topological invariants by computing the Causal Topological Manifold:
        Z_topo = Phi(G) = ||_{k=0}^K Z_topo^{(k)}
    where Z_topo^{(k)} = sum_{(b,p) in Dgm_k(G)} f_phi(p) * kappa_theta(b, p)
    """
    
    def __init__(self, hidden_dim: int, topo_dim: int = 64, max_persistence: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topo_dim = topo_dim
        
        # Learnable scalar node-weight function w(v) = u^T MLP(H^{(L)}_v)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.persistence = DifferentiablePersistenceDiagram(max_persistence)
        self.gating = GatingFunction(max_degree=3)
        self.kernel = AnisotropicKernel(dim=topo_dim)
        
        apply_deterministic_init(self)

    def forward(self, node_embs: torch.Tensor, edge_index: torch.Tensor, 
                num_nodes: int, batch: torch.Tensor) -> torch.Tensor:
        """
        Computes the Causal Topological Manifold Z_topo.
        
        Args:
            node_embs (torch.Tensor): Node embeddings H^{(L)} from the backbone.
            edge_index (torch.Tensor): Edge indices.
            num_nodes (int): Total number of nodes in the batch.
            batch (torch.Tensor): Batch vector mapping nodes to graphs.
        Returns:
            torch.Tensor: Z_topo of shape (B, 2 * topo_dim).
        """
        # 1. Compute filtration weights w(v)
        w = self.mlp(node_embs) # (N, 1)
        
        # 2. Compute persistence diagrams for k=0 and k=1
        b_0, p_0, b_1, p_1 = self.persistence(w, edge_index, num_nodes)
        
        # 3. Vectorization for k=0 (Nodes)
        f_phi_0 = self.gating(p_0) # (N, 1)
        kappa_0 = self.kernel(b_0, p_0) # (N, D)
        z_topo_0_per_node = f_phi_0 * kappa_0 # (N, D)
        
        # 4. Vectorization for k=1 (Edges)
        f_phi_1 = self.gating(p_1) # (E, 1)
        kappa_1 = self.kernel(b_1, p_1) # (E, D)
        z_topo_1_per_edge = f_phi_1 * kappa_1 # (E, D)
        
        # 5. Aggregate to graph level via sum pooling
        num_graphs = batch.max().item() + 1
        device = node_embs.device
        
        # Aggregate k=0 over nodes
        z_topo_0 = torch.zeros((num_graphs, self.topo_dim), device=device)
        z_topo_0.scatter_add_(0, batch.unsqueeze(1).expand_as(z_topo_0_per_node), z_topo_0_per_node)
        
        # Aggregate k=1 over edges
        edge_batch = batch[edge_index[0]]
        z_topo_1 = torch.zeros((num_graphs, self.topo_dim), device=device)
        z_topo_1.scatter_add_(0, edge_batch.unsqueeze(1).expand_as(z_topo_1_per_edge), z_topo_1_per_edge)
        
        # 6. Concatenate k=0 and k=1 to form Z_topo
        z_topo = torch.cat([z_topo_0, z_topo_1], dim=1) # (B, 2*D)
        
        return z_topo


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID Differentiable Topological Causal Projection (TCP)")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}, Nodes: {batch.x.size(0)}, Edges: {batch.edge_index.size(1)}")
    
    # 2. Initialize GIN Backbone to get node embeddings H^{(L)}
    print("\n--- Initializing GIN Backbone ---")
    num_node_features = batch.x.size(1)
    hidden_dim = 64
    backbone = GINBackbone(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=3)
    apply_deterministic_init(backbone)
    
    # 3. Initialize TCP Module
    print("\n--- Initializing TCP Module ---")
    topo_dim = 64
    tcp = TCPModule(hidden_dim=hidden_dim, topo_dim=topo_dim)
    
    # 4. Forward Pass
    print("\n--- Executing Forward Pass ---")
    backbone.eval()
    tcp.eval()
    
    with torch.no_grad():
        node_embs, _ = backbone(batch)
        z_topo = tcp(node_embs, batch.edge_index, batch.x.size(0), batch.batch)
        
    print(f"Node Embeddings H^{{(L)}} Shape: {node_embs.shape}")
    print(f"Causal Topological Manifold Z_{{topo}} Shape: {z_topo.shape}")
    print(f"Z_{{topo}} Mean: {z_topo.mean().item():.4f}, Std: {z_topo.std().item():.4f}")
    
    # 5. Backward Pass Verification
    print("\n--- Verifying Differentiability (Backward Pass) ---")
    tcp.train()
    backbone.train()
    
    node_embs, _ = backbone(batch)
    z_topo = tcp(node_embs, batch.edge_index, batch.x.size(0), batch.batch)
    
    # Create a dummy loss to compute gradients
    dummy_loss = z_topo.sum()
    dummy_loss.backward()
    
    grad_norm = 0.0
    for p in tcp.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Total TCP Gradient Norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("TCP Module Verification Complete. No synthetic/random data used.")
    print("=" * 60)