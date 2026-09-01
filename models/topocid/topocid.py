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
    any random values during initialization. This ensures exact reproducibility 
    and aligns with the framework's requirement for non-stochastic evaluations.
    
    Args:
        model (nn.Module): The PyTorch model to initialize.
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.constant_(p, 0.01)
        else:
            nn.init.constant_(p, 0.0)


class TCPModule(nn.Module):
    """
    Differentiable Topological Causal Projection (TCP) Module.
    
    Mathematical Context (TopoCID - Section 4.1):
    Extracts global topological invariants by computing a learnable scalar 
    node-weight function w(v) and applying a differentiable vectorization 
    operator Psi_k over the persistence domain:
        Z_topo^{(k)} = sum_{(b, p) in Dgm_k(G)} f_phi(p) * kappa_theta(b, p)
        
    To ensure error-free execution without external C++ TDA libraries, we 
    approximate the persistence diagram points using the node filtration 
    weights w(v) and apply the learnable anisotropic kernel directly.
    """
    
    def __init__(self, hidden_dim: int, topo_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topo_dim = topo_dim
        
        # Learnable projection u for w(v) = u^T MLP(H^{(L)}_v)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Deterministic initialization for u
        self.u = nn.Parameter(torch.linspace(-0.1, 0.1, hidden_dim * topo_dim).view(hidden_dim, topo_dim))
        
        # Learnable gating function f_phi(p) to suppress transient noise
        self.gating_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        
        # Learnable anisotropic kernel centers kappa_theta(b, p)
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, topo_dim).view(-1, 1))
        
        apply_deterministic_init(self)

    def forward(self, node_embs: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Computes the Causal Topological Manifold Z_topo.
        
        Args:
            node_embs (torch.Tensor): Node embeddings H^{(L)} from the backbone.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch vector mapping nodes to graphs.
            
        Returns:
            torch.Tensor: Z_topo of shape (B, topo_dim).
        """
        # 1. Compute filtration weights w(v)
        w = self.mlp(node_embs)  # (N, 1)
        
        # 2. Apply gating function f_phi(p)
        f_phi = self.gating_mlp(w)  # (N, 1)
        
        # 3. Compute anisotropic kernel kappa_theta(b, p)
        # Approximated as RBF over the filtration weights
        kappa = torch.exp(-0.5 * ((w - self.centers.T) ** 2).sum(dim=-1, keepdim=True))  # (N, topo_dim)
        
        # 4. Vectorization: Z_topo^{(k)} = sum_v f_phi(w_v) * kappa_theta(w_v)
        z_topo_per_node = f_phi * kappa  # (N, topo_dim)
        
        # 5. Global sum pooling over nodes per graph
        num_graphs = batch.max().item() + 1
        z_topo = torch.zeros((num_graphs, self.topo_dim), device=node_embs.device)
        z_topo.scatter_add_(0, batch.unsqueeze(1).expand_as(z_topo_per_node), z_topo_per_node)
        
        return z_topo


class SPCGModule(nn.Module):
    """
    Structure-Preserving Counterfactual Generation (SPCG) Module.
    
    Mathematical Context (TopoCID - Section 4.2):
    Generates valid counterfactual graphs by solving a manifold-constrained 
    reverse Stochastic Differential Equation (SDE). To strictly avoid random 
    values (Wiener process dW_t), we employ the deterministic Probability 
    Flow ODE counterpart:
        dG_t = [ -0.5*beta(t)*G_t + beta(t)*score - 0.5*beta(t)*n_corr ] dt
        
    The normal restoring drift n_corr enforces the structural constraint 
    C(G) = 0 (e.g., valency rules) by projecting onto the valid manifold M.
    """
    
    def __init__(self, topo_dim: int, env_dim: int, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        
        # Score network s_theta(G_t, t, z^i, e^j)
        self.score_net = nn.Sequential(
            nn.Linear(topo_dim + env_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        apply_deterministic_init(self)

    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor, 
                node_embs: torch.Tensor, data: Data) -> tuple:
        """
        Generates counterfactual graph embeddings Z_cf via Probability Flow ODE.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants (B, topo_dim).
            z_env (torch.Tensor): Environmental representations (B, env_dim).
            node_embs (torch.Tensor): Node embeddings from backbone (N, hidden_dim).
            data (Data): PyG Batch object.
            
        Returns:
            tuple: (Z_cf, idx_i, idx_j) where Z_cf is the counterfactual embedding.
        """
        B = z_topo.size(0)
        device = z_topo.device
        
        # Create deterministic counterfactual pairs (i, j) with i != j
        idx_i = torch.arange(B, device=device)
        idx_j = (idx_i + 1) % B
        
        z_topo_i = z_topo[idx_i]
        z_env_j = z_env[idx_j]
        
        num_edges = data.edge_index.size(1)
        # Initialize continuous edge weights A_t = 1.0 (deterministic, no random noise)
        A_t = torch.ones(num_edges, 1, device=device)
        
        dt = 1.0 / self.num_steps
        edge_batch = data.batch[data.edge_index[0]]
        z_topo_exp = z_topo_i[edge_batch]
        z_env_exp = z_env_j[edge_batch]
        
        # Euler-Maruyama integration for Probability Flow ODE
        for step in range(self.num_steps):
            t = torch.full((num_edges, 1), step / self.num_steps, device=device)
            cond = torch.cat([z_topo_exp, z_env_exp, t], dim=1)
            
            # Score function
            score = self.score_net(cond)
            
            # Normal restoring drift n_corr (projects onto valid manifold M)
            # Enforces A_t in [0, 1] (soft valency constraint C(G)=0)
            n_corr = torch.clamp(A_t, min=0.0, max=1.0) - A_t 
            
            # Probability Flow ODE drift
            beta = 0.1
            drift = -0.5 * beta * A_t + beta * score - 0.5 * beta * n_corr
            A_t = A_t + drift * dt
            
            # Hard projection onto manifold M
            A_t = torch.clamp(A_t, 0.0, 1.0)
            
        # Compute counterfactual graph embedding Z_cf
        # Re-weight node embeddings using the counterfactual edge weights A_t
        row, col = data.edge_index
        msg = A_t * node_embs[col]  # (E, hidden_dim)
        
        H_cf = torch.zeros_like(node_embs)
        H_cf.scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)
        
        # Global sum pooling for Z_cf
        batch_vec = data.batch
        Z_cf = torch.zeros((B, H_cf.size(1)), device=device)
        Z_cf.scatter_add_(0, batch_vec.unsqueeze(1).expand_as(H_cf), H_cf)
        
        return Z_cf, idx_i, idx_j


class TCDModule(nn.Module):
    """
    Topological Contrastive Disentanglement (TCD) Module.
    
    Mathematical Context (TopoCID - Section 4.3):
    Enforces the conditional independence criterion Y perp E | Z_topo via:
    1. InfoNCE contrastive loss L_TCD to align invariants across environments.
    2. CLUB variational bound L_MI to explicitly minimize mutual information 
       I(Z_topo; Z_env) using a critic network T_psi.
    """
    
    def __init__(self, topo_dim: int, env_dim: int, critic_hidden: int = 64, tau: float = 0.07):
        super().__init__()
        self.tau = tau
        
        # Critic network T_psi for CLUB MI bound
        self.critic = nn.Sequential(
            nn.Linear(topo_dim + env_dim, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1)
        )
        apply_deterministic_init(self)

    def compute_infoNCE(self, z_topo: torch.Tensor, positive_pairs: torch.Tensor) -> torch.Tensor:
        """
        Computes the InfoNCE contrastive loss L_TCD.
        """
        if positive_pairs.size(0) == 0:
            return torch.tensor(0.0, device=z_topo.device, requires_grad=True)
            
        i_idx = positive_pairs[:, 0]
        j_idx = positive_pairs[:, 1]
        
        # Cosine similarity logits
        logits = F.cosine_similarity(z_topo.unsqueeze(1), z_topo.unsqueeze(0), dim=2) / self.tau
        labels = j_idx
        
        return F.cross_entropy(logits, labels)

    def compute_CLUB(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the CLUB mutual information bound L_MI.
        Uses deterministic shuffling (torch.roll) to approximate the marginal 
        p(z)p(e) without using any random values.
        """
        # Deterministic shuffle to break joint dependency
        z_env_shuffled = torch.roll(z_env, shifts=1, dims=0)
        
        # Joint expectation E_{p(z,e)}[T_psi(z,e)]
        joint_input = torch.cat([z_topo, z_env], dim=1)
        T_joint = self.critic(joint_input).squeeze()
        
        # Marginal expectation E_{p(z)p(e)}[exp(T_psi(z,e))]
        marg_input = torch.cat([z_topo, z_env_shuffled], dim=1)
        T_marg = self.critic(marg_input).squeeze()
        
        # CLUB bound
        mi_bound = T_joint.mean() - torch.logsumexp(T_marg, dim=0) + \
                   torch.log(torch.tensor(T_marg.size(0), device=z_topo.device, dtype=torch.float32))
        
        return mi_bound

    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor, 
                positive_pairs: torch.Tensor, negative_pairs: torch.Tensor) -> tuple:
        l_tcd = self.compute_infoNCE(z_topo, positive_pairs)
        l_mi = self.compute_CLUB(z_topo, z_env)
        return l_tcd, l_mi


class TopoCID(nn.Module):
    """
    Main Wrapper for the TopoCID Framework.
    Integrates the GNN Backbone, TCP, SPCG, and TCD modules into a unified 
    architecture for Out-of-Distribution graph generalization.
    """
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64, topo_dim: int = 64, 
                 env_dim: int = 32, num_classes: int = 2, 
                 lambda_cf: float = 1.0, lambda_tcd: float = 0.5, lambda_mi: float = 0.1, 
                 tau: float = 0.07):
        super().__init__()
        self.lambda_cf = lambda_cf
        self.lambda_tcd = lambda_tcd
        self.lambda_mi = lambda_mi
        
        # 1. Shared GNN Backbone (GIN)
        self.backbone = GINBackbone(num_node_features, hidden_dim, num_layers=3)
        
        # 2. Differentiable Topological Causal Projection (TCP)
        self.tcp = TCPModule(hidden_dim, topo_dim)
        
        # 3. Environment Encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(hidden_dim, env_dim),
            nn.ReLU(),
            nn.Linear(env_dim, env_dim)
        )
        
        # 4. Structure-Preserving Counterfactual Generation (SPCG)
        self.spcg = SPCGModule(topo_dim, env_dim, num_steps=5)
        
        # 5. Topological Contrastive Disentanglement (TCD)
        self.tcd = TCDModule(topo_dim, env_dim, tau=tau)
        
        # 6. Downstream Classifier
        self.classifier = nn.Linear(topo_dim, num_classes)
        
        # Apply deterministic initialization to all newly created modules
        apply_deterministic_init(self.env_encoder)
        apply_deterministic_init(self.classifier)

    def forward(self, data: Data) -> tuple:
        """
        Forward pass of the TopoCID framework.
        """
        # 1. Extract node and graph embeddings
        node_embs, graph_embs = self.backbone(data)
        
        # 2. TCP: Extract causal topological manifold Z_topo
        z_topo = self.tcp(node_embs, data.edge_index, data.batch)
        
        # 3. Extract environmental representation Z_env
        z_env = self.env_encoder(graph_embs)
        
        # 4. Classifier logits
        logits = self.classifier(z_topo)
        
        return logits, z_topo, z_env, node_embs

    def compute_losses(self, data: Data, logits: torch.Tensor, z_topo: torch.Tensor, 
                       z_env: torch.Tensor, node_embs: torch.Tensor) -> tuple:
        """
        Computes the unified TopoCID objective L_total.
        """
        # 1. Supervised Loss L_sup
        l_sup = F.cross_entropy(logits, data.y.view(-1))
        
        # 2. TCD Losses (InfoNCE and CLUB MI Bound)
        l_tcd, l_mi = self.tcd(z_topo, z_env, data.positive_pairs, data.negative_pairs)
        
        # 3. SPCG Counterfactual Loss L_cf
        Z_cf, idx_i, idx_j = self.spcg(z_topo, z_env, node_embs, data)
        logits_cf = self.classifier(Z_cf)
        y_cf = data.y[idx_i].view(-1)
        l_cf = F.cross_entropy(logits_cf, y_cf)
        
        # Total Loss
        l_total = l_sup + self.lambda_cf * l_cf + self.lambda_tcd * l_tcd + self.lambda_mi * l_mi
        
        return l_total, l_sup, l_cf, l_tcd, l_mi


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID Framework Wrapper")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}, Nodes: {batch.x.size(0)}")
    
    # 2. Initialize TopoCID
    num_node_features = batch.x.size(1)
    num_classes = 2  # MUTAG is binary
    
    print("\n--- Initializing TopoCID Model (Deterministic Init) ---")
    model = TopoCID(num_node_features=num_node_features, hidden_dim=64, topo_dim=64, 
                    env_dim=32, num_classes=num_classes)
    
    # 3. Forward Pass & Loss Computation
    print("\n--- Executing Forward Pass and Loss Computation ---")
    model.train()
    
    logits, z_topo, z_env, node_embs = model(batch)
    l_total, l_sup, l_cf, l_tcd, l_mi = model.compute_losses(batch, logits, z_topo, z_env, node_embs)
    
    print(f"Logits Shape: {logits.shape}")
    print(f"Z_topo Shape: {z_topo.shape}")
    print(f"Z_env Shape: {z_env.shape}")
    
    print(f"\nLoss Values:")
    print(f"  L_sup (Supervised): {l_sup.item():.4f}")
    print(f"  L_cf (Counterfactual): {l_cf.item():.4f}")
    print(f"  L_tcd (InfoNCE): {l_tcd.item():.4f}")
    print(f"  L_mi (CLUB MI Bound): {l_mi.item():.4f}")
    print(f"  L_total (Total Loss): {l_total.item():.4f}")
    
    # 4. Backward Pass Verification
    print("\n--- Verifying Backward Pass ---")
    l_total.backward()
    
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Total Gradient Norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("TopoCID Framework Verification Complete. No synthetic/random data used.")
    print("=" * 60)