import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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


class NoiseSchedule:
    """
    Variance-Preserving (VP) Linear Noise Schedule.
    
    Mathematical Context (TopoCID - Section 4.2):
    Defines the noise schedule beta(t) for the forward and reverse SDEs:
        beta(t) = beta_min + 0.5 * (beta_max - beta_min) * t
    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Computes beta(t)."""
        return self.beta_min + 0.5 * (self.beta_max - self.beta_min) * t

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the log mean coefficient for the forward SDE."""
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min


class ManifoldProjector(nn.Module):
    """
    Orthogonal Projection onto the Tangent Space of the Valid Manifold M.
    
    Mathematical Context (TopoCID - Section 4.2):
    The valid data manifold M is defined by the constraint C(G) = 0, where 
    C_i(A) = max(0, sum_j A_ij - v_i).
    The orthogonal projection matrix is:
        Pi_M = I - J_C^T (J_C J_C^T + delta I)^{-1} J_C
    The normal restoring drift is:
        n_corr = J_C^T C(G_t)
        
    This implementation computes the projection efficiently without explicit 
    matrix inversion by exploiting the row-sum structure of J_C.
    """
    def __init__(self, max_valency: float = 4.0, delta: float = 1e-5):
        super().__init__()
        self.max_valency = max_valency
        self.delta = delta

    def constraint_function(self, G_t: torch.Tensor) -> torch.Tensor:
        """C(G_t) = max(0, degree(G_t) - v)"""
        degree = G_t.sum(dim=-1)
        return F.relu(degree - self.max_valency)

    def active_constraints(self, G_t: torch.Tensor) -> torch.Tensor:
        """Indicator of active constraints (degree > v)"""
        degree = G_t.sum(dim=-1)
        return (degree > self.max_valency).float()

    def project_matrix(self, M: torch.Tensor, G_t: torch.Tensor) -> torch.Tensor:
        """
        Applies Pi_M to a matrix M (e.g., drift or diffusion).
        Pi_M M = M - J_C^T (J_C J_C^T + delta I)^{-1} J_C M
        """
        B, N, _ = G_t.shape
        active = self.active_constraints(G_t) # (B, N)
        
        # J_C M is the row sum of M
        JC_M = M.sum(dim=-1) # (B, N)
        
        # (J_C J_C^T + delta I)^{-1} is diagonal. 
        # J_C J_C^T has diagonal entries N for active constraints, 0 otherwise.
        inv_term = torch.where(active > 0, 
                               1.0 / (N + self.delta), 
                               1.0 / self.delta) # (B, N)
        
        # J_C^T (inv * JC_M)
        correction = inv_term * active * JC_M # (B, N)
        
        # Broadcast to columns and subtract
        M_proj = M - correction.unsqueeze(-1)
        return M_proj

    def normal_restoring_drift(self, G_t: torch.Tensor) -> torch.Tensor:
        """
        Computes n_corr = J_C^T C(G_t).
        """
        C = self.constraint_function(G_t) # (B, N)
        # J_C^T broadcasts C to all columns
        return C.unsqueeze(-1).expand_as(G_t)


class BaseSDESolver:
    """
    Base class for SDE and ODE solvers.
    """
    def __init__(self, num_steps: int = 50, beta_min: float = 0.1, beta_max: float = 20.0):
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
        self.noise_schedule = NoiseSchedule(beta_min, beta_max)

    def step(self, t: float, G_t: torch.Tensor, score: torch.Tensor, projector: ManifoldProjector) -> torch.Tensor:
        raise NotImplementedError

    def integrate(self, G_T: torch.Tensor, score_fn: callable, projector: ManifoldProjector, 
                  z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Integrates the reverse-time process from t=1 to t=0.
        """
        G_t = G_T
        for step in range(self.num_steps, 0, -1):
            t = step / self.num_steps
            t_tensor = torch.full((G_t.size(0), 1), t, device=G_t.device)
            score = score_fn(G_t, t_tensor, z_topo, z_env)
            G_t = self.step(t, G_t, score, projector)
        return G_t


class EulerMaruyamaSolver(BaseSDESolver):
    """
    Euler-Maruyama Solver for the Manifold-Constrained Reverse SDE.
    
    Mathematical Context (TopoCID - Section 4.2):
    Integrates the reverse-time SDE:
        dG_t = [ -0.5*beta(t)*G_t + beta(t)*score - 0.5*beta(t)*n_corr ] dt + sqrt(beta(t)) * Pi_M dW_t
        
    To strictly avoid random values (the Wiener process dW_t), this implementation 
    sets dW_t = 0, effectively reducing the SDE to its deterministic drift component 
    while maintaining the Euler-Maruyama integration structure.
    """
    def step(self, t: float, G_t: torch.Tensor, score: torch.Tensor, projector: ManifoldProjector) -> torch.Tensor:
        beta = self.noise_schedule.beta(torch.tensor(t, device=G_t.device))
        
        # 1. Unconstrained drift
        drift_unconstrained = -0.5 * beta * G_t + beta * score
        
        # 2. Project drift onto tangent space: Pi_M * drift
        drift_proj = projector.project_matrix(drift_unconstrained, G_t)
        
        # 3. Normal restoring drift: -0.5 * beta * n_corr
        n_corr = projector.normal_restoring_drift(G_t)
        drift = drift_proj - 0.5 * beta * n_corr
        
        # 4. Diffusion term (Strictly deterministic: dW_t = 0 to avoid random values)
        dW = torch.zeros_like(G_t)
        diffusion_proj = projector.project_matrix(dW, G_t)
        diffusion = torch.sqrt(beta) * diffusion_proj
        
        # 5. Euler-Maruyama update
        G_next = G_t + drift * self.dt + diffusion * torch.sqrt(torch.tensor(self.dt, device=G_t.device))
        
        # Hard projection to [0, 1] to maintain valid probability matrix
        G_next = torch.clamp(G_next, 0.0, 1.0)
        
        return G_next


class ProbabilityFlowODESolver(BaseSDESolver):
    """
    Probability Flow ODE Solver for the Manifold-Constrained Reverse Process.
    
    Mathematical Context (TopoCID - Section 4.2):
    The deterministic counterpart to the reverse SDE, obtained by setting dW_t = 0 
    and halving the score coefficient:
        dG_t = [ -0.5*beta(t)*G_t + 0.5*beta(t)*score - 0.5*beta(t)*n_corr ] dt
    """
    def step(self, t: float, G_t: torch.Tensor, score: torch.Tensor, projector: ManifoldProjector) -> torch.Tensor:
        beta = self.noise_schedule.beta(torch.tensor(t, device=G_t.device))
        
        # 1. Unconstrained ODE drift (note the 0.5 factor on the score)
        drift_unconstrained = -0.5 * beta * G_t + 0.5 * beta * score
        
        # 2. Project drift onto tangent space
        drift_proj = projector.project_matrix(drift_unconstrained, G_t)
        
        # 3. Normal restoring drift
        n_corr = projector.normal_restoring_drift(G_t)
        drift = drift_proj - 0.5 * beta * n_corr
        
        # 4. Euler update for ODE
        G_next = G_t + drift * self.dt
        
        # Hard projection to [0, 1]
        G_next = torch.clamp(G_next, 0.0, 1.0)
        
        return G_next


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    from torch_geometric.utils import to_dense_adj
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing SDE/ODE Solvers for TopoCID SPCG Module")
    print("=" * 60)
    
    # 1. Load original dataset
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}, Nodes: {batch.x.size(0)}")
    
    # Compute max nodes for dense adjacency
    max_nodes = batch.batch.bincount().max().item()
    
    # Initialize dense adjacency A_0
    A_0 = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=max_nodes).float()
    B, N, _ = A_0.shape
    
    # Dummy conditioning variables
    z_topo = torch.ones(B, 64, device=A_0.device)
    z_env = torch.ones(B, 32, device=A_0.device)
    
    # Dummy score function
    def dummy_score_fn(G_t, t, z_topo, z_env):
        return torch.zeros_like(G_t)
    
    # 2. Initialize Projector and Solvers
    print("\n--- Initializing Manifold Projector and Solvers ---")
    projector = ManifoldProjector(max_valency=4.0, delta=1e-5)
    
    em_solver = EulerMaruyamaSolver(num_steps=10, beta_min=0.1, beta_max=20.0)
    ode_solver = ProbabilityFlowODESolver(num_steps=10, beta_min=0.1, beta_max=20.0)
    
    # 3. Forward Pass (Integration)
    print("\n--- Executing Reverse Integration (Deterministic) ---")
    
    # Add deterministic perturbation to A_0 to start the reverse process (NO RANDOM NOISE)
    A_T = A_0 + 0.1 * torch.sin(torch.arange(A_0.numel()).view(A_0.shape).to(A_0.device))
    A_T = torch.clamp(A_T, 0.0, 1.0)
    
    # Euler-Maruyama (with dW=0)
    A_cf_em = em_solver.integrate(A_T.clone(), dummy_score_fn, projector, z_topo, z_env)
    print(f"Euler-Maruyama (dW=0) Final Adjacency Shape: {A_cf_em.shape}")
    
    # Probability Flow ODE
    A_cf_ode = ode_solver.integrate(A_T.clone(), dummy_score_fn, projector, z_topo, z_env)
    print(f"Probability Flow ODE Final Adjacency Shape: {A_cf_ode.shape}")
    
    # 4. Verify Structural Validity
    print("\n--- Verifying Structural Validity (Valency Constraints) ---")
    degrees_em = A_cf_em.sum(dim=-1)
    degrees_ode = A_cf_ode.sum(dim=-1)
    
    max_valency = 4.0
    violations_em = F.relu(degrees_em - max_valency).max().item()
    violations_ode = F.relu(degrees_ode - max_valency).max().item()
    
    print(f"Max Valency Violation (EM): {violations_em:.4f}")
    print(f"Max Valency Violation (ODE): {violations_ode:.4f}")
    
    print("\n" + "=" * 60)
    print("SDE/ODE Solvers Verification Complete. No random values used.")
    print("=" * 60)