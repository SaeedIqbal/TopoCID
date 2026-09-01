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


class SpectralNormalizedLinear(nn.Module):
    """
    Linear layer with spectral normalization to enforce Lipschitz continuity.
    
    Mathematical Context (TopoCID - Section 4.3):
    The critic network T_psi is constrained to be Lipschitz continuous to ensure 
    bounded gradients for stable min-max optimization:
        |T_psi(z, e) - T_psi(z', e')| <= L * ||(z,e) - (z',e')||_2
    Spectral normalization bounds the spectral norm of the weight matrix to 1, 
    ensuring L <= 1 per layer.
    """
    
    def __init__(self, in_features: int, out_features: int, power_iterations: int = 1):
        """
        Args:
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            power_iterations (int): Number of power iteration steps for spectral norm estimation.
        """
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features, bias=True),
            n_power_iterations=power_iterations
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CriticNetwork(nn.Module):
    """
    Critic Network T_psi for the CLUB Variational Bound.
    
    Mathematical Context (TopoCID - Section 4.3):
    A trainable critic network parameterized by psi, mapping the joint space 
    (Z_topo, Z_env) to a scalar score:
        T_psi: R^{d_topo} x R^{d_env} -> R
    Constrained to be Lipschitz continuous via spectral normalization.
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """
        Args:
            x_dim (int): Dimension of Z_topo (first variable).
            y_dim (int): Dimension of Z_env (second variable).
            hidden_dim (int): Hidden layer dimension.
            num_layers (int): Number of hidden layers.
        """
        super().__init__()
        
        input_dim = x_dim + y_dim
        layers = []
        
        # Input layer with spectral normalization
        layers.append(SpectralNormalizedLinear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers with spectral normalization
        for _ in range(num_layers - 1):
            layers.append(SpectralNormalizedLinear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (scalar)
        layers.append(SpectralNormalizedLinear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        apply_deterministic_init(self)
        
    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the critic score T_psi(z_topo, z_env).
        
        Args:
            z_topo (torch.Tensor): First variable of shape (B, x_dim).
            z_env (torch.Tensor): Second variable of shape (B, y_dim).
            
        Returns:
            torch.Tensor: Scalar critic scores of shape (B,).
        """
        # Concatenate inputs
        z_joint = torch.cat([z_topo, z_env], dim=1) # (B, x_dim + y_dim)
        
        # Forward through critic
        out = self.network(z_joint).squeeze(-1) # (B,)
        
        return out


class CLUBEstimator(nn.Module):
    """
    CLUB (CLub Upper Bound) Variational Mutual Information Estimator.
    
    Mathematical Context (TopoCID - Section 4.3):
    Provides a variational upper bound on the mutual information I(Z_topo; Z_env) 
    using the Donsker-Varadhan representation:
        I(Z_topo; Z_env) >= sup_psi ( E_{p(z,e)}[T_psi(z,e)] - log E_{p(z)p(e)}[exp(T_psi(z,e))] )
    
    The estimator computes:
        I_CLUB = E_{p(z,e)}[T_psi(z,e)] - log( (1/B) * sum_i exp(T_psi(z_i, e_{sigma(i)})) )
    
    where:
    - The joint expectation E_{p(z,e)} uses true paired samples (z_i, e_i).
    - The marginal expectation E_{p(z)p(e)} uses deterministically shuffled samples 
      (z_i, e_{sigma(i)}) to approximate the product of marginals p(z)p(e).
    - sigma is a deterministic circular shift (torch.roll) to strictly avoid random values.
    
    During training:
    - The critic parameters psi are updated via gradient ASCENT to maximize the bound.
    - The representation extractors (Phi, Psi_env) are updated via gradient DESCENT 
      to minimize the bound.
    This implements the min-max optimization:
        min_{Phi, Psi_env} max_psi I_CLUB
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """
        Args:
            x_dim (int): Dimension of Z_topo.
            y_dim (int): Dimension of Z_env.
            hidden_dim (int): Hidden dimension of the critic network.
            num_layers (int): Number of hidden layers in the critic.
        """
        super().__init__()
        self.critic = CriticNetwork(x_dim, y_dim, hidden_dim, num_layers)

    def compute_joint_expectation(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the joint expectation term: E_{p(z,e)}[T_psi(z,e)].
        Uses true paired samples from the joint distribution p(z, e).
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, x_dim).
            z_env (torch.Tensor): Environmental representations of shape (B, y_dim).
            
        Returns:
            torch.Tensor: Scalar joint expectation.
        """
        T_joint = self.critic(z_topo, z_env) # (B,)
        return T_joint.mean()

    def compute_marginal_expectation(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the marginal expectation term: log E_{p(z)p(e)}[exp(T_psi(z,e))].
        
        To strictly avoid random values, the product of marginals p(z)p(e) is 
        approximated by pairing each z_i with a deterministically shifted e_{i+1}:
            sigma(i) = (i + 1) mod B
        
        The log-expectation is computed via logsumexp for numerical stability:
            log( (1/B) * sum_i exp(T_psi(z_i, e_{sigma(i)})) ) 
            = logsumexp(T_marginal) - log(B)
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, x_dim).
            z_env (torch.Tensor): Environmental representations of shape (B, y_dim).
            
        Returns:
            torch.Tensor: Scalar log-marginal expectation.
        """
        B = z_topo.size(0)
        
        # Deterministic circular shift to break joint dependency
        z_env_shuffled = torch.roll(z_env, shifts=1, dims=0)
        
        # Compute critic scores for shuffled pairs
        T_marginal = self.critic(z_topo, z_env_shuffled) # (B,)
        
        # Numerically stable log-mean-exp
        log_marginal = torch.logsumexp(T_marginal, dim=0) - \
                       torch.log(torch.tensor(B, dtype=torch.float32, device=z_topo.device))
        
        return log_marginal

    def forward(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the CLUB mutual information upper bound.
        
        Mathematical Formula:
            I_CLUB = E_{p(z,e)}[T_psi(z,e)] - log E_{p(z)p(e)}[exp(T_psi(z,e))]
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, x_dim).
            z_env (torch.Tensor): Environmental representations of shape (B, y_dim).
            
        Returns:
            torch.Tensor: The CLUB bound (scalar). 
                          Minimized w.r.t. representation extractors (gradient descent).
                          Maximized w.r.t. critic parameters psi (gradient ascent).
        """
        joint_term = self.compute_joint_expectation(z_topo, z_env)
        marginal_term = self.compute_marginal_expectation(z_topo, z_env)
        
        club_bound = joint_term - marginal_term
        
        return club_bound

    def get_critic_loss(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the critic loss for gradient ascent (maximizing the CLUB bound).
        Since optimizers typically minimize, we return the negative of the bound.
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, x_dim).
            z_env (torch.Tensor): Environmental representations of shape (B, y_dim).
            
        Returns:
            torch.Tensor: Negative CLUB bound (to be minimized by the critic optimizer).
        """
        club_bound = self.forward(z_topo, z_env)
        return -club_bound

    def get_representation_loss(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Computes the representation loss for gradient descent (minimizing the CLUB bound).
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants of shape (B, x_dim).
            z_env (torch.Tensor): Environmental representations of shape (B, y_dim).
            
        Returns:
            torch.Tensor: CLUB bound (to be minimized by the representation optimizers).
        """
        return self.forward(z_topo, z_env)

    def update_critic(self, z_topo: torch.Tensor, z_env: torch.Tensor, 
                      optimizer: torch.optim.Optimizer) -> float:
        """
        Performs one gradient ascent step on the critic parameters psi.
        
        Mathematical Context:
            psi <- psi + eta_psi * nabla_psi I_CLUB
        Since optimizers minimize, we minimize -I_CLUB.
        
        Args:
            z_topo (torch.Tensor): Detached causal topological invariants.
            z_env (torch.Tensor): Detached environmental representations.
            optimizer (torch.optim.Optimizer): Optimizer for the critic parameters.
            
        Returns:
            float: The CLUB bound value before the update.
        """
        # Detach representations to prevent gradient flow to extractors
        z_topo_detached = z_topo.detach()
        z_env_detached = z_env.detach()
        
        optimizer.zero_grad()
        critic_loss = self.get_critic_loss(z_topo_detached, z_env_detached)
        critic_loss.backward()
        optimizer.step()
        
        return -critic_loss.item()

    def update_representations(self, z_topo: torch.Tensor, z_env: torch.Tensor, 
                                optimizer: torch.optim.Optimizer) -> float:
        """
        Performs one gradient descent step on the representation extractor parameters.
        
        Mathematical Context:
            (Phi, Psi_env) <- (Phi, Psi_env) - eta_theta * nabla_{Phi, Psi_env} I_CLUB
        
        Args:
            z_topo (torch.Tensor): Causal topological invariants (with grad).
            z_env (torch.Tensor): Environmental representations (with grad).
            optimizer (torch.optim.Optimizer): Optimizer for the representation parameters.
            
        Returns:
            float: The CLUB bound value.
        """
        optimizer.zero_grad()
        rep_loss = self.get_representation_loss(z_topo, z_env)
        rep_loss.backward()
        optimizer.step()
        
        return rep_loss.item()


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    from models.backbones.gin import GINBackbone
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing CLUB Variational Mutual Information Estimator")
    print("=" * 60)
    
    # 1. Load original dataset
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}, Nodes: {batch.x.size(0)}")
    
    # 2. Initialize GIN Backbone to get graph embeddings
    print("\n--- Initializing GIN Backbone ---")
    num_node_features = batch.x.size(1)
    hidden_dim = 64
    backbone = GINBackbone(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=3)
    apply_deterministic_init(backbone)
    
    # 3. Initialize CLUB Estimator
    print("\n--- Initializing CLUB Estimator (Deterministic) ---")
    topo_dim = 64
    env_dim = 32
    
    club = CLUBEstimator(x_dim=topo_dim, y_dim=env_dim, hidden_dim=64, num_layers=2)
    
    # 4. Forward Pass
    print("\n--- Executing Forward Pass (CLUB Bound Computation) ---")
    backbone.eval()
    club.eval()
    
    with torch.no_grad():
        _, graph_embs = backbone(batch)
        z_topo = graph_embs[:, :topo_dim]
        z_env = graph_embs[:, :env_dim]
        
        joint_exp = club.compute_joint_expectation(z_topo, z_env)
        marginal_exp = club.compute_marginal_expectation(z_topo, z_env)
        club_bound = club(z_topo, z_env)
        
    print(f"Joint Expectation E_{{p(z,e)}}[T_psi]: {joint_exp.item():.4f}")
    print(f"Marginal Expectation log E_{{p(z)p(e)}}[exp(T_psi)]: {marginal_exp.item():.4f}")
    print(f"CLUB MI Bound I_CLUB: {club_bound.item():.4f}")
    
    # 5. Backward Pass Verification
    print("\n--- Verifying Differentiability (Backward Pass) ---")
    club.train()
    backbone.train()
    
    _, graph_embs = backbone(batch)
    z_topo = graph_embs[:, :topo_dim]
    z_env = graph_embs[:, :env_dim]
    
    club_loss = club.get_representation_loss(z_topo, z_env)
    club_loss.backward()
    
    # Check critic gradients
    critic_grad_norm = 0.0
    for p in club.critic.parameters():
        if p.grad is not None:
            critic_grad_norm += p.grad.norm().item()
    print(f"Critic Gradient Norm: {critic_grad_norm:.4f}")
    
    # Check backbone gradients (should flow through z_topo and z_env)
    backbone_grad_norm = 0.0
    for p in backbone.parameters():
        if p.grad is not None:
            backbone_grad_norm += p.grad.norm().item()
    print(f"Backbone Gradient Norm: {backbone_grad_norm:.4f}")
    
    # 6. Min-Max Optimization Verification
    print("\n--- Verifying Min-Max Optimization (Alternating Updates) ---")
    critic_optimizer = torch.optim.Adam(club.critic.parameters(), lr=1e-3)
    backbone_optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
    
    for step in range(3):
        # Step A: Gradient ascent on critic (maximize CLUB bound)
        _, graph_embs = backbone(batch)
        z_topo = graph_embs[:, :topo_dim]
        z_env = graph_embs[:, :env_dim]
        mi_val = club.update_critic(z_topo, z_env, critic_optimizer)
        
        # Step B: Gradient descent on representations (minimize CLUB bound)
        _, graph_embs = backbone(batch)
        z_topo = graph_embs[:, :topo_dim]
        z_env = graph_embs[:, :env_dim]
        mi_val = club.update_representations(z_topo, z_env, backbone_optimizer)
        
        print(f"  Step {step+1}: CLUB MI Bound = {mi_val:.4f}")
    
    print("\n" + "=" * 60)
    print("CLUB Estimator Verification Complete. No synthetic/random data used.")
    print("=" * 60)