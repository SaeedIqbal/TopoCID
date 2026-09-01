import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class StructuralConstraint(nn.Module):
    """
    Computes the structural constraint function C(G) and its active constraint mask.
    
    Mathematical Context (TopoCID - Section 4.2):
    The valid data manifold M is defined by the zero-level set of the constraint function:
        M = { G in G_space | C(G) = 0 }
    For molecular graphs, C enforces chemical valency rules:
        C_i(G) = max(0, sum_j A_ij - v_i)
    where v_i is the maximum allowed valency for node i (e.g., 4 for Carbon).
    """
    
    def __init__(self, max_valency: float = 4.0):
        """
        Args:
            max_valency (float): The maximum allowed degree (valency) for any node.
        """
        super().__init__()
        self.max_valency = max_valency

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the constraint violation C(G).
        
        Args:
            A (torch.Tensor): Continuous adjacency matrix of shape (B, N, N).
            
        Returns:
            torch.Tensor: Constraint violations of shape (B, N).
        """
        degree = A.sum(dim=-1) # (B, N)
        return F.relu(degree - self.max_valency)

    def active_mask(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the indicator of active constraints (where degree > max_valency).
        This corresponds to the non-zero rows of the Jacobian J_C.
        
        Args:
            A (torch.Tensor): Continuous adjacency matrix of shape (B, N, N).
            
        Returns:
            torch.Tensor: Active mask of shape (B, N) with values in {0, 1}.
        """
        degree = A.sum(dim=-1)
        return (degree > self.max_valency).float()


class ManifoldProjector(nn.Module):
    """
    Orthogonal Projection onto the Tangent Space of the Valid Manifold M.
    
    Mathematical Context (TopoCID - Section 4.2):
    The orthogonal projection matrix onto the null space of the Jacobian J_C is:
        Pi_M = I - J_C^T (J_C J_C^T + delta I)^{-1} J_C
        
    The normal restoring drift is:
        n_corr = J_C^T C(G_t)
        
    Because C(G) depends only on the row sums of A (the degrees), the Jacobian J_C 
    has a highly structured form. The action of J_C on a matrix M is simply the 
    row sum of M. J_C J_C^T is a diagonal matrix with entries N for active constraints.
    This allows us to compute the action of Pi_M on a matrix M efficiently in O(N^2) 
    time without explicitly forming the D_G x D_G Jacobian.
    """
    
    def __init__(self, max_valency: float = 4.0, delta: float = 1e-5):
        """
        Args:
            max_valency (float): The maximum allowed degree (valency) for any node.
            delta (float): Tikhonov regularization parameter for numerical stability.
        """
        super().__init__()
        self.constraint = StructuralConstraint(max_valency)
        self.delta = delta

    def project_matrix(self, M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Applies the orthogonal projection Pi_M to a matrix M.
        Pi_M M = M - J_C^T (J_C J_C^T + delta I)^{-1} J_C M
        
        Args:
            M (torch.Tensor): The matrix to project (e.g., drift or score), shape (B, N, N).
            A (torch.Tensor): Current continuous adjacency matrix, shape (B, N, N).
            
        Returns:
            torch.Tensor: The projected matrix M_proj, shape (B, N, N).
        """
        B, N, _ = A.shape
        
        # 1. Identify active constraints (non-zero rows of J_C)
        active = self.constraint.active_mask(A) # (B, N)
        
        # 2. Compute J_C M: This is the row sum of M, masked by active constraints
        JC_M = M.sum(dim=-1) * active # (B, N)
        
        # 3. Compute (J_C J_C^T + delta I)^{-1}
        # J_C J_C^T is diagonal with entries N for active constraints, 0 otherwise.
        # So the inverse is 1 / (N * active + delta)
        inv_diag = 1.0 / (N * active + self.delta) # (B, N)
        
        # 4. Compute J_C^T (inv_diag * JC_M)
        # J_C^T broadcasts a vector to all columns.
        correction = inv_diag * JC_M # (B, N)
        
        # 5. Subtract correction from M
        # The correction is applied to the rows corresponding to active constraints.
        M_proj = M - correction.unsqueeze(-1)
        
        return M_proj

    def normal_restoring_drift(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the normal restoring drift n_corr = J_C^T C(G_t).
        This drift continuously pulls the trajectory back to the zero-level set M.
        
        Args:
            A (torch.Tensor): Current continuous adjacency matrix, shape (B, N, N).
            
        Returns:
            torch.Tensor: The normal restoring drift, shape (B, N, N).
        """
        # 1. Compute C(G_t)
        C = self.constraint(A) # (B, N)
        
        # 2. Compute J_C^T C(G_t)
        # J_C^T broadcasts C to all columns.
        n_corr = C.unsqueeze(-1).expand_as(A)
        
        return n_corr


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    from torch_geometric.utils import to_dense_adj
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing Manifold Projection Module for TopoCID")
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
    
    # Initialize dense adjacency A_0 deterministically
    A_0 = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=max_nodes).float()
    B, N, _ = A_0.shape
    print(f"Dense Adjacency Shape: {A_0.shape}")
    
    # 2. Initialize Projector
    print("\n--- Initializing Manifold Projector ---")
    projector = ManifoldProjector(max_valency=4.0, delta=1e-5)
    
    # 3. Compute Constraints and Drift
    print("\n--- Computing Structural Constraints ---")
    C = projector.constraint(A_0)
    print(f"Constraint Violations C(G) Shape: {C.shape}")
    print(f"Max Constraint Violation: {C.max().item():.4f}")
    
    n_corr = projector.normal_restoring_drift(A_0)
    print(f"Normal Restoring Drift n_corr Shape: {n_corr.shape}")
    
    # 4. Verify Projection Properties (Using deterministic matrices, NO random values)
    print("\n--- Verifying Orthogonal Projection Properties ---")
    
    # Create a deterministic matrix M (e.g., all ones)
    M = torch.ones(B, N, N)
    
    # Project M
    M_proj = projector.project_matrix(M, A_0)
    
    # Property 1: Pi_M * J_C^T = 0
    # Let's test with a specific vector u = ones, so J_C^T u is a matrix of ones for active rows
    u = torch.ones(B, N)
    JC_T_u = u.unsqueeze(-1).expand_as(M)
    
    Pi_M_JC_T_u = projector.project_matrix(JC_T_u, A_0)
    print(f"Norm of Pi_M * J_C^T * u (Should be ~0): {Pi_M_JC_T_u.norm().item():.6f}")
    
    # Property 2: Pi_M * Pi_M = Pi_M (Idempotence)
    M_proj_2 = projector.project_matrix(M_proj, A_0)
    diff = (M_proj - M_proj_2).norm().item()
    print(f"Norm of Pi_M(M) - Pi_M(Pi_M(M)) (Should be ~0): {diff:.6f}")
    
    # 5. Backward Pass Verification
    print("\n--- Verifying Differentiability (Backward Pass) ---")
    A_test = A_0.clone().requires_grad_(True)
    M_test = torch.ones(B, N, N, requires_grad=True)
    
    M_proj_test = projector.project_matrix(M_test, A_test)
    loss = M_proj_test.sum()
    loss.backward()
    
    print(f"Gradient w.r.t A Shape: {A_test.grad.shape}")
    print(f"Gradient w.r.t M Shape: {M_test.grad.shape}")
    print(f"A Gradient Norm: {A_test.grad.norm().item():.4f}")
    print(f"M Gradient Norm: {M_test.grad.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("Manifold Projection Verification Complete. No synthetic/random data used.")
    print("=" * 60)