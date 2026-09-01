import os
import sys
import unittest
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.manifold_projection import ManifoldProjector
from data.dataloaders import TopoCIDDataModule
from torch_geometric.utils import to_dense_adj


class TestManifoldProjection(unittest.TestCase):
    """
    Verifies the mathematical guarantee of the Orthogonal Projection Matrix.
    
    Mathematical Context (TopoCID - Section 4.2):
    The orthogonal projection matrix is defined as:
        Pi_M = I - J_C^T (J_C J_C^T + delta I)^{-1} J_C
        
    By definition, the projection of any vector in the column space of J_C^T 
    onto the null space of J_C must be zero:
        Pi_M * J_C^T * u = 0  for any vector u.
    """
    
    def setUp(self):
        """Loads the original MUTAG dataset and initializes the projector."""
        self.root = "/home/phd/datasets/"
        self.data_module = TopoCIDDataModule(dataset_name='MUTAG', root=self.root, batch_size=16)
        self.data_module.prepare_data()
        self.loader = self.data_module.train_dataloader()
        self.batch = next(iter(self.loader))
        
        # Compute max nodes for dense adjacency representation
        self.max_nodes = self.batch.batch.bincount().max().item()
        self.A = to_dense_adj(self.batch.edge_index, batch=self.batch.batch, max_num_nodes=self.max_nodes).float()
        
        # Initialize Manifold Projector with Tikhonov regularization delta
        self.projector = ManifoldProjector(max_valency=4.0, delta=1e-5)

    def test_orthogonality_property(self):
        """Tests that Pi_M * J_C^T * u approx 0."""
        B, N, _ = self.A.shape
        
        # 1. Construct a deterministic vector u (NO random values)
        u = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        
        # 2. Construct M = J_C^T u
        # For the valency constraint, J_C^T broadcasts u_i to all columns of row i 
        # for active constraints (where degree > max_valency).
        active = self.projector.constraint.active_mask(self.A)
        M = u.unsqueeze(-1) * active.unsqueeze(-1)
        
        # 3. Apply the orthogonal projection Pi_M
        M_proj = self.projector.project_matrix(M, self.A)
        
        # 4. Verify the norm is close to 0 
        # (It will be exactly 0 if delta=0; with delta=1e-5, it is bounded by ~1e-5)
        norm = M_proj.norm().item()
        self.assertLess(norm, 1e-4, 
                        f"Projection of J_C^T u should be ~0 due to orthogonality, got norm={norm}")


if __name__ == '__main__':
    unittest.main()