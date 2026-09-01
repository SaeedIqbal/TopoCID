import os
import sys
import unittest
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.topocid.spcg import SPCGModule, apply_deterministic_init
from evaluation.structural_validity import StructuralValidityEvaluator
from data.dataloaders import TopoCIDDataModule


class TestSPCGStructuralValidity(unittest.TestCase):
    """
    Verifies the 100% Structural Validity Rate (SVR) guarantee of SPCG.
    
    Mathematical Context (TopoCID - Section 4.2):
    The SPCG module integrates a manifold-constrained reverse SDE. The normal 
    restoring drift n_corr = J_C^T C(G_t) and the orthogonal projection Pi_M 
    mathematically guarantee that the generated continuous adjacency matrix A_cf 
    strictly satisfies the valency constraints:
        C(A_cf) = max(0, degree(A_cf) - v) = 0
    """
    
    def setUp(self):
        """Loads the original MUTAG dataset and initializes SPCG and the evaluator."""
        self.root = "/home/phd/datasets/"
        self.data_module = TopoCIDDataModule(dataset_name='MUTAG', root=self.root, batch_size=16)
        self.data_module.prepare_data()
        self.loader = self.data_module.train_dataloader()
        self.batch = next(iter(self.loader))
        
        self.max_nodes = self.batch.batch.bincount().max().item()
        self.topo_dim = 32
        self.env_dim = 16
        
        # Initialize SPCG with deterministic weights
        self.spcg = SPCGModule(max_nodes=self.max_nodes, topo_dim=self.topo_dim, 
                               env_dim=self.env_dim, num_steps=10)
        apply_deterministic_init(self.spcg)
        
        # Initialize SVR Evaluator
        self.evaluator = StructuralValidityEvaluator(max_valency=4.0)

    def test_100_percent_svr(self):
        """Tests that SPCG generates counterfactuals with 100% structural validity."""
        B = self.batch.batch_size
        
        # Deterministic conditioning variables (NO random values)
        z_topo = torch.ones(B, self.topo_dim)
        z_env = torch.ones(B, self.env_dim)
        
        # Deterministic node embeddings
        node_embs = torch.ones(self.batch.x.size(0), 64)
        
        # Generate counterfactual adjacency matrices A_cf via Probability Flow ODE
        _, _, _, A_cf = self.spcg(z_topo, z_env, node_embs, self.batch.edge_index, self.batch.batch)
        
        # Evaluate Structural Validity Rate
        svr = self.evaluator.compute_svr(A_cf)
        
        # Assert strict 100% validity
        self.assertEqual(svr, 1.0, 
                         f"SPCG must mathematically guarantee 100% valid graphs, got SVR={svr}")


if __name__ == '__main__':
    unittest.main()