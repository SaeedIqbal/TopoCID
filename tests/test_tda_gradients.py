import os
import sys
import unittest
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.differentiable_tda import DifferentiableTDAModule
from data.dataloaders import TopoCIDDataModule


class TestTDAGradients(unittest.TestCase):
    """
    Verifies exact gradient flow through the differentiable persistence diagram.
    
    Mathematical Context (TopoCID - Section 4.1):
    Standard persistent homology relies on discrete matrix reduction, which breaks 
    gradient flow. TopoCID implements a differentiable proxy using soft-order statistics.
    This test verifies that the gradient of the topological representation Z_topo 
    with respect to the initial node filtration weights w is non-zero:
        nabla_w Z_topo != 0
    """
    
    def setUp(self):
        """Loads the original MUTAG dataset and initializes the TDA module."""
        self.root = "/home/phd/datasets/"
        self.data_module = TopoCIDDataModule(dataset_name='MUTAG', root=self.root, batch_size=16)
        self.data_module.prepare_data()
        self.loader = self.data_module.train_dataloader()
        self.batch = next(iter(self.loader))
        
        # Initialize Differentiable TDA Module
        self.tda = DifferentiableTDAModule(topo_dim=32)

    def test_backpropagation_through_persistence(self):
        """Tests that gradients successfully flow back to the node filtration weights."""
        # 1. Compute deterministic initial node weights w(v) = ||H_v||_2
        # Enable gradient tracking
        w = torch.norm(self.batch.x, p=2, dim=1, keepdim=True).requires_grad_(True)
        
        # 2. Forward pass through the differentiable persistence diagram
        z_topo = self.tda(w, self.batch.edge_index, self.batch.x.size(0), self.batch.batch)
        
        # 3. Compute a dummy scalar loss to trigger backpropagation
        loss = z_topo.sum()
        loss.backward()
        
        # 4. Verify that gradients flowed back to w
        self.assertIsNotNone(w.grad, "Gradients did not flow back to node weights w. Differentiability broken.")
        
        grad_norm = w.grad.norm().item()
        self.assertGreater(grad_norm, 0.0, 
                           f"Gradient norm should be strictly > 0, got {grad_norm}. Soft-min/max might be saturated.")


if __name__ == '__main__':
    unittest.main()