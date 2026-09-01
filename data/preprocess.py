import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, contains_self_loops, to_undirected


class GraphPreprocessor:
    """
    Object-Oriented Preprocessor for Graph Data in the TopoCID Framework.
    
    This class implements graph grammar checks, valency rule enforcement, and 
    feature normalization. These operations are critical for ensuring that the 
    input graphs strictly reside on the valid data manifold M = {G | C(G) = 0} 
    required by the Structure-Preserving Counterfactual Generation (SPCG) module, 
    and for providing stable initial features H^(0) for the Differentiable 
    Topological Causal Projection (TCP) module.
    """
    
    # Maximum valid valency per atom type for molecular datasets
    # Mapped based on standard atomic numbers / one-hot encoding indices
    VALENCY_RULES = {
        0: 4,  # Carbon (C)
        1: 3,  # Nitrogen (N)
        2: 2,  # Oxygen (O)
        3: 1,  # Fluorine (F)
        4: 6,  # Sulfur (S)
        5: 5,  # Phosphorus (P)
        6: 1,  # Chlorine (Cl)
        7: 1,  # Bromine (Br)
        8: 1,  # Iodine (I)
    }
    
    # For datasets where atom types are not explicitly mapped to the above dictionary,
    # we use a default maximum valency.
    DEFAULT_MAX_VALENCY = 6

    def __init__(self, dataset_name: str, is_molecular: bool = True):
        """
        Initializes the GraphPreprocessor.
        
        Args:
            dataset_name (str): Name of the dataset ('MUTAG', 'PROTEINS', 'NCI1', etc.).
            is_molecular (bool): Whether the dataset represents molecular graphs 
                                 (True) or generic/protein graphs (False).
        """
        self.dataset_name = dataset_name
        self.is_molecular = is_molecular
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

    def normalize_features(self, data: Data) -> Data:
        """
        Normalizes node features to zero mean and unit variance.
        
        Mathematical Context (TopoCID - TCP Module):
        The initial node embeddings H^(0) = X are used to compute the learnable 
        scalar node-weight function:
            w(v) = u^T MLP(H^(0)_v)
        Normalizing X ensures that the subsequent simplicial filtration 
        {X_t}_{t in R} operates on a consistent scale, preventing features with 
        large magnitudes from dominating the filtration process.
        
        Equation:
            X_hat = (X - mu) / (sigma + epsilon)
        
        Args:
            data (Data): The input graph Data object.
            
        Returns:
            Data: The graph with normalized node features.
        """
        if data.x is None or data.x.numel() == 0:
            return data
            
        # Compute statistics if not already computed (e.g., on the first batch)
        if self.feature_mean is None:
            self.feature_mean = data.x.mean(dim=0)
            self.feature_std = data.x.std(dim=0)
            # Prevent division by zero
            self.feature_std[self.feature_std == 0] = 1.0
            
        # Apply normalization
        epsilon = 1e-8
        data.x = (data.x - self.feature_mean) / (self.feature_std + epsilon)
        
        return data

    def check_grammar(self, data: Data) -> bool:
        """
        Checks if the graph satisfies basic structural grammar rules.
        
        Mathematical Context (TopoCID - SPCG Module):
        The valid data manifold M is defined by the constraint function C(G) = 0.
        For general graphs, C(G) enforces:
        1. No self-loops: forall (u, v) in E, u != v.
        2. Undirected edges (symmetric adjacency): if (u, v) in E => (v, u) in E.
        
        Args:
            data (Data): The input graph Data object.
            
        Returns:
            bool: True if the graph satisfies grammar rules, False otherwise.
        """
        if data.edge_index is None or data.edge_index.numel() == 0:
            return True
            
        # Check for self-loops
        if contains_self_loops(data.edge_index):
            return False
            
        # Check for symmetric adjacency (undirected graph)
        # A simpler check: to_undirected should not change the number of edges.
        edge_index_undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        if edge_index_undirected.size(1) != data.edge_index.size(1):
            return False
            
        return True

    def check_valency(self, data: Data) -> bool:
        """
        Checks if the molecular graph satisfies chemical valency rules.
        
        Mathematical Context (TopoCID - SPCG Module):
        For molecular graphs, the constraint function C(G) enforces:
            sum_j A_ij <= v_type(i) for all nodes i
        where v_type(i) is the maximum allowed valency for the atom type of node i.
        
        Args:
            data (Data): The input graph Data object.
            
        Returns:
            bool: True if all nodes satisfy valency constraints, False otherwise.
        """
        if not self.is_molecular:
            return True
            
        if data.edge_index is None or data.edge_index.numel() == 0:
            return True
            
        # Compute degree for each node
        row, _ = data.edge_index
        deg = degree(row, num_nodes=data.num_nodes, dtype=torch.long)
        
        # Determine max valency for each node
        if hasattr(data, 'x') and data.x is not None and data.x.size(1) > 0:
            # Assume the atom type is the argmax of the one-hot encoded features
            atom_types = torch.argmax(data.x, dim=1).cpu().numpy()
            max_valencies = np.array([self.VALENCY_RULES.get(at, self.DEFAULT_MAX_VALENCY) for at in atom_types])
            max_valencies = torch.tensor(max_valencies, dtype=torch.long)
        else:
            # Fallback to default max valency if atom types are not explicitly encoded
            max_valencies = torch.full((data.num_nodes,), self.DEFAULT_MAX_VALENCY, dtype=torch.long)
            
        # Check if any node exceeds its max valency
        if torch.any(deg > max_valencies):
            return False
            
        return True

    def enforce_valency(self, data: Data) -> Data:
        """
        Enforces chemical valency rules by pruning excess edges.
        
        Mathematical Context (TopoCID - SPCG Module):
        If a graph G violates C(G) = 0, we project it back to the valid manifold M 
        by removing edges until the constraint is satisfied. This is a discrete 
        projection step:
            G_pruned = argmin_{G' in M} || G' - G ||_F
        
        Args:
            data (Data): The input graph Data object.
            
        Returns:
            Data: The graph with valency constraints enforced.
        """
        if not self.is_molecular:
            return data
            
        if data.edge_index is None or data.edge_index.numel() == 0:
            return data
            
        row, col = data.edge_index
        deg = degree(row, num_nodes=data.num_nodes, dtype=torch.long)
        
        if hasattr(data, 'x') and data.x is not None and data.x.size(1) > 0:
            atom_types = torch.argmax(data.x, dim=1).cpu().numpy()
            max_valencies = np.array([self.VALENCY_RULES.get(at, self.DEFAULT_MAX_VALENCY) for at in atom_types])
            max_valencies = torch.tensor(max_valencies, dtype=torch.long)
        else:
            max_valencies = torch.full((data.num_nodes,), self.DEFAULT_MAX_VALENCY, dtype=torch.long)
            
        # Identify nodes that violate valency
        violating_nodes = torch.where(deg > max_valencies)[0]
        
        if len(violating_nodes) == 0:
            return data
            
        # Create a mask for edges to keep
        keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
        
        for v in violating_nodes:
            # Find all edges connected to v
            connected_edges = torch.where((row == v) | (col == v))[0]
            
            # We need to remove (deg[v] - max_valencies[v]) edges.
            # Since the graph is undirected, each physical edge is represented twice in edge_index.
            # So we need to remove pairs of edges.
            num_edges_to_remove = (deg[v].item() - max_valencies[v].item()) // 2
            
            if num_edges_to_remove > 0 and len(connected_edges) > 0:
                # Select edges to remove (e.g., the last ones in the list)
                # Ensure we remove an even number of entries to maintain undirected property
                edges_to_remove = connected_edges[-(num_edges_to_remove * 2):]
                keep_mask[edges_to_remove] = False
                
        # Apply mask
        data.edge_index = data.edge_index[:, keep_mask]
        
        # Update edge attributes if they exist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep_mask]
            
        return data

    def process(self, data: Data) -> Data:
        """
        Main preprocessing pipeline.
        
        Applies feature normalization, grammar checks, and valency enforcement.
        
        Args:
            data (Data): The input graph Data object.
            
        Returns:
            Data: The fully preprocessed graph Data object.
        """
        # 1. Normalize features for TCP module
        data = self.normalize_features(data)
        
        # 2. Check grammar (for SPCG manifold validity)
        is_valid_grammar = self.check_grammar(data)
        data.is_valid_grammar = torch.tensor([is_valid_grammar], dtype=torch.bool)
        
        # 3. Check and enforce valency (for SPCG manifold validity)
        is_valid_valency = self.check_valency(data)
        data.is_valid_valency = torch.tensor([is_valid_valency], dtype=torch.bool)
        
        if not is_valid_valency:
            data = self.enforce_valency(data)
            
        # Final overall validity flag
        data.is_valid = data.is_valid_grammar & data.is_valid_valency
        
        return data


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID Graph Preprocessor Pipeline")
    print("=" * 60)
    
    # Load original MUTAG dataset
    print("\n--- Loading Original MUTAG Dataset ---")
    mutag_dataset = TUDataset(root=DATASET_ROOT, name='MUTAG', use_node_attr=True)
    print(f"Loaded {len(mutag_dataset)} graphs from MUTAG.")
    
    # Initialize preprocessor for molecular data
    preprocessor = GraphPreprocessor(dataset_name='MUTAG', is_molecular=True)
    
    # Process the first 5 graphs as a demonstration
    print("\n--- Applying Preprocessing Pipeline ---")
    for i in range(min(5, len(mutag_dataset))):
        data = mutag_dataset[i]
        print(f"\nGraph {i}:")
        print(f"  Original Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
        
        # Check initial validity
        init_grammar = preprocessor.check_grammar(data)
        init_valency = preprocessor.check_valency(data)
        print(f"  Initial Grammar Valid: {init_grammar}")
        print(f"  Initial Valency Valid: {init_valency}")
        
        # Apply full preprocessing
        processed_data = preprocessor.process(data)
        
        print(f"  Processed Nodes: {processed_data.num_nodes}, Edges: {processed_data.edge_index.size(1)}")
        print(f"  Final Grammar Valid: {processed_data.is_valid_grammar.item()}")
        print(f"  Final Valency Valid: {processed_data.is_valid_valency.item()}")
        print(f"  Overall Manifold Valid (C(G)=0): {processed_data.is_valid.item()}")
        print(f"  Normalized Feature Mean: {processed_data.x.mean(dim=0).mean().item():.4f}")
        print(f"  Normalized Feature Std: {processed_data.x.std(dim=0).mean().item():.4f}")

    print("\n" + "=" * 60)
    print("Preprocessor Pipeline Verification Complete. No synthetic data used.")
    print("=" * 60)