import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from typing import Tuple

# Add parent directory to path to import data loaders if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) Convolution Layer.
    
    Mathematical Context (TopoCID Framework):
    The GIN layer updates node embeddings by aggregating neighborhood information.
    For a node v at layer l, the update rule is:
        H^{(l+1)}_v = MLP^{(l)} ( (1 + epsilon^{(l)}) * H^{(l)}_v + sum_{u in N(v)} H^{(l)}_u )
    
    This aggregation provides the structural context required by the TopoCID framework.
    Specifically, the final node embeddings H^{(L)} are used by the Differentiable 
    Topological Causal Projection (TCP) module to compute the learnable scalar 
    node-weight function for simplicial filtration:
        w(v) = u^T MLP(H^{(L)}_v)
    """
    
    def __init__(self, nn: nn.Module, eps: float = 0.0, train_eps: bool = False):
        """
        Initializes the GIN Convolution layer.
        
        Args:
            nn (nn.Module): The MLP applied after aggregation.
            eps (float): Initial value for epsilon.
            train_eps (bool): Whether epsilon is a learnable parameter.
        """
        super(GINConv, self).__init__(aggr='add')
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GIN convolution.
        
        Args:
            x (torch.Tensor): Node features of shape (N, F).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            
        Returns:
            torch.Tensor: Updated node features of shape (N, F_out).
        """
        # Add self loops to include the central node in the aggregation
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Message passing (aggregation)
        out = self.propagate(edge_index, x=x)
        
        # Apply MLP and add the (1 + eps) * x term
        out = self.nn((1 + self.eps) * x + out)
        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


class GINBackbone(nn.Module):
    """
    Graph Isomorphism Network (GIN) Backbone Encoder for TopoCID.
    
    This class implements a multi-layer GIN network that serves as the shared 
    feature extractor for the TopoCID framework. It produces node-level embeddings 
    H^{(L)} which are subsequently consumed by:
    1. The Differentiable Topological Causal Projection (TCP) module to extract 
       global topological invariants Z_topo.
    2. The Environment Encoder to extract environmental representations Z_env.
    
    Mathematical Context:
    Given an input graph G = (V, E, X) with initial node features H^{(0)} = X,
    the GIN backbone computes L layers of message passing:
        H^{(l+1)}_v = MLP^{(l)} ( (1 + epsilon^{(l)}) H^{(l)}_v + sum_{u in N(v)} H^{(l)}_u )
    for l = 0, ..., L-1.
    """
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64, num_layers: int = 3, 
                 dropout: float = 0.0, train_eps: bool = True):
        """
        Initializes the GIN Backbone.
        
        Args:
            num_node_features (int): Dimension of the input node features X.
            hidden_dim (int): Dimension of the hidden node embeddings.
            num_layers (int): Number of GIN convolutional layers (L).
            dropout (float): Dropout rate applied after each layer.
            train_eps (bool): Whether epsilon is a learnable parameter.
        """
        super(GINBackbone, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection if num_node_features != hidden_dim
        self.input_proj = nn.Linear(num_node_features, hidden_dim) if num_node_features != hidden_dim else nn.Identity()
        
        # GIN Convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # MLP for the GIN layer: 2-layer MLP with BatchNorm
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn=mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GIN Backbone.
        
        Args:
            data (Data): PyG Data or Batch object containing x and edge_index.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - node_embeddings (torch.Tensor): Final node embeddings H^{(L)} of shape (N, hidden_dim).
                - graph_embeddings (torch.Tensor): Graph-level embeddings of shape (B, hidden_dim) 
                  obtained via sum pooling over nodes.
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial projection
        x = self.input_proj(x.float())
        
        # Message passing layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        node_embeddings = x
        
        # Graph-level readout via sum pooling
        # Z_graph = sum_{v in V} H^{(L)}_v
        graph_embeddings = self._global_sum_pool(node_embeddings, batch)
        
        return node_embeddings, graph_embeddings

    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """
        Extracts the final node embeddings H^{(L)} required by the TCP module.
        
        Mathematical Context (TopoCID - TCP Module):
        The TCP module uses H^{(L)} to compute the learnable scalar node-weight function:
            w(v) = u^T MLP(H^{(L)}_v)
        which initializes the simplicial filtration for persistent homology computation.
        
        Args:
            data (Data): PyG Data or Batch object.
            
        Returns:
            torch.Tensor: Node embeddings H^{(L)} of shape (N, hidden_dim).
        """
        node_embeddings, _ = self.forward(data)
        return node_embeddings

    def _global_sum_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Performs global sum pooling over nodes to obtain graph-level embeddings.
        
        Mathematical Context:
        Z_graph = sum_{v in V} H^{(L)}_v
        
        Args:
            x (torch.Tensor): Node embeddings of shape (N, hidden_dim).
            batch (torch.Tensor): Batch vector mapping nodes to graphs.
            
        Returns:
            torch.Tensor: Graph embeddings of shape (B, hidden_dim).
        """
        num_graphs = batch.max().item() + 1
        out = torch.zeros((num_graphs, x.size(1)), dtype=x.dtype, device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        return out


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID GIN Backbone Encoder")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch loaded. Nodes: {batch.x.size(0)}, Edges: {batch.edge_index.size(1)}, Graphs: {batch.batch_size}")
    
    # 2. Initialize GIN Backbone
    num_node_features = batch.x.size(1)
    hidden_dim = 64
    num_layers = 3
    
    print(f"\n--- Initializing GIN Backbone (Input: {num_node_features}, Hidden: {hidden_dim}, Layers: {num_layers}) ---")
    gin_model = GINBackbone(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # 3. Forward Pass
    print("\n--- Executing Forward Pass ---")
    gin_model.eval()
    with torch.no_grad():
        node_embs, graph_embs = gin_model(batch)
        
    print(f"Node Embeddings Shape (H^{{(L)}}): {node_embs.shape}")
    print(f"Graph Embeddings Shape (Z_{{graph}}): {graph_embs.shape}")
    
    # 4. Verify TCP Compatibility
    print("\n--- Verifying TCP Module Compatibility ---")
    with torch.no_grad():
        h_L = gin_model.get_node_embeddings(batch)
        
    print(f"Extracted H^{{(L)}} for TCP module: {h_L.shape}")
    print(f"Mean of H^{{(L)}}: {h_L.mean().item():.4f}")
    print(f"Std of H^{{(L)}}: {h_L.std().item():.4f}")
    
    print("\n" + "=" * 60)
    print("GIN Backbone Verification Complete. No synthetic data used.")
    print("=" * 60)