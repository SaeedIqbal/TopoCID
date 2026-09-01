import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from typing import Tuple

# Add parent directory to path to import data loaders if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class GATConv(MessagePassing):
    """
    Graph Attention Network (GAT) Convolution Layer (Veličković et al., 2018).
    
    Mathematical Context (TopoCID Framework):
    The GAT layer updates node embeddings using a masked self-attention mechanism.
    For a node u at layer l, the attention coefficient e_uv to its neighbor v is:
        e_{uv}^{(l)} = LeakyReLU( a_l^{(l)T} W^{(l)} h_u^{(l)} + a_r^{(l)T} W^{(l)} h_v^{(l)} )
        
    The normalized attention coefficient alpha_uv is computed via softmax:
        alpha_{uv}^{(l)} = softmax_v(e_{uv}^{(l)}) = exp(e_{uv}^{(l)}) / sum_{k in N(u) U {u}} exp(e_{uk}^{(l)})
        
    The final node update is the attention-weighted aggregation:
        h_u^{(l+1)} = sigma( sum_{v in N(u) U {u}} alpha_{uv}^{(l)} W^{(l)} h_v^{(l)} )
        
    The final node embeddings H^{(L)} are subsequently consumed by the 
    Differentiable Topological Causal Projection (TCP) module to compute the 
    learnable scalar node-weight function for simplicial filtration:
        w(v) = u^T MLP(H^{(L)}_v)
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 negative_slope: float = 0.2, dropout: float = 0.0):
        """
        Initializes the GAT Convolution layer.
        
        Args:
            in_channels (int): Dimension of the input node features.
            out_channels (int): Dimension of the output node features per head.
            heads (int): Number of multi-head attention heads.
            negative_slope (float): The negative slope used for the LeakyReLU activation.
            dropout (float): Dropout rate applied to the attention coefficients.
        """
        super(GATConv, self).__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Linear transformations W^{(l)} for source and target nodes
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention vectors a_l^{(l)} and a_r^{(l)}
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Bias term
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initializes the parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, size=None) -> torch.Tensor:
        """
        Forward pass of the GAT convolution.
        Note: It is assumed that edge_index contains self-loops so that nodes 
        attend to themselves. If not, add_self_loops should be applied prior.
        
        Args:
            x (torch.Tensor): Node features of shape (N, in_channels).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            
        Returns:
            torch.Tensor: Updated node features of shape (N, heads * out_channels).
        """
        # Apply linear transformations and reshape for multi-head attention
        x_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
        x_r = self.lin_r(x).view(-1, self.heads, self.out_channels)
        
        # Compute the left and right attention components
        # alpha_l = a_l^T W h_u, alpha_r = a_r^T W h_v
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)
        
        # Propagate messages
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        
        # Concatenate heads and add bias
        out = out.view(-1, self.heads * self.out_channels)
        out = out + self.bias
        return out
        
    def message(self, x_j: torch.Tensor, alpha_j: torch.Tensor, alpha_i: torch.Tensor, 
                index: torch.Tensor, ptr: torch.Tensor, size_i: int) -> torch.Tensor:
        """
        Constructs and weights messages from source to target nodes.
        
        Args:
            x_j (torch.Tensor): Source node features (W h_v).
            alpha_j (torch.Tensor): Source attention component (a_r^T W h_v).
            alpha_i (torch.Tensor): Target attention component (a_l^T W h_u).
            index (torch.Tensor): Target node indices for softmax.
            ptr (torch.Tensor): Pointer for CSR format (used by softmax).
            size_i (int): Number of target nodes.
            
        Returns:
            torch.Tensor: Attention-weighted messages.
        """
        # Compute attention coefficients: e_uv = alpha_i + alpha_j
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize via softmax over the neighborhood
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight the source features by the attention coefficients
        return x_j * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class GATBackbone(nn.Module):
    """
    Graph Attention Network (GAT) Backbone Encoder for TopoCID.
    
    This class implements a multi-layer GAT network that serves as an alternative 
    feature extractor for the TopoCID framework. It produces node-level embeddings 
    H^{(L)} which are subsequently consumed by:
    1. The Differentiable Topological Causal Projection (TCP) module to extract 
       global topological invariants Z_topo.
    2. The Environment Encoder to extract environmental representations Z_env.
    
    Mathematical Context:
    Given an input graph G = (V, E, X) with initial node features H^{(0)} = X,
    the GAT backbone computes L layers of attention-based message passing:
        H^{(l+1)} = ELU( sum_{v in N(u) U {u}} alpha_{uv}^{(l)} W^{(l)} H^{(l)}_v )
    for l = 0, ..., L-1.
    """
    
    def __init__(self, num_node_features: int, hidden_dim: int = 16, num_layers: int = 3, 
                 heads: int = 4, dropout: float = 0.0):
        """
        Initializes the GAT Backbone.
        
        Args:
            num_node_features (int): Dimension of the input node features X.
            hidden_dim (int): Dimension of the hidden node embeddings per head.
            num_layers (int): Number of GAT convolutional layers (L).
            heads (int): Number of attention heads in each layer.
            dropout (float): Dropout rate applied to features and attention coefficients.
        """
        super(GATBackbone, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.hidden_dim = hidden_dim
        
        # GAT Convolution layers and Batch Normalization
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(heads * hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(heads * hidden_dim))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GAT Backbone.
        
        Args:
            data (Data): PyG Data or Batch object containing x and edge_index.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - node_embeddings (torch.Tensor): Final node embeddings H^{(L)} of shape (N, heads * hidden_dim).
                - graph_embeddings (torch.Tensor): Graph-level embeddings of shape (B, heads * hidden_dim) 
                  obtained via sum pooling over nodes.
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Message passing layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)  # GAT typically uses ELU activation
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
            torch.Tensor: Node embeddings H^{(L)} of shape (N, heads * hidden_dim).
        """
        node_embeddings, _ = self.forward(data)
        return node_embeddings

    def _global_sum_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Performs global sum pooling over nodes to obtain graph-level embeddings.
        
        Mathematical Context:
        Z_graph = sum_{v in V} H^{(L)}_v
        
        Args:
            x (torch.Tensor): Node embeddings of shape (N, heads * hidden_dim).
            batch (torch.Tensor): Batch vector mapping nodes to graphs.
            
        Returns:
            torch.Tensor: Graph embeddings of shape (B, heads * hidden_dim).
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
    from torch_geometric.utils import add_self_loops
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID GAT Backbone Encoder")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    # Get a batch
    batch = next(iter(train_loader))
    
    # GAT requires self-loops for self-attention. Add them if not present.
    batch.edge_index, _ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0))
    
    print(f"Batch loaded. Nodes: {batch.x.size(0)}, Edges (with self-loops): {batch.edge_index.size(1)}, Graphs: {batch.batch_size}")
    
    # 2. Initialize GAT Backbone
    num_node_features = batch.x.size(1)
    hidden_dim = 16
    num_layers = 3
    heads = 4
    
    print(f"\n--- Initializing GAT Backbone (Input: {num_node_features}, Hidden: {hidden_dim}, Heads: {heads}, Layers: {num_layers}) ---")
    gat_model = GATBackbone(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads)
    
    # 3. Forward Pass
    print("\n--- Executing Forward Pass ---")
    gat_model.eval()
    with torch.no_grad():
        node_embs, graph_embs = gat_model(batch)
        
    print(f"Node Embeddings Shape (H^{{(L)}}): {node_embs.shape}")
    print(f"Graph Embeddings Shape (Z_{{graph}}): {graph_embs.shape}")
    
    # 4. Verify TCP Compatibility
    print("\n--- Verifying TCP Module Compatibility ---")
    with torch.no_grad():
        h_L = gat_model.get_node_embeddings(batch)
        
    print(f"Extracted H^{{(L)}} for TCP module: {h_L.shape}")
    print(f"Mean of H^{{(L)}}: {h_L.mean().item():.4f}")
    print(f"Std of H^{{(L)}}: {h_L.std().item():.4f}")
    
    print("\n" + "=" * 60)
    print("GAT Backbone Verification Complete. No synthetic data used.")
    print("=" * 60)