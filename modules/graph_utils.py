import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from typing import Tuple, Dict

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class GraphLaplacian(nn.Module):
    """
    Computes the Graph Laplacian matrices.
    
    Mathematical Context (TopoCID Framework):
    The unnormalized Graph Laplacian is defined as:
        L = D - A
    where D is the diagonal degree matrix and A is the adjacency matrix.
    
    The symmetric normalized Graph Laplacian is defined as:
        L_sym = I - D^{-1/2} A D^{-1/2}
        
    The eigenvalues of L are non-negative: 0 = lambda_1 <= lambda_2 <= ... <= lambda_n.
    The multiplicity of lambda_1 = 0 equals the number of connected components (Betti-0).
    """
    
    def __init__(self, normalized: bool = False):
        """
        Args:
            normalized (bool): If True, compute the symmetric normalized Laplacian L_sym.
        """
        super().__init__()
        self.normalized = normalized

    def forward(self, edge_index: torch.Tensor, num_nodes: int, 
                batch: torch.Tensor = None, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the dense Graph Laplacian for a batch of graphs.
        
        Args:
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            num_nodes (int): Total number of nodes in the batch.
            batch (torch.Tensor): Batch vector mapping nodes to graphs.
            edge_weight (torch.Tensor): Optional edge weights for weighted Laplacian.
            
        Returns:
            torch.Tensor: Dense Laplacian matrix of shape (B, max_N, max_N) if batched, 
                          or (N, N) if unbatched.
        """
        # 1. Compute dense adjacency matrix A
        if batch is not None:
            A = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)
            # A shape: (B, max_N, max_N)
        else:
            A = to_dense_adj(edge_index, max_num_nodes=num_nodes, edge_attr=edge_weight)
            A = A.squeeze(0) # (N, N)
            
        # 2. Compute degree matrix D and Laplacian L
        if A.dim() == 3:
            # Batched
            D_diag = A.sum(dim=-1) # (B, max_N)
            D = torch.diag_embed(D_diag) # (B, max_N, max_N)
            I = torch.eye(A.size(1), device=A.device).unsqueeze(0).expand_as(A)
            
            if self.normalized:
                D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D_diag + 1e-8))
                L = I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
            else:
                L = D - A
        else:
            # Unbatched
            D_diag = A.sum(dim=-1) # (N,)
            D = torch.diag(D_diag) # (N, N)
            I = torch.eye(A.size(0), device=A.device)
            
            if self.normalized:
                D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_diag + 1e-8))
                L = I - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
            else:
                L = D - A
                
        return L


class SpectralFeatureExtractor(nn.Module):
    """
    Extracts spectral features from the Graph Laplacian.
    
    Mathematical Context (TopoCID Framework):
    Computes the eigenvalues and eigenvectors of the Laplacian L.
    The spectral gap (algebraic connectivity) is defined as:
        gap = lambda_2 - lambda_1
    where lambda_1 = 0 for connected graphs. The Fiedler vector is the eigenvector 
    corresponding to lambda_2, which provides a continuous relaxation of the graph 
    bipartition and is a key global topological invariant captured by the TCP module.
    """
    
    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k (int): Number of smallest eigenvalues/eigenvectors to extract.
        """
        super().__init__()
        self.top_k = top_k

    def forward(self, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the spectral features.
        
        Args:
            L (torch.Tensor): Dense Laplacian matrix of shape (B, N, N) or (N, N).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - eigenvalues (B, top_k) or (top_k,)
                - eigenvectors (B, N, top_k) or (N, top_k)
                - spectral_gap (B,) or scalar
        """
        if L.dim() == 3:
            # Batched
            B, N, _ = L.shape
            # Symmetrize to ensure real eigenvalues (numerical stability)
            L_sym = (L + L.transpose(1, 2)) / 2.0
            
            # Compute eigenvalues and eigenvectors
            # torch.linalg.eigh returns them in ascending order
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym) # (B, N), (B, N, N)
            
            # Extract top_k smallest
            k = min(self.top_k, N)
            top_eigenvalues = eigenvalues[:, :k] # (B, k)
            top_eigenvectors = eigenvectors[:, :, :k] # (B, N, k)
            
            # Spectral gap: lambda_2 - lambda_1
            if k >= 2:
                spectral_gap = top_eigenvalues[:, 1] - top_eigenvalues[:, 0] # (B,)
            else:
                spectral_gap = torch.zeros(B, device=L.device)
                
            return top_eigenvalues, top_eigenvectors, spectral_gap
            
        else:
            # Unbatched
            N = L.size(0)
            L_sym = (L + L.transpose(0, 1)) / 2.0
            
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
            
            k = min(self.top_k, N)
            top_eigenvalues = eigenvalues[:k]
            top_eigenvectors = eigenvectors[:, :k]
            
            if k >= 2:
                spectral_gap = top_eigenvalues[1] - top_eigenvalues[0]
            else:
                spectral_gap = torch.tensor(0.0, device=L.device)
                
            return top_eigenvalues, top_eigenvectors, spectral_gap


class CliqueComplexBuilder(nn.Module):
    """
    Constructs the Clique Complex (Flag Complex) from a graph.
    
    Mathematical Context (TopoCID Framework):
    The clique complex Cl(G) is formed by filling in every clique (complete subgraph) 
    of G with a simplex. 
    - 0-simplices: Nodes
    - 1-simplices: Edges
    - 2-simplices: Triangles (3-cliques)
    - k-simplices: (k+1)-cliques
    
    This module identifies the simplices up to a maximum dimension k_max, which are 
    then used by the SimplicialFiltration module to compute persistent homology.
    """
    
    def __init__(self, max_dim: int = 2):
        """
        Args:
            max_dim (int): Maximum dimension of simplices to extract (e.g., 2 for triangles).
        """
        super().__init__()
        self.max_dim = max_dim

    def _get_adjacency_matrix(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Helper to get dense adjacency matrix for a single graph."""
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes)
        return A.squeeze(0)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> Dict[int, torch.Tensor]:
        """
        Extracts simplices up to max_dim for a single graph.
        
        Args:
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            num_nodes (int): Number of nodes in the graph.
            
        Returns:
            Dict[int, torch.Tensor]: Dictionary mapping dimension k to a tensor of 
                                     shape (num_simplices_k, k+1) containing node indices.
        """
        A = self._get_adjacency_matrix(edge_index, num_nodes)
        
        simplices = {}
        
        # 0-simplices (Nodes)
        simplices[0] = torch.arange(num_nodes, device=A.device).unsqueeze(1) # (N, 1)
        
        # 1-simplices (Edges)
        # Extract upper triangle to avoid duplicates and self-loops
        row, col = torch.triu(A, diagonal=1).nonzero(as_tuple=True)
        simplices[1] = torch.stack([row, col], dim=1) # (E, 2)
        
        if self.max_dim >= 2:
            # 2-simplices (Triangles)
            # A triangle exists if A[i,j]=1, A[j,k]=1, A[i,k]=1 for i < j < k
            triangles = []
            edges = simplices[1]
            for i in range(edges.size(0)):
                u, v = edges[i].tolist()
                # Find common neighbors of u and v
                common = (A[u] * A[v]).nonzero(as_tuple=True)[0]
                # Filter for w > v to maintain strict order u < v < w
                w_candidates = common[common > v]
                for w in w_candidates.tolist():
                    triangles.append([u, v, w])
                    
            if len(triangles) > 0:
                simplices[2] = torch.tensor(triangles, dtype=torch.long, device=A.device)
            else:
                simplices[2] = torch.empty((0, 3), dtype=torch.long, device=A.device)
                
        return simplices


class GraphUtilsModule(nn.Module):
    """
    Unified Graph Utilities Module for TopoCID.
    Integrates Laplacian computation, spectral feature extraction, and clique complex 
    construction to provide the foundational topological and spectral invariants.
    """
    
    def __init__(self, top_k: int = 5, max_clique_dim: int = 2, normalized_laplacian: bool = True):
        """
        Args:
            top_k (int): Number of smallest eigenvalues to extract for spectral features.
            max_clique_dim (int): Maximum dimension of simplices to extract for the clique complex.
            normalized_laplacian (bool): Whether to use the normalized Laplacian.
        """
        super().__init__()
        self.laplacian = GraphLaplacian(normalized=normalized_laplacian)
        self.spectral = SpectralFeatureExtractor(top_k=top_k)
        self.clique = CliqueComplexBuilder(max_dim=max_clique_dim)
        
    def forward(self, edge_index: torch.Tensor, num_nodes: int, 
                batch: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Computes all graph utilities.
        
        Args:
            edge_index (torch.Tensor): Edge indices.
            num_nodes (int): Total number of nodes.
            batch (torch.Tensor): Batch vector.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing Laplacian, eigenvalues, 
                                     eigenvectors, spectral gap, and simplices.
        """
        # 1. Laplacian (Batched)
        L = self.laplacian(edge_index, num_nodes, batch)
        
        # 2. Spectral Features (Batched)
        eigenvalues, eigenvectors, spectral_gap = self.spectral(L)
        
        # 3. Clique Complex (Unbatched / Single Graph)
        # Note: Clique complex construction is computationally intensive. 
        # Here we demonstrate it for the first graph in the batch.
        if batch is not None:
            first_graph_mask = batch == 0
            edge_index_first = edge_index[:, first_graph_mask]
            num_nodes_first = first_graph_mask.sum().item()
            simplices = self.clique(edge_index_first, num_nodes_first)
        else:
            simplices = self.clique(edge_index, num_nodes)
            
        return {
            'laplacian': L,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'spectral_gap': spectral_gap,
            'simplices': simplices
        }


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    from data.dataloaders import TopoCIDDataModule
    
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing Graph Utilities Module (Laplacian, Spectral, Clique)")
    print("=" * 60)
    
    # 1. Load original dataset
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs: {batch.batch_size}, Nodes: {batch.x.size(0)}, Edges: {batch.edge_index.size(1)}")
    
    # 2. Initialize Graph Utils
    print("\n--- Initializing Graph Utils Module ---")
    graph_utils = GraphUtilsModule(top_k=5, max_clique_dim=2, normalized_laplacian=True)
    
    # 3. Forward Pass
    print("\n--- Executing Forward Pass ---")
    graph_utils.eval()
    
    with torch.no_grad():
        # Batched Laplacian and Spectral Features
        L_batched = graph_utils.laplacian(batch.edge_index, batch.x.size(0), batch.batch)
        evals, evecs, sgap = graph_utils.spectral(L_batched)
        
        print(f"Batched Laplacian Shape: {L_batched.shape}")
        print(f"Batched Eigenvalues Shape: {evals.shape}")
        print(f"Batched Spectral Gap Shape: {sgap.shape}")
        print(f"Mean Spectral Gap (Algebraic Connectivity): {sgap.mean().item():.4f}")
        
        # Unbatched Clique Complex for the first graph
        first_graph_mask = batch.batch == 0
        edge_index_first = batch.edge_index[:, first_graph_mask]
        num_nodes_first = first_graph_mask.sum().item()
        
        simplices = graph_utils.clique(edge_index_first, num_nodes_first)
        print(f"\nClique Complex for First Graph:")
        print(f"  0-simplices (Nodes): {simplices[0].shape}")
        print(f"  1-simplices (Edges): {simplices[1].shape}")
        print(f"  2-simplices (Triangles): {simplices[2].shape}")
        
    # 4. Backward Pass Verification
    print("\n--- Verifying Differentiability (Backward Pass) ---")
    graph_utils.train()
    
    # Create a dummy edge weight tensor that requires grad to test Laplacian differentiability
    edge_weight = torch.ones(batch.edge_index.size(1), requires_grad=True, device=batch.x.device)
    
    L_test = graph_utils.laplacian(batch.edge_index, batch.x.size(0), batch.batch, edge_weight)
    evals_test, _, _ = graph_utils.spectral(L_test)
    
    loss = evals_test.sum()
    loss.backward()
    
    print(f"Edge Weight Gradient Norm: {edge_weight.grad.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("Graph Utils Verification Complete. No synthetic/random data used.")
    print("=" * 60)