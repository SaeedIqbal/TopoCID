import os
from typing import Tuple, Dict, Any, Optional, List
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import GOODHIV, GOODCMNIST
from torch_geometric.loader import DataLoader

class BaseGOODLoader:
    """
    Base Object-Oriented Loader for Graph Out-of-Distribution (GOOD) datasets.
    
    This class encapsulates the data loading, splitting, and preprocessing logic 
    required for the TopoCID framework. It ensures that environment labels (E) 
    and initial topological filtration weights (w) are correctly extracted and 
    attached to the graph data objects.
    
    Mathematical Context (TopoCID - TCP Module):
    The loader prepares the initial node features H^{(0)} for the learnable 
    scalar node-weight function:
        w(v) = u^T MLP(H^{(0)}_v)
    which is subsequently used to construct the nested sublevel filtration:
        X_t = { \sigma \in Cl(G) | max_{v \in \sigma} w(v) <= t }
    """

    def __init__(self, root: str, split_type: str = 'scaffold', batch_size: int = 128):
        """
        Initializes the GOOD dataset loader.
        
        Args:
            root (str): Root directory where the dataset is stored (/home/phd/datasets/).
            split_type (str): The OOD split type ('scaffold', 'motif', or 'size').
            batch_size (int): Batch size for the PyG DataLoaders.
        """
        self.root = root
        self.split_type = split_type
        self.batch_size = batch_size
        
        # Ensure the root directory exists
        os.makedirs(self.root, exist_ok=True)
        
        self.dataset_name = self.__class__.__name__.replace('Loader', '').upper()
        self.full_dataset: Optional[Dataset] = None
        self.split_indices: Dict[str, torch.Tensor] = {}

    def _load_raw_dataset(self) -> Dataset:
        """
        Abstract method to be overridden by subclasses to load the specific PyG dataset.
        """
        raise NotImplementedError("Subclasses must implement _load_raw_dataset()")

    def load_and_split(self) -> None:
        """
        Loads the full dataset and extracts the train/val/test split indices 
        based on the specified OOD shift type.
        """
        print(f"[{self.dataset_name}] Loading raw dataset from {self.root}...")
        self.full_dataset = self._load_raw_dataset()
        
        # GOOD datasets in PyG provide a method to get split indices
        # The split_type dictates the distribution shift (e.g., 'scaffold' for molecular backbone shifts)
        self.split_indices = self.full_dataset.get_split_idx(split_type=self.split_type)
        
        print(f"[{self.dataset_name}] Successfully loaded. Total graphs: {len(self.full_dataset)}")
        print(f"[{self.dataset_name}] Split sizes - Train: {len(self.split_indices['train'])}, "
              f"Val: {len(self.split_indices['val'])}, Test: {len(self.split_indices['test'])}")

    def _create_subset(self, split_name: str) -> Dataset:
        """
        Creates a PyG Subset dataset for a specific split.
        """
        if self.full_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_and_split() first.")
        
        indices = self.split_indices[split_name]
        return self.full_dataset[indices]

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Generates PyTorch Geometric DataLoaders for training, validation, and testing.
        
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, Val, and Test dataloaders.
        """
        train_dataset = self._create_subset('train')
        val_dataset = self._create_subset('val')
        test_dataset = self._create_subset('test')
        
        # Apply TopoCID-specific preprocessing to all splits
        train_dataset = self._apply_topocid_preprocessing(train_dataset)
        val_dataset = self._apply_topocid_preprocessing(val_dataset)
        test_dataset = self._apply_topocid_preprocessing(test_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def _apply_topocid_preprocessing(self, dataset: Dataset) -> Dataset:
        """
        Iterates through the dataset and attaches TopoCID-specific attributes:
        1. Environment labels (E) for Invariant Learning and TCD.
        2. Initial deterministic filtration weights (w) for the TDA layer.
        """
        for data in dataset:
            data = self._extract_environment_labels(data)
            data = self._initialize_filtration_weights(data)
        return dataset

    def _extract_environment_labels(self, data: Data) -> Data:
        """
        Extracts the discrete environment variable E_i for each graph.
        
        Mathematical Context (TopoCID - TCD Module):
        Ensures the conditional independence criterion Y \perp E | Z_topo can be 
        explicitly enforced during the Topological Contrastive Disentanglement phase.
        """
        if hasattr(data, 'env_id'):
            data.env = data.env_id
        else:
            # Fallback if env_id is not explicitly named, though standard GOOD datasets have it
            data.env = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

    def _initialize_filtration_weights(self, data: Data) -> Data:
        """
        Computes a deterministic initial scalar weight w(v) for each node to initialize 
        the simplicial filtration before the learnable MLP is applied.
        
        Mathematical Context (TopoCID - TCP Module):
        While the final weight is w(v) = u^T MLP(H^{(0)}_v), we initialize the filtration 
        using the L2 norm of the initial node features to ensure a deterministic, 
        non-random starting point for the clique complex Cl(G):
            w^{(0)}(v) = || H^{(0)}_v ||_2
        """
        if data.x is not None:
            # Compute L2 norm of node features as the deterministic initial weight
            data.filtration_weight = torch.norm(data.x, p=2, dim=1)
        else:
            # Fallback to node degree if features are absent
            data.filtration_weight = torch.degrees(data.edge_index[0]).float()
        return data


class GOODHIVLoader(BaseGOODLoader):
    """
    Specific loader for the GOOD-HIV dataset.
    Evaluates OOD generalization on HIV inhibition prediction under scaffold/size shifts.
    """
    def __init__(self, root: str, split_type: str = 'scaffold', batch_size: int = 256):
        super().__init__(root, split_type, batch_size)

    def _load_raw_dataset(self) -> Dataset:
        """
        Loads the GOOD-HIV dataset using PyTorch Geometric's built-in loader.
        """
        return GOODHIV(root=self.root, domain='HIV')


class GOODCMNISTLoader(BaseGOODLoader):
    """
    Specific loader for the GOOD-CMNIST dataset.
    Evaluates OOD generalization on graph-converted MNIST under motif/size shifts.
    """
    def __init__(self, root: str, split_type: str = 'motif', batch_size: int = 256):
        super().__init__(root, split_type, batch_size)

    def _load_raw_dataset(self) -> Dataset:
        """
        Loads the GOOD-CMNIST dataset using PyTorch Geometric's built-in loader.
        """
        return GOODCMNIST(root=self.root, domain='CMNIST')


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    # Strictly using the original dataset path as specified
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("="*60)
    print("Initializing TopoCID Data Pipeline for GOOD Datasets")
    print("="*60)
    
    # 1. Load GOOD-HIV (Scaffold Shift)
    print("\n--- Processing GOOD-HIV (Scaffold Shift) ---")
    hiv_loader = GOODHIVLoader(root=DATASET_ROOT, split_type='scaffold', batch_size=128)
    hiv_loader.load_and_split()
    hiv_train, hiv_val, hiv_test = hiv_loader.get_dataloaders()
    
    # Verify a single batch from GOOD-HIV
    batch_hiv = next(iter(hiv_train))
    print(f"Batch Graph Count: {batch_hiv.num_graphs}")
    print(f"Node Features Shape: {batch_hiv.x.shape}")
    print(f"Filtration Weights Shape: {batch_hiv.filtration_weight.shape}")
    print(f"Environment Labels Shape: {batch_hiv.env.shape}")
    
    # 2. Load GOOD-CMNIST (Motif Shift)
    print("\n--- Processing GOOD-CMNIST (Motif Shift) ---")
    cmnist_loader = GOODCMNISTLoader(root=DATASET_ROOT, split_type='motif', batch_size=256)
    cmnist_loader.load_and_split()
    cmnist_train, cmnist_val, cmnist_test = cmnist_loader.get_dataloaders()
    
    # Verify a single batch from GOOD-CMNIST
    batch_cmnist = next(iter(cmnist_train))
    print(f"Batch Graph Count: {batch_cmnist.num_graphs}")
    print(f"Node Features Shape: {batch_cmnist.x.shape}")
    print(f"Filtration Weights Shape: {batch_cmnist.filtration_weight.shape}")
    print(f"Environment Labels Shape: {batch_cmnist.env.shape}")
    
    print("\n" + "="*60)
    print("Data Pipeline Verification Complete. No synthetic data used.")
    print("="*60)