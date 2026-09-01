import os
import torch
from typing import List, Optional, Tuple, Union
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class TopoCIDCollater:
    """
    Custom Collater for the TopoCID Framework.
    
    This collater extends the standard PyTorch Geometric batching process to 
    explicitly compute and attach the positive and negative pair indices required 
    by the Topological Contrastive Disentanglement (TCD) module.
    
    Mathematical Context (TopoCID - TCD Module):
    The TCD module utilizes an InfoNCE contrastive objective to align topological 
    invariants across distinct environments. For a given batch B, the positive 
    pair set P and negative pair set N are defined as:
        P = {(i, j) in B x B | y_i = y_j, E_i != E_j}
        N = {(i, k) in B x B | y_i != y_k} U {(i, k) in B x B | y_i = y_k, E_i = E_k}
        
    The InfoNCE loss is then computed as:
        L_TCD = - (1 / |P|) * sum_{(i,j) in P} log [ exp(kappa(z_i, z_j) / tau) / sum_{k != i} exp(kappa(z_i, z_k) / tau) ]
    """
    
    def __init__(self, follow_batch: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None):
        """
        Initializes the TopoCIDCollater.
        
        Args:
            follow_batch (Optional[List[str]]): Keys for which to generate batch indices.
            exclude_keys (Optional[List[str]]): Keys to exclude from batching.
        """
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, data_list: List[Data]) -> Batch:
        """
        Collates a list of Data objects into a single Batch object, computing 
        the TCD positive and negative pair indices.
        
        Args:
            data_list (List[Data]): List of graph Data objects.
            
        Returns:
            Batch: The collated batch with attached pair indices.
        """
        # 1. Standard PyG Batching
        batch = Batch.from_data_list(
            data_list, 
            follow_batch=self.follow_batch, 
            exclude_keys=self.exclude_keys
        )
        
        # 2. Extract Labels and Environments
        # Ensure y and env are present and correctly shaped
        if not hasattr(batch, 'y') or batch.y is None:
            raise ValueError("Batch does not contain target labels 'y'.")
        if not hasattr(batch, 'env') or batch.env is None:
            raise ValueError("Batch does not contain environment labels 'env'. Required for TCD.")
            
        y = batch.y.view(-1)
        env = batch.env.view(-1)
        batch_size = y.size(0)
        device = y.device
        
        # 3. Compute Pair Masks for TCD
        # y_eq[i, j] is True if y_i == y_j
        y_eq = (y.unsqueeze(0) == y.unsqueeze(1))
        
        # env_neq[i, j] is True if E_i != E_j
        env_neq = (env.unsqueeze(0) != env.unsqueeze(1))
        
        # Positive pairs: y_i == y_j AND E_i != E_j
        pos_mask = y_eq & env_neq
        
        # Negative pairs: NOT (y_i == y_j AND E_i != E_j)
        neg_mask = ~pos_mask
        
        # Remove diagonal (i == j) from both masks
        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = pos_mask & diag_mask
        neg_mask = neg_mask & diag_mask
        
        # 4. Extract Pair Indices
        # pos_pairs shape: (N_pos, 2), where each row is [i, j]
        pos_pairs = torch.nonzero(pos_mask, as_tuple=False)
        neg_pairs = torch.nonzero(neg_mask, as_tuple=False)
        
        # Attach to batch
        batch.positive_pairs = pos_pairs
        batch.negative_pairs = neg_pairs
        
        # Store batch size for convenience
        batch.batch_size = batch_size
        
        return batch


class TopoCIDDataLoader(DataLoader):
    """
    Custom DataLoader wrapper for the TopoCID Framework.
    
    This DataLoader utilizes the TopoCIDCollater to ensure that every batch 
    contains the necessary positive and negative pair indices for the TCD module, 
    alongside the standard batched graph data.
    """
    
    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = False, 
                 follow_batch: Optional[List[str]] = None, 
                 exclude_keys: Optional[List[str]] = None, 
                 **kwargs):
        """
        Initializes the TopoCIDDataLoader.
        
        Args:
            dataset: The PyG dataset (or list of Data objects) to load from.
            batch_size (int): Number of graphs per batch.
            shuffle (bool): Whether to shuffle the dataset at every epoch.
            follow_batch (Optional[List[str]]): Keys for which to generate batch indices.
            exclude_keys (Optional[List[str]]): Keys to exclude from batching.
            **kwargs: Additional arguments passed to the PyG DataLoader.
        """
        self.collater = TopoCIDCollater(follow_batch=follow_batch, exclude_keys=exclude_keys)
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self.collater, 
            **kwargs
        )


class TopoCIDDataModule:
    """
    High-level Data Module for the TopoCID Framework.
    
    This class abstracts the dataset loading and DataLoader creation process, 
    ensuring that the correct dataset-specific loaders (from good.py, drugood.py, 
    tu_dataset.py) are instantiated and wrapped with the TopoCIDDataLoader to 
    provide environment-label batching.
    """
    
    SUPPORTED_DATASETS = {
        'GOOD-HIV', 'GOOD-CMNIST', 'DrugOOD-IC50', 
        'MUTAG', 'PROTEINS', 'NCI1'
    }
    
    def __init__(self, dataset_name: str, root: str = "/home/phd/datasets/", 
                 batch_size: int = 128, split_type: str = 'scaffold', **kwargs):
        """
        Initializes the TopoCIDDataModule.
        
        Args:
            dataset_name (str): Name of the dataset (must be in SUPPORTED_DATASETS).
            root (str): Root directory for dataset storage.
            batch_size (int): Batch size for the DataLoaders.
            split_type (str): The OOD split type (e.g., 'scaffold', 'motif', 'assay').
            **kwargs: Additional arguments for the dataset loaders.
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Must be one of {self.SUPPORTED_DATASETS}")
            
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.split_type = split_type
        self.kwargs = kwargs
        
        self.train_loader: Optional[TopoCIDDataLoader] = None
        self.val_loader: Optional[TopoCIDDataLoader] = None
        self.test_loader: Optional[TopoCIDDataLoader] = None
        
    def _load_datasets(self) -> Tuple:
        """
        Loads the train, val, and test datasets based on the dataset_name.
        Reuses the specific loader classes defined in good.py, drugood.py, and tu_dataset.py.
        
        Returns:
            Tuple: (train_dataset, val_dataset, test_dataset)
        """
        # Importing locally to avoid circular dependencies and ensure modularity
        if self.dataset_name in ['GOOD-HIV', 'GOOD-CMNIST']:
            from data.good import GOODHIVLoader, GOODCMNISTLoader
            loader_cls = GOODHIVLoader if self.dataset_name == 'GOOD-HIV' else GOODCMNISTLoader
            loader = loader_cls(root=self.root, split_type=self.split_type, batch_size=self.batch_size)
            loader.load_and_split()
            return loader._create_subset('train'), loader._create_subset('val'), loader._create_subset('test')
            
        elif self.dataset_name == 'DrugOOD-IC50':
            from data.drugood import DrugOODIC50Loader
            loader = DrugOODIC50Loader(root=self.root, split_type=self.split_type, batch_size=self.batch_size)
            loader.load_and_split()
            return loader.datasets['train'], loader.datasets['val'], loader.datasets['test']
            
        elif self.dataset_name in ['MUTAG', 'PROTEINS', 'NCI1']:
            from data.tu_dataset import MUTAGLoader, PROTEINSLoader, NCI1Loader
            if self.dataset_name == 'MUTAG':
                loader = MUTAGLoader(root=self.root, batch_size=self.batch_size)
            elif self.dataset_name == 'PROTEINS':
                loader = PROTEINSLoader(root=self.root, batch_size=self.batch_size)
            else:
                loader = NCI1Loader(root=self.root, batch_size=self.batch_size)
                
            loader.load_and_split()
            # For TUDatasets, apply the preprocessing and environment derivation
            all_env_labels = loader._derive_environment_labels()
            train_data = loader._apply_topocid_preprocessing(loader._create_subset('train'), loader.split_indices['train'], all_env_labels)
            val_data = loader._apply_topocid_preprocessing(loader._create_subset('val'), loader.split_indices['val'], all_env_labels)
            test_data = loader._apply_topocid_preprocessing(loader._create_subset('test'), loader.split_indices['test'], all_env_labels)
            return train_data, val_data, test_data
            
        else:
            raise NotImplementedError(f"Loading logic for {self.dataset_name} is not implemented.")

    def prepare_data(self) -> None:
        """
        Loads the datasets and initializes the TopoCIDDataLoaders.
        """
        print(f"[TopoCIDDataModule] Preparing data for {self.dataset_name}...")
        train_data, val_data, test_data = self._load_datasets()
        
        # Wrap the datasets in our custom TopoCIDDataLoader
        self.train_loader = TopoCIDDataLoader(train_data, batch_size=self.batch_size, shuffle=True, **self.kwargs)
        self.val_loader = TopoCIDDataLoader(val_data, batch_size=self.batch_size, shuffle=False, **self.kwargs)
        self.test_loader = TopoCIDDataLoader(test_data, batch_size=self.batch_size, shuffle=False, **self.kwargs)
        
        print(f"[TopoCIDDataModule] DataLoaders initialized with environment-label batching for TCD.")

    def train_dataloader(self) -> TopoCIDDataLoader:
        """Returns the training DataLoader."""
        if self.train_loader is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        return self.train_loader

    def val_dataloader(self) -> TopoCIDDataLoader:
        """Returns the validation DataLoader."""
        if self.val_loader is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        return self.val_loader

    def test_dataloader(self) -> TopoCIDDataLoader:
        """Returns the test DataLoader."""
        if self.test_loader is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        return self.test_loader


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    DATASET_ROOT = "/home/phd/datasets/"
    
    print("=" * 60)
    print("Initializing TopoCID DataLoaders with Environment-Label Batching")
    print("=" * 60)
    
    # Test with MUTAG (TUDataset)
    print("\n--- Testing MUTAG DataModule ---")
    mutag_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    mutag_module.prepare_data()
    
    train_loader = mutag_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch Graph Count: {batch.batch_size}")
    print(f"Positive Pairs Shape (for TCD): {batch.positive_pairs.shape}")
    print(f"Negative Pairs Shape (for TCD): {batch.negative_pairs.shape}")
    
    if batch.positive_pairs.shape[0] > 0:
        i, j = batch.positive_pairs[0]
        print(f"Example Positive Pair: Graph {i.item()} and Graph {j.item()}")
        print(f"  Label y_{i.item()} == y_{j.item()}: {batch.y[i].item() == batch.y[j].item()}")
        print(f"  Env E_{i.item()} != E_{j.item()}: {batch.env[i].item() != batch.env[j].item()}")
        
    print("\n" + "=" * 60)
    print("DataLoader Verification Complete. No synthetic data used.")
    print("=" * 60)