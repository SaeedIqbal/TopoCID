import os
import yaml
import torch
from typing import Dict, Any

# Import framework classes
from data.dataloaders import TopoCIDDataModule
from models.topocid.topocid import TopoCID
from train.trainer import TopoCIDTrainer


class TopoCIDConfigLoader:
    """
    Object-Oriented Configuration Manager.
    Parses YAML configs and instantiates the corresponding framework objects.
    """
    
    def __init__(self, dataset_config_path: str, model_config_path: str):
        self.dataset_config = self._load_yaml(dataset_config_path)
        self.model_config = self._load_yaml(model_config_path)
        
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Safely loads a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def build_data_module(self) -> TopoCIDDataModule:
        """Instantiates the TopoCIDDataModule from dataset config."""
        ds_cfg = self.dataset_config['dataset']
        return TopoCIDDataModule(
            dataset_name=ds_cfg['name'],
            root=ds_cfg['root'],
            batch_size=ds_cfg['batch_size'],
            split_type=ds_cfg['split_type']
        )

    def build_model(self, num_node_features: int) -> TopoCID:
        """Instantiates the TopoCID model from model config."""
        m_cfg = self.model_config['model']
        ds_cfg = self.dataset_config['dataset']
        
        return TopoCID(
            num_node_features=num_node_features,
            hidden_dim=m_cfg['backbone']['hidden_dim'],
            topo_dim=m_cfg['tcp']['topo_dim'],
            env_dim=m_cfg['env_encoder']['env_dim'],
            num_classes=ds_cfg['num_classes'],
            lambda_cf=m_cfg['trainer']['lambda_cf'],
            lambda_tcd=m_cfg['trainer']['lambda_tcd'],
            lambda_mi=m_cfg['trainer']['lambda_mi'],
            tau=m_cfg['tcd']['temperature']
        )

    def build_trainer(self, model: TopoCID, data_module: TopoCIDDataModule, device: torch.device) -> TopoCIDTrainer:
        """Instantiates the TopoCIDTrainer from model config."""
        t_cfg = self.model_config['trainer']
        
        return TopoCIDTrainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            test_loader=data_module.test_dataloader(),
            lr_descent=t_cfg['lr_descent'],
            lr_ascent=t_cfg['lr_ascent'],
            weight_decay=t_cfg['weight_decay'],
            device=device,
            critic_update_steps=t_cfg['critic_update_steps'],
            lambda_cf=t_cfg['lambda_cf'],
            lambda_tcd=t_cfg['lambda_tcd'],
            lambda_mi=t_cfg['lambda_mi']
        )


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Initializing TopoCID Object-Oriented Config Loader")
    print("=" * 60)
    
    # Paths to the newly created YAML configs
    ds_config_path = "configs/datasets/mutag.yaml"
    model_config_path = "configs/models/topocid.yaml"
    
    # 1. Initialize Loader
    loader = TopoCIDConfigLoader(ds_config_path, model_config_path)
    
    # 2. Build Data Module (Strictly loads from /home/phd/datasets/)
    print("\n--- Building Data Module from YAML ---")
    data_module = loader.build_data_module()
    data_module.prepare_data()
    
    # 3. Build Model
    print("\n--- Building TopoCID Model from YAML ---")
    batch = next(iter(data_module.train_dataloader()))
    num_node_features = batch.x.size(1)
    model = loader.build_model(num_node_features)
    
    # 4. Build Trainer
    print("\n--- Building Trainer from YAML ---")
    device = torch.device('cpu') # Strictly deterministic
    trainer = loader.build_trainer(model, data_module, device)
    
    print("\n" + "=" * 60)
    print("Config Loader Verification Complete. All objects instantiated successfully.")
    print("=" * 60)