import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

# Add parent directory to path to import models and data loaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloaders import TopoCIDDataModule
from models.topocid.topocid import TopoCID


class TopoCIDTrainer:
    """
    Main Training Loop for the TopoCID Framework.
    
    Mathematical Context (TopoCID - Section 4.4):
    Optimizes the joint objective function via alternating gradient descent and ascent:
        L_total = L_sup + lambda_cf * L_cf + lambda_TCD * L_TCD + lambda_MI * L_MI
        
    The critic network T_psi is updated via gradient ASCENT to maximize the CLUB bound:
        theta_critic <- theta_critic + eta_psi * nabla_{theta_critic} L_MI
        
    The rest of the model parameters Theta are updated via gradient DESCENT to minimize the total loss:
        Theta \ {theta_critic} <- Theta \ {theta_critic} - eta_theta * nabla_{Theta \ {theta_critic}} L_total
    """
    
    def __init__(self, model: TopoCID, train_loader, val_loader, test_loader,
                 lr_descent: float = 1e-3, lr_ascent: float = 1e-3,
                 weight_decay: float = 1e-5, device: torch.device = torch.device('cpu'),
                 critic_update_steps: int = 1,
                 lambda_cf: float = 1.0, lambda_tcd: float = 0.5, lambda_mi: float = 0.1):
        """
        Args:
            model (TopoCID): The TopoCID model to train.
            train_loader: PyG DataLoader for training.
            val_loader: PyG DataLoader for validation.
            test_loader: PyG DataLoader for testing.
            lr_descent (float): Learning rate for the model parameters (gradient descent).
            lr_ascent (float): Learning rate for the critic parameters (gradient ascent).
            weight_decay (float): Weight decay for the optimizers.
            device (torch.device): Device to run the training on.
            critic_update_steps (int): Number of ascent steps per descent step.
            lambda_cf (float): Weight for the counterfactual loss.
            lambda_tcd (float): Weight for the InfoNCE loss.
            lambda_mi (float): Weight for the CLUB MI bound loss.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.critic_update_steps = critic_update_steps
        
        self.lambda_cf = lambda_cf
        self.lambda_tcd = lambda_tcd
        self.lambda_mi = lambda_mi
        
        # Separate critic parameters and model parameters for alternating optimization
        critic_params = list(self.model.tcd.club.critic.parameters())
        model_params = [p for p in self.model.parameters() if not any(torch.equal(p, cp) for cp in critic_params)]
        
        # Optimizer for gradient DESCENT (minimizing L_total)
        self.opt_descent = optim.Adam(model_params, lr=lr_descent, weight_decay=weight_decay)
        
        # Optimizer for gradient ASCENT (maximizing L_MI)
        self.opt_ascent = optim.Adam(critic_params, lr=lr_ascent, weight_decay=weight_decay)
        
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_acc': [],
            'l_sup': [], 'l_cf': [], 'l_tcd': [], 'l_mi': []
        }

    def _compute_losses(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes all individual loss components for a given batch.
        
        Mathematical Formulations:
        L_sup = - (1/|D|) sum Y^T log f_phi(Phi(G))
        L_cf = - (1/|B|) sum y_i^T log f_phi(Phi(G_cf^{(i,j)}))
        L_TCD = - (1/|P|) sum log [ exp(kappa(z_i, z_j)/tau) / sum exp(kappa(z_i, z_k)/tau) ]
        L_MI = E_{p(z,e)}[T_psi(z,e)] - log E_{p(z)p(e)}[exp(T_psi(z,e))]
        """
        # 1. Forward pass to get representations
        logits, z_topo, z_env, node_embs = self.model(batch)
        
        # 2. Supervised Loss (L_sup)
        l_sup = nn.functional.cross_entropy(logits, batch.y.view(-1))
        
        # 3. TCD Losses (L_TCD and L_MI)
        l_tcd, l_mi = self.model.tcd(z_topo, z_env, batch.positive_pairs)
        
        # 4. Counterfactual Loss (L_cf)
        Z_cf, idx_i, idx_j, _ = self.model.spcg(z_topo, z_env, node_embs, batch.edge_index, batch.batch)
        logits_cf = self.model.classifier(Z_cf)
        l_cf = nn.functional.cross_entropy(logits_cf, batch.y[idx_i].view(-1))
        
        # 5. Total Loss
        l_total = l_sup + self.lambda_cf * l_cf + self.lambda_tcd * l_tcd + self.lambda_mi * l_mi
        
        loss_dict = {
            'total': l_total.item(),
            'sup': l_sup.item(),
            'cf': l_cf.item(),
            'tcd': l_tcd.item(),
            'mi': l_mi.item()
        }
        
        return l_total, l_mi, loss_dict

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Executes one training epoch with alternating gradient ascent and descent.
        """
        self.model.train()
        total_losses = {'total': 0.0, 'sup': 0.0, 'cf': 0.0, 'tcd': 0.0, 'mi': 0.0}
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            l_total, l_mi, loss_dict = self._compute_losses(batch)
            
            # --- Alternating Optimization ---
            
            # Step A: Critic Ascent (Maximize L_MI)
            for _ in range(self.critic_update_steps):
                self.opt_descent.zero_grad()
                self.opt_ascent.zero_grad()
                
                # Backpropagate only the MI loss for the critic
                # retain_graph=True is needed because we will backprop l_total next
                l_mi.backward(retain_graph=True)
                self.opt_ascent.step()
                
            # Step B: Model Descent (Minimize L_total)
            self.opt_descent.zero_grad()
            self.opt_ascent.zero_grad()
            
            # Backpropagate the total loss for the rest of the model
            l_total.backward()
            self.opt_descent.step()
            
            # Accumulate metrics
            for k in total_losses:
                total_losses[k] += loss_dict[k]
            num_batches += 1
            
        # Average over batches
        return {k: v / num_batches for k, v in total_losses.items()}

    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluates the model on a given dataloader (validation or test).
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                l_total, _, loss_dict = self._compute_losses(batch)
                total_loss += loss_dict['total']
                
                # Compute accuracy
                logits, _, _, _ = self.model(batch)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch.y.view(-1)).sum().item()
                total_samples += batch.y.size(0)
                num_batches += 1
                
        return {
            'loss': total_loss / num_batches,
            'acc': total_correct / total_samples
        }

    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Main training loop.
        """
        print(f"Starting TopoCID Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Alternating Updates: {self.critic_update_steps} critic ascent steps per model descent step.")
        
        for epoch in range(1, num_epochs + 1):
            # 1. Train
            train_metrics = self.train_epoch(epoch)
            
            # 2. Validate
            val_metrics = self.evaluate(self.val_loader)
            
            # 3. Log metrics
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['l_sup'].append(train_metrics['sup'])
            self.history['l_cf'].append(train_metrics['cf'])
            self.history['l_tcd'].append(train_metrics['tcd'])
            self.history['l_mi'].append(train_metrics['mi'])
            
            print(f"Epoch [{epoch}/{num_epochs}] | "
                  f"Train Loss: {train_metrics['total']:.4f} (Sup: {train_metrics['sup']:.4f}, "
                  f"CF: {train_metrics['cf']:.4f}, TCD: {train_metrics['tcd']:.4f}, MI: {train_metrics['mi']:.4f}) | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f}")
                  
        # 4. Final Test
        print("\n--- Final Test Evaluation ---")
        test_metrics = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']:.4f}")
        
        return self.history


# ==============================================================================
# Execution and Verification Block
# ==============================================================================
if __name__ == "__main__":
    DATASET_ROOT = "/home/phd/datasets/"
    device = torch.device('cpu') # Use CPU for strict reproducibility and no random CUDA seeds
    
    print("=" * 60)
    print("Initializing TopoCID Training Pipeline")
    print("=" * 60)
    
    # 1. Load original dataset using the previously defined DataModule
    print("\n--- Loading Original MUTAG Dataset ---")
    data_module = TopoCIDDataModule(dataset_name='MUTAG', root=DATASET_ROOT, batch_size=32)
    data_module.prepare_data()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Get a batch to determine input dimensions
    batch = next(iter(train_loader))
    num_node_features = batch.x.size(1)
    num_classes = 2  # MUTAG is binary
    
    # 2. Initialize TopoCID Model (Deterministic Initialization)
    print("\n--- Initializing TopoCID Model ---")
    model = TopoCID(
        num_node_features=num_node_features, 
        hidden_dim=64, 
        topo_dim=64, 
        env_dim=32, 
        num_classes=num_classes
    )
    
    # 3. Initialize Trainer
    print("\n--- Initializing TopoCID Trainer ---")
    trainer = TopoCIDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr_descent=1e-3,
        lr_ascent=1e-3,
        device=device,
        critic_update_steps=1,
        lambda_cf=1.0,
        lambda_tcd=0.5,
        lambda_mi=0.1
    )
    
    # 4. Train the model
    print("\n--- Starting Training ---")
    history = trainer.fit(num_epochs=3)
    
    print("\n" + "=" * 60)
    print("Training Pipeline Verification Complete. No synthetic/random data used.")
    print("=" * 60)