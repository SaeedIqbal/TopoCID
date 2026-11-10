import torch
from typing import Dict

class MIEstimationEvaluator:
    """
    Evaluates empirical Mutual Information post-training using the CLUB estimator.
    
    Mathematical Context:
    I(Z_{topo}; Z_{env}) \ge \sup_\psi \left( \mathbb{E}_{p(z,e)}[T_\psi(z,e)] - \log \mathbb{E}_{p(z)p(e)}[\exp(T_\psi(z,e))] \right)
    """
    
    def __init__(self, club_estimator: torch.nn.Module):
        self.club = club_estimator

    def compute_empirical_mi(self, z_topo: torch.Tensor, z_env: torch.Tensor) -> Dict[str, float]:
        """Computes the empirical CLUB MI bound."""
        self.club.eval()
        with torch.no_grad():
            joint_exp = self.club.compute_joint_expectation(z_topo, z_env).item()
            marginal_exp = self.club.compute_marginal_expectation(z_topo, z_env).item()
            mi_bound = joint_exp - marginal_exp
        return {'joint_exp': joint_exp, 'marginal_exp': marginal_exp, 'mi_bound': mi_bound}
