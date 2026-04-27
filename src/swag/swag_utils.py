import copy
import math
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def initialize_swa_model(model):
    """
    Create a copy of the model to hold SWA weights.
    """
    return copy.deepcopy(model)


def update_swa_model(swa_model, model, num_models):
    """
    Update SWA model parameters using running average.
    """
    with torch.no_grad():
        for swa_param, param in zip(swa_model.parameters(), model.parameters()):
            swa_param.data.mul_(num_models / (num_models + 1.0))
            swa_param.data.add_(param.data / (num_models + 1.0))


class SWAGPosterior:
    """
    Stores SWAG statistics:
    - running mean of parameters
    - running second moment of parameters
    - low-rank deviation matrix
    """

    def __init__(self, max_rank=20, var_clamp=1e-30):
        self.max_rank = max_rank
        self.var_clamp = var_clamp

        self.n_models = 0
        self.mean = None
        self.sq_mean = None
        self.deviations = []

    def collect_model(self, model):
        """
        Collect one SGD/SWA-phase model snapshot.
        """
        vec = parameters_to_vector(model.parameters()).detach().cpu()

        if self.mean is None:
            self.mean = torch.zeros_like(vec)
            self.sq_mean = torch.zeros_like(vec)

        old_mean = self.mean.clone()

        self.n_models += 1

        # running first moment
        self.mean += (vec - self.mean) / self.n_models

        # running second uncentered moment
        self.sq_mean += (vec.pow(2) - self.sq_mean) / self.n_models

        # low-rank deviation
        deviation = vec - old_mean

        self.deviations.append(deviation)

        if len(self.deviations) > self.max_rank:
            self.deviations.pop(0)

    def sample(self, scale=1.0):
        """
        Draw a parameter vector from the SWAG Gaussian approximation.

        SWAG posterior:
            theta ~ N(theta_SWA, 1/2 * (Sigma_diag + Sigma_lowrank))
        """
        if self.n_models == 0:
            raise RuntimeError("No models collected in SWAG posterior.")

        diag_var = self.sq_mean - self.mean.pow(2)
        diag_var = torch.clamp(diag_var, min=self.var_clamp)

        # diagonal sample
        z_diag = torch.randn_like(self.mean)
        diag_sample = (1.0 / math.sqrt(2.0)) * torch.sqrt(diag_var) * z_diag

        # low-rank sample
        if len(self.deviations) > 1:
            D = torch.stack(self.deviations, dim=0)  # [rank, num_params]
            z_lowrank = torch.randn(D.size(0))
            lowrank_sample = z_lowrank @ D
            lowrank_sample = lowrank_sample / math.sqrt(2.0 * (D.size(0) - 1))
        else:
            lowrank_sample = torch.zeros_like(self.mean)

        sample = self.mean + scale * (diag_sample + lowrank_sample)
        return sample

    def set_weights(self, model, vector, device):
        """
        Load sampled vector into model parameters.
        """
        vector = vector.to(device)
        vector_to_parameters(vector, model.parameters())
        
    def state_dict(self):
        """
        Save enough information to reconstruct the SWAG posterior later.
        """
        return {
            "max_rank": self.max_rank,
            "var_clamp": self.var_clamp,
            "n_models": self.n_models,
            "mean": self.mean,
            "sq_mean": self.sq_mean,
            "deviations": self.deviations,
        }

    def load_state_dict(self, state):
        """
        Load SWAG posterior statistics from a saved state dictionary.
        """
        self.max_rank = state["max_rank"]
        self.var_clamp = state["var_clamp"]
        self.n_models = state["n_models"]
        self.mean = state["mean"]
        self.sq_mean = state["sq_mean"]
        self.deviations = state["deviations"]    