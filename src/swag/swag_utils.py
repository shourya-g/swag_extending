import copy
import torch


def initialize_swa_model(model):
    """
    Create a copy of the model to hold SWA weights.
    """
    return copy.deepcopy(model)


def update_swa_model(swa_model, model, num_models):
    """
    Update SWA model parameters using running average.

    SWA update:
        theta_swa = (num_models * theta_swa + theta_current) / (num_models + 1)
    """
    with torch.no_grad():
        for swa_param, param in zip(swa_model.parameters(), model.parameters()):
            swa_param.data.mul_(num_models / (num_models + 1.0))
            swa_param.data.add_(param.data / (num_models + 1.0))