import torch


def update_bn(loader, model, device):
    """
    Recompute BatchNorm running statistics for the given model
    using one forward pass over the training data.

    Args:
        loader: training dataloader
        model: model whose BatchNorm stats need to be updated
        device: torch device
    """
    model.train()

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            momenta[module] = module.momentum

    if not momenta:
        return

    num_samples = 0

    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked.zero_()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)

            momentum = batch_size / float(num_samples + batch_size)
            for module in momenta.keys():
                module.momentum = momentum

            model(images)
            num_samples += batch_size

    for module, momentum in momenta.items():
        module.momentum = momentum