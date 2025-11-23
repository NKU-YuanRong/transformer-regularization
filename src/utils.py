import torch
import numpy as np
import random

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda

    Implementation adapted for standard MixUp: draw lambda from Beta(alpha, alpha)
    and create mixed inputs and target pairs.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    batch_size = x.size(0)
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """Compute the MixUp loss given a base criterion."""
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
