import random
import torch
import numpy as np


def set_random_seed(seed):
    seed = seed if seed >= 0 else random.randint(0, 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def sample_t_logit_normal(B, device, mu_t=-0.5, sigma_t=1.5, eps=1e-4):
    # logit(t) ~ N(mu_t, sigma_t^2)  =>  t = sigmoid(N(mu_t, sigma_t))
    z = mu_t + sigma_t * torch.randn(B, device=device)
    t = torch.sigmoid(z)
    # avoid extreme values for numerical stability (your loss has (1 - t) in denom)
    t = t.clamp(min=eps, max=1.0 - eps)
    return t