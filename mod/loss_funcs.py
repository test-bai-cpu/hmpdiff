#### Define loss

import math
import torch
from torch import nn
import numpy as np
from scipy.stats import multivariate_normal


def circdiff(circular1, circular2):
    """
    Compute the circular difference between two angles.
    """
    return torch.abs(torch.atan2(torch.sin(circular1 - circular2), torch.cos(circular1 - circular2)))


def angle_diff_signed(a, b):
    # pi = torch.tensor(np.pi, dtype=torch.float32, device=a.device)
    # return (a - b + pi) % (2 * pi) - pi   # in [-pi, pi]
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


# Function to calculate the distance in 2D space (speed and motion_angle)
def distance_wrap_2d(p1, p2):
    # p1 and p2 are tensors with [speed, motion_angle]
    speed_diff = torch.abs(p1[:, 0] - p2[:, 0])  # Linear difference in speed
    angle_diff = circdiff(p1[:, 1], p2[:, 1])    # Circular difference in motion angle
    return torch.sqrt(speed_diff**2 + angle_diff**2)  # Euclidean distance


class CircularLinearLoss(nn.Module):
    def __init__(self):
        super(CircularLinearLoss, self).__init__()
        
    def forward(self, output, target):
        return distance_wrap_2d(output, target).mean()


def wrapped_gaussian_nll(means, vars, corr_coef, target, device):
    """
    Compute the negative log-likelihood for a wrapped bivariate Gaussian distribution.

    Args:
        means: Tensor of shape (batch_size, 2), means for speed and motion_angle.
        vars: Tensor of shape (batch_size, 2), variances for speed and motion_angle.
        targets: Tensor of shape (batch_size, 2), target speed and motion_angle.
        correlations: Tensor of shape (batch_size,), correlation coefficients (rho) between speed and motion_angle.

    Returns:
        Tensor: Scalar loss (mean negative log-likelihood over the batch).
    """
    # Unpack means and variances
    batch_size = target.shape[0]
    speed_mean, angle_mean = means[:, 0], means[:, 1]
    speed_var, angle_var = vars[:, 0], vars[:, 1]

    # Unpack targets
    speed_target, angle_target = target[:, 0], target[:, 1]

    # Initialize wrapped probabilities
    wrapped_probs = torch.zeros(batch_size, device=device)

    pi = torch.tensor(np.pi, dtype=torch.float32, device=device)

    for wrap_num in [-1, 0, 1]:
        wrapped_angle = angle_target + 2 * pi * wrap_num

        # Difference vector (target - mean)
        diff = torch.stack([
            speed_target - speed_mean,  # Speed difference
            # wrapped_angle - angle_mean  # Angle difference
            circdiff(wrapped_angle, angle_mean)  # Angle difference
        ], dim=-1)  # Shape: (batch_size, 2)
        
        det = speed_var * angle_var * (1 - corr_coef**2)

        inv_cov = torch.stack([
            torch.stack([angle_var, -corr_coef * torch.sqrt(speed_var * angle_var)], dim=-1),
            torch.stack([-corr_coef * torch.sqrt(speed_var * angle_var), speed_var], dim=-1)
        ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)

        # Mahalanobis distance
        mahalanobis = torch.einsum("bi,bij,bj->b", diff, inv_cov, diff)  # (batch_size,)

        # Bivariate Gaussian PDF
        norm_const = torch.sqrt((2 * pi)**2 * det)
        prob = torch.exp(-0.5 * mahalanobis) / norm_const
        
        # Accumulate probabilities across wraps
        wrapped_probs += prob

    wrapped_probs = torch.clamp(wrapped_probs, min=1e-9)

    # Compute negative log-likelihood
    nll = -torch.log(wrapped_probs)

    return nll.mean()


def wrapped_GMM_nll_v1(GMM_params, target, device):
    """
    Compute the negative log-likelihood for a wrapped bivariate GMM distribution.

    Args:
        means: Tensor of shape (batch_size, 2), means for speed and motion_angle.
        vars: Tensor of shape (batch_size, 2), variances for speed and motion_angle.
        targets: Tensor of shape (batch_size, 2), target speed and motion_angle.
        correlations: Tensor of shape (batch_size,), correlation coefficients (rho) between speed and motion_angle.

    Returns:
        Tensor: Scalar loss (mean negative log-likelihood over the batch).
    """
    pi = torch.tensor(3.141592653589793, dtype=torch.float32)
    
    # Unpack means and variances
    batch_size = target.shape[0]
    num_components = GMM_params.shape[1]

    # Unpack targets
    speed_target, angle_target = target[:, 0], target[:, 1]
    
    # Initialize wrapped probabilities
    wrapped_probs = torch.zeros(batch_size, device=device)

    for i in range(num_components):
        weight = GMM_params[:, i, 0]       # Shape: (batch_size, num_components)
        speed_mean = GMM_params[:, i, 1]   # Shape: (batch_size, num_components)
        angle_mean = GMM_params[:, i, 2]   # Shape: (batch_size, num_components)
        speed_var = GMM_params[:, i, 3]    # Shape: (batch_size, num_components)
        angle_var = GMM_params[:, i, 4]    # Shape: (batch_size, num_components)
        corr_coef = GMM_params[:, i, 5]    # Shape: (batch_size, num_components)

        total_prob_for_component = torch.zeros(batch_size, device=device)
        
        for wrap_num in [-1, 0, 1]:
            wrapped_angle = angle_target + 2 * pi * wrap_num

            # Difference vector (target - mean)
            diff = torch.stack([
                speed_target - speed_mean,  # Speed difference
                # wrapped_angle - angle_mean  # Angle difference
                # circdiff(wrapped_angle, angle_mean)  # Angle difference
                angle_diff_signed(wrapped_angle, angle_mean)  # Angle difference
            ], dim=-1)  # Shape: (batch_size, 2)
            
            det = speed_var * angle_var * (1 - corr_coef**2)

            inv_cov = torch.stack([
                torch.stack([angle_var, -corr_coef * torch.sqrt(speed_var * angle_var)], dim=-1),
                torch.stack([-corr_coef * torch.sqrt(speed_var * angle_var), speed_var], dim=-1)
            ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)

            # Mahalanobis distance
            mahalanobis = torch.einsum("bi,bij,bj->b", diff, inv_cov, diff)  # (batch_size,)

            # Bivariate Gaussian PDF
            norm_const = torch.sqrt((2 * pi)**2 * det)
            prob = torch.exp(-0.5 * mahalanobis) / norm_const
            
            total_prob_for_component += prob
        
        wrapped_probs += weight * total_prob_for_component
    
    wrapped_probs = torch.clamp(wrapped_probs, min=1e-9)

    # Compute negative log-likelihood
    nll = -torch.log(wrapped_probs)

    return nll.mean()


def wrapped_GMM_nll(GMM_params, target, reduction='mean'):
    """
    Compute the negative log-likelihood for a wrapped bivariate GMM distribution.

    Args:
        GMM_params: (B, K, 6) with [w, mu_s, mu_a, var_s, var_a, rho]
            - w should already be softmaxed over K and >= 0
            - var_* must be > 0
            - rho in (-1, 1)
        target:     (B, 2) with [speed, angle]
        reduction:  'mean' | 'none'
        Returns:    scalar if 'mean', else (B,) per-sample NLL
        
        means: Tensor of shape (batch_size, 2), means for speed and motion_angle.
        vars: Tensor of shape (batch_size, 2), variances for speed and motion_angle.
        targets: Tensor of shape (batch_size, 2), target speed and motion_angle.
        correlations: Tensor of shape (batch_size,), correlation coefficients (rho) between speed and motion_angle.

    Returns:
        Tensor: Scalar loss (mean negative log-likelihood over the batch).
    """
    
    B, K, P = GMM_params.shape
    assert P == 6, f"Expected last dim 6, got {P}"

    # Unpack
    w     = GMM_params[:, :, 0].clamp_min(1e-12)             # (B,K)
    mu_s  = GMM_params[:, :, 1]                               # (B,K)
    mu_a  = GMM_params[:, :, 2]                               # (B,K)
    var_s = GMM_params[:, :, 3].clamp_min(1e-12)              # (B,K)
    var_a = GMM_params[:, :, 4].clamp_min(1e-12)              # (B,K)
    rho   = GMM_params[:, :, 5].clamp(-0.999, 0.999)          # (B,K)

    std_s = var_s.sqrt()
    std_a = var_a.sqrt()
    denom = (1 - rho**2).clamp_min(1e-12)                    # (B,K)

    s = target[:, 0].unsqueeze(-1)                           # (B,1)
    a = target[:, 1].unsqueeze(-1)                           # (B,1)

    # 3 wraps: a-2pi, a, a+2pi  -> (B,3,1)
    twopi = 2 * torch.pi
    a_wraps = torch.stack([a - twopi, a, a + twopi], dim=1)  # (B,3,1)

    # Expand to (B,3,K)
    s_b     = s.unsqueeze(1).expand(B, 3, 1)
    
    w_b     = w.unsqueeze(1).expand(B, 3, K)
    mu_s_b  = mu_s.unsqueeze(1).expand(B, 3, K)
    mu_a_b  = mu_a.unsqueeze(1).expand(B, 3, K)
    std_s_b = std_s.unsqueeze(1).expand(B, 3, K)
    std_a_b = std_a.unsqueeze(1).expand(B, 3, K)
    rho_b   = rho.unsqueeze(1).expand(B, 3, K)
    denom_b = denom.unsqueeze(1).expand(B, 3, K)
    

    # Residuals (signed angle!)
    ds = (s_b - mu_s_b)                                      # (B,3,K)
    da = angle_diff_signed(a_wraps, mu_a_b)                  # (B,3,K)

    ns = ds / std_s_b
    na = da / std_a_b

    quad = (ns**2 - 2*rho_b*ns*na + na**2) / denom_b         # (B,3,K)
    log_norm = torch.log(2 * torch.pi * std_s_b * std_a_b) + 0.5 * torch.log(denom_b)
    comp_logp = -0.5 * quad - log_norm                       # (B,3,K)

    # Sum over components in log-space (log-sum-exp with mixture weights)
    log_mix_over_K = torch.logsumexp(torch.log(w_b) + comp_logp, dim=-1)  # (B,3)

    # Sum over the 3 wraps in probability space
    prob = torch.exp(log_mix_over_K).sum(dim=1).clamp_min(1e-12)          # (B,)
    nll = -torch.log(prob)                                                 # (B,)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class NLLLoss(nn.Module):
    def __init__(self, device):
        super(NLLLoss, self).__init__()
        self.device = device
        
    def forward(self, output, target):
        means, vars, corr_coef = output  # Expect output as a tuple (means, variances)
        return wrapped_gaussian_nll(means, vars, corr_coef, target, self.device)
    
    
class NLLLoss_Density(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        means, vars, corr_coef, density = output  # Expect output as a tuple (means, variances)
        nll_loss = wrapped_gaussian_nll(means, vars, corr_coef, target)
        density_loss = nn.MSELoss()(density, target[:, 2])
        return nll_loss + density_loss
    
    
class NLLGMMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, reduction='mean'):
        # # output == GMM_params (B,K,6)
        return wrapped_GMM_nll(output, target, reduction=reduction)

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, coords, grad_threshold=5.0, lambda_penalty=0.1):
        GMM_loss = wrapped_GMM_nll(output, target)
        
        # # inputs = input.clone().detach().requires_grad_(True)  # Ensure input gradients are separate from model params
        # inputs = inputs.clone().detach().requires_grad_(True)  # Ensure inputs require gradients

        # grad_outputs = torch.ones_like(output, device=self.device)
        # gradients = torch.autograd.grad(
        #     outputs=output,
        #     inputs=[inputs],  # Compute gradient w.r.t. inputs
        #     grad_outputs=grad_outputs,
        #     create_graph=True,  # No second-order gradients (to save memory)
        #     allow_unused=True
        # )[0]
        
        # print(f"coords.requires_grad: {coords.requires_grad}")
        # print(f"output.requires_grad: {output.requires_grad}")
        
        if not output.requires_grad:
            return GMM_loss
        
        gradients = gradient(GMM_loss, coords)
        
        grad_norm = torch.norm(gradients, dim=-1, p=2)  # L2 norm
        # gradient_loss = torch.norm(gradients, p=2, dim=-1).mean()
        penalty = torch.relu(grad_norm - grad_threshold) ** 2
        penalty = penalty.mean()

        # Final loss = Original loss + λ * Gradient Penalty
        total_loss = GMM_loss + lambda_penalty * penalty
        
        return total_loss


class NLLGMMLoss_Density(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, density, target):
        nll_gmm_loss = wrapped_GMM_nll(output, target)
        density_loss = nn.MSELoss()(density, target[:, 2])
        
        return nll_gmm_loss + density_loss