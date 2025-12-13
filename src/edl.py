import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def get_evidence(y, evidence_func=relu_evidence):
    """
    Convert logits to evidence.
    """
    return evidence_func(y)

def calculate_uncertainty(logits, evidence_func=relu_evidence):
    """
    Calculate uncertainty and probabilities using EDL.
    
    Args:
        logits: (Batch, Num_Classes, ...)
        evidence_func: Function to convert logits to evidence.
        
    Returns:
        uncertainty: (Batch, ...) - Uncertainty score usually between 0 and 1.
        probabilities: (Batch, Num_Classes, ...) - Class probabilities.
    """
    evidence = get_evidence(logits, evidence_func)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = logits.shape[1]
    
    uncertainty = K / S
    probabilities = alpha / S
    
    return uncertainty.squeeze(1), probabilities

# --- Loss Functions ---

def kl_divergence(alpha, num_classes, device=None):
    if device is None:
        device = alpha.device
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term

def loglikelihood_loss(y, alpha, device=None):
    if device is None:
        device = y.device
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True
    )
    return loglikelihood_err

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if device is None:
        device = y.device
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    loglikelihood = loglikelihood_loss(y, alpha, device)
    
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """
    Generic EDL loss wrapper.
    """
    return func(y, alpha, epoch_num, num_classes, annealing_step, device)
