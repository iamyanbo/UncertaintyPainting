import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def get_evidence(y, evidence_func=relu_evidence):
    return evidence_func(y)

def calculate_uncertainty(logits, evidence_func=relu_evidence):
    """
    Categorical Uncertainty (Dirichlet).
    """
    evidence = get_evidence(logits, evidence_func)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = logits.shape[1]
    
    uncertainty = K / S
    probabilities = alpha / S
    
    return uncertainty.squeeze(1), probabilities

def calculate_uncertainty_binary(logits):
    """
    Binary Uncertainty (Beta) per channel.
    logits: [Batch, Classes]
    Returns:
        uncertainty: [Batch, Classes]
        probabilities: [Batch, Classes]
    """
    alpha = F.relu(logits) + 1
    beta = F.relu(-logits) + 1
    S = alpha + beta
    
    probabilities = alpha / S
    uncertainty = 2 / S
    
    return uncertainty, probabilities

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
    return func(y, alpha, epoch_num, num_classes, annealing_step, device)

# --- Binary EDL Loss ---

def kl_divergence_beta(alpha, beta, device=None):
    """KL Divergence for Beta Distribution (Binary case)"""
    if device is None:
        device = alpha.device
        
    # Standard Beta KL against Uniform (1,1)
    # KL(Beta(a,b) || Beta(1,1))
    # = ln(B(1,1)/B(a,b)) + (a-1)psi(a) + (b-1)psi(b) - (a+b-2)psi(a+b)
    # B(1,1) = 1
    # ln(1/B(a,b)) = -ln(B(a,b)) = lgamma(a+b) - lgamma(a) - lgamma(b)
    
    first_term = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    second_term = (alpha - 1) * torch.digamma(alpha) + (beta - 1) * torch.digamma(beta)
    third_term = -(alpha + beta - 2) * torch.digamma(alpha + beta)
    
    return first_term + second_term + third_term

def binary_edl_loss(logits, targets, epoch_num, annealing_step, device=None):
    """
    EDL Loss for Binary Classification (per channel).
    Args:
        logits: [Batch, Classes] - Logits from network
        targets: [Batch, Classes] - 0/1 Targets
    """
    if device is None:
        device = logits.device
        
    # 1. Evidence
    alpha = F.relu(logits) + 1
    beta = F.relu(-logits) + 1
    S = alpha + beta
    
    # 2. Probability
    prob = alpha / S
    
    # 3. MSE Loss (Expected Squared Error)
    # L = (y - p)^2 + p(1-p)/(S+1)  <-- Variance term often included in strict EDL derivation, 
    # but standard simple implementation often uses just MSE of mean.
    # We will use the simple MSE of mean for stability: (y - p)^2
    mse_loss = (targets - prob) ** 2
    
    # 4. KL Regularization
    # For positive target (y=1): minimize KL(Beta(a,b) || Beta(Uncertainty_High)) ?
    # Actually standard EDL uses a specific regularization:
    # If y=1, we want approx Beta(Inf, 1). We regularize deviations from this?
    # No, the paper "Evidential Deep Learning to Quantify Classification Uncertainty"
    # suggests KL(Predicted || Uniform) scaled by (1-alpha_tilde) etc.
    # Simplified approach:
    # Just regularize towards Uniform Beta(1,1).
    kl = kl_divergence_beta(alpha, beta, device)
    
    # Weight KL by annealing
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    # IMPORTANT: We only calculate KL for "misleading" evidence?
    # Or just regular KL. The original paper does KL regularization to prevent infinite evidence accumulation.
    
    # The standard "MSE" EDL formulation:
    # L = Sum (y-p)^2 + lambda * KL
    total_loss = mse_loss + annealing_coef * kl
    
    return total_loss
