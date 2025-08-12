import torch

# ============ KL Divergence Helper ============
def KL_DIV(mu_q, sig_q, mu_p, sig_p):
    return 0.5 * (2 * torch.log(sig_p / sig_q) - 1 +
                  (sig_q / sig_p).pow(2) +
                  ((mu_p - mu_q) / sig_p).pow(2)).sum()