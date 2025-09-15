import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import enable_dropout

# ============= BALD (Bayesian AL by Disagreement) ============= 
def bald_sampling(X, model_backbone, model_fc, k, batch_size, device, unlabeled_pool, T):
    model_backbone.eval()
    model_fc.eval()
    enable_dropout(model_backbone)
    enable_dropout(model_fc)

    indices = list(unlabeled_pool)
    X_pool = X[indices]
    mc_probs = []

    with torch.no_grad():
        for _ in tqdm(range(T), desc="MC Dropout Simulations (BALD)"):
            logits = []
            for i in range(0, len(X_pool), batch_size):
                x_batch = torch.tensor(X_pool[i:i+batch_size]).float().permute(0, 2, 1).to(device)  # (B, F, T)
                feats = model_backbone(x_batch)
                out = model_fc(feats)
                probs = F.softmax(out, dim=1).cpu()
                logits.append(probs)
            mc_probs.append(torch.cat(logits, dim=0))

    mc_probs = torch.stack(mc_probs)  # [T, N, C]
    avg_probs = mc_probs.mean(dim=0)  # [N, C]

    # Predictive Entropy
    H = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=1)  # [N]

    # Expected Entropy
    E_H = -torch.sum(mc_probs * torch.log(mc_probs + 1e-8), dim=2).mean(dim=0)  # [N]

    # BALD = H - E_H
    bald_score = H - E_H
    topk = torch.topk(bald_score, k=k).indices
    return [indices[i.item()] for i in topk]

# ============= BvSB ============= 
def bvsb_sampling(X, model_backbone, model_fc, k, batch_size, device, unlabeled_pool, T):
    model_backbone.eval()
    model_fc.eval()
    enable_dropout(model_backbone)
    enable_dropout(model_fc)

    indices = list(unlabeled_pool)
    X_pool = X[indices]
    mc_probs = []

    with torch.no_grad():
        for _ in tqdm(range(T), desc="MC Dropout Simulations (BvSB)"):
            logits = []
            for i in range(0, len(X_pool), batch_size):
                x_batch = torch.tensor(X_pool[i:i+batch_size]).float().permute(0, 2, 1).to(device)  # (B, F, T)
                feats = model_backbone(x_batch)
                out = model_fc(feats)
                probs = F.softmax(out, dim=1).cpu()
                logits.append(probs)
            mc_probs.append(torch.cat(logits, dim=0))

    mc_probs = torch.stack(mc_probs)        # [T, N, C]
    avg_probs = mc_probs.mean(dim=0)        # [N, C]

    # Sort class probabilities per sample
    sorted_probs, _ = torch.sort(avg_probs, dim=1)  # [N, C]

    # Best vs Second Best margin = highest prob - second highest prob
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]  # [N]

    # Smaller margins = more uncertainty → select them
    topk = torch.topk(-margins, k=k).indices
    return [indices[i.item()] for i in topk]
