import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Enable dropout at inference/prediction (MC dropout)
def enable_dropout(model):
    """Enable dropout layers during test time."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # Keep dropout in training mode

# ============= Predictive Entropy ============= 
def predictive_entropy_sampling(X, model_backbone, model_fc, k, device, unlabeled_pool, sim=20):
    model_backbone.eval()
    model_fc.eval()
    enable_dropout(model_backbone)
    enable_dropout(model_fc)

    indices = list(unlabeled_pool)
    X_pool = X[indices]
    mc_probs = []

    with torch.no_grad():
        for _ in tqdm(range(sim), desc='MC Dropout Simulations (PredEntropy)'):
            logits = []
            for i in range(0, len(X_pool), 32):
                x_batch = torch.tensor(X_pool[i:i+32]).float().to(device)
                feats = model_backbone(x_batch)
                out = model_fc(feats)
                logits.append(F.softmax(out, dim=1).cpu())
            probs = torch.cat(logits, dim=0)
            mc_probs.append(probs)

    mc_probs = torch.stack(mc_probs)  # Shape: [T, N, C]
    avg_probs = mc_probs.mean(dim=0)  # Shape: [N, C]
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=1)

    topk = torch.topk(entropy, k=k).indices
    return [indices[i.item()] for i in topk]


# ============= BALD (Bayesian AL by Disagreement) ============= 
def bald_sampling(X, model_backbone, model_fc, k, device, unlabeled_pool, sim=20):
    model_backbone.eval()
    model_fc.eval()
    enable_dropout(model_backbone)
    enable_dropout(model_fc)

    indices = list(unlabeled_pool)
    X_pool = X[indices]
    mc_probs = []

    with torch.no_grad():
        for _ in tqdm(range(sim), desc="MC Dropout Simulations (BALD)"):
            logits = []
            for i in range(0, len(X_pool), 32):
                x_batch = torch.tensor(X_pool[i:i+32]).float().to(device)
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



# ============= Variation Ratios ============= 
def variation_ratio_sampling(X, model_backbone, model_fc, k, device, unlabeled_pool, sim=20):
    model_backbone.eval()
    model_fc.eval()
    enable_dropout(model_backbone)
    enable_dropout(model_fc)

    indices = list(unlabeled_pool)
    X_pool = X[indices]
    preds = []

    with torch.no_grad():
        for _ in tqdm(range(sim), desc="MC Dropout Simulations (VarRatio)"):
            logits = []
            for i in range(0, len(X_pool), 32):
                x_batch = torch.tensor(X_pool[i:i+32]).float().to(device)
                feats = model_backbone(x_batch)
                out = model_fc(feats)
                pred = torch.argmax(out, dim=1).cpu()
                logits.append(pred)
            preds.append(torch.cat(logits, dim=0))

    preds = torch.stack(preds)  # [T, N]
    mode_preds = torch.mode(preds, dim=0)[0]  # [N]
    agreement = (preds == mode_preds.unsqueeze(0)).sum(dim=0)  # [N]
    variation_ratio = 1 - agreement.float() / sim

    topk = torch.topk(variation_ratio, k=k).indices
    return [indices[i.item()] for i in topk]
