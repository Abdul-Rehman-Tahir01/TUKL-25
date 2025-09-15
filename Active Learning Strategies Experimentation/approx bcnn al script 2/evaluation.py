import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from utils import enable_dropout

def mc_dropout_eval(backbone, fc, dataloader, class_weights, device, T, round_num, total_rounds):
    backbone.eval()
    fc.eval()
    enable_dropout(backbone)
    enable_dropout(fc)

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation - AL Round: [{round_num}/{total_rounds}]"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            all_labels.append(y.cpu())

            logits_T = []
            mc_outputs = []
            for _ in range(T):
                feat = backbone(x)
                logits = fc(feat)
                logits_T.append(logits)
                probs = F.softmax(logits, dim=1)  # (B, num_classes)
                mc_outputs.append(probs.unsqueeze(0))  # (1, B, C)

            # Stack and average: (T, B, C) → (B, C)
            logits_T = torch.stack(logits_T, dim=0)  # (T, B, C)
            mc_outputs = torch.cat(mc_outputs, dim=0)  # (T, B, C)
            mc_mean = mc_outputs.mean(dim=0)  # (B, C)

            # ===== Compute loss =====
            ce_t = [F.cross_entropy(l, y, weight=class_weights, reduction="mean") for l in logits_T]
            loss = torch.stack(ce_t).mean()
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(mc_mean, dim=1)
            all_preds.append(preds.cpu())

    # Concat all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # ====== Metrics ======
    kappa = cohen_kappa_score(all_labels.numpy(), all_preds.numpy())
    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    avg_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
    per_class_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average=None)
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())

    avg_loss = total_loss / total_samples

    return avg_loss, acc, avg_f1, per_class_f1, kappa, cm
