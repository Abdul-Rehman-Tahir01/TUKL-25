import torch
import numpy as np
import sklearn.metrics


# Evaluate performance
def eval_perf(dataloader, backbone, fc, device):
    backbone.eval()
    fc.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            xt = batch["x"].to(device)
            yt = batch["y"].to(device)
            outputs = fc(backbone(xt))
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yt).sum().item()
            total += yt.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yt.cpu().numpy())


    # Computing metrics
    f1s = sklearn.metrics.f1_score(all_labels, all_preds, average=None)
    acc = correct / total
    avg_f1 = f1s.mean()

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(all_labels, all_preds)

    # Set models back to train mode
    backbone.train()
    fc.train()

    return f1s, acc, avg_f1, cm
