import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


# -----------------------------------------------------
# Evaluation function (for MM-F2F-style multimodal setup)
# -----------------------------------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=3, class_names=["keep", "turn", "backchannel"]):
    model.eval()
    model.to(device)

    all_labels, all_preds, all_probs = [], [], []

    for batch in dataloader:
        # Move only tensor values to device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        waveforms = batch.get("waveforms")
        labels = batch["labels"]

        # Flexible forward for text / audio / fusion models
        if hasattr(model, "text_model") and hasattr(model, "audio_model"):
            logits = model(batch)
        elif hasattr(model, "student"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
        elif hasattr(model, "hubert"):
            outputs = model(waveforms)
            logits = outputs["logits"]
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if hasattr(outputs, "logits") else outputs

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # Convert to numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # --- Compute metrics ---
    results = {}
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    for i, name in enumerate(class_names):
        results[f"{name}_precision"] = precision[i]
        results[f"{name}_recall"] = recall[i]
        results[f"{name}_f1"] = f1[i]

    # Macro & weighted averages
    results["macro_f1"] = f1_score(all_labels, all_preds, average="macro")
    results["weighted_f1"] = f1_score(all_labels, all_preds, average="weighted")
    # ✅ Balanced accuracy (one number per model)
    results["balanced_accuracy"] = balanced_accuracy_score(all_labels, all_preds)



    # PR-AUC per class
    for i, name in enumerate(class_names):
        y_true_bin = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        results[f"{name}_pr_auc"] = average_precision_score(y_true_bin, y_score)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    results["confusion_matrix"] = cm

    # --- Print results ---
    print("\n===== Per-Class Metrics =====")
    for i, name in enumerate(class_names):
        print(f"{name:<12s} | P: {precision[i]:.3f}  R: {recall[i]:.3f}  F1: {f1[i]:.3f}  PR-AUC: {results[f'{name}_pr_auc']:.3f}")

    print("\n===== Overall =====")
    print(f"Macro F1:    {results['macro_f1']:.3f}")
    print(f"Weighted F1: {results['weighted_f1']:.3f}")
    print(f"balanced_accuracy:{results['balanced_accuracy']:.3f}")

    print("\n===== Confusion Matrix =====")
    print(pd.DataFrame(cm, index=[f"true_{n}" for n in class_names],
                          columns=[f"pred_{n}" for n in class_names]))


    return results
