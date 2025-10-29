

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\modality_fusion\TurnGPT")


from Evaluation.eval import evaluate_model
from Modules.context_collate import ChunkedFusionContext, collate_context
from models.distilhub import HubertSmall
from teacher_trp_script.teacher_model import TurnGPT3Class
from models.fusion_with_modality import FusionStudent
from models.text_head_1 import TextClassifier
#from unified_training.audio_head import HubertLightningModule
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Modules.weight_sampler import get_sampler

#train_df = pd.read_csv("/home/student/m/mnsiah/modality_fusion/csv_files/final_train_1.csv")
#sampler = get_sampler(train_df['label'].values)
###########################################################################################
from teacher_trp_script.teacher_model import TurnGPT3Class
from models.text_head_1 import TextClassifier
import torch
from turngpt.tokenizer import SpokenDialogTokenizer
# 1. Load TurnGPT tokenizer used during training
pretrained_model_name_or_path = "distilgpt2"
tokenizer = SpokenDialogTokenizer(pretrained_model_name_or_path)



###################################################################################
###################################################################################
# for MMF2F datasets

#   for current audio segment only
train_dataset = ChunkedFusionContext(r"D:\modality_fusion\train_pt_files_final\train")
# path to the pt files
#single collate function for sinle utterance preprocessing
val_dataset = ChunkedFusionContext(r"D:\modality_fusion\val_pt_files_final\val")
#test_dataset = ChunkedFusionDataset("/share/users/student/m/mnsiah/processed_files/test_pt_files_hubtext/test")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_context, num_workers=15)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_context, num_workers=15)
test_dataset=  ChunkedFusionContext(r"D:\modality_fusion\test_pt_files_final\test")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_context,num_workers=15)


###############################################################################
#LOAD MODEL CHECKPOINTSAND STATE DICT
###########################################################################


text_ckpt_path=r"D:\modality_fusion\checkpoints\text_only\text_model-epoch=04-val_min_f1=0.74.ckpt"
import os
os.path.exists(text_ckpt_path)  # Should return True

text_model=TextClassifier.load_from_checkpoint(text_ckpt_path,tokenizer=tokenizer)
audio_ckpt_path=r"D:\modality_fusion\checkpoints\audio_only\distil_hub-epoch=04-val_min_f1=0.79.ckpt"
audio_model=HubertSmall.load_from_checkpoint(audio_ckpt_path)


fusion_chkpt=r"D:\modality_fusion\checkpoints\fusion\final_model-epoch=06-val_min_f1=0.87.ckpt"
fusion_model = FusionStudent.load_from_checkpoint(fusion_chkpt, text_ckpt_path=text_ckpt_path,
    audio_ckpt_path=audio_ckpt_path)
########################################################################################
########################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(fusion_results)
################################################################
import matplotlib.pyplot as plt
import numpy as np

def plot_per_class_metrics(results_dicts, model_names, class_names):
    metrics = ["f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        width = 0.25
        x = np.arange(len(class_names))

        for i, (res, name) in enumerate(zip(results_dicts, model_names)):
            values = [res[f"{cls}_{metric}"] for cls in class_names]
            bars = ax.bar(x + i*width, values, width, label=name)

            # Annotate each bar with its value
            for rect in bars:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width()/2,
                    height,
                    f"{height:.2f}",
                    ha="center", va="bottom", fontsize=8
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names)
        ax.set_title(f"Test-set-Per-class {metric.upper()}")
        ax.legend()

    plt.tight_layout()
    savepath=r"D:\modality_fusion\Evaluation\new_plots\test_per_class_f1.png"
    plt.savefig(savepath)
    plt.close()



    ####################################################################
    #balanced accuracy
'''def plot_bal_acc(results_dicts, model_names):
    plt.figure(figsize=(6,5))
    values = [res["balanced_accuracy"] for res in results_dicts]
    bars = plt.bar(model_names, values, color=["skyblue","salmon","limegreen"])
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Acuuracy")

    # Annotate values
    for rect in bars:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width()/2,
            height,
            f"{height:.2f}",
            ha="center", va="bottom", fontsize=9
        )
    

    savepath="/home/student/m/mnsiah/modality_fusion/Evaluation/new_plots/test_macro_f1.png"
    plt.savefig(savepath)
    plt.close()


plot_bal_acc(
    [text_results, audio_results, fusion_results],   # list of results dicts
    ["Text", "Audio", "Fusion"],
)'''
    ###################################
    #cm matrix
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

'''def save_confusion_matrices(results_dicts, model_names, class_names, outdir):
    """
    Save one confusion matrix per model as a separate PNG file.
    """
    os.makedirs(outdir, exist_ok=True)

    for res, name in zip(results_dicts, model_names):
        cm = res["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(5,5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} Confusion Matrix")

        savepath = os.path.join(outdir, f"conf_matrix_{name.lower()}.png")
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {savepath}")


save_confusion_matrices(
    [text_results, audio_results, fusion_results],
    ["Text", "Audio", "Fusion"],
    ["keep", "turn", "backchannel"],
    "/home/student/m/mnsiah/modality_fusion/Evaluation/new_plots"
)'''
# Main execution (Windows-safe)
# -----------------------------------------------------
if __name__ == "__main__":
    # Make sure your models and DataLoaders are defined here
    # text_model, audio_model, fusion_model, test_loader, device

    text_results = evaluate_model(text_model, test_loader, device)
    audio_results = evaluate_model(audio_model, test_loader, device)
    fusion_results = evaluate_model(fusion_model, test_loader, device)

    # Plot metrics
    plot_per_class_metrics(
        [text_results, audio_results, fusion_results],
        ["Text", "Audio", "Fusion"],
        ["keep", "turn", "backchannel"])