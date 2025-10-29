import pandas as pd 

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_train_val_losses(csv_paths, model_names, savepath=None):
    """
    Plot training and validation losses per epoch for multiple models.
    Left panel = training loss, Right panel = validation loss.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharex=True)

    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        df = df.dropna(subset=["epoch"])

        if df.empty:
            print(f"⚠️ {name}: No epoch data found in {path}, skipping.")
            continue

        # Auto-detect column names
        train_key = "train_loss_epoch" if "train_loss_epoch" in df.columns else "train_loss"
        val_key = "val_loss_epoch" if "val_loss_epoch" in df.columns else "val_loss"

        epoch_losses = df.groupby("epoch").agg({train_key: "last", val_key: "last"})

        if not epoch_losses.empty:
            if train_key in epoch_losses:
                axes[0].plot(epoch_losses.index, epoch_losses[train_key], marker="o", label=name)
            if val_key in epoch_losses:
                axes[1].plot(epoch_losses.index, epoch_losses[val_key], marker="o", label=name)

    # Titles, labels, grids
    axes[0].set_title("Training Loss per Epoch")
    axes[1].set_title("Validation Loss per Epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax.get_lines():  # Only show legend if there are plotted lines
            ax.legend()

    plt.tight_layout()

    # Save figure
    if savepath is None:
        savepath = "/home/student/m/mnsiah/modality_fusion/Evaluation/new_plots/losses.png"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved loss plot to {savepath}")


# ------------------- Usage -------------------

compare_train_val_losses(
    [
        "/share/users/student/m/mnsiah/new_results/text_logs_1/lightning_logs/version_0/metrics.csv",
        "/share/users/student/m/mnsiah/new_results/distilhub1_logs/lightning_logs/version_0/metrics.csv",
        "/share/users/student/m/mnsiah/fusion_model_logs/model_logs/lightning_logs/version_0/metrics.csv"
    ],
    ["Text", "Audio", "Fusion"]
)

