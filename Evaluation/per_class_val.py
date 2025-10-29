import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_per_class_val_f1(csv_paths, model_names, class_names, savepath=None):
    """
    Plot validation F1 per class for multiple models.
    
    csv_paths: list of paths to metrics.csv logs
    model_names: list of model names
    class_names: list of class names, e.g. ["keep","turn","backchannel"]
    savepath: path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(8,5))

    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        df = df.dropna(subset=["epoch"])

        if df.empty:
            print(f"⚠️ {name}: No epoch data found in {path}, skipping.")
            continue

        # Plot each class
        for idx, cls_name in enumerate(class_names):
            col_name = f"val_f1_class_{idx}"
            if col_name in df.columns:
                ax.plot(df["epoch"], df[col_name], marker="o", label=f"{name}-{cls_name}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("Validation F1 per Class")
    ax.grid(True, linestyle="--", alpha=0.5)
    if ax.get_lines():
        ax.legend(fontsize=8, ncol=len(class_names))
    
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot to {savepath}")
    else:
        plt.show()


plot_per_class_val_f1(
    [
        "/share/users/student/m/mnsiah/new_results/text_logs_1/lightning_logs/version_0/metrics.csv",
        "/share/users/student/m/mnsiah/new_results/audio_logs/lightning_logs/version_0/metrics.csv",
        "/share/users/student/m/mnsiah/fusion_model_logs/final_fused_model_3/lightning_logs/version_1/metrics.csv"
    ],
    ["Text", "Audio", "Fusion"],
    ["keep", "turn", "backchannel"],
    savepath="/home/student/m/mnsiah/modality_fusion/Evaluation/train_cm_plots/val_f1_per_class.png"
)
