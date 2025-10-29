import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.text_head_1 import TextClassifier
from models.distilhub import HubertSmall
import os
from Modules.focal_loss import FocalLoss
from transformers import get_linear_schedule_with_warmup

#latest code
class FusionStudent(pl.LightningModule):
    def __init__(self,text_ckpt_path=None,
                 audio_ckpt_path=None, num_classes=3, lr=1e-4,
                 plot_dir="/home/student/m/mnsiah/modality_fusion/Evaluation/train_cm_plots/modality_fusion_plots", modality_dropout_p=0.2):
        super().__init__()
        self.save_hyperparameters()

        if text_ckpt_path is not None:
            self.text_model = TextClassifier.load_from_checkpoint(text_ckpt_path)
        else:
            self.text_model = TextClassifier()

        if audio_ckpt_path is not None:
            self.audio_model = HubertSmall.load_from_checkpoint(audio_ckpt_path)
        else:
            self.audio_model =HubertSmall()
        self.num_classes = num_classes
        self.lr = lr
        self.modality_dropout_p = modality_dropout_p  # added dropout probability

        # Freeze encoders at start
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.audio_model.parameters():
            p.requires_grad = False

        self.text_model.eval()
        self.audio_model.eval()

        # Metrics
        self.train_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        
        # Focal loss
        self.focal_loss = FocalLoss(
            gamma=1.5,
            alpha=[0.293, 0.293, 0.414], #keep, turn ,bk #modified focal loss previous #was 0.3,0.3,0.7 with alpha 2.0
            
            reduction='mean',
            task_type="multi-class",
            num_classes=num_classes
        )
        self.ce=nn.CrossEntropyLoss()
        # Normalization
        self.text_norm = nn.LayerNorm(256)
        self.audio_norm = nn.LayerNorm(256)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Validation buffers
        self.val_preds, self.val_labels = [], []
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_dir = plot_dir

    # --------------------- NEW: Modality Dropout ----------------------
    def modality_dropout(self, text_emb, audio_emb):
        if self.training:
            drop_text = torch.rand(1).item() < self.modality_dropout_p
            drop_audio = torch.rand(1).item()  < self.modality_dropout_p

            # Prevent both modalities being dropped at once
            if drop_text and drop_audio:
                if torch.rand(1).item()  < 0.5:
                    drop_text = False
                else:
                    drop_audio = False

            if drop_text:
                text_emb = torch.zeros_like(text_emb)
            if drop_audio:
                audio_emb = torch.zeros_like(audio_emb)

        return text_emb, audio_emb

    # ---------------------- Forward pass ----------------------
    def forward(self, batch):
        text_out = self.text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        audio_out = self.audio_model(batch["waveforms"])

        text_emb = self.text_norm(text_out["pooled_embeds"])
        audio_emb = self.audio_norm(audio_out["pooled_embeds"])

        # Apply controlled modality dropout here
        text_emb, audio_emb = self.modality_dropout(text_emb, audio_emb)

        fused = torch.cat([text_emb, audio_emb], dim=-1)
        logits = self.classifier(fused)
        return logits

    # ---------------------- Training Step ----------------------
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.focal_loss(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)

        self.train_macro_f1.update(preds, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---------------------- Validation Step ----------------------
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.ce(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)

        self.val_macro_f1.update(preds, batch["labels"])
        self.val_per_class_f1.update(preds, batch["labels"])
        self.val_preds.append(preds.cpu())
        self.val_labels.append(batch["labels"].cpu())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # ---------------------- Validation Epoch End ----------------------
    def on_validation_epoch_end(self):
        macro_f1 = self.val_macro_f1.compute()
        per_class_f1 = self.val_per_class_f1.compute()

        # Log metrics
        self.log("val_macro_f1", macro_f1, prog_bar=True)
        for i, class_f1 in enumerate(per_class_f1):
            self.log(f"val_f1_class_{i}", class_f1)
        val_min_f1=torch.min(per_class_f1)
        self.log("val_min_f1", val_min_f1, prog_bar=True)

        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(self.num_classes)))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch}')
        plt.savefig(os.path.join(self.plot_dir, f"cm_epoch{self.current_epoch}.png"))
        plt.close()

        # Plot F1 per class
        classes = list(range(self.num_classes))
        scores = per_class_f1.detach().cpu().numpy()

        plt.figure(figsize=(6, 4))
        plt.bar(classes, scores, color="skyblue", edgecolor="black")
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.title(f"Per Class F1 Scores - Epoch {self.current_epoch}")
        plt.ylim(0, 1)

        # Show class indices (or replace with semantic names)
        plt.xticks(classes, [str(c) for c in classes])

         # Overlay the actual F1 values above each bar
        for i, v in enumerate(scores):
            y = min(0.98, v + 0.02)  # avoid going outside the plot
            plt.text(i, y, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"per_f1_epoch{self.current_epoch}.png"))
        plt.close()


        # Reset metrics
        self.val_macro_f1.reset()
        self.val_per_class_f1.reset()
        self.val_preds, self.val_labels = [], []

    # ---------------------- End of Epochs ----------------------
    def on_train_epoch_end(self):
        f1 = self.train_macro_f1.compute()
        self.log("train_macro_f1", f1, prog_bar=True)
        self.train_macro_f1.reset()

    # ---------------------- Optimizers ----------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
           optimizer,
           num_warmup_steps=int(0.1 * num_training_steps),
           num_training_steps=num_training_steps
          )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

0