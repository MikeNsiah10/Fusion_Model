import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from Modules.focal_loss import FocalLoss
from Modules.audio_pooling import AudioPooling
from transformers import get_linear_schedule_with_warmup
# Load model directly
from transformers import AutoProcessor, AutoModel

#processor = AutoProcessor.from_pretrained("ntu-spml/distilhubert")
#model = AutoModel.from_pretrained("ntu-spml/distilhubert")

class HubertSmall(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        lr_head=1e-4,
        lr_backbone=1e-5,
        pool_type="attention",
        plot_dir="/home/student/m/mnsiah/modality_fusion/Evaluation/train_cm_plots/hub_1",
        hubert_model_name="ntu-spml/distilhubert",
        sampling_rate=16000,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Processor + model
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        self.hubert = AutoModel.from_pretrained(hubert_model_name)

        # Pooler
        self.pool = AudioPooling(method=pool_type, hidden_dim=768)

        # Projection and classifier
        self.proj = nn.Linear(768 * 3, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # --- UnFreeze everything at start ---
        for p in self.hubert.parameters():
            p.requires_grad = True
        
       

        # --- Hyperparams ---
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.num_classes = num_classes
        self.plot_dir = plot_dir
        self.sampling_rate = sampling_rate
        os.makedirs(self.plot_dir, exist_ok=True)

        # --- Criterion ---
        self.criterion= FocalLoss(
            gamma=1.5,
            alpha=[0.293, 0.293, 0.414], #keep, turn ,bk #modified focal loss previous #was 0.3,0.3,0.7 with alpha 2.0
            
            reduction='mean',
            task_type="multi-class",
            num_classes=num_classes
        )
        self.ce = nn.CrossEntropyLoss()

        # --- Metrics ---
        self.train_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        self.val_preds, self.val_labels = [], []  


    # -----------------------------
    # Forward and feature extraction
    # -----------------------------
    def _extract_embeddings_batched(self, waveforms):
        B, M, C, T = waveforms.shape
        assert M == 3 and C == 1, "Expecting shape [B, 3, 1, T]"
        flat = waveforms.view(B * M, C, T)
        inputs = self.processor(
            [w.squeeze(0).cpu().numpy() for w in flat],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.hubert(**inputs)
        feats = out.last_hidden_state
        pooled = self.pool(feats)
        pooled = pooled.view(B, M, 768)
        return pooled

    def forward(self, waveform):
        B, M, C, T = waveform.shape
        utt_embs = self._extract_embeddings_batched(waveform)
        concat_emb = utt_embs.reshape(B, 768 * 3)
        projected = self.proj(concat_emb)
        logits = self.classifier(projected)
        return {"pooled_embeds": projected, "logits": logits}

    # -----------------------------
    # Training / Validation
    # -----------------------------
    def training_step(self, batch, batch_idx):
        outputs = self(batch["waveforms"])
        logits = outputs["logits"]
        loss = self.criterion(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        self.train_macro_f1.update(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["waveforms"])
        logits = outputs["logits"]
        loss = self.ce(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        self.val_macro_f1.update(preds, batch["labels"])
        self.val_per_class_f1.update(preds, batch["labels"])
        self.val_preds.append(preds.cpu().view(-1))
        self.val_labels.append(batch["labels"].cpu().view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # -----------------------------
    # Validation End
    # -----------------------------
    def on_validation_epoch_end(self):
        macro_f1 = self.val_macro_f1.compute()
        per_class_f1 = self.val_per_class_f1.compute()
        self.log("val_macro_f1", macro_f1, prog_bar=True)
        for i, class_f1 in enumerate(per_class_f1):
            self.log(f"val_f1_class_{i}", class_f1)
        val_min_f1 = torch.min(per_class_f1)
        self.log("val_min_f1", val_min_f1, prog_bar=True)

        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        disp = ConfusionMatrixDisplay(cm, display_labels=["keep", "turn", "backchannel"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Epoch {self.current_epoch}")
        plt.savefig(os.path.join(self.plot_dir, f"cm_epoch{self.current_epoch}.png"))
        plt.close()

        plt.bar(["keep", "turn", "backchannel"], per_class_f1.cpu().numpy())
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.title(f"Per Class F1 - Epoch {self.current_epoch}")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(self.plot_dir, f"per_f1_epoch{self.current_epoch}.png"))
        plt.close()

        self.val_macro_f1.reset()
        self.val_per_class_f1.reset()
        self.val_preds, self.val_labels = [], []

    def on_train_epoch_end(self):
        f1 = self.train_macro_f1.compute()
        self.log("train_macro_f1", f1, prog_bar=True)
        self.train_macro_f1.reset()

    # -----------------------------
    # Optimizer + Scheduler
    # -----------------------------
   

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
           {"params": list(self.proj.parameters()) + list(self.classifier.parameters()), "lr": self.lr_head},
           {"params": self.hubert.parameters(), "lr": self.lr_backbone}

         ], weight_decay=1e-2)

        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
           optimizer,
           num_warmup_steps=int(0.1 * num_training_steps),
           num_training_steps=num_training_steps
          )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

