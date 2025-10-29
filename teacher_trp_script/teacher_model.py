import os, sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau
from turngpt.model import TurnGPT
from Modules.text_pooling import PoolingStrategy
from Modules.focal_loss import FocalLoss
import numpy as np

# ----------------------------------------------------------------------
# Modified TurnGPT for 3-class TRP prediction
# ----------------------------------------------------------------------
class TurnGPT3Class(TurnGPT):
    """
    Modified TurnGPT with 3-class head for:
    0 = continue_speech, 1 = turn, 2 = backchannel
    """

    def __init__(self,
                 pretrained_model_name_or_path="microsoft/DialoGPT-small", #never distilled so i set to small
                 num_classes=3,
                 **turngpt_kwargs):
        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path,
                         **turngpt_kwargs)

        hidden_size = self.transformer.config.hidden_size

        # === Pooling strategy ===
        self.pooler = PoolingStrategy(hidden_dim=hidden_size, pooling_type="last")

        # === Classification head ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes
        self.val_preds = []
        self.val_targets = []
        self.save_hyperparameters()

        # === Loss functions ===
        self.focal_loss = FocalLoss(
            gamma=2.0,
            alpha=[0.3, 0.3, 0.7],  # keep, turn, backchannel
            reduction='mean',
            num_classes=num_classes,
            task_type="multi-class"
        )
        self.ce_loss = nn.CrossEntropyLoss()

        # === Freeze GPT-2 backbone initially ===
        for param in self.transformer.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, speaker_ids=None):
        outputs = self.transformer.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=speaker_ids,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        pooled_repr = self.pooler(hidden_states, attention_mask)
        task_logits = self.classifier(pooled_repr)

        return {
            "task_logits": task_logits,
            "hidden_states": hidden_states
        }

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            speaker_ids=batch.get('speaker_ids')
        )

        logits = outputs["task_logits"]
        loss = self.focal_loss(logits, batch['labels'])

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            speaker_ids=batch.get('speaker_ids')
        )

        logits = outputs["task_logits"]
        loss = self.ce_loss(logits, batch['labels'])

        preds = torch.argmax(logits, dim=1).cpu()
        targets = batch['labels'].cpu()

        self.val_preds.extend(preds.tolist())
        self.val_targets.extend(targets.tolist())

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self):
        if not self.val_preds or not self.val_targets:
            return

        all_preds = np.array(self.val_preds)
        all_targets = np.array(self.val_targets)

        # Metrics
        val_f1_per_class = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2])
        macro_f1 = f1_score(all_targets, all_preds, average='macro', labels=[0, 1, 2])
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

        # Log metrics
        for i, f1 in enumerate(val_f1_per_class):
            self.log(f'val_f1_class{i}', f1, prog_bar=True, on_epoch=True)
        self.log('val_macro_f1', macro_f1, prog_bar=True, on_epoch=True)
        val_min_f1=float(np.min(val_f1_per_class))

        self.log("val_min_f1", val_min_f1, prog_bar=True)
        

        # Save confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Epoch {self.current_epoch}")
        save_path = f"/home/student/m/mnsiah/modality_fusion/teacher_trp_script/cm_epoch_{self.current_epoch}.png"
        plt.savefig(save_path)
        plt.close()

        # Reset buffers
        self.val_preds.clear()
        self.val_targets.clear()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": self.classifier.parameters(), "lr": 1e-4}],
            weight_decay=1e-2
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }