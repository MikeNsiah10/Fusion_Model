import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel
from Modules.text_pooling import PoolingStrategy
from Modules.focal_loss import FocalLoss
import pandas as pd
from transformers import get_linear_schedule_with_warmup
#now exploring full finetuning
#cuurett implementtion layer wise unfreezing
class TextClassifier(pl.LightningModule):
    def __init__(
        self,
        student_model_name="distilgpt2",
        num_classes=3,
        vocab_size=None,
        tokenizer=None,
        lr_backbone=1e-5,
        lr_classifier=1e-4,
        #unfreeze_start_epoch=2,   # 👈 unfreeze after this many epochs
        plot_dir="/home/student/m/mnsiah/modality_fusion/Evaluation/train_cm_plots/text_plots",
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Load pretrained model ---
        if tokenizer is not None:
            vocab_size = len(tokenizer)
        self.student = GPT2LMHeadModel.from_pretrained(student_model_name)
        if vocab_size is not None:
            self.student.resize_token_embeddings(vocab_size)

        # --- Initially freeze all layers ---
        #for p in self.student.parameters():
         #   p.requires_grad = True

        hidden_size = self.student.config.hidden_size
        self.pooler = PoolingStrategy(hidden_dim=hidden_size, pooling_type="last")

        # --- Loss ---
        self.criterion = FocalLoss(
            gamma=1.5,
            alpha=[0.293, 0.293, 0.414], #keep, turn ,bk #modified focal loss previous #was 0.3,0.3,0.7 with alpha 2.0
            
            reduction='mean',
            task_type="multi-class",
            num_classes=num_classes
        )

        # --- Classifier head ---
        self.proj = nn.Linear(hidden_size, 256)
        self.student_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        # --- Training hyperparams ---
        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_classifier = lr_classifier
        #self.unfreeze_start_epoch = unfreeze_start_epoch
        self.ce = nn.CrossEntropyLoss()

        # --- Metrics ---
        self.train_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)

        self.val_preds, self.val_labels = [], []
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_dir = plot_dir

    # ------------------------------------
    # Forward pass
    # ------------------------------------
    def forward(self, input_ids, attention_mask=None):
        outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        pooled_repr = self.pooler(hidden_states, attention_mask)
        pooled_embeds = self.proj(pooled_repr)
        logits = self.student_classifier(pooled_embeds)
        return {"logits": logits, "hidden_states": hidden_states, "pooled_embeds": pooled_embeds}

    # ------------------------------------
    # Training & Validation
    # ------------------------------------
    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch.get("attention_mask", None))
        logits = outputs["logits"]
        loss = self.criterion(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        self.train_macro_f1.update(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch.get("attention_mask", None))
        logits = outputs["logits"]
        loss = self.ce(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        self.val_macro_f1.update(preds, batch["labels"])
        self.val_per_class_f1.update(preds, batch["labels"])
        self.val_preds.append(preds.cpu())
        self.val_labels.append(batch["labels"].cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------
    # Gradual unfreezing (main addition)
    # ------------------------------------
    '''def on_train_epoch_start(self):
        """Unfreeze progressively higher layers after a few epochs."""
        if self.current_epoch == self.unfreeze_start_epoch:
            print(f"\n Unfreezing top transformer layers at epoch {self.current_epoch}\n")
            for i, block in enumerate(self.student.transformer.h):
                requires_grad = i >= 4  # unfreeze top 2 layers (4,5 for DistilGPT2)
                for p in block.parameters():
                    p.requires_grad = requires_grad
            for p in self.student.transformer.ln_f.parameters():
                p.requires_grad = True
                                                '''
    # ------------------------------------
    def on_train_epoch_start(self):
        blocks = self.student.transformer.h
        ln_f = self.student.transformer.ln_f
        depth = len(blocks)  # should be 12 in your case
    
    # Freeze everything first  
        for block in blocks:
           for p in block.parameters():
            p.requires_grad = False
        for p in ln_f.parameters():
            p.requires_grad = False
        
    # Epoch-based schedule
        if self.current_epoch == 0:
          unfrozen = 0 
        elif self.current_epoch == 1:
           unfrozen = 2   # last 2 layers
        elif self.current_epoch == 2:
           unfrozen = 4   # last 4 layers
        elif self.current_epoch == 3:
           unfrozen = 6   # last 6 layers
        elif self.current_epoch == 4:
            unfrozen = 8   # last 8 layers
        elif self.current_epoch == 5:
           unfrozen = 10  # last 10 layers
        else:  # epoch >= 6
           unfrozen = depth  # all 12 layers
        
    # Unfreeze the chosen number of top layers
        if unfrozen > 0:
            for block in blocks[-unfrozen:]:
              for p in block.parameters():
                p.requires_grad = True
            for p in ln_f.parameters():
                p.requires_grad = True
            
    # Logging
        unfrozen_idxs = list(range(depth - unfrozen, depth)) if unfrozen > 0 else []
        self.print(f"[Epoch {self.current_epoch}] Unfroze blocks {unfrozen_idxs} of {depth}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.log("trainable_params", float(trainable), prog_bar=False, on_epoch=True)
    # ------------------------------------
    # Validation Epoch End (same as before)
    # ------------------------------------
    def on_validation_epoch_end(self):
        macro_f1 = self.val_macro_f1.compute()
        per_class_f1 = self.val_per_class_f1.compute()
        self.log("val_macro_f1", macro_f1, prog_bar=True)
        for i, class_f1 in enumerate(per_class_f1):
            self.log(f"val_f1_class_{i}", class_f1)

        val_min_f1 = torch.min(per_class_f1)
        self.log("val_min_f1", val_min_f1, prog_bar=True)

        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(self.num_classes)))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Epoch {self.current_epoch}")
        plt.savefig(os.path.join(self.plot_dir, f"cm_epoch{self.current_epoch}.png"))
        plt.close()

        classes = ["keep", "turn", "backchannel"]
        plt.bar(classes, per_class_f1.cpu().numpy())   # use class names directly
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.title(f"Per Class F1 Scores - Epoch {self.current_epoch}")
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

    # Optimizer and Scheduler
    # ------------------------------------
    

    def configure_optimizers(self):
    # Separate LRs for classifier and backbone
        optimizer = torch.optim.AdamW(
           [
            #{"params": [p for p in self.student.parameters() if p.requires_grad], "lr": self.lr_backbone},
            {"params": self.student.parameters(), "lr": self.lr_backbone},
            {"params": list(self.proj.parameters()) + list(self.student_classifier.parameters()), "lr": self.lr_classifier},
          ],
          weight_decay=1e-2,
      )

    # Total training steps = number of batches per epoch * number of epochs
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=num_warmup_steps,
          num_training_steps=num_training_steps,
        )

        return {
          "optimizer": optimizer,
          "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",   # update every step
            "frequency": 1,
        },
    }

