import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from unified_training.unified_fusion import JointTrainer
############################################################################
#from unified_training.fusion_head import FusionModel
from models.context_audio_head import HubertAudioClassifier
#from unified_training.text_head import TextHead
from teacher_trp_script.teacher_model import TurnGPT3Class
from torch.utils.data import DataLoader
from Modules.context_collate import ChunkedFusionContext, collate_context
#from unified_training.audio_head import HubertLightningModule
teacher=TurnGPT3Class()
teacher.init_tokenizer()
teacher.initialize_special_embeddings()
import pandas as pd
import numpy as np
from Modules.weight_sampler import get_sampler
train_df = pd.read_csv("/home/student/m/mnsiah/modality_fusion/csv_files/final_train.csv")
sampler = get_sampler(train_df['label'].values)

###################################################################################
# for MMF2F datasets

#   for current audio segment only
train_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/train_pt_files_hubc3text/train")
# path to the pt files
#single collate function for sinle utterance preprocessing
val_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/val_pt_files_hubc3text/val")
#test_dataset = ChunkedFusionDataset("/share/users/student/m/mnsiah/processed_files/test_pt_files_hubtext/test")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_context, num_workers=15)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_context, num_workers=15)
# test dataset would be used later
#test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn,num_workers=15)

for batch in train_loader:
    waveform = batch["waveforms"]
    print(type(waveform), waveform.shape)
    
    break

#audio model
audio_model=HubertAudioClassifier()
#audio_model=HubertLightningMod()


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

# Define checkpoint callback to save based on val_ppl
checkpoint_callback = ModelCheckpoint(
    monitor="val_macro_f1",           # Metric to monitor
    dirpath=  "/share/users/student/m/mnsiah/results/ctx_audio_chkpoint_3",    # Directory to save checkpoints
    filename="unfrozen_audio_ctx-{epoch:02d}-{val_macro_f1:.2f}",  # Checkpoint file naming
    save_top_k=1,                # Save only the best model
    mode="max",                  # Minimize val_ppl
          # Save only model weights
)
early_stopping_callback = EarlyStopping(
    monitor="val_macro_f1",  # <-- focus on mcro F1 score average
    mode="max",
    verbose=True
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=25,               # Number of epochs
    accelerator="gpu",           # Use GPU if available
    devices=1,                   # Number of GPUs
    precision="bf16-mixed",                # Use mixed precision (16-bit) for faster training
    callbacks=[checkpoint_callback,early_stopping_callback],  # Add checkpoint callback
    log_every_n_steps=10,        # Log metrics every 10 steps
    val_check_interval=1.0,     # Run validation 4 times per epoch
    default_root_dir="/share/users/student/m/mnsiah/results/ctx_audio_logs_3",
    accumulate_grad_batches=4
)

trainer.fit(audio_model,train_loader,val_loader)
#/share/users/student/m/mnsiah/fusion_model_checkpoints/text_only
#/share/users/student/m/mnsiah/fusion_model_checkpoints/audio_only
#/share/users/student/m/mnsiah/fusion_model_logs/text_only
#/share/users/student/m/mnsiah/fusion_model_logs/audio_only