import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from unified_training.unified_fusion import JointTrainer
############################################################################
#from unified_training.fusion_head import FusionModel
from models.audio_head_1 import HubertClassifier
from models.fusion_head import FusionStudent_1
from models.fusion_with_modality import FusionStudent
from models.text_head_1 import TextClassifier
#from unified_training.text_head import TextHead
from teacher_trp_script.teacher_model import TurnGPT3Class
from torch.utils.data import DataLoader
from models.distilhub import HubertSmall
from Modules.context_collate import ChunkedFusionContext, collate_context
from teacher_trp_script.teacher_model import TurnGPT3Class
teacher=TurnGPT3Class()
teacher.init_tokenizer()
teacher.initialize_special_embeddings()
import pandas as pd
import numpy as np
from Modules.weight_sampler import get_sampler
train_df = pd.read_csv("/home/student/m/mnsiah/modality_fusion/csv_files/final_train_1.csv")
sampler = get_sampler(train_df['label'].values)

###################################################################################
# for MMF2F datasets

#   for current audio segment only
train_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/train_pt_files_final/train")
# path to the pt files
#single collate function for sinle utterance preprocessing
val_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/val_final")
#test_dataset = ChunkedFusionDataset("/share/users/student/m/mnsiah/processed_files/test_pt_files_hubtext/test")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_context, num_workers=15)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_context, num_workers=15)
# test dataset would be used later
#test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn,num_workers=15)
# Example iteration
####################################################################
#for previous two turns + current audio segment
#from fusion_script.fusion_context_collate import ChunkedFusionContext,collate_context

#train_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/train_pt_files_hubctext/train")
# path to the pt files
#single collate function for sinle utterance preprocessing
#val_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/val_pt_files_hubctext/val")
#test_dataset = ChunkedFusionDataset("/share/users/student/m/mnsiah/processed_files/test_pt_files_hubtext/test")
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_context, num_workers=15)
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_context, num_workers=15)


















###################################################################################
for batch in train_loader:
    waveform = batch["waveforms"]
    print(type(waveform), waveform.shape)
    
    break

#audio model
#audio_chktpoint="/share/users/student/m/mnsiah/results/ctx_audio_chkpoint_1/unfrozen_audio_ctx-epoch=03-val_macro_f1=0.83.ckpt"
#audio_chktpoint="/share/users/student/m/mnsiah/new_results/audio_chkpoint/audio_ctx-epoch=10-val_min_f1=0.78.ckpt"
#audio_model=HubertClassifier.load_from_checkpoint(audio_chktpoint)
#text model
#text_chktpoint="/share/users/student/m/mnsiah/results/ctx_text_chkpt_1/ctx_text_model_focal-epoch=07-val_macro_f1=0.85.ckpt"
text_chktpoint="/share/users/student/m/mnsiah/final_results/text_context/text_model-epoch=04-val_min_f1=0.74.ckpt"
#text_model=TextClassifier.load_from_checkpoint(text_chktpoint)
audio_chktpoint="/share/users/student/m/mnsiah/final_results/distilhubv2_chkpt/distil_hub-epoch=04-val_min_f1=0.79.ckpt"
#audio_model=HubertSmall.load_from_checkpoint(audio_chktpoint)

#fusion model
fusion_model=FusionStudent(text_ckpt_path=text_chktpoint,audio_ckpt_path=audio_chktpoint)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

# Define checkpoint callback to save based on val_ppl
checkpoint_callback = ModelCheckpoint(
    monitor="val_min_f1",           # Metric to monitor
    dirpath=  "/share/users/student/m/mnsiah/final_results/model_chkt_1",    # Directory to save checkpoints
    filename="final_model-{epoch:02d}-{val_min_f1:.2f}",  # Checkpoint file naming
    save_top_k=1,                # Save only the best model
    mode="max",                  # Minimize val_ppl
          # Save only model weights
)
early_stopping_callback = EarlyStopping(
    monitor="val_min_f1",  # <-- focus on mcro F1 score average
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
    default_root_dir="/share/users/student/m/mnsiah/final_results/model_logs_1",
    accumulate_grad_batches=4
)

trainer.fit(fusion_model,train_loader,val_loader)
#/share/users/student/m/mnsiah/fusion_model_checkpoints/text_only
#/share/users/student/m/mnsiah/fusion_model_checkpoints/audio_only
#/share/users/student/m/mnsiah/fusion_model_logs/text_only
#/share/users/student/m/mnsiah/fusion_model_logs/audio_only