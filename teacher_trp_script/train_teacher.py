import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from teacher_trp_script.teacher_model import TurnGPT3Class
#from student_model import StudentLM
from turngpt.model import TurnGPT
from torch.utils.data import DataLoader
#from datasets import load_from_disk
import torch
#from distil_data import TurnTakingDataset
#from teacher_trp_script.weight_sampler import get_sampler

 
#create teacher model
#teacher_model=teacher_model

# Load dataset
#print("Loading dataset...")
#dataset = load_from_disk("/home/student/m/mnsiah/modality_fusion/dialogsum_preprocessed")
#print(f"Dataset splits: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")



# Data collator for dialogshow datasets
#train_dataset = DialogSumDataset(dataset['train']['dialogue'], teacher_model.tokenizer)
#val_dataset=DialogSumDataset(dataset['test']['dialogue'], teacher_model.tokenizer)
#test_dataset = DialogSumDataset(dataset['validation']['dialogue'], teacher_model.tokenizer)

# DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,collate_fn=custom_collate_fn, num_workers=4)
#test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,collate_fn=custom_collate_fn, num_workers=4)
#print(f"Train loader batches: {len(train_loader)}")
#print("DataLoaders created successfully")
##########################################################################################
import pandas as pd
#train_df = pd.read_csv("/home/student/m/mnsiah/modality_fusion/unified_training/final_train.csv")
#sampler = get_sampler(train_df['label'].values)


###########################################################################################
from teacher_model import TurnGPT3Class
#instantial model
model=TurnGPT3Class()
model.init_tokenizer()
model.initialize_special_embeddings()
model.eval()
############################################################################################



###########################################################################################
# for MMF2F datasets 
from Modules.context_collate import ChunkedFusionContext, collate_context
#using four previous utterances as context
train_dataset = ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/train_pt_files_final/train")
 # path to the pt files 
val_dataset=ChunkedFusionContext("/share/users/student/m/mnsiah/processed_files/val_final")
#test_dataset=ChunkedFusionDataset("/share/users/student/m/mnsiah/processed_files/test_pt_files/test")
train_loader = DataLoader(train_dataset, batch_size=8,shuffle=True, collate_fn=collate_context,num_workers=15)
val_loader= DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_context,num_workers=15)
#test dataset would be used later
#test_loader= DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Example usage for testing
for batch in train_loader:
    print(batch["waveforms"].shape)  # [B, max_num_chunks, 2, 8000]
    print(batch["input_ids"].shape) 
    print(batch['labels'].shape)       # [B, max_length]
    break



##########################################################################################
#student_model=StudentWithTRP(freeze_backbone=False,teacher=teacher_model)



########################################################
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

# Define checkpoint callback to save based on val_ppl
checkpoint_callback = ModelCheckpoint(
    monitor="val_min_f1",           # Metric to monitor
    dirpath=  "/share/users/student/m/mnsiah/final_results/teacher_model_checkpoint",    # Directory to save checkpoints
    filename="turngpt_1{epoch:02d}-{val_min_f1:.3f}",  # Checkpoint file naming
    save_top_k=1,                # Save only the best model
    mode="max",                  # Minimize val_ppl
          # Save only model weights
)
early_stopping_callback = EarlyStopping(
    monitor="val_min_f1",  # <-- focus on F1 score average
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
    default_root_dir="/share/users/student/m/mnsiah/final_results/teacher_model_logs",
    accumulate_grad_batches=4
)




#   #Train the model
#Turngpt3class or Turngpt3multihead
trainer.fit(model, train_loader, val_loader)