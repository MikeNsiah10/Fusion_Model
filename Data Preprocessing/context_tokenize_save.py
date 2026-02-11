import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
from tqdm import tqdm

# TurnGPT imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turngpt.model import TurnGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Initialize Teacher (TurnGPT)
# ----------------------------
teacher = TurnGPT(
    pretrained_model_name_or_path="gpt2",
    pretrained=True,
    trp_projection_steps=1,
    trp_projection_type="linear",
    omit_dialog_states=False,
    no_train_first_n=0,
    learning_rate=1e-4,
    weight_loss=True,
    weight_regular_token=0.5,
    weight_eos_token=1.0,
)
teacher.init_tokenizer()
teacher.initialize_special_embeddings()
teacher.to(device)
tokenizer = teacher.tokenizer



# ----------------------------
# Load CSV splits
# ----------------------------
df_train = pd.read_csv(r"C:\Users\nsiah\Desktop\Preprocessing_audio\refining_preprocessing\updated_train.csv")
df_val   = pd.read_csv(r"C:\Users\nsiah\Desktop\Preprocessing_audio\refining_preprocessing\updated_val.csv")
df_test  = pd.read_csv(r"C:\Users\nsiah\Desktop\Preprocessing_audio\refining_preprocessing\updated_test.csv")

splits = {"train": df_train, "val": df_val, "test": df_test}

audio_dirs = {
    "train": r"D:\audio_train",
    "val": r"D:\audio_val",
    "test": r"D:\audio_test"
}

save_roots = {
    "train": r"D:\train_pt_files_final",
    "val": r"D:\val_pt_files_final",
    "test": r"D:\test_pt_files_final"
}

target_sr = 16000
max_length = 256  # for text tokenizer
context_window = 2  # number of previous turns


# ----------------------------
# Audio loader
# ----------------------------
def load_audio_segment(path, start_sec, end_sec):
    info = torchaudio.info(path)
    sr = info.sample_rate
    start_frame = int(start_sec * sr)
    num_frames = int((end_sec - start_sec) * sr)
    waveform, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)
    return waveform, sr

# ----------------------------
# Speaker mapping
# ----------------------------
speaker_map = {
    0: "<speaker1>",
    1: "<speaker2>"
}

# ----------------------------
# Processing loop
# ----------------------------
for split_name, split_df in splits.items():
    print(f"\nProcessing {split_name} split ({len(split_df)} samples)...")

    audio_dir = audio_dirs[split_name]
    save_root = save_roots[split_name]
    save_dir = os.path.join(save_root, split_name)
    os.makedirs(save_dir, exist_ok=True)

    # Group by video_id so context never crosses conversations
    for vid, vid_df in split_df.groupby("video_id"):
        vid_df = vid_df.reset_index(drop=True)  # reset row indices inside each video

        for i, row in enumerate(tqdm(vid_df.itertuples(index=False),
                                     total=len(vid_df),
                                     desc=f"{split_name}-{vid}")):

            # Collect current + previous rows (within this video only)
            context_rows = vid_df.iloc[max(0, i-context_window):i]
            current_row = row

            # ---- TEXT ----
            text_parts = []
            for ctx_row in context_rows.itertuples(index=False):
                speaker_token = speaker_map[int(ctx_row.speaker)]
                text_parts.append(f"{speaker_token} {ctx_row.text} <SEP>")
            curr_speaker_token = speaker_map[int(current_row.speaker)]
            text_parts.append(f"{curr_speaker_token} {current_row.text}")
            full_text = " ".join(text_parts)

            tokenized = tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # ---- AUDIO ----
            waveforms = []
            for ctx_row in list(context_rows.itertuples(index=False)) + [current_row]:
                audio_path = f"{audio_dir}/{ctx_row.video_id}.wav"
                waveform, sr = load_audio_segment(audio_path, ctx_row.start, ctx_row.end)
                if sr != target_sr:
                    waveform = Resample(orig_freq=sr, new_freq=target_sr)(waveform)
                if waveform.shape[0] != 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveforms.append(waveform)

            # ---- SAVE ----
            torch.save({
                "waveforms": waveforms,  # list of tensors [prev1, prev2, current]
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": current_row.label,
                "sr": target_sr
            }, os.path.join(save_dir, f"{vid}_{i}.pt"))


        
