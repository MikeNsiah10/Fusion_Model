import torch
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
class ChunkedFusionContext(Dataset):
    def __init__(self, split_dir):
        self.files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data


def collate_context(batch):
    # enforce exactly 3 utterances
    max_utts = 3

    # global max length across all utterances in the batch
    global_max_len = max(w.shape[-1] for sample in batch for w in sample["waveforms"])

    padded_waveforms = []
    utt_masks = []

    for sample in batch:
        utts = sample["waveforms"]
        num_utts = len(utts)

        # truncate if more than 3
        if len(utts) > max_utts:
            utts = utts[-max_utts:]  # keep the last 3 (prev2, prev1, current)

        # pad if fewer than 3
        while len(utts) < max_utts:
            utts.insert(0, torch.zeros_like(utts[0]))  # pad at the front with silence

        padded_utts = []
        for w in utts:
            pad_len = global_max_len - w.shape[-1]
            if pad_len > 0:
                w = F.pad(w, (0, pad_len))
            padded_utts.append(w)

        padded_utts = torch.stack(padded_utts)  # [3, 1, global_max_len]
        padded_waveforms.append(padded_utts)

        # mask: 1 for real utterances, 0 for padded ones
        utt_mask = torch.zeros(max_utts, dtype=torch.long)
        utt_mask[-num_utts:] = 1  # mark the last num_utts as real
        utt_masks.append(utt_mask)

    waveform_batch = torch.stack(padded_waveforms)  # [B, 3, 1, global_max_len]
    audio_mask = torch.stack(utt_masks)             # [B, 3]

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)

    return {
        "waveforms": waveform_batch,   # [B, 3, 1, global_max_len]
        "audio_mask": audio_mask,      # [B, 3]
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sr": batch[0]["sr"]
    }



