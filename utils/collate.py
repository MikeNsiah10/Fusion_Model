#from transformers import DataCollatorWithPadding
#data_collator = DataCollatorWithPadding(tokenizer=teacher.tokenizer,padding=True,truncation=True)
#train_dataloader = DataLoader(processed["train"], batch_size=16, collate_fn=data_collator)
from torch.nn.utils.rnn import pad_sequence
import torch
# for daily dialog
class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.padding = padding
        #self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        batch = {
            "input_ids": [torch.tensor(f["input_ids"], dtype=torch.long) for f in features],
            "attention_mask": [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features],
            "speaker_ids": [torch.tensor(f["speaker_ids"], dtype=torch.long) for f in features],
            "labels": [torch.tensor(f["labels"], dtype=torch.long) for f in features],
        }
        batch["input_ids"] = pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        batch["attention_mask"] = pad_sequence(
            batch["attention_mask"], batch_first=True, padding_value=0
        )
        batch["speaker_ids"] = pad_sequence(
            batch["speaker_ids"], batch_first=True, padding_value=0  # Adjust if needed
        )
        batch["labels"] = pad_sequence(
            batch["labels"], batch_first=True, padding_value=0
        )
        
        return batch
