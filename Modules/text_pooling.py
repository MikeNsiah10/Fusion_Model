import torch
import torch.nn as nn

class PoolingStrategy(nn.Module):
    def __init__(self, hidden_dim, pooling_type='attention'):
        super().__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim

        if pooling_type == "attention":
            # Learnable query vector initialized with Xavier uniform
            self.query_vector = nn.Parameter(torch.empty(hidden_dim))
            nn.init.xavier_uniform_(self.query_vector.unsqueeze(0))

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, N, D]
        attention_mask: [B, N] (optional, 1=real token, 0=pad)
        """
        if self.pooling_type == "cls":
            pooled = hidden_states[:, 0]

        elif self.pooling_type == "max":
            pooled = torch.max(hidden_states, dim=1)[0]

        elif self.pooling_type == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = torch.sum(hidden_states * mask, dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                pooled = sum_hidden / lengths
            else:
                pooled = torch.mean(hidden_states, dim=1)

        elif self.pooling_type == "cls_max":
            cls = hidden_states[:, 0]
            max_pooled = torch.max(hidden_states[:, 1:], dim=1)[0]
            pooled = torch.cat([cls, max_pooled], dim=1)

        elif self.pooling_type == "mean_max":
            mean_pooled = torch.mean(hidden_states, dim=1)
            max_pooled = torch.max(hidden_states, dim=1)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        elif self.pooling_type == "attention":
            # [B, N, D] · [D] → [B, N]
            attn_scores = torch.matmul(hidden_states, self.query_vector)
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, N, 1]
            pooled = torch.sum(hidden_states * attn_weights, dim=1)

        elif self.pooling_type == "last":
            if attention_mask is None:
                pooled = hidden_states[:, -1, :]
            else:
                lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0))
                pooled = hidden_states[batch_indices, lengths, :]

        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return pooled