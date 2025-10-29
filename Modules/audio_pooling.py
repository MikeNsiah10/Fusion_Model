import torch
import torch.nn as nn

class AudioPooling(nn.Module):
    """
    Pools frame-level audio embeddings [B, T, D] into a single vector [B, D].
    Supports mean, max, or attention pooling.
    """

    def __init__(self, method: str = "mean", hidden_dim: int = None):
        super().__init__()
        self.method = method
        if method == "attention":
            assert hidden_dim is not None, "hidden_dim required for attention pooling"
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor [B, T, D] — frame-level embeddings
            mask: Optional [B, T] — 1 for valid frames, 0 for padding
        Returns:
            pooled: Tensor [B, D]
        """
        if self.method == "mean":
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                return x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-6)
            return x.mean(1)

        elif self.method == "max":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            return x.max(1).values

        elif self.method == "attention":
            scores = self.attn(x).squeeze(-1)  # [B, T]
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1)  # [B, T]
            pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]
            return pooled

        else:
            raise ValueError(f"Unknown pooling method: {self.method}")