from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#
# unnormalised weights for smapler
#Classs weight calculation
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_sampler(labels):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    sample_weights = np.array([class_weights[label] for label in labels])
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler



