"""Small tabular network: 28 -> 64 -> 32 -> 1 logit (no sigmoid during loss)."""
import torch
from torch import nn


class ShotNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def choose_device(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run check_gpu.py; install a CUDA-enabled torch wheel and compatible NVIDIA driver, or explicitly use --device cpu.')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested))
