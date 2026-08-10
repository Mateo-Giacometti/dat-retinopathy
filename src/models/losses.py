"""Focal Loss multi-clase (Lin et al., 2017) para el desbalance de grados de RD."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt).

    Args:
        gamma: factor de foco (0 == CE plano).
        alpha: None, escalar, o tensor de pesos por clase [num_classes]
               (p.ej. frecuencia inversa normalizada).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha if torch.is_tensor(alpha) else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # probabilidad asignada a la clase verdadera
        loss = (1.0 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        return loss.mean()


def class_inverse_freq_alpha(train_csv_col, num_classes: int = 5) -> torch.Tensor:
    """Alpha = frecuencia inversa normalizada a media 1 (suaviza el desbalance)."""
    counts = train_csv_col.value_counts().sort_index().values.astype('float64')
    assert len(counts) == num_classes
    inv = 1.0 / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)
