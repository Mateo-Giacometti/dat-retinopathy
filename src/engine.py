"""Loops de entrenamiento y evaluacion con AMP bf16 y gradient accumulation."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None,
                    accum_steps: int = 1, amp: bool = True, epoch: int = 0,
                    num_epochs: int = 0, logger=None):
    model.train()
    running_loss, n_seen = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f'epoch {epoch + 1}/{num_epochs} [train]', dynamic_ncols=True)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            logits = model(imgs)
            loss = criterion(logits, labels) / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = imgs.size(0)
        running_loss += loss.item() * accum_steps * bs
        n_seen += bs
        pbar.set_postfix(loss=f'{running_loss / n_seen:.4f}')

    return running_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = True, desc: str = '[eval]'):
    model.eval()
    all_probs, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=desc, dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            logits = model(imgs)
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_prob, num_classes=y_prob.shape[1])
    metrics['y_true'] = y_true
    metrics['y_prob'] = y_prob
    return metrics
