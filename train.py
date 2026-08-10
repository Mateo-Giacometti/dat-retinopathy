"""Entrenamiento del clasificador de RD (DAT-Tiny por defecto).

Uso:
    python train.py --config configs/dat_tiny_smoke.yaml
    python train.py --config configs/dat_tiny_512.yaml
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.data.dataset import DRDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.engine import evaluate, train_one_epoch
from src.models import build_model
from src.models.losses import FocalLoss, class_inverse_freq_alpha
from src.utils import (copy_config, get_logger, save_checkpoint, save_metrics,
                       set_seed)


def build_optimizer(model, lr, head_lr_mult, weight_decay):
    """LR discriminativo: la head nueva (cls_head) aprende mas rapido."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'cls_head' in name or 'classifier' in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    groups = [
        {'params': backbone_params, 'lr': lr},
        {'params': head_params, 'lr': lr * head_lr_mult},
    ]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    logger = get_logger()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_name = cfg.get('run_name', 'run')
    out_dir = os.path.join('checkpoints', run_name)
    tb_dir = os.path.join('runs', run_name)
    os.makedirs(out_dir, exist_ok=True)

    # img_size unico: data manda, se inyecta en model
    img_size = cfg['data']['img_size']
    cfg['model']['img_size'] = img_size

    # ---------------- datasets ----------------
    train_ds = DRDataset(
        cfg['data']['train_csv'], cfg['data']['img_dir'],
        transform=get_train_transforms(img_size),
        img_ext=cfg['data'].get('img_ext', '.jpg'),
        subset=cfg['data'].get('subset_train', 0), seed=cfg.get('seed', 42),
    )
    val_ds = DRDataset(
        cfg['data']['val_csv'], cfg['data']['img_dir'],
        transform=get_val_transforms(img_size),
        img_ext=cfg['data'].get('img_ext', '.jpg'),
        subset=cfg['data'].get('subset_val', 0), seed=cfg.get('seed', 42),
    )
    tcfg = cfg['train']
    train_loader = DataLoader(
        train_ds, batch_size=tcfg['batch_size'], shuffle=True,
        num_workers=cfg['data'].get('num_workers', 8), pin_memory=True,
        persistent_workers=True, prefetch_factor=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg['batch_size'] * 2, shuffle=False,
        num_workers=cfg['data'].get('num_workers', 8), pin_memory=True,
        persistent_workers=True,
    )
    logger.info(f'train: {len(train_ds)} imgs | val: {len(val_ds)} imgs | '
                f'img_size: {img_size} | device: {device}')

    # ---------------- modelo / loss / optim ----------------
    model = build_model(cfg['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f'modelo: {cfg["model"]["name"]} ({n_params:.1f}M params)')

    alpha = cfg['loss'].get('alpha')
    if alpha == 'inv_freq':
        train_levels = pd.read_csv(cfg['data']['train_csv'])['level']
        alpha = class_inverse_freq_alpha(train_levels, cfg['model'].get('num_classes', 5))
        logger.info(f'focal alpha (inv_freq): {alpha.tolist()}')
    criterion = FocalLoss(gamma=cfg['loss'].get('gamma', 2.0), alpha=alpha)

    optimizer = build_optimizer(model, tcfg['lr'], tcfg.get('head_lr_mult', 10),
                                tcfg.get('weight_decay', 1e-4))
    steps_per_epoch = math.ceil(len(train_loader) / tcfg.get('grad_accum', 1))
    total_steps = steps_per_epoch * tcfg['epochs']
    warmup_steps = steps_per_epoch * tcfg.get('warmup_epochs', 1)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    writer = SummaryWriter(tb_dir)
    copy_config(args.config, out_dir)

    # ---------------- loop ----------------
    best_qwk = -1.0
    best_metrics = None
    patience = tcfg.get('early_stop_patience', 5)
    bad_epochs = 0

    for epoch in range(tcfg['epochs']):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, accum_steps=tcfg.get('grad_accum', 1),
            amp=tcfg.get('amp', True), epoch=epoch,
            num_epochs=tcfg['epochs'], logger=logger,
        )
        val_metrics = evaluate(model, val_loader, device,
                               amp=tcfg.get('amp', True), desc=f'epoch {epoch + 1} [val]')
        dt = time.time() - t0

        qwk, auc = val_metrics['qwk'], val_metrics['auc_macro']
        logger.info(
            f'epoch {epoch + 1}/{tcfg["epochs"]} ({dt / 60:.1f} min) | '
            f'loss {train_loss:.4f} | val QWK {qwk:.4f} | AUC {auc:.4f} | '
            f'AUC ref {val_metrics["auc_referable"]:.4f} | acc {val_metrics["accuracy"]:.4f}'
        )
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/qwk', qwk, epoch)
        writer.add_scalar('val/auc_macro', auc, epoch)
        writer.add_scalar('val/auc_referable', val_metrics['auc_referable'], epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if qwk > best_qwk:
            best_qwk = qwk
            best_metrics = val_metrics
            save_checkpoint({
                'model': model.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'qwk': qwk,
            }, out_dir, 'best.pt')
            logger.info(f'  -> mejor QWK, checkpoint guardado en {out_dir}/best.pt')
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f'early stopping (sin mejora en {patience} epocas)')
                break

    writer.close()

    # metrics.json sin arrays pesados
    final = {k: v for k, v in (best_metrics or {}).items()
             if k not in ('y_true', 'y_prob')}
    final['best_qwk'] = best_qwk
    save_metrics(final, out_dir)
    logger.info(f'fin. mejor val QWK: {best_qwk:.4f}. Artefactos en {out_dir}/')


if __name__ == '__main__':
    main()
