"""Evaluacion de un checkpoint sobre cualquier CSV (p.ej. APTOS = validacion externa).

Uso:
    python scripts/evaluate.py --checkpoint checkpoints/dat_tiny_512/best.pt \
        --csv data/aptos/train.csv --img-dir data/processed/aptos_512 \
        --image-col id_code --label-col diagnosis
"""
import argparse
import os
import sys

# permitir import de src/ desde scripts/ (repo root al path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.data.transforms import get_val_transforms
from src.engine import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--image-col', default='image')
    ap.add_argument('--label-col', default='level')
    ap.add_argument('--img-ext', default='.jpg')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--out', default=None, help='donde guardar reporte (default: junto al checkpoint)')
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    img_size = cfg['data']['img_size']
    cfg['model']['img_size'] = img_size

    from src.models import build_model
    model = build_model({**cfg['model'], 'pretrained': None})
    model.load_state_dict(ckpt['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'checkpoint: {args.checkpoint} (epoch {ckpt.get("epoch")}, QWK {ckpt.get("qwk"):.4f})')

    # normalizar nombres de columnas a (image, level) para DRDataset
    import pandas as pd
    df = pd.read_csv(args.csv)
    df = df.rename(columns={args.image_col: 'image', args.label_col: 'level'})

    ds = DRDataset.__new__(DRDataset)  # construccion manual para no reescribir el CSV
    ds.df, ds.img_dir, ds.img_ext = df, args.img_dir, args.img_ext
    ds.transform = get_val_transforms(img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    metrics = evaluate(model, loader, device, desc='[eval externo]')

    print('\n===== RESULTADOS =====')
    print(f'imagenes:        {len(df)}')
    print(f'QWK:             {metrics["qwk"]:.4f}')
    print(f'Accuracy:        {metrics["accuracy"]:.4f}')
    print(f'AUC macro OVR:   {metrics["auc_macro"]:.4f}')
    print(f'AUC por clase:   {np.round(metrics["auc_per_class"], 4).tolist()}')
    print(f'AUC referible:   {metrics["auc_referable"]:.4f}')
    print(f'Sens por clase:  {np.round(metrics["sensitivity_per_class"], 4).tolist()}')
    print(f'Spec por clase:  {np.round(metrics["specificity_per_class"], 4).tolist()}')

    # matriz de confusion
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    ax.set_title(f'Matriz de confusion - QWK {metrics["qwk"]:.3f}')
    fig.colorbar(im)
    fig.tight_layout()

    out_dir = args.out or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.csv))[0]
    fig.savefig(os.path.join(out_dir, f'confusion_{tag}.png'), dpi=150)

    import json
    report = {k: v for k, v in metrics.items() if k not in ('y_true', 'y_prob')}
    report['csv'] = args.csv
    with open(os.path.join(out_dir, f'eval_{tag}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nreporte guardado en {out_dir}/eval_{tag}.json + confusion_{tag}.png')


if __name__ == '__main__':
    main()
