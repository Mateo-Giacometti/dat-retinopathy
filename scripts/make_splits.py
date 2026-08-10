"""Split train/val de EyePACS AGRUPADO POR PACIENTE (evita leakage ojo izq/der).

Los nombres son '<patient>_<left|right>' -> se agrupa por <patient> y se
estratifica por nivel de RD con StratifiedGroupKFold.

Uso:
    python scripts/make_splits.py
"""
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS_CSV = 'data/eyepacs/trainLabels.csv'
OUT_DIR = 'data/splits'
VAL_FRACTION = 0.10
SEED = 42


def main():
    df = pd.read_csv(LABELS_CSV)
    df['patient'] = df['image'].str.replace(r'_(left|right)$', '', regex=True)
    print(f'imagenes: {len(df)} | pacientes unicos: {df["patient"].nunique()}')

    n_splits = round(1 / VAL_FRACTION)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(df['image'], df['level'], df['patient']))

    train_df = df.iloc[train_idx][['image', 'level']].reset_index(drop=True)
    val_df = df.iloc[val_idx][['image', 'level']].reset_index(drop=True)

    # sanity check: ningun paciente en ambos lados
    p_train = set(df.iloc[train_idx]['patient'])
    p_val = set(df.iloc[val_idx]['patient'])
    assert len(p_train & p_val) == 0, 'LEAKAGE: pacientes compartidos entre train y val'

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUT_DIR, 'eyepacs_train.csv'), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, 'eyepacs_val.csv'), index=False)

    print(f'train: {len(train_df)} | val: {len(val_df)}')
    print('dist train:', train_df['level'].value_counts().sort_index().tolist())
    print('dist val:  ', val_df['level'].value_counts().sort_index().tolist())
    print(f'guardados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
