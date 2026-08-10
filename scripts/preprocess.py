"""Preprocesamiento offline de retinografias (propuesta SS3.2):

  1. Crop adaptativo de bordes negros (bbox del umbral sobre gris)
  2. Resize a SIZExSIZE
  3. CLAHE sobre el canal verde (mayor contraste vascular/lesiones)
  4. Guardado JPEG q90 en cache -> entrenamiento sin bottleneck de CPU

Uso:
    python scripts/preprocess.py --csv data/eyepacs/trainLabels.csv \
        --img-dir data/eyepacs/train --in-ext .jpeg \
        --out-dir data/processed/eyepacs_512 --size 512 --workers 12

    python scripts/preprocess.py --csv data/aptos/train.csv --image-col id_code \
        --label-col diagnosis --img-dir data/aptos/train_images --in-ext .png \
        --out-dir data/processed/aptos_512 --size 512 --workers 12
"""
import argparse
import os
import re
from functools import partial
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm


def crop_black_borders(img, thresh: int = 10, pad_frac: float = 0.02):
    """Recorta al bounding box de los pixeles no-negros (circulo util de retina)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > thresh
    if mask.sum() < 100:  # imagen practicamente vacia: no recortar
        return img
    coords = cv2.findNonZero(mask.astype('uint8'))
    x, y, w, h = cv2.boundingRect(coords)
    pad = int(max(w, h) * pad_frac)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1 = min(x + w + pad, img.shape[1])
    y1 = min(y + h + pad, img.shape[0])
    return img[y0:y1, x0:x1]


def apply_clahe_green(img, clip_limit: float = 2.0, tile: int = 8):
    """CLAHE en el canal verde; rojo y azul quedan iguales."""
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    g = clahe.apply(g)
    return cv2.merge([b, g, r])


EXT_FALLBACK = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def process_one(name, img_dir, in_ext, out_dir, size):
    """Devuelve None si OK, o el nombre si falla (para reporte correcto con
    imap_unordered)."""
    try:
        # el CSV puede incluir la extension en el nombre (p.ej. Messidor):
        # la quitamos para concatenar in_ext de forma consistente
        name = re.sub(r'(?i)\.(jpe?g|png)$', '', name)
        src = os.path.join(img_dir, name + in_ext)
        if not os.path.exists(src):  # datasets mixtos (PNG+JPEG): buscar extension real
            for ext in EXT_FALLBACK:
                cand = os.path.join(img_dir, name + ext)
                if os.path.exists(cand):
                    src = cand
                    break
            else:
                return name
        dst = os.path.join(out_dir, name + '.jpg')
        if os.path.exists(dst):
            return None
        img = cv2.imread(src)
        if img is None:
            return name
        img = crop_black_borders(img)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = apply_clahe_green(img)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return None
    except Exception:
        return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--image-col', default='image')
    ap.add_argument('--label-col', default='level')
    ap.add_argument('--in-ext', default='.jpeg')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    names = df[args.image_col].astype(str).tolist()
    os.makedirs(args.out_dir, exist_ok=True)

    fn = partial(process_one, img_dir=args.img_dir, in_ext=args.in_ext,
                 out_dir=args.out_dir, size=args.size)
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(fn, names, chunksize=32),
                            total=len(names), desc='preprocess'))

    failed = [r for r in results if r]
    ok = len(names) - len(failed)
    print(f'OK: {ok}/{len(names)} -> {args.out_dir}')
    if failed:
        print(f'fallaron {len(failed)} (ej: {failed[:5]})')


if __name__ == '__main__':
    main()
