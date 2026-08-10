# dat-retinopathy

**Deformable Attention Transformer para clasificación y segmentación explicable de retinopatía diabética.**

Tesis de grado — Mateo Giacometti, Ingeniería en IA, Universidad de San Andrés.

## Objetivo

Un modelo DAT (Deformable Attention Transformer, Xia et al. CVPR 2022) que en una sola pasada:
1. Clasifica severidad de RD en 5 grados (0-4).
2. Segmenta lesiones (microaneurismas, hemorragias, exudados) con pseudo-máscaras débilmente supervisadas.
3. Genera mapas de calor Deform-CAM desde los offsets deformables del propio modelo (explicabilidad nativa).

## Estado actual

- [x] Fase 1 (parcial): curaduría de EyePACS, APTOS, Messidor-2 (preprocesados a 512px con crop + CLAHE canal verde).
- [x] Clasificador base: DAT-Tiny vendorizado (commit `797f187` de LeapLabTHU/DAT) con pesos ImageNet-1K, Focal Loss, AMP bf16.
- [ ] Entrenamiento completo @512 (en curso).
- [ ] Validación externa APTOS / Messidor-2.
- [ ] Deform-CAM, pseudo-máscaras + CRF, multi-tarea FPN, ONNX, Streamlit.

## Setup

```bash
mamba create -n dat-ret python=3.11 -y
mamba activate dat-ret
pip install -r requirements.txt
```

## Datos

```
data/
├── eyepacs/            # 35.126 imágenes originales + trainLabels.csv
├── aptos/              # 3.662 train + 1.928 test
├── messidor/           # 1.748 imágenes + labels.csv (google-brain/messidor2-dr-grades)
├── ddr/                # 12.524 imágenes grading (máscaras: fase 3)
├── idrid/              # 3 zips extraídos (A: segmentación, B: grading, C: localización)
└── processed/
    ├── eyepacs_512/    # cache preprocesado (crop + CLAHE + 512px)
    ├── aptos_512/
    └── messidor_512/
```

Splits por paciente (sin leakage izq/der): `data/splits/eyepacs_{train,val}.csv` (90/10 estratificado).

## Uso

```bash
# Preprocesar un dataset (idempotente)
python scripts/preprocess.py --csv <labels.csv> --img-dir <imgs> --in-ext .jpeg \
    --out-dir data/processed/<nombre>_512 --size 512 --workers 12

# Generar splits por paciente
python scripts/make_splits.py

# Entrenar (smoke test o run serio)
python train.py --config configs/dat_tiny_smoke.yaml
python train.py --config configs/dat_tiny_512.yaml

# Monitorear
tensorboard --logdir runs/

# Evaluar en validación externa (p.ej. APTOS o Messidor)
python scripts/evaluate.py --checkpoint checkpoints/dat_tiny_512/best.pt \
    --csv data/aptos/train.csv --img-dir data/processed/aptos_512 \
    --image-col id_code --label-col diagnosis
```

## Métricas

QWK (Quadratic Weighted Kappa), AUC macro OVR y por clase, AUC "RD referible" (grado ≥2 vs <2), accuracy, sensibilidad/especificidad por clase, matriz de confusión.

## Notas técnicas

- **DAT no está en timm**: el backbone vive en `src/models/vendor/dat/` (código oficial v1, sin dependencia `natten`, solo `torch` + `einops` + `timm`). La atención deformable usa `F.grid_sample` (portable a torch 2.x).
- **Resolución**: la atención local exige feature maps divisibles por `window_size`. A 512px se usa `window_size=8`; a 384px, `window_size=12` (como el DAT-Base@384 oficial). Las tablas de positional bias se interpolan al cargar los pesos de 224px (`load_pretrained`).
- **VRAM (RTX 4060 8 GB)**: DAT-Tiny @512 batch 4 + grad_accum 8 (efectivo 32) con AMP bf16 → ~4.5 GB.
- Los offsets deformables (`forward(x, return_offsets=True)` → `positions`, `references` por etapa) son la materia prima de Deform-CAM (fase 2).
