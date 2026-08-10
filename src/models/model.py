"""Factory de modelos: DAT (vendorizado) y baselines timm (EfficientNet)."""
import torch
import torch.nn as nn

from .vendor.dat import DAT

# Config exacta del DAT-Tiny v1 (configs/dat_tiny.yaml @ commit 797f187)
DAT_VARIANTS = {
    'dat_tiny': dict(
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        heads=[3, 6, 12, 24],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        strides=[-1, -1, 1, 1],
        offset_range_factor=[-1, -1, 2, 2],
        drop_path_rate=0.2,
    ),
}


class DATClassifier(nn.Module):
    """Wrapper del DAT v1 para clasificacion.

    - num_classes configurable (reemplaza la head de 1000 clases).
    - window_sizes se adapta a la resolucion de entrada (los feature maps deben
      ser divisibles por la ventana: img_size/4, /8, /16, /32).
    - forward(x, return_offsets=True) expone (logits, positions, references),
      la materia prima para Deform-CAM.
    """

    def __init__(self, variant='dat_tiny', img_size=512, window_size=8,
                 num_classes=5, pretrained_path=None):
        super().__init__()
        assert variant in DAT_VARIANTS, f'Variante desconocida: {variant}'
        cfg = DAT_VARIANTS[variant]
        self.backbone = DAT(
            img_size=img_size,
            patch_size=4,
            num_classes=num_classes,
            window_sizes=[window_size] * 4,
            **cfg,
        )
        if pretrained_path:
            self._load_imagenet_weights(pretrained_path)

    def _load_imagenet_weights(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        # checkpoints estilo Swin: {'model': state_dict, ...}
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        # load_pretrained interpola RPE/bias tables y saltea la head de 1000 clases
        self.backbone.load_pretrained(state_dict)

    def forward(self, x, return_offsets=False):
        logits, positions, references = self.backbone(x)
        if return_offsets:
            return logits, positions, references
        return logits


def build_model(model_cfg: dict) -> nn.Module:
    """Construye el modelo desde la seccion `model` del config YAML."""
    name = model_cfg['name']
    num_classes = model_cfg.get('num_classes', 5)

    if name.startswith('dat'):
        return DATClassifier(
            variant=name,
            img_size=model_cfg.get('img_size', 512),
            window_size=model_cfg.get('window_size', 8),
            num_classes=num_classes,
            pretrained_path=model_cfg.get('pretrained'),
        )

    # Baselines timm (p.ej. efficientnet_b4 para la tabla comparativa de la tesis)
    import timm
    return timm.create_model(name, pretrained=model_cfg.get('pretrained', True),
                             num_classes=num_classes)
