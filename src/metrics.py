"""Metricas de evaluacion: QWK, AUC OVR por clase y macro, AUC referible, sens/spec."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 5) -> dict:
    """y_true: [N] ints, y_prob: [N, C] probabilidades."""
    y_pred = y_prob.argmax(axis=1)
    out = {}

    out['qwk'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    out['accuracy'] = float((y_pred == y_true).mean())

    # AUC one-vs-rest por clase y macro (robusto a clases ausentes en el subset)
    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucs.append(float('nan'))
            continue
        aucs.append(float(roc_auc_score(y_bin, y_prob[:, c])))
    out['auc_per_class'] = aucs
    valid = [a for a in aucs if not np.isnan(a)]
    out['auc_macro'] = float(np.mean(valid)) if valid else float('nan')

    # AUC "RD referible" (grados 2-3-4 vs 0-1): probabilidad de grado >= 2
    y_ref = (y_true >= 2).astype(int)
    if 0 < y_ref.sum() < len(y_ref):
        out['auc_referable'] = float(roc_auc_score(y_ref, y_prob[:, 2:].sum(axis=1)))
    else:
        out['auc_referable'] = float('nan')

    # Sensibilidad / especificidad por clase desde la matriz de confusion
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, spec = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan'))
        spec.append(float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan'))
    out['sensitivity_per_class'] = sens
    out['specificity_per_class'] = spec
    out['confusion_matrix'] = cm.tolist()

    return out
