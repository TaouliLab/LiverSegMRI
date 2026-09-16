"""Segmentation metrics: Dice, HD95, ASSD, and absolute volume error, computed on the reference grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

METRICS = ("dice", "hd95_mm", "assd_mm", "volume_error_pct")


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    return mask & ~(distance_transform_edt(mask) > 1)


def _surface_distances(reference: np.ndarray, prediction: np.ndarray, spacing: tuple[float, ...]):
    ref_b, pred_b = _boundary(reference), _boundary(prediction)
    if not ref_b.any() or not pred_b.any():
        return np.array([0.0]), np.array([0.0])
    to_ref = distance_transform_edt(~ref_b, sampling=spacing)
    to_pred = distance_transform_edt(~pred_b, sampling=spacing)
    return to_pred[ref_b], to_ref[pred_b]


def _diagonal(shape, spacing) -> float:
    return float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, shape))))


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = reference.sum() + prediction.sum()
    return 1.0 if denominator == 0 else float(2.0 * (reference & prediction).sum() / denominator)


def hd95(reference: np.ndarray, prediction: np.ndarray, spacing: tuple[float, ...]) -> float:
    if not reference.any() and not prediction.any():
        return 0.0
    if not reference.any() or not prediction.any():
        return _diagonal(reference.shape, spacing)
    d_ref, d_pred = _surface_distances(reference, prediction, spacing)
    return float(np.percentile(np.concatenate([d_ref, d_pred]), 95))


def assd(reference: np.ndarray, prediction: np.ndarray, spacing: tuple[float, ...]) -> float:
    if not reference.any() and not prediction.any():
        return 0.0
    if not reference.any() or not prediction.any():
        return _diagonal(reference.shape, spacing)
    d_ref, d_pred = _surface_distances(reference, prediction, spacing)
    return float((d_ref.mean() + d_pred.mean()) / 2.0)


def volume_error(reference: np.ndarray, prediction: np.ndarray, spacing: tuple[float, ...]) -> float:
    voxel = float(np.prod(spacing))
    v_ref, v_pred = reference.sum() * voxel, prediction.sum() * voxel
    if v_ref == 0:
        return 100.0 if v_pred > 0 else 0.0
    return float(abs(v_pred - v_ref) / v_ref * 100.0)


def evaluate_pair(reference_path: str | Path, prediction_path: str | Path) -> dict[str, float]:
    """All metrics for one prediction; the prediction is resampled (nearest neighbor) to the reference grid if needed."""
    ref_img = sitk.ReadImage(str(reference_path), sitk.sitkUInt8)
    pred_img = sitk.ReadImage(str(prediction_path), sitk.sitkUInt8)
    if pred_img.GetSize() != ref_img.GetSize():
        pred_img = sitk.Resample(pred_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    reference = sitk.GetArrayFromImage(ref_img).astype(bool)
    prediction = sitk.GetArrayFromImage(pred_img).astype(bool)
    spacing = tuple(reversed(ref_img.GetSpacing()))
    return {"dice": dice(reference, prediction), "hd95_mm": hd95(reference, prediction, spacing),
            "assd_mm": assd(reference, prediction, spacing), "volume_error_pct": volume_error(reference, prediction, spacing)}
