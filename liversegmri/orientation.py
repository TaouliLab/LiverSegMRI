"""Detect and correct volumes whose slice order disagrees with their header.

Some public MRI collections are distributed with an identity direction matrix while the slices actually run
superior->inferior, i.e. opposite to what the header states. Segmentation models that rely on anatomical
orientation then see an inverted abdomen and fail, often silently. The order is not necessarily wrong for every
case in a collection, so it is checked per examination.

The check uses no reference masks: a fast multi-organ pass gives the mean slice index of thoracic structures minus
that of pelvic structures, which is positive when the slice order matches the header and negative when it is
reversed. Sequences of the same examination vote, because slice order is a property of the examination.

Requires the optional comparator dependency (`pip install "liversegmri[comparators]"`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SUPERIOR = ("lung", "heart", "trachea", "aorta_arch", "atrial", "pulmonary")
INFERIOR = ("urinary_bladder", "kidney", "hip", "femur", "iliac", "prostate", "sacrum")


def slice_order_cue(image_path: str | Path, device: str = "gpu") -> float:
    """Thoracic minus pelvic mean slice index, normalized to [0, 1]. NaN when neither group is found."""
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    names = class_map["total_mr"]
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "image.nii.gz", Path(tmp) / "seg.nii.gz"
        sitk.WriteImage(sitk.ReadImage(str(image_path)), str(source))
        totalsegmentator(input=str(source), output=str(target), task="total_mr", ml=True, fast=True,
                         output_type="nifti", device=device, quiet=True)
        labels = sitk.GetArrayFromImage(sitk.ReadImage(str(target), sitk.sitkUInt16))

    def centroid(keys):
        ids = [i for i, name in names.items() if any(k in name for k in keys)]
        mask = np.isin(labels, ids)
        return float(np.nonzero(mask)[0].mean() / labels.shape[0]) if mask.any() else np.nan

    return centroid(SUPERIOR) - centroid(INFERIOR)


def detect(images, sample: int = 3, device: str = "gpu") -> dict:
    """Check up to `sample` volumes of one examination and decide by majority.

    Returns {"cues": {file name: cue}, "reversed": bool}.
    """
    images = [Path(p) for p in images][:sample] if sample else [Path(p) for p in images]
    cues = {p.name: slice_order_cue(p, device=device) for p in images}
    votes = [c for c in cues.values() if not np.isnan(c)]
    return {"cues": cues, "reversed": bool(votes) and float(np.mean([v < 0 for v in votes])) > 0.5}


def reverse_slices(image_path: str | Path, output_path: str | Path) -> Path:
    """Write a copy of the volume with its slice order reversed, keeping the header."""
    image = sitk.ReadImage(str(image_path))
    flipped = sitk.GetImageFromArray(sitk.GetArrayFromImage(image)[::-1])
    flipped.CopyInformation(image)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(flipped, str(output_path), useCompression=True)
    return Path(output_path)


def correct(images, output_dir: str | Path, sample: int = 3, device: str = "gpu") -> dict:
    """Detect the slice order of an examination and, if reversed, write corrected copies of all its volumes."""
    images = [Path(p) for p in images]
    result = detect(images, sample=sample, device=device)
    if result["reversed"]:
        result["written"] = [str(reverse_slices(p, Path(output_dir) / p.name)) for p in images]
    return result
