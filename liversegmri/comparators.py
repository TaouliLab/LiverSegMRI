"""Liver masks from the comparator models, run off the shelf as in the study.

TotalSegmentator MRI  https://github.com/wasserth/TotalSegmentator  (TotalSegmentator 2.12.0, task "total_mr")
MRAnnotator          https://github.com/lzl199704/MRAnnotator      (abdomen model "Dataset001_Abdomen", liver = label 5)

Please cite the original publications and follow the licenses of these projects.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _save_binary(mask: np.ndarray, reference: sitk.Image, path: str | Path) -> Path:
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path), useCompression=True)
    return Path(path)


def totalsegmentator_liver(image_path: str | Path, output_path: str | Path, device: str = "gpu") -> Path:
    """Liver mask from TotalSegmentator MRI (task "total_mr"; no postprocessing).

    TotalSegmentator configures nnU-Net paths when it runs; call it in a separate process from the other models.
    """
    from totalsegmentator.python_api import totalsegmentator

    with tempfile.TemporaryDirectory() as tmp:
        nifti = Path(tmp) / "image.nii.gz"
        sitk.WriteImage(sitk.ReadImage(str(image_path)), str(nifti))
        totalsegmentator(input=str(nifti), output=str(Path(tmp) / "segmentations"), task="total_mr",
                         output_type="nifti", device=device, quiet=True)
        liver = sitk.ReadImage(str(Path(tmp) / "segmentations" / "liver.nii.gz"))
        return _save_binary(sitk.GetArrayFromImage(liver) > 0, liver, output_path)


class MRAnnotatorLiver:
    """Liver masks from the MRAnnotator abdomen model (fold 0, final checkpoint; no postprocessing).

    Args:
        model_dir: `.../Dataset001_Abdomen/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres`, obtained from the
            MRAnnotator repository.
        device: "cuda" or "cpu".
    """

    LIVER_LABEL = 5

    def __init__(self, model_dir: str | Path, device: str = "cuda", fold: int = 0,
                 checkpoint: str = "checkpoint_final.pth"):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                         perform_everything_on_device=device.startswith("cuda"),
                                         device=torch.device(device), verbose=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint)

    @staticmethod
    def _flip_first_axis(path_in: Path, path_out: Path, is_mask: bool = False) -> None:
        """Flip the voxel array along its first axis while keeping the header (MRAnnotator orientation convention)."""
        import nibabel as nib

        img = nib.load(str(path_in))
        data = np.flip(img.get_fdata(), axis=0)
        if is_mask:
            data = data.astype(np.uint8)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(path_out))

    def predict(self, image_path: str | Path, output_path: str | Path) -> Path:
        with tempfile.TemporaryDirectory() as tmp:
            nifti = Path(tmp) / "image.nii.gz"
            sitk.WriteImage(sitk.ReadImage(str(image_path)), str(nifti))
            self._flip_first_axis(nifti, nifti)
            target = Path(tmp) / "prediction"
            self.predictor.predict_from_files([[str(nifti)]], [str(target)], save_probabilities=False, overwrite=True,
                                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            prediction = Path(str(target) + self.predictor.dataset_json["file_ending"])
            self._flip_first_axis(prediction, prediction, is_mask=True)
            labels = sitk.ReadImage(str(prediction), sitk.sitkUInt8)
            return _save_binary(sitk.GetArrayFromImage(labels) == self.LIVER_LABEL, labels, output_path)
