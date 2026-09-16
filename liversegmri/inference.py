"""LiverSegMRI inference: five-fold nnU-Net v2 ensemble with largest-connected-component filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

HF_REPO_ID = "TaouliLab/LiverSegMRI"
IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")
FOLDS = (0, 1, 2, 3, 4)


def strip_extension(name: str) -> str:
    """File name without its image extension (handles double extensions such as .nii.gz)."""
    for ext in IMAGE_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def list_images(path: str | Path) -> list[Path]:
    """A single image file, or all image files directly inside a folder."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.name.endswith(IMAGE_EXTENSIONS))


def download_weights(repo_id: str = HF_REPO_ID, revision: str | None = None, cache_dir: str | None = None) -> Path:
    """Download the model folder (dataset.json, plans.json, fold_*/checkpoint_final.pth) from Hugging Face."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir,
                                  allow_patterns=["dataset.json", "plans.json", "fold_*/checkpoint_final.pth"]))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 3D connected component of a binary mask."""
    labeled, n = ndimage.label(mask > 0)
    if n <= 1:
        return (mask > 0).astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


class LiverSegMRI:
    """LiverSegMRI predictor.

    Args:
        weights_dir: local model folder; downloaded from Hugging Face when omitted.
        device: "cuda" or "cpu".
        folds: ensemble folds (the study used all five).
        checkpoint: checkpoint file name within each fold folder.
        step_size: sliding-window step as a fraction of the patch size.
    """

    def __init__(self, weights_dir: str | Path | None = None, device: str = "cuda", folds: Sequence[int] = FOLDS,
                 checkpoint: str = "checkpoint_final.pth", step_size: float = 0.5):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        weights_dir = Path(weights_dir) if weights_dir else download_weights()
        self.predictor = nnUNetPredictor(tile_step_size=step_size, use_gaussian=True, use_mirroring=False,
                                         perform_everything_on_device=device.startswith("cuda"),
                                         device=torch.device(device), verbose=False, allow_tqdm=True)
        self.predictor.initialize_from_trained_model_folder(str(weights_dir), use_folds=tuple(folds),
                                                            checkpoint_name=checkpoint)

    def predict(self, images: Iterable[str | Path], output_dir: str | Path, suffix: str = "_liver",
                extension: str = ".nii.gz", postprocess: bool = True, num_workers: int = 2) -> list[Path]:
        """Segment each image and write `<image name><suffix><extension>` to `output_dir`."""
        images = [Path(p) for p in images]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = [str(output_dir / f"{strip_extension(p.name)}{suffix}") for p in images]

        self.predictor.predict_from_files([[str(p)] for p in images], targets, save_probabilities=False, overwrite=True,
                                          num_processes_preprocessing=num_workers,
                                          num_processes_segmentation_export=num_workers,
                                          folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

        written = []
        nnunet_ending = self.predictor.dataset_json["file_ending"]
        for target in targets:
            raw = Path(target + nnunet_ending)
            image = sitk.ReadImage(str(raw))
            mask = sitk.GetArrayFromImage(image)
            mask = keep_largest_component(mask) if postprocess else (mask > 0).astype(np.uint8)
            out = sitk.GetImageFromArray(mask)
            out.CopyInformation(image)
            final = Path(target + extension)
            sitk.WriteImage(out, str(final), useCompression=True)
            if final != raw:
                raw.unlink()
            written.append(final)
        return written
