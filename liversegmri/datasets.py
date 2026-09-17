"""
Conversion of the three public datasets into the per-examination layout this package expects.

Every dataset is distributed in its own layout: CirrMRI600+ as NIfTI split across train/valid/test folders, the Duke
Liver Dataset as DICOM series, and LLD-MMRI already unpacked as NIfTI. The functions here normalize all of them to

    <output>/<case_id>/<sequence>.nrrd
    <output>/<case_id>/<sequence>_LiverSegManual.seg.nrrd     (reference mask, when the dataset ships one)

which is the layout read by `liversegmri.data.discover_cases`, by the evaluation code, and by the inference CLI. The
sequence names are the ones used throughout the paper (`pvp`, `t2`, `pre`, `in`, `out`, `diff_800`, ...), so figures
and tables can be reproduced from the converted folders without further renaming.

LLD-MMRI needs no conversion but does need an orientation check: in the public release most examinations are stored
with the slice order opposite to the direction recorded in the image header. `lld_mmri_slice_order()` returns the
per-case decision table used in the study, so the exact set of reversed examinations can be reproduced without
re-running the detector. See `liversegmri.orientation` for the reference-free detector itself.

Examples
--------
    liversegmri convert-cirrmri600 --root /data/CirrMRI600 --output /data/CirrMRI600/NRRD
    liversegmri convert-duke       --root /data/DukeData   --output /data/DukeData/NRRD
    liversegmri lld-slice-order    --output lld_mmri_slice_order.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MASK_SUFFIX = "_LiverSegManual.seg"

# CirrMRI600+ ships each contrast under its own tree, split into train/valid/test; the study pooled all three
# because the whole dataset is used as an external test set here.
CIRRMRI600_SOURCES = [
    ("Cirrhosis", "pvp", "Cirrhosis_T1_3D", ["train", "valid", "test"]),
    ("Cirrhosis", "t2", "Cirrhosis_T2_3D", ["train", "valid", "test"]),
    ("Healthy", "pvp", "Healthy_subjects/T1_W_Healthy", None),
    ("Healthy", "t2", "Healthy_subjects/T2_W_Healthy", None),
]


def _write(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def _binarize(mask: sitk.Image) -> sitk.Image:
    """Any positive label becomes 1, so masks from different datasets share one convention."""
    array = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(mask)
    return out


def _series(folder: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder)))
    return reader.Execute()


def convert_cirrmri600(root: Path, output: Path, overwrite: bool = False) -> int:
    """CirrMRI600+ NIfTI volumes to <output>/CirrMRI600-<subgroup>-<stem>/<sequence>.nrrd. Returns cases written."""
    root, output, written = Path(root), Path(output), 0
    for subgroup, sequence, subdir, splits in CIRRMRI600_SOURCES:
        folders = ([(root / subdir / f"{s}_images", root / subdir / f"{s}_masks") for s in splits] if splits else
                   [(root / subdir / f"{sequence[:2].upper().replace('PV', 'T1')}_images",
                     root / subdir / f"{sequence[:2].upper().replace('PV', 'T1')}_masks")])
        for image_dir, mask_dir in folders:
            if not image_dir.is_dir():
                print(f"  missing, skipped: {image_dir}")
                continue
            for image_path in sorted(image_dir.glob("*.nii.gz")):
                stem = image_path.name.split(".")[0]
                mask_path = mask_dir / f"{stem}.nii.gz"
                if not mask_path.exists():
                    print(f"  no mask for {image_path.name}, skipped")
                    continue
                case = output / f"CirrMRI600-{subgroup}-{stem}"
                target = case / f"{sequence}.nrrd"
                if target.exists() and not overwrite:
                    raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
                _write(sitk.ReadImage(str(image_path)), target)
                _write(_binarize(sitk.ReadImage(str(mask_path))), case / f"{sequence}{MASK_SUFFIX}.nrrd")
                written += 1
    return written


def convert_duke(root: Path, output: Path, overwrite: bool = False) -> int:
    """Duke Liver Dataset DICOM series to <output>/DLDS_<id>/<sequence>.nrrd. Returns series written.

    Reads SegmentationKey.csv (columns DLDS, Series, Label) and SequenceTypes.csv (columns Label, SeriesMap); the
    SeriesMap value is the sequence name used in the paper.
    """
    root, output, written = Path(root), Path(output), 0
    with open(root / "SequenceTypes.csv", newline="", encoding="utf-8-sig") as fh:
        sequence_of = {r["Label"]: r["SeriesMap"] for r in csv.DictReader(fh)}
    with open(root / "SegmentationKey.csv", newline="", encoding="utf-8-sig") as fh:
        key = list(csv.DictReader(fh))

    for entry in key:
        sequence = sequence_of.get(entry["Label"])
        if not sequence:
            print(f"  no SeriesMap for label {entry['Label']}, skipped")
            continue
        study = str(entry["DLDS"]).zfill(4)
        base = root / "Segmentation" / study / entry["Series"]
        case = output / f"DLDS_{study}"
        target = case / f"{sequence}.nrrd"
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        _write(_series(base / "images"), target)

        mask = _series(base / "masks")
        levels = np.unique(sitk.GetArrayFromImage(mask))
        if len(levels) != 2:                 # a liver mask must be background plus one label
            raise ValueError(f"{study}/{sequence}: mask has {len(levels)} levels {levels}, expected 2")
        _write(_binarize(mask), case / f"{sequence}{MASK_SUFFIX}.nrrd")
        written += 1
    return written


def lld_mmri_slice_order() -> list[dict]:
    """The per-case slice-order decisions used in the study: case_id, slice_order_reversed, sequences_voting.

    `slice_order_reversed` is True where the examination is stored with the slice order opposite to the direction in
    its header and was reversed before inference; `sequences_voting` is how many sequences of that examination
    carried a usable orientation cue (the decision is a majority vote over them).
    """
    path = Path(__file__).resolve().parent / "resources" / "lld_mmri_slice_order.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return [{"case_id": r["case_id"],
                 "slice_order_reversed": r["slice_order_reversed"] == "True",
                 "sequences_voting": int(r["sequences_voting"])} for r in csv.DictReader(fh)]
