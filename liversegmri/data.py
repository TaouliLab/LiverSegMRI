"""Dataset preparation: case discovery, patient-level split, and nnU-Net v2 dataset export."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")
SUBGROUPS = ("cirrhotic_appearance", "irregular_liver_borders", "prior_hepatic_resection",
             "left_lobe_extension", "exophytic_lesion", "presence_of_ascites")
_SEQUENCE_GROUPS = {"pre": "T1WI pre-contrast", "eap": "T1WI early arterial", "ap": "T1WI arterial",
                    "aphe": "T1WI arterial", "pvp": "T1WI portal venous", "tp": "T1WI transitional/delayed",
                    "hbp": "T1WI hepatobiliary", "in": "T1WI in-phase", "out": "T1WI out-of-phase"}


def sequence_group(name: str) -> str:
    """Map a sequence name to the sequence types used in the analysis (e.g., 'diff_800' -> 'DWI')."""
    s = str(name).lower()
    if "adc" in s:
        return "ADC"
    if "diff" in s or "dwi" in s:
        return "DWI"
    if s.startswith("t2"):
        return "T2WI"
    return _SEQUENCE_GROUPS.get(s, str(name))


def case_id(source: str, patient_id: str, sequence: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{source}_{patient_id}-{sequence}")


def _size(path: str | Path) -> tuple[int, ...]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader.GetSize()


def discover_cases(data_root: str | Path, mask_suffix: str = "_mask") -> pd.DataFrame:
    """Find `<source>/<patient_id>/<sequence><mask_suffix>.*` masks and their images; flag invalid pairs."""
    rows = []
    for mask in sorted(Path(data_root).rglob(f"*{mask_suffix}.*")):
        if not mask.name.endswith(IMAGE_EXTENSIONS):
            continue
        sequence = mask.name.split(mask_suffix)[0]
        image = next((mask.parent / f"{sequence}{e}" for e in IMAGE_EXTENSIONS if (mask.parent / f"{sequence}{e}").exists()), None)
        status = "ok"
        if image is None:
            status = "missing image"
        elif _size(image) != _size(mask):
            status = "grid mismatch"
        rows.append(dict(source=mask.parent.parent.name, patient_id=mask.parent.name, sequence=sequence,
                         image=str(image) if image else None, mask=str(mask), status=status))
    return pd.DataFrame(rows)


def split_patients(cases: pd.DataFrame, subgroups: pd.DataFrame | None = None, test_fraction: float = 0.15,
                   seed: int = 42) -> pd.DataFrame:
    """Assign whole patients to 'train' or 'test', stratified by subgroup features and data source."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    patients = cases[["source", "patient_id"]].drop_duplicates().reset_index(drop=True)
    labels = pd.get_dummies(patients["source"], prefix="source").astype(int)
    if subgroups is not None:
        features = patients.merge(subgroups, on=["source", "patient_id"], how="left")
        features = features[[c for c in SUBGROUPS if c in features]].fillna(0).astype(int)
        labels = pd.concat([features, labels], axis=1)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    _, test_idx = next(splitter.split(patients, labels.values))
    test = set(map(tuple, patients.iloc[test_idx][["source", "patient_id"]].values))
    out = cases.copy()
    out["split"] = ["test" if (s, p) in test else "train" for s, p in zip(out.source, out.patient_id)]
    return out


def write_nnunet_dataset(cases: pd.DataFrame, nnunet_raw: str | Path,
                         dataset: str = "Dataset001_LiverSegmentation") -> Path:
    """Write images (`<case>_0000.nrrd`), binary labels, dataset.json, and cases.csv in nnU-Net v2 format."""
    root = Path(nnunet_raw) / dataset
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cases = cases.copy()
    cases["case_id"] = [case_id(*r) for r in cases[["source", "patient_id", "sequence"]].itertuples(index=False)]
    for r in cases.itertuples():
        tag = "Tr" if r.split == "train" else "Ts"
        sitk.WriteImage(sitk.ReadImage(r.image), str(root / f"images{tag}" / f"{r.case_id}_0000.nrrd"), useCompression=True)
        mask = sitk.ReadImage(r.mask)
        label = sitk.GetImageFromArray((sitk.GetArrayFromImage(mask) > 0).astype(np.uint8))
        label.CopyInformation(mask)
        sitk.WriteImage(label, str(root / f"labels{tag}" / f"{r.case_id}.nrrd"), useCompression=True)
    dataset_json = {"channel_names": {"0": "LiverSegmentation"}, "labels": {"background": 0, "liver": 1},
                    "numTraining": int((cases.split == "train").sum()), "file_ending": ".nrrd",
                    "overwrite_image_reader_writer": "SimpleITKIO"}
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    cases.drop(columns=["image", "mask"]).to_csv(root / "cases.csv", index=False)
    return root
