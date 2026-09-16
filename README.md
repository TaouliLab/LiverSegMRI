# LiverSegMRI

**Sequence-agnostic whole-liver segmentation on abdominal MRI**

[![Paper](https://img.shields.io/badge/Paper-Radiology%3A%20Artificial%20Intelligence-1f4e79)](https://doi.org/10.1148/ryai.XXXXXXX)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LiverSegMRI-ffcc4d)](https://huggingface.co/TaouliLab/LiverSegMRI)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

LiverSegMRI is a deep learning model for automated whole-liver segmentation on routine abdominal MRI. It is a five-fold nnU-Net v2 ensemble trained on 10,494 sequences from 1,058 patients with and without chronic liver disease. Each sequence is used as an independent single-channel input, so the same model segments T1-weighted in- and opposed-phase, pre- and post-contrast dynamic, hepatobiliary, T2-weighted, diffusion-weighted, and ADC images without sequence labels.

This repository contains the inference code, the code used to run the comparator models, the evaluation metrics, and the statistical analysis of the study:

> Yuce M, Tordjman M, Meribout A, Ozkaya E, Lee JO, Lee JM, Akinci D'Antonoli T, Wasserthal J, Mei X, Taouli B.
> **LiverSegMRI: Development and External Validation of a Deep Learning Model for Automated Liver Segmentation across Multiparametric MRI Sequences and Comparison with Generalist Segmentation Models.**
> *Radiology: Artificial Intelligence* 2026. doi: [10.1148/ryai.XXXXXXX](https://doi.org/10.1148/ryai.XXXXXXX)

| Test set | Patients | LiverSegMRI | TotalSegmentator MRI | MRAnnotator |
|---|---:|---:|---:|---:|
| External (Duke Liver Dataset, CirrMRI600+) | 487 | **0.944** | 0.886 | 0.889 |
| Internal (held out) | 188 | **0.984** | 0.728 | 0.760 |

*Patient-wise mean Dice similarity coefficient. See the paper for boundary metrics, volume error, and sequence-wise and subgroup analyses.*

---

## Contents

```
LiverSegMRI/
├── liversegmri/
│   ├── inference.py      # LiverSegMRI prediction (weights downloaded from Hugging Face)
│   ├── comparators.py    # TotalSegmentator MRI and MRAnnotator liver masks, as run in the study
│   ├── orientation.py    # slice-order check for datasets whose headers disagree with the image
│   ├── data.py           # nnU-Net dataset preparation and patient-level split
│   ├── metrics.py        # Dice, HD95, ASSD, volume error
│   ├── evaluation.py     # metrics for a manifest of cases
│   ├── statistics.py     # patient-level paired analysis, bootstrap CIs, Holm correction
│   └── cli.py            # `liversegmri` command-line interface
├── examples/
│   └── manifest_template.csv
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.10 or later. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA version first, then:

```bash
git clone https://github.com/TaouliLab/LiverSegMRI.git
cd LiverSegMRI
pip install -e .                  # LiverSegMRI inference, evaluation, statistics
pip install -e ".[comparators]"   # optional: TotalSegmentator, for comparator runs
```

## Quick start

Segment a single image or every image in a folder:

```bash
liversegmri predict -i /path/to/images -o /path/to/output
```

On first use, the model weights are downloaded from [Hugging Face](https://huggingface.co/TaouliLab/LiverSegMRI). Use `--weights-dir` to point to a local copy.

- **Input:** 3D axial MRI volumes (`.nii.gz`, `.nii`, `.nrrd`, `.mha`), one sequence per file.
- **Output:** a binary liver mask per image, named `<image>_liver.nii.gz`.

Options:

| Option | Description | Default |
|---|---|---|
| `--device` | `cuda` or `cpu` | `cuda` |
| `--folds` | Ensemble folds | `0,1,2,3,4` |
| `--extension` | Output format | `.nii.gz` |
| `--no-postprocessing` | Keep all connected components | off (largest component is kept) |

Python:

```python
from liversegmri.inference import LiverSegMRI

model = LiverSegMRI(device="cuda")            # or LiverSegMRI(weights_dir="/path/to/weights")
masks = model.predict(["/path/to/pvp.nii.gz", "/path/to/t2.nii.gz"], output_dir="/path/to/output")
```

## Segmentation convention

The liver label includes the liver parenchyma and all intrahepatic lesions, including exophytic lesions arising from the liver. The gallbladder and major vessels (inferior vena cava, extrahepatic portal vein) are excluded. This matches the liver labels of TotalSegmentator MRI and MRAnnotator.

## Reproducing the study

### 1. Data layout

```
/path/to/data/
└── <source>/                  # data source or site
    └── <patient_id>/
        ├── pvp.nii.gz         # image: one file per sequence
        ├── pvp_mask.nii.gz    # manual liver mask (same grid)
        ├── t2.nrrd
        └── t2_mask.nrrd
```

Optional subgroup table for stratified splitting (`subgroups.csv`):

```
source,patient_id,cirrhotic_appearance,irregular_liver_borders,prior_hepatic_resection,left_lobe_extension,exophytic_lesion,presence_of_ascites
```

### 2. Build the nnU-Net dataset

```bash
liversegmri prepare-nnunet --data-root /path/to/data --subgroups subgroups.csv \
    --nnunet-raw /path/to/nnUNet_raw --mask-suffix _mask
```

This command:
- Validates every image and mask pair, and skips pairs whose grids differ.
- Splits patients into training (85%) and internal test (15%) sets. The split is stratified by data source and subgroup features, with seed 42.
- Writes `Dataset001_LiverSegmentation` in nnU-Net format, plus a `cases.csv` with each case's split.

### 3. Train

The study used the nnU-Net v2 `3d_fullres` configuration with 1 mm isotropic target spacing. That spacing is set in the published `plans.json`.

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

nnUNetv2_extract_fingerprint -d 1 --verify_dataset_integrity
huggingface-cli download TaouliLab/LiverSegMRI plans.json --local-dir "$nnUNet_preprocessed/Dataset001_LiverSegmentation"
mv "$nnUNet_preprocessed/Dataset001_LiverSegmentation/plans.json" "$nnUNet_preprocessed/Dataset001_LiverSegmentation/nnUNetPlans.json"
nnUNetv2_preprocess -d 1 -c 3d_fullres

for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 1 3d_fullres $FOLD -tr nnUNetTrainerNoMirroring
done
```

In the study, each fold was trained for 1,000 epochs on one NVIDIA A100 (40 GB), taking about 45 hours per fold. Inference uses the ensemble of the five final checkpoints without test-time mirroring, followed by largest-connected-component filtering.

### 4. Comparator models

Both comparators are run off the shelf, as in the study. Please cite and follow the licenses of the original works.

| Model | Source | Settings used in the study |
|---|---|---|
| TotalSegmentator MRI | [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | TotalSegmentator 2.12.0, task `total_mr`, `liver` output |
| MRAnnotator | [github.com/lzl199704/MRAnnotator](https://github.com/lzl199704/MRAnnotator) | Abdomen model `Dataset001_Abdomen`, fold 0, final checkpoint, liver label 5 |

```bash
liversegmri comparator totalsegmentator -i /path/to/images -o /path/to/predictions/totalsegmentator

# download the MRAnnotator abdomen weights following the MRAnnotator instructions, then:
liversegmri comparator mrannotator -i /path/to/images -o /path/to/predictions/mrannotator \
    --model-dir /path/to/MRAnnotator/Dataset001_Abdomen/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres
```

### 5. Slice order of public datasets

Some public collections are distributed with an identity direction matrix while the slices actually run
superior→inferior, opposite to the header. Generalist models that rely on anatomical orientation then segment an
inverted abdomen and fail, often without any obvious error; LiverSegMRI is insensitive to it. The order is not
wrong for every examination in a collection, so check per examination before running comparators:

```bash
liversegmri check-orientation -i /path/to/data/site_a/P0001                      # report only
liversegmri check-orientation -i /path/to/data/site_a/P0001 -o /path/to/corrected # write corrected copies
```

The check uses no reference masks: it runs a fast multi-organ pass and compares the mean slice index of thoracic
structures with that of pelvic structures, which is positive when the slice order matches the header and negative
when it is reversed. A few volumes of the examination vote. If a volume is corrected, reverse the model's output
back to the original grid before scoring, so that metrics stay on the reference mask's geometry.

### 6. Evaluation

List reference masks and predictions in a manifest; see [`examples/manifest_template.csv`](examples/manifest_template.csv).
- Each `pred_<model>` column adds one model.
- Additional columns, such as `test_set` or subgroup flags, are carried through to the output.

```bash
liversegmri evaluate --manifest manifest.csv --output metrics.csv --workers 8
```

| Metric | Definition |
|---|---|
| Dice | Dice similarity coefficient |
| HD95 | 95th percentile Hausdorff distance, mm |
| ASSD | Average symmetric surface distance, mm |
| Volume error | Absolute volume error, % of the reference volume |

All metrics are computed on the reference mask grid.

### 7. Statistical analysis

```bash
liversegmri analyze --metrics metrics.csv --output-dir results --reference LiverSegMRI --by test_set \
    --flags cirrhotic_appearance,irregular_liver_borders,prior_hepatic_resection,left_lobe_extension,exophytic_lesion,presence_of_ascites
```

Patients contribute multiple sequences, so metrics are first averaged per patient. The analysis then produces:

| Output | Content |
|---|---|
| `summary_patientwise.csv` | Mean ± SD per model and metric with bootstrap 95% CIs (5,000 resamples of patients) |
| `paired_comparisons.csv` | Friedman omnibus test; Wilcoxon signed-rank tests of the reference model against each comparator; bootstrap CIs of the mean paired difference; Holm correction |
| `sequence_wise_comparisons.csv` | The same comparisons within each sequence type |
| `sequence_class_comparisons.csv` | Conventional T1-weighted versus functional sequences (DWI, ADC): each model's within-patient drop, and the difference between a comparator's drop and the reference model's drop |
| `failure_rates.csv` | Sequences and patients below Dice thresholds and above a volume-error threshold |
| `subgroup_comparisons.csv` | Models within each morphologic subgroup, and each subgroup against patients with none of the flags (requires `--flags`) |

## Data availability

The public datasets used in the study are available from their providers:
- [LLD-MMRI](https://github.com/LMMMEng/LLD-MMRI-Dataset)
- [Duke Liver Dataset](https://zenodo.org/records/7774566)
- [CirrMRI600+](https://osf.io/cuk24/)

Institutional data are not publicly available.

## Citation

```bibtex
@article{yuce2026liversegmri,
  title   = {LiverSegMRI: Development and External Validation of a Deep Learning Model for Automated Liver Segmentation across Multiparametric MRI Sequences and Comparison with Generalist Segmentation Models},
  author  = {Yuce, Murat and Tordjman, Mickael and Meribout, Anis and Ozkaya, Efe and Lee, Jung-Oh and Lee, Jeong Min and Akinci D'Antonoli, Tugba and Wasserthal, Jakob and Mei, Xueyan and Taouli, Bachir},
  journal = {Radiology: Artificial Intelligence},
  year    = {2026},
  doi     = {10.1148/ryai.XXXXXXX}
}
```

Please also cite [nnU-Net](https://doi.org/10.1038/s41592-020-01008-z), and [TotalSegmentator MRI](https://doi.org/10.1148/radiol.241613) and [MRAnnotator](https://doi.org/10.1093/radadv/umae035) when using the comparator code.

## License

Code and model weights are released under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). Comparator models are subject to their own licenses.

LiverSegMRI is intended for research use only. It is not a medical device and must not be used for clinical decision-making.
