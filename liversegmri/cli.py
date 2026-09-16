"""Command-line interface: `liversegmri {predict, comparator, prepare-nnunet, evaluate, analyze}`."""

from __future__ import annotations

import argparse
from pathlib import Path


def _predict(args):
    from .inference import LiverSegMRI, list_images

    model = LiverSegMRI(weights_dir=args.weights_dir, device=args.device, folds=[int(f) for f in args.folds.split(",")])
    written = model.predict(list_images(args.input), args.output, extension=args.extension,
                            postprocess=not args.no_postprocessing)
    print(f"Wrote {len(written)} liver masks to {args.output}")


def _comparator(args):
    from . import comparators
    from .inference import list_images, strip_extension

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images = list_images(args.input)
    if args.model == "totalsegmentator":
        device = "gpu" if args.device.startswith("cuda") else "cpu"
        run = lambda image, out: comparators.totalsegmentator_liver(image, out, device=device)  # noqa: E731
    else:
        if not args.model_dir:
            raise SystemExit("--model-dir is required for MRAnnotator")
        run = comparators.MRAnnotatorLiver(args.model_dir, device=args.device).predict
    for image in images:
        run(image, output / f"{strip_extension(image.name)}_liver.nii.gz")
    print(f"Wrote {len(images)} {args.model} liver masks to {output}")


def _orientation(args):
    from .inference import list_images
    from .orientation import correct, detect

    images = list_images(args.input)
    device = "gpu" if args.device.startswith("cuda") else "cpu"
    result = correct(images, args.output, sample=args.sample, device=device) if args.output else \
        detect(images, sample=args.sample, device=device)
    for name, cue in result["cues"].items():
        print(f"{name}: cue {cue:+.2f}" if cue == cue else f"{name}: cue not determined")
    print("slice order is REVERSED relative to the header" if result["reversed"] else "slice order matches the header")
    if result.get("written"):
        print(f"Wrote {len(result['written'])} corrected volumes to {args.output}")


def _prepare(args):
    import pandas as pd

    from . import data

    cases = data.discover_cases(args.data_root, mask_suffix=args.mask_suffix)
    invalid = cases[cases.status != "ok"]
    if len(invalid):
        print(f"Skipping {len(invalid)} cases:\n{invalid.groupby('status').size().to_string()}")
    cases = cases[cases.status == "ok"].drop(columns="status")
    subgroups = pd.read_csv(args.subgroups) if args.subgroups else None
    cases = data.split_patients(cases, subgroups, test_fraction=args.test_fraction, seed=args.seed)
    root = data.write_nnunet_dataset(cases, args.nnunet_raw)
    summary = cases.groupby("split").agg(patients=("patient_id", "nunique"), sequences=("sequence", "size"))
    print(f"nnU-Net dataset written to {root}\n{summary.to_string()}")


def _evaluate(args):
    from .evaluation import evaluate_manifest

    results = evaluate_manifest(args.manifest, args.output, workers=args.workers)
    print(f"Wrote metrics for {len(results)} predictions to {args.output}")


def _analyze(args):
    import pandas as pd

    from . import statistics as st

    metrics = pd.read_csv(args.metrics)
    if "prediction_available" in metrics:
        metrics = metrics[metrics["prediction_available"].astype(bool)]
    by = [args.by] if args.by else []
    flags = [f.strip() for f in args.flags.split(",")] if args.flags else []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    patients = st.patient_level(metrics, by=by)
    written = {"summary_patientwise.csv": st.describe(patients, by=by),
               "paired_comparisons.csv": st.paired_comparisons(patients, args.reference, by=by),
               "sequence_wise_comparisons.csv": st.sequence_wise(metrics, args.reference, by=by),
               "sequence_class_comparisons.csv": st.sequence_class(metrics, args.reference, by=by),
               "failure_rates.csv": st.failure_rates(metrics, by=by)}
    if flags:
        written["subgroup_comparisons.csv"] = st.subgroup_comparisons(metrics, args.reference, flags, by=by)
    empty = [name for name, table in written.items() if table.empty]
    for name, table in written.items():
        if not table.empty:
            table.to_csv(out / name, index=False)
    print(f"Wrote {', '.join(n for n in written if n not in empty)} to {out}"
          + (f"; no results for {', '.join(empty)}" if empty else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="liversegmri", description="LiverSegMRI: whole-liver segmentation on abdominal MRI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="segment the liver with LiverSegMRI")
    p.add_argument("-i", "--input", required=True, help="image file or folder of images")
    p.add_argument("-o", "--output", required=True, help="output folder")
    p.add_argument("--weights-dir", help="local model folder (default: download from Hugging Face)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--extension", default=".nii.gz", choices=[".nii.gz", ".nrrd"])
    p.add_argument("--no-postprocessing", action="store_true", help="keep all connected components")
    p.set_defaults(func=_predict)

    p = sub.add_parser("comparator", help="liver masks from TotalSegmentator MRI or MRAnnotator")
    p.add_argument("model", choices=["totalsegmentator", "mrannotator"])
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model-dir", help="MRAnnotator Dataset001_Abdomen model folder")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.set_defaults(func=_comparator)

    p = sub.add_parser("check-orientation", help="check whether an examination's slice order matches its header")
    p.add_argument("-i", "--input", required=True, help="folder with the volumes of one examination")
    p.add_argument("-o", "--output", help="write corrected copies here when the slice order is reversed")
    p.add_argument("--sample", type=int, default=3, help="volumes used to decide (default 3)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.set_defaults(func=_orientation)

    p = sub.add_parser("prepare-nnunet", help="build the nnU-Net v2 training/test dataset")
    p.add_argument("--data-root", required=True, help="folder with <source>/<patient_id>/<sequence> images and masks")
    p.add_argument("--nnunet-raw", required=True)
    p.add_argument("--mask-suffix", default="_mask")
    p.add_argument("--subgroups", help="CSV with source, patient_id, and binary subgroup columns")
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_prepare)

    p = sub.add_parser("evaluate", help="compute metrics for a manifest of predictions")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("analyze", help="patient-level paired statistics")
    p.add_argument("--metrics", required=True, help="output of `liversegmri evaluate`")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reference", default="LiverSegMRI", help="model compared against all others")
    p.add_argument("--by", help="optional column to stratify by (e.g., test_set)")
    p.add_argument("--flags", help="comma-separated binary subgroup columns for the subgroup analysis")
    p.set_defaults(func=_analyze)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
