"""Evaluate predictions listed in a manifest CSV.

Required columns: `case_id`, `patient_id`, `sequence`, `reference`, and one `pred_<model>` column per model.
Any other columns (e.g., `test_set`, subgroup flags) are carried through to the output.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS, evaluate_pair


def _job(args):
    meta, model, reference, prediction = args
    row = {**meta, "model": model}
    if isinstance(prediction, str) and prediction and Path(prediction).exists():
        row.update(evaluate_pair(reference, prediction), prediction_available=True)
    else:
        row.update({m: np.nan for m in METRICS}, prediction_available=False)
    return row


def evaluate_manifest(manifest: str | Path, output: str | Path, workers: int = 8) -> pd.DataFrame:
    table = pd.read_csv(manifest)
    required = {"case_id", "patient_id", "sequence", "reference"}
    if missing := required - set(table.columns):
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    model_columns = [c for c in table.columns if c.startswith("pred_")]
    meta_columns = [c for c in table.columns if c not in model_columns and c != "reference"]
    jobs = [({c: r[c] for c in meta_columns}, col[len("pred_"):], r["reference"], r[col])
            for _, r in table.iterrows() for col in model_columns]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pd.DataFrame(pool.map(_job, jobs, chunksize=4))
    results.to_csv(output, index=False)
    return results
