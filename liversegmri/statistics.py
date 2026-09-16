"""Patient-level statistics used in the study.

Patients contribute multiple sequences, so metrics are averaged per patient before inference. Models are compared
pairwise with the Wilcoxon signed-rank test (Friedman omnibus test across all models), with bootstrap 95% CIs of the
mean paired difference and Holm correction within each family of tests.

Analyses provided here, all patient level:
    describe / paired_comparisons   headline performance and model comparisons
    sequence_wise                   comparisons within each sequence type
    sequence_class                  conventional T1-weighted imaging versus functional sequences (DWI, ADC)
    subgroup_comparisons            models within morphologic subgroups, and each subgroup against the reference group
    failure_rates                   proportion of sequences below Dice thresholds
"""

from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

from .data import sequence_group
from .metrics import METRICS

CONVENTIONAL_T1 = ("T1WI pre-contrast", "T1WI early arterial", "T1WI arterial", "T1WI portal venous",
                   "T1WI transitional/delayed", "T1WI hepatobiliary", "T1WI in-phase", "T1WI out-of-phase")
FUNCTIONAL = ("DWI", "ADC")


def bootstrap_ci(values, statistic=np.mean, n_resamples: int = 5000, confidence: float = 0.95,
                 seed: int = 20260914) -> tuple[float, float]:
    """Percentile bootstrap CI; the random stream is seeded by the data for reproducibility."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng([seed, zlib.crc32(np.ascontiguousarray(x).tobytes())])
    draws = np.concatenate([statistic(x[rng.integers(0, len(x), (min(500, n_resamples - i), len(x)))], axis=1)
                            for i in range(0, n_resamples, 500)])
    alpha = (1 - confidence) / 2
    return tuple(np.quantile(draws, [alpha, 1 - alpha]))


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-adjusted P values (NaN entries are ignored)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (len(valid) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def patient_level(metrics: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Mean of each metric over a patient's sequences, per model (and per `by` group)."""
    values = [m for m in METRICS if m in metrics]
    return metrics.groupby([*by, "patient_id", "model"], as_index=False)[values].mean()


def describe(patients: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    rows = []
    for keys, group in patients.groupby([*by, "model"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in (m for m in METRICS if m in group):
            x = group[metric].dropna()
            low, high = bootstrap_ci(x)
            rows.append({**dict(zip([*by, "model"], keys)), "metric": metric, "n_patients": len(x), "mean": x.mean(),
                         "sd": x.std(), "ci_low": low, "ci_high": high, "median": x.median()})
    return pd.DataFrame(rows)


def _paired(wide: pd.DataFrame, a: str, b: str, min_patients: int = 6) -> dict | None:
    """Paired difference a - b over the patients present in both columns."""
    g = wide[[a, b]].dropna()
    if len(g) < min_patients:
        return None
    diff = g[a] - g[b]
    low, high = bootstrap_ci(diff)
    return {"n_patients": len(diff), "mean_a": g[a].mean(), "mean_b": g[b].mean(), "mean_difference": diff.mean(),
            "ci_low": low, "ci_high": high, "p": wilcoxon(g[a], g[b]).pvalue if (diff != 0).any() else np.nan}


def paired_comparisons(patients: pd.DataFrame, reference: str, by: Sequence[str] = (),
                       min_patients: int = 6) -> pd.DataFrame:
    """Reference model versus each other model, per metric (and per `by` group); Holm correction within metric."""
    rows = []
    groups = patients.groupby(list(by)) if by else [((), patients)]
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in (m for m in METRICS if m in group):
            wide = group.pivot_table(index="patient_id", columns="model", values=metric).dropna()
            if reference not in wide or len(wide) < min_patients:
                continue
            friedman = friedmanchisquare(*[wide[m] for m in wide]).pvalue if wide.shape[1] >= 3 else np.nan
            for other in (m for m in wide if m != reference):
                result = _paired(wide, reference, other, min_patients)
                if result:
                    rows.append({**dict(zip(by, keys)), "metric": metric, "comparison": f"{reference} - {other}",
                                 **{k: v for k, v in result.items() if k not in ("mean_a", "mean_b")},
                                 f"mean_{reference}": result["mean_a"], "mean_other": result["mean_b"],
                                 "wilcoxon_p": result["p"], "friedman_p": friedman})
    result = pd.DataFrame(rows).drop(columns="p", errors="ignore")
    if not result.empty:
        result["p_holm"] = result.groupby("metric")["wilcoxon_p"].transform(lambda s: holm(s.values))
    return result


def sequence_wise(metrics: pd.DataFrame, reference: str, by: Sequence[str] = (), min_patients: int = 10) -> pd.DataFrame:
    """Paired comparisons within each sequence type after averaging per patient within that type."""
    table = metrics.assign(sequence_group=metrics["sequence"].map(sequence_group))
    patients = patient_level(table, by=[*by, "sequence_group"])
    return paired_comparisons(patients, reference, by=[*by, "sequence_group"], min_patients=min_patients)


def sequence_class(metrics: pd.DataFrame, reference: str, by: Sequence[str] = (), metric: str = "dice",
                   min_patients: int = 6) -> pd.DataFrame:
    """Conventional T1-weighted imaging versus functional sequences (DWI, ADC).

    For every model, the within-patient drop from conventional T1WI to each functional class; then the difference
    between the comparator's drop and the reference model's drop over the same patients (difference in differences),
    which tests whether a model degrades specifically on the functional sequences.
    """
    table = metrics.assign(sequence_group=metrics["sequence"].map(sequence_group))
    table["sequence_class"] = np.where(table.sequence_group.isin(CONVENTIONAL_T1), "Conventional T1WI", table.sequence_group)
    patients = patient_level(table, by=[*by, "sequence_class"])
    rows = []
    groups = patients.groupby(list(by)) if by else [((), patients)]
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        wide = group.pivot_table(index=["patient_id", "model"], columns="sequence_class", values=metric)
        if "Conventional T1WI" not in wide:
            continue
        for functional in (c for c in FUNCTIONAL if c in wide.columns):
            drops = {}
            for model in wide.index.get_level_values("model").unique():
                w = wide.xs(model, level="model")
                result = _paired(w, "Conventional T1WI", functional, min_patients)
                if result:
                    rows.append({**dict(zip(by, keys)), "model": model, "sequence_class": functional,
                                 "comparison": f"Conventional T1WI - {functional}", "metric": metric,
                                 "mean_conventional": result["mean_a"], "mean_functional": result["mean_b"],
                                 "drop": result["mean_difference"], "ci_low": result["ci_low"],
                                 "ci_high": result["ci_high"], "n_patients": result["n_patients"], "p": result["p"]})
                    drops[model] = (w["Conventional T1WI"] - w[functional]).dropna()
            for model, drop in drops.items():
                if model == reference or reference not in drops:
                    continue
                pair = pd.concat([drops[reference].rename("reference"), drop.rename("model")], axis=1).dropna()
                if len(pair) < min_patients:
                    continue
                did = pair["model"] - pair["reference"]
                low, high = bootstrap_ci(did)
                rows.append({**dict(zip(by, keys)), "model": model, "sequence_class": functional,
                             "comparison": f"drop({model}) - drop({reference})", "metric": metric,
                             "drop": did.mean(), "ci_low": low, "ci_high": high, "n_patients": len(did),
                             "p": wilcoxon(pair["model"], pair["reference"]).pvalue if (did != 0).any() else np.nan})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = result.groupby([*by, "sequence_class"])["p"].transform(lambda s: holm(s.values)) \
            if by else holm(result["p"].values)
    return result


def subgroup_comparisons(metrics: pd.DataFrame, reference: str, flags: Sequence[str], by: Sequence[str] = (),
                         reference_group: str = "Normal morphology", min_patients: int = 6) -> pd.DataFrame:
    """Models within each morphologic subgroup, and each subgroup against patients with none of the flags.

    `flags` are binary patient-level columns of the metrics table. Model comparisons within a subgroup are paired
    (Wilcoxon); subgroup versus reference-group comparisons are unpaired (Mann-Whitney), as those are different
    patients. Holm correction runs within each `by` group.
    """
    available = [f for f in flags if f in metrics]
    if not available:
        return pd.DataFrame()
    rows = []
    groups = metrics.groupby(list(by)) if by else [((), metrics)]
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        patients = patient_level(group, by=())
        flag_values = group.groupby("patient_id")[available].max()
        normal = flag_values.index[flag_values.sum(axis=1) == 0]
        for subgroup in [reference_group, *available]:
            members = normal if subgroup == reference_group else flag_values.index[flag_values[subgroup] == 1]
            sub = patients[patients.patient_id.isin(members)]
            for metric in (m for m in METRICS if m in sub):
                wide = sub.pivot_table(index="patient_id", columns="model", values=metric).dropna()
                if reference not in wide or len(wide) < min_patients:
                    continue
                for other in (m for m in wide if m != reference):
                    result = _paired(wide, reference, other, min_patients)
                    if result:
                        rows.append({**dict(zip(by, keys)), "subgroup": subgroup, "metric": metric,
                                     "analysis": "models within subgroup", "comparison": f"{reference} - {other}",
                                     "n_patients": result["n_patients"], "value": result["mean_difference"],
                                     "ci_low": result["ci_low"], "ci_high": result["ci_high"], "p": result["p"]})
                if subgroup == reference_group:
                    continue
                for model in sub.model.unique():  # subgroup versus the reference group, within model
                    a = sub[sub.model == model][metric].dropna()
                    b = patients[(patients.patient_id.isin(normal)) & (patients.model == model)][metric].dropna()
                    if len(a) < min_patients or len(b) < min_patients:
                        continue
                    rows.append({**dict(zip(by, keys)), "subgroup": subgroup, "metric": metric,
                                 "analysis": f"subgroup vs {reference_group.lower()}", "comparison": model,
                                 "n_patients": len(a), "value": a.mean() - b.mean(), "ci_low": np.nan,
                                 "ci_high": np.nan, "p": mannwhitneyu(a, b).pvalue})
    result = pd.DataFrame(rows)
    if not result.empty:
        keys = [*by, "analysis"] if by else ["analysis"]
        result["p_holm"] = result.groupby(keys)["p"].transform(lambda s: holm(s.values))
    return result


def failure_rates(metrics: pd.DataFrame, by: Sequence[str] = (), thresholds: Sequence[float] = (0.90, 0.50),
                  volume_error_threshold: float = 10.0) -> pd.DataFrame:
    """Sequence-level and patient-level failure rates per model."""
    rows = []
    patients = patient_level(metrics, by=by)
    for keys, group in metrics.groupby([*by, "model"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip([*by, "model"], keys))
        per_patient = patients
        for column, value in meta.items():
            if column != "model":
                per_patient = per_patient[per_patient[column] == value]
        per_patient = per_patient[per_patient.model == meta["model"]]
        for threshold in thresholds:
            rows.append({**meta, "definition": f"Dice < {threshold:.2f}", "sequences": len(group),
                         "n_sequences": int((group.dice < threshold).sum()),
                         "pct_sequences": 100 * float((group.dice < threshold).mean()),
                         "n_patients_by_mean": int((per_patient.dice < threshold).sum()),
                         "pct_patients_by_mean": 100 * float((per_patient.dice < threshold).mean())})
        if "volume_error_pct" in group:
            over = group.volume_error_pct > volume_error_threshold
            rows.append({**meta, "definition": f"Volume error > {volume_error_threshold:.0f}%", "sequences": len(group),
                         "n_sequences": int(over.sum()), "pct_sequences": 100 * float(over.mean()),
                         "n_patients_by_mean": int((per_patient.volume_error_pct > volume_error_threshold).sum()),
                         "pct_patients_by_mean": 100 * float((per_patient.volume_error_pct > volume_error_threshold).mean())})
    return pd.DataFrame(rows)
