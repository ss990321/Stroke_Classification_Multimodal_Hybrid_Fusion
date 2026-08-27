#!/usr/bin/env python3
"""Prepare selected MIMIC-IV-ECG records for external validation.

Outputs are aligned to this project's stroke ECG experiment format:

1) mimic_external_signal.npz
   X, y, file_name, PatientID, study_id

2) mimic_external_multimodal.npz
   ecg, signal, feature, y, file_name, PatientID, study_id, feature_cols

3) mimic_external_features.csv
   file_name, PatientID, study_id, label, FEATURES_41...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

try:
    import neurokit2 as nk
except ImportError:
    nk = None


FEATURES_41 = [
    "Gender",
    "VentricularRate",
    "AtrialRate",
    "PRInterval",
    "QRSDuration",
    "QTInterval",
    "QTCorrected",
    "PAxis",
    "RAxis",
    "TAxis",
    "QRSCount",
    "QOnset",
    "QOffset",
    "POnset",
    "POffset",
    "TOffset",
    "QTcFrederica",
    "GlobalRR",
    "PharmaRRinterval",
    "PharmaPPinterval",
    "PRInterval_missing",
    "POnset_missing",
    "POffset_missing",
    "PAxis_missing",
    "QTc (ms)",
    "QT (ms)",
    "QRS (ms)",
    "ST (ms)",
    "Impulse factor",
    "PRQ (ms)",
    "Kurtosis",
    "R-H (mV)",
    "Crest factor",
    "pNN50 (%)",
    "P-H (mV)",
    "RMSSD (ms)",
    "SDSD (ms)",
    "Skewness",
    "Peak value",
    "HR (bpm)",
    "RR-I (ms)",
]

STANDARD_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def gender_to_project_code(value: object) -> float:
    text = str(value).strip().upper()
    if text in {"F", "FEMALE", "0"}:
        return 0.0
    if text in {"M", "MALE", "1"}:
        return 1.0
    return np.nan


def numeric(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column, np.nan), errors="coerce")
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def interval_or_nan(end: float, start: float) -> float:
    if not np.isfinite(end) or not np.isfinite(start):
        return np.nan
    value = end - start
    return float(value) if value >= 0 else np.nan


def is_missing_measure(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return 1.0 if value <= 0 else 0.0


def window_max(signal: np.ndarray, start: float, end: float) -> float:
    if not np.isfinite(start) or not np.isfinite(end):
        return np.nan
    left = int(max(0, round(start)))
    right = int(min(len(signal), round(end)))
    if right <= left:
        return np.nan
    return float(np.nanmax(signal[left:right]))


def waveform_stats(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "Kurtosis": np.nan,
            "Skewness": np.nan,
            "Peak value": np.nan,
            "Impulse factor": np.nan,
            "Crest factor": np.nan,
        }

    abs_x = np.abs(x)
    peak = float(np.nanmax(abs_x))
    mean_abs = float(np.nanmean(abs_x))
    rms = float(np.sqrt(np.nanmean(np.square(x))))

    return {
        "Kurtosis": float(kurtosis(x, fisher=True, bias=False))
        if x.size > 3
        else np.nan,
        "Skewness": float(skew(x, bias=False)) if x.size > 2 else np.nan,
        "Peak value": peak,
        "Impulse factor": peak / mean_abs if mean_abs > 0 else np.nan,
        "Crest factor": peak / rms if rms > 0 else np.nan,
    }


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < fs:
        return np.asarray([], dtype=int)

    x = x.copy()
    x[~finite] = np.nanmedian(x[finite])

    if nk is not None:
        try:
            cleaned = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
            peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            if len(peaks) >= 3:
                return peaks
        except Exception:
            pass

    x = x - np.nanmedian(x)
    std = float(np.nanstd(x))
    if std <= 0:
        return np.asarray([], dtype=int)

    distance = int(round(0.25 * fs))
    prominence = max(0.05, 0.6 * std)

    peak_sets = []
    for candidate in (x, -x):
        peaks, properties = find_peaks(
            candidate,
            distance=distance,
            prominence=prominence,
        )
        if len(peaks) > 0:
            score = float(np.nanmedian(properties.get("prominences", [0.0])))
        else:
            score = 0.0
        peak_sets.append((score, peaks))

    return max(peak_sets, key=lambda item: (len(item[1]), item[0]))[1]


def hrv_time_features(
    signal: np.ndarray,
    fs: int,
    reference_rr_ms: float | None = None,
) -> dict[str, float]:
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 3:
        return {
            "RMSSD (ms)": np.nan,
            "SDSD (ms)": np.nan,
            "pNN50 (%)": np.nan,
            "R_peak_count_detected": int(len(peaks)),
        }

    rr_ms = np.diff(peaks) * (1000.0 / float(fs))
    keep = (rr_ms >= 300.0) & (rr_ms <= 2000.0)
    if reference_rr_ms is not None and np.isfinite(reference_rr_ms) and reference_rr_ms > 0:
        keep &= rr_ms >= max(300.0, 0.70 * reference_rr_ms)
        keep &= rr_ms <= min(2000.0, 1.30 * reference_rr_ms)
    rr_ms = rr_ms[keep]
    if len(rr_ms) < 2:
        return {
            "RMSSD (ms)": np.nan,
            "SDSD (ms)": np.nan,
            "pNN50 (%)": np.nan,
            "R_peak_count_detected": int(len(peaks)),
        }

    diff_rr = np.diff(rr_ms)
    if len(diff_rr) == 0:
        return {
            "RMSSD (ms)": np.nan,
            "SDSD (ms)": np.nan,
            "pNN50 (%)": np.nan,
            "R_peak_count_detected": int(len(peaks)),
        }

    return {
        "RMSSD (ms)": float(np.sqrt(np.mean(np.square(diff_rr)))),
        "SDSD (ms)": float(np.std(diff_rr, ddof=1)) if len(diff_rr) > 1 else 0.0,
        "pNN50 (%)": float(100.0 * np.mean(np.abs(diff_rr) > 50.0)),
        "R_peak_count_detected": int(len(peaks)),
    }


def load_record(record_root: Path, path: str, target_length: int) -> tuple[np.ndarray, list[str]]:
    record = wfdb.rdrecord(str(record_root / path))
    signal = np.asarray(record.p_signal, dtype=np.float32)
    lead_names = [str(name) for name in record.sig_name]

    if signal.ndim != 2:
        raise ValueError(f"Unexpected signal shape for {path}: {signal.shape}")

    if signal.shape[0] < target_length:
        pad = np.zeros((target_length - signal.shape[0], signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, pad])
    elif signal.shape[0] > target_length:
        signal = signal[:target_length]

    lead_to_index = {lead: idx for idx, lead in enumerate(lead_names)}
    missing_leads = [lead for lead in STANDARD_LEADS if lead not in lead_to_index]
    if missing_leads:
        raise ValueError(f"{path}: missing leads {missing_leads}; found {lead_names}")

    signal = signal[:, [lead_to_index[lead] for lead in STANDARD_LEADS]]
    return signal.T.astype(np.float32, copy=False), lead_names


def build_feature_row(row: pd.Series, lead_ii: np.ndarray, fs: int) -> dict[str, float]:
    rr = numeric(row, "rr_interval")
    p_on = numeric(row, "p_onset")
    p_end = numeric(row, "p_end")
    qrs_on = numeric(row, "qrs_onset")
    qrs_end = numeric(row, "qrs_end")
    t_end = numeric(row, "t_end")
    p_axis = numeric(row, "p_axis")
    qrs_axis = numeric(row, "qrs_axis")
    t_axis = numeric(row, "t_axis")

    pr = interval_or_nan(qrs_on, p_on)
    qrs = interval_or_nan(qrs_end, qrs_on)
    qt = interval_or_nan(t_end, qrs_on)
    st = interval_or_nan(t_end, qrs_end)
    hr = 60000.0 / rr if np.isfinite(rr) and rr > 0 else np.nan
    qt_corrected = qt / np.sqrt(rr / 1000.0) if np.isfinite(qt) and np.isfinite(rr) and rr > 0 else np.nan
    qtc_f = qt / np.cbrt(rr / 1000.0) if np.isfinite(qt) and np.isfinite(rr) and rr > 0 else np.nan
    qrs_count = int(round(10000.0 / rr)) if np.isfinite(rr) and rr > 0 else np.nan

    stats = waveform_stats(lead_ii)
    hrv = hrv_time_features(lead_ii, fs, reference_rr_ms=rr)
    out = {
        "Gender": gender_to_project_code(row.get("gender", np.nan)),
        "VentricularRate": hr,
        "AtrialRate": hr,
        "PRInterval": pr,
        "QRSDuration": qrs,
        "QTInterval": qt,
        "QTCorrected": qt_corrected,
        "PAxis": p_axis,
        "RAxis": qrs_axis,
        "TAxis": t_axis,
        "QRSCount": qrs_count,
        "QOnset": qrs_on,
        "QOffset": qrs_end,
        "POnset": p_on,
        "POffset": p_end,
        "TOffset": t_end,
        "QTcFrederica": qtc_f,
        "GlobalRR": rr,
        "PharmaRRinterval": rr,
        "PharmaPPinterval": rr,
        "PRInterval_missing": 1.0 if not np.isfinite(pr) else 0.0,
        "POnset_missing": is_missing_measure(p_on),
        "POffset_missing": is_missing_measure(p_end),
        "PAxis_missing": is_missing_measure(p_axis),
        "QTc (ms)": qt_corrected,
        "QT (ms)": qt,
        "QRS (ms)": qrs,
        "ST (ms)": st,
        "Impulse factor": stats["Impulse factor"],
        "PRQ (ms)": pr,
        "Kurtosis": stats["Kurtosis"],
        "R-H (mV)": window_max(lead_ii, qrs_on, qrs_end),
        "Crest factor": stats["Crest factor"],
        "pNN50 (%)": hrv["pNN50 (%)"],
        "P-H (mV)": window_max(lead_ii, p_on, p_end),
        "RMSSD (ms)": hrv["RMSSD (ms)"],
        "SDSD (ms)": hrv["SDSD (ms)"],
        "Skewness": stats["Skewness"],
        "Peak value": stats["Peak value"],
        "HR (bpm)": hr,
        "RR-I (ms)": rr,
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort_csv",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_subset_manifest\mimic_stroke_control_cohort.csv"),
    )
    parser.add_argument(
        "--machine_measurements_csv",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\machine_measurements.csv"),
    )
    parser.add_argument(
        "--record_root",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_subset"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_external_prepared"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target_length", type=int, default=5000)
    parser.add_argument("--fs", type=int, default=500)
    parser.add_argument("--progress_every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort_csv)
    if args.limit is not None:
        cohort = cohort.head(args.limit).copy()

    cohort["study_id"] = pd.to_numeric(cohort["study_id"], errors="coerce").astype("Int64")
    measurement_cols = [
        "study_id",
        "rr_interval",
        "p_onset",
        "p_end",
        "qrs_onset",
        "qrs_end",
        "t_end",
        "p_axis",
        "qrs_axis",
        "t_axis",
    ]
    measurements = pd.read_csv(args.machine_measurements_csv, usecols=measurement_cols)
    measurements["study_id"] = pd.to_numeric(
        measurements["study_id"], errors="coerce"
    ).astype("Int64")

    data = cohort.merge(measurements, on="study_id", how="left", validate="one_to_one")
    data = data.sort_values(["label", "subject_id", "study_id"]).reset_index(drop=True)

    n = len(data)
    signals = np.empty((n, 12, args.target_length), dtype=np.float32)
    feature_rows = []
    failed = []
    lead_name_examples = {}

    for idx, row in enumerate(data.itertuples(index=False), start=0):
        path = str(row.path).replace("\\", "/")
        try:
            signal, lead_names = load_record(args.record_root, path, args.target_length)
            lead_ii = signal[STANDARD_LEADS.index("II")]
            row_series = data.iloc[idx]
            features = build_feature_row(row_series, lead_ii, args.fs)
            signals[idx] = signal
            feature_rows.append(features)
            if len(lead_name_examples) < 5:
                lead_name_examples[path] = lead_names
        except Exception as exc:  # noqa: BLE001
            failed.append({"index": idx, "path": path, "error": str(exc)})
            signals[idx] = np.nan
            feature_rows.append({feature: np.nan for feature in FEATURES_41})

        done = idx + 1
        if done == 1 or done % args.progress_every == 0 or done == n:
            print(f"[{done}/{n}] prepared")

    feature_frame = pd.DataFrame(feature_rows, columns=FEATURES_41)
    meta_frame = pd.DataFrame(
        {
            "file_name": data["path"].astype(str).to_numpy(),
            "PatientID": data["subject_id"].astype(str).to_numpy(),
            "study_id": data["study_id"].astype(str).to_numpy(),
            "label": data["label"].astype(int).to_numpy(),
            "gender": data["gender"].astype(str).to_numpy(),
            "age": pd.to_numeric(data["age"], errors="coerce").to_numpy(),
        }
    )
    feature_csv = pd.concat([meta_frame, feature_frame], axis=1)

    y = data["label"].astype(int).to_numpy()
    file_names = data["path"].astype(str).to_numpy(dtype=object)
    patient_ids = data["subject_id"].astype(str).to_numpy(dtype=object)
    study_ids = data["study_id"].astype(str).to_numpy(dtype=object)
    features = feature_frame.to_numpy(dtype=np.float32)

    np.savez_compressed(
        args.out_dir / "mimic_external_signal.npz",
        X=signals,
        y=y,
        file_name=file_names,
        PatientID=patient_ids,
        study_id=study_ids,
    )
    np.savez_compressed(
        args.out_dir / "mimic_external_multimodal.npz",
        ecg=signals,
        signal=signals,
        feature=features,
        y=y,
        file_name=file_names,
        PatientID=patient_ids,
        study_id=study_ids,
        feature_cols=np.asarray(FEATURES_41, dtype=object),
    )
    feature_csv.to_csv(
        args.out_dir / "mimic_external_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(failed).to_csv(args.out_dir / "failed_records.csv", index=False)

    summary = {
        "n_records": int(n),
        "n_stroke": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "signal_shape": list(signals.shape),
        "feature_shape": list(features.shape),
        "feature_columns": FEATURES_41,
        "failed_records": int(len(failed)),
        "hrv_nan_counts": {
            column: int(feature_frame[column].isna().sum())
            for column in ["RMSSD (ms)", "SDSD (ms)", "pNN50 (%)"]
        },
        "hrv_extractor": "neurokit2_ecg_peaks_with_scipy_fallback"
        if nk is not None
        else "scipy_find_peaks_fallback_only",
        "lead_order_output": STANDARD_LEADS,
        "lead_name_examples": lead_name_examples,
        "notes": [
            "MIMIC machine measurements are harmonized to project FEATURES_41.",
            "RMSSD, SDSD, and pNN50 are estimated from lead II R-peak intervals over the 10-second waveform.",
            "RR intervals outside 70-130% of the machine rr_interval are excluded before HRV calculation.",
            "QTCorrected uses Bazett and QTcFrederica uses Fridericia from QT and RR.",
            "Short-window HRV estimates should be interpreted cautiously.",
        ],
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
