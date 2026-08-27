#!/usr/bin/env python3
"""Build a MIMIC-IV-ECG stroke/control waveform download manifest.

The script uses local metadata CSV files only. It does not download waveforms.
It creates a balanced subset and writes PhysioNet URLs for the selected .hea
and .dat files so the full waveform archive does not need to be downloaded.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ACUTE_STROKE_PREFIXES = ("I60", "I61", "I62", "I63", "I64")
TIA_PREFIXES = ("G45",)
STROKE_HISTORY_PREFIXES = ("Z8673",)


def normalize_code(value: object) -> str:
    return str(value).strip().replace(".", "").upper()


def parse_code_list(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text == "[]":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, (list, tuple, set)):
        return []

    return [normalize_code(code) for code in parsed]


def has_any_prefix(codes: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(str(code).startswith(prefixes) for code in codes)


def derive_path_from_file_name(value: object) -> str:
    text = str(value).replace("\\", "/").strip()
    marker = "/files/"
    if marker in text:
        return "files/" + text.split(marker, 1)[1]
    if text.startswith("files/"):
        return text
    return text


def add_age_bin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    age = pd.to_numeric(out["age"], errors="coerce")
    out["age_bin"] = (np.floor(age / 10.0) * 10).clip(lower=0, upper=90)
    out["age_bin"] = out["age_bin"].fillna(-1).astype(int)
    out["gender_bin"] = (
        out["gender"].astype(str).str.strip().str.upper().replace({"": "UNK"})
    )
    return out


def stratified_control_sample(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    n_controls: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cases = add_age_bin(cases)
    controls = add_age_bin(controls)

    requested_by_stratum = (
        cases.groupby(["gender_bin", "age_bin"], dropna=False)
        .size()
        .rename("requested")
        .reset_index()
    )

    selected_parts = []
    selected_indices: set[int] = set()

    for row in requested_by_stratum.itertuples(index=False):
        pool = controls[
            (controls["gender_bin"] == row.gender_bin)
            & (controls["age_bin"] == row.age_bin)
        ]
        if pool.empty:
            continue
        take = min(int(row.requested), len(pool))
        chosen = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        selected_indices.update(int(idx) for idx in chosen)
        selected_parts.append(controls.loc[chosen])

    selected_n = sum(len(part) for part in selected_parts)
    deficit = int(n_controls) - int(selected_n)
    if deficit > 0:
        remaining = controls.loc[~controls.index.isin(selected_indices)]
        if len(remaining) < deficit:
            raise ValueError(
                f"Not enough controls: selected={selected_n}, "
                f"remaining={len(remaining)}, requested={n_controls}"
            )
        chosen = rng.choice(remaining.index.to_numpy(), size=deficit, replace=False)
        selected_parts.append(remaining.loc[chosen])

    selected = pd.concat(selected_parts, ignore_index=False)
    selected = selected.sample(frac=1.0, random_state=seed).head(n_controls)
    return selected.drop(columns=["age_bin", "gender_bin"], errors="ignore")


def load_records(args: argparse.Namespace) -> pd.DataFrame:
    usecols = [
        "file_name",
        "study_id",
        "subject_id",
        "ecg_time",
        args.diag_col,
        "gender",
        "age",
        "ecg_no_within_stay",
        "ecg_taken_in_ed_or_hosp",
    ]
    frame = pd.read_csv(args.records_diag_csv, usecols=usecols, nrows=args.max_rows)
    frame = frame.rename(columns={args.diag_col: "diagnosis_codes_raw"})

    frame["study_id"] = pd.to_numeric(frame["study_id"], errors="coerce").astype("Int64")
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce").astype("Int64")
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["ecg_no_within_stay"] = pd.to_numeric(
        frame["ecg_no_within_stay"], errors="coerce"
    )
    frame["ecg_time"] = pd.to_datetime(frame["ecg_time"], errors="coerce")

    codes = frame["diagnosis_codes_raw"].map(parse_code_list)
    frame["acute_stroke"] = codes.map(
        lambda values: has_any_prefix(values, ACUTE_STROKE_PREFIXES)
    )
    frame["tia"] = codes.map(lambda values: has_any_prefix(values, TIA_PREFIXES))
    frame["stroke_history"] = codes.map(
        lambda values: has_any_prefix(values, STROKE_HISTORY_PREFIXES)
    )
    frame["stroke_codes"] = codes.map(
        lambda values: [
            code for code in values if code.startswith(ACUTE_STROKE_PREFIXES)
        ]
    )

    keep = frame["study_id"].notna() & frame["subject_id"].notna()
    keep &= frame["age"].ge(args.adult_min_age)

    if args.first_ecg_within_stay:
        keep &= frame["ecg_no_within_stay"].eq(0)

    if args.require_ed_or_hosp:
        keep &= frame["ecg_taken_in_ed_or_hosp"].astype(str).str.lower().eq("true")

    return frame.loc[keep].copy()


def maybe_one_ecg_per_subject(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["subject_id", "ecg_time", "study_id"]).copy()
    return ordered.drop_duplicates("subject_id", keep="first")


def attach_record_paths(cohort: pd.DataFrame, record_list_csv: Path) -> pd.DataFrame:
    record_cols = ["subject_id", "study_id", "path"]
    records = pd.read_csv(record_list_csv, usecols=record_cols)
    records["study_id"] = pd.to_numeric(records["study_id"], errors="coerce").astype("Int64")
    records["subject_id"] = pd.to_numeric(records["subject_id"], errors="coerce").astype("Int64")

    out = cohort.merge(
        records,
        on=["subject_id", "study_id"],
        how="left",
        validate="one_to_one",
    )
    missing_path = out["path"].isna()
    if missing_path.any():
        out.loc[missing_path, "path"] = out.loc[missing_path, "file_name"].map(
            derive_path_from_file_name
        )
    return out


def write_outputs(cohort: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url.rstrip("/")
    cohort = cohort.copy()
    cohort["path"] = cohort["path"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    cohort["hea_url"] = base_url + "/" + cohort["path"] + ".hea"
    cohort["dat_url"] = base_url + "/" + cohort["path"] + ".dat"

    cohort_out = out_dir / "mimic_stroke_control_cohort.csv"
    manifest_out = out_dir / "download_manifest.csv"
    urls_out = out_dir / "download_urls.txt"
    summary_out = out_dir / "summary.json"

    keep_cols = [
        "label",
        "subject_id",
        "study_id",
        "ecg_time",
        "gender",
        "age",
        "file_name",
        "path",
        "acute_stroke",
        "tia",
        "stroke_history",
        "stroke_codes",
    ]
    cohort[keep_cols].to_csv(cohort_out, index=False, encoding="utf-8-sig")
    cohort[
        ["label", "subject_id", "study_id", "path", "hea_url", "dat_url"]
    ].to_csv(manifest_out, index=False, encoding="utf-8-sig")

    urls = []
    for row in cohort.itertuples(index=False):
        urls.append(row.hea_url)
        urls.append(row.dat_url)
    urls_out.write_text("\n".join(urls) + "\n", encoding="utf-8")

    summary = {
        "n_total_selected_ecg": int(len(cohort)),
        "n_stroke": int(cohort["label"].sum()),
        "n_control": int((cohort["label"] == 0).sum()),
        "unique_subjects": int(cohort["subject_id"].nunique()),
        "diagnosis_column": args.diag_col,
        "stroke_definition": "ICD-10 prefixes I60-I64",
        "control_definition": (
            "no I60-I64, no G45, no Z86.73"
            if args.exclude_tia_history_from_controls
            else "no I60-I64"
        ),
        "first_ecg_within_stay": bool(args.first_ecg_within_stay),
        "one_ecg_per_subject": bool(args.one_ecg_per_subject),
        "base_url": base_url,
        "outputs": {
            "cohort_csv": str(cohort_out),
            "download_manifest_csv": str(manifest_out),
            "download_urls_txt": str(urls_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    print("Next command if GNU wget.exe is installed:")
    print(
        "wget.exe --continue --input-file "
        f'"{urls_out}" --directory-prefix "{args.download_dir}"'
    )


def build_manifest(args: argparse.Namespace) -> None:
    records = load_records(args)

    cases = records.loc[records["acute_stroke"]].copy()
    controls = records.loc[~records["acute_stroke"]].copy()

    if args.exclude_tia_history_from_controls:
        controls = controls.loc[~controls["tia"] & ~controls["stroke_history"]].copy()

    if args.one_ecg_per_subject:
        cases = maybe_one_ecg_per_subject(cases)
        controls = maybe_one_ecg_per_subject(controls)

    if cases.empty:
        raise ValueError("No stroke cases found after filters.")

    n_cases = min(args.n_cases, len(cases)) if args.n_cases else len(cases)
    cases = cases.sort_values(["subject_id", "ecg_time", "study_id"]).head(n_cases)

    n_controls = args.n_controls if args.n_controls is not None else n_cases
    n_controls = min(int(n_controls), len(controls))
    controls = stratified_control_sample(cases, controls, n_controls, args.seed)

    cases["label"] = 1
    controls["label"] = 0
    cohort = pd.concat([cases, controls], ignore_index=True)
    cohort = cohort.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    cohort = attach_record_paths(cohort, args.record_list_csv)

    missing_path = cohort["path"].isna() | cohort["path"].astype(str).str.strip().eq("")
    if missing_path.any():
        raise ValueError(f"Missing waveform paths for {int(missing_path.sum())} rows.")

    write_outputs(cohort, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records_diag_csv",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\records_w_diag_icd10.csv"),
    )
    parser.add_argument(
        "--record_list_csv",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\record_list.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_subset_manifest"),
    )
    parser.add_argument(
        "--download_dir",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_subset"),
    )
    parser.add_argument(
        "--base_url",
        default="https://physionet.org/files/mimic-iv-ecg/1.0",
    )
    parser.add_argument("--diag_col", default="all_diag_all")
    parser.add_argument("--n_cases", type=int, default=None)
    parser.add_argument("--n_controls", type=int, default=None)
    parser.add_argument("--adult_min_age", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--first_ecg_within_stay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require_ed_or_hosp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exclude_tia_history_from_controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--one_ecg_per_subject",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    build_manifest(parse_args())


if __name__ == "__main__":
    main()
