#!/usr/bin/env python3
"""Build a hemorrhagic-vs-ischemic dataset from the prepared MIMIC stroke data.

The stroke-vs-control experiment prepares 17,560 records; this script keeps the
stroke half, drops records carrying both a hemorrhagic and an ischemic code, and
relabels the rest (1 = hemorrhagic I60-I62, 0 = ischemic I63). Output uses the
same layout as prepare_mimic_external_data.py, so the rest of the pipeline runs
against it unchanged via --prepared_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


HEMORRHAGIC_PREFIXES = {"I60", "I61", "I62"}
ISCHEMIC_PREFIX = "I63"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", type=Path, required=True)
    parser.add_argument("--cohort_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stroke_prefixes(raw) -> set:
    if pd.isna(raw):
        return set()
    return {code[:3] for code in re.findall(r"I6[0-4][0-9A-Za-z.]*", str(raw).upper())}


def subtype_table(cohort_csv: Path) -> pd.DataFrame:
    """study_id -> subtype, keeping only records with exactly one of the two."""
    cohort = pd.read_csv(cohort_csv)
    cohort = cohort.loc[cohort["label"] == 1].copy()

    prefixes = cohort["stroke_codes"].map(stroke_prefixes)
    hemorrhagic = prefixes.map(lambda ps: bool(HEMORRHAGIC_PREFIXES & ps))
    ischemic = prefixes.map(lambda ps: ISCHEMIC_PREFIX in ps)

    both = int((hemorrhagic & ischemic).sum())
    neither = int((~hemorrhagic & ~ischemic).sum())
    print(f"[COHORT] stroke records={len(cohort)} both={both} neither={neither}")

    keep = hemorrhagic ^ ischemic
    table = cohort.loc[keep, ["study_id"]].copy()
    table["subtype"] = hemorrhagic.loc[keep].astype(int).to_numpy()
    return table


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    signal_path = args.out_dir / "mimic_external_signal.npz"
    if signal_path.exists() and not args.overwrite:
        raise SystemExit(f"{signal_path} already exists; pass --overwrite to replace it.")

    print("[LOAD]", args.prepared_dir / "mimic_external_signal.npz")
    npz = np.load(args.prepared_dir / "mimic_external_signal.npz", allow_pickle=True)
    features = pd.read_csv(args.prepared_dir / "mimic_external_features.csv")

    study_id = np.asarray(npz["study_id"]).astype(str)
    if len(features) != len(study_id):
        raise ValueError(f"row mismatch: features={len(features)} signal={len(study_id)}")
    if not (features["study_id"].astype(str).to_numpy() == study_id).all():
        raise ValueError("features csv and signal npz are not row-aligned on study_id")

    table = subtype_table(args.cohort_csv)
    mapping = dict(zip(table["study_id"].astype(str), table["subtype"].astype(int)))

    subtype = np.array([mapping.get(sid, -1) for sid in study_id], dtype=np.int64)
    mask = subtype >= 0
    kept = int(mask.sum())
    if kept == 0:
        raise ValueError("no records matched the cohort subtype table")
    print(f"[SELECT] kept {kept} of {len(mask)} records "
          f"(hemorrhagic={int((subtype[mask] == 1).sum())} ischemic={int((subtype[mask] == 0).sum())})")

    out_features = features.loc[mask].copy()
    out_features["label"] = subtype[mask]
    out_features.to_csv(args.out_dir / "mimic_external_features.csv", index=False)

    arrays = {"X": npz["X"][mask], "y": subtype[mask]}
    for key in ("file_name", "PatientID", "study_id"):
        if key in npz.files:
            arrays[key] = np.asarray(npz[key])[mask]
    print("[SAVE]", signal_path)
    np.savez_compressed(signal_path, **arrays)

    summary = {
        "source_prepared_dir": str(args.prepared_dir),
        "cohort_csv": str(args.cohort_csv),
        "task": "hemorrhagic (1) vs ischemic (0) stroke",
        "label_definition": {
            "1": "any of ICD-10 I60, I61, I62",
            "0": "ICD-10 I63",
            "excluded": "records carrying both, or neither",
        },
        "n_records": kept,
        "n_hemorrhagic": int((subtype[mask] == 1).sum()),
        "n_ischemic": int((subtype[mask] == 0).sum()),
        "n_patients": int(pd.Series(np.asarray(npz["PatientID"])[mask]).nunique())
        if "PatientID" in npz.files else None,
        "signal_shape": list(arrays["X"].shape),
    }
    with open(args.out_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
