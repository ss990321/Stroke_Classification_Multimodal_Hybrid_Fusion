#!/usr/bin/env python3
"""Create patient-level MIMIC folds for the public training pipeline.

Input is the prepared MIMIC external dataset produced by
prepare_mimic_external_data.py. Output matches the train scripts:

data/
  signal_folds/fold0/{train,val,test}.npz
  features/fold0/{train,val,test}.csv
  multimodal_folds/fold0/{train,val,test}.npz
  features/feature_list.txt
  features/all_features_state.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


DEFAULT_PREPARED_DIR = Path(r"C:\Users\ISY\Downloads\mimic_ecg_external_prepared")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "mimic_pipeline"
NUM_PARTITIONS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Leave unset for the full MIMIC dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fold files.",
    )
    parser.add_argument(
        "--index_only",
        action="store_true",
        help="Write split indices and feature CSVs only. Signal/multimodal train scripts can slice the original NPZ.",
    )
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_prepared(prepared_dir: Path, max_records: int | None, load_signal_array: bool):
    signal_path = require(prepared_dir / "mimic_external_signal.npz")
    feature_path = require(prepared_dir / "mimic_external_features.csv")

    print("[LOAD] signal:", signal_path)
    sig = np.load(signal_path, allow_pickle=True)
    x = sig["X"] if load_signal_array else None
    y = sig["y"].astype(np.int64)
    file_name = sig["file_name"].astype(str)
    patient_id = sig["PatientID"].astype(str)

    print("[LOAD] features:", feature_path)
    features_df = pd.read_csv(feature_path)

    if max_records is not None:
        if x is not None:
            x = x[:max_records]
        y = y[:max_records]
        file_name = file_name[:max_records]
        patient_id = patient_id[:max_records]
        features_df = features_df.iloc[:max_records].copy()

    if len(features_df) != len(y):
        raise ValueError(
            f"Length mismatch: signal rows={len(y)}, feature rows={len(features_df)}"
        )

    if "label" not in features_df.columns:
        raise ValueError("mimic_external_features.csv is missing 'label'")

    labels_from_csv = features_df["label"].astype(int).to_numpy()
    if not np.array_equal(y.astype(int), labels_from_csv):
        raise ValueError("Labels differ between signal npz and feature csv")

    return x, y, file_name, patient_id, features_df


def infer_feature_columns(features_df: pd.DataFrame) -> list[str]:
    metadata = {"file_name", "PatientID", "study_id", "label", "gender", "age"}
    cols = [c for c in features_df.columns if c not in metadata]
    if not cols:
        raise ValueError("No feature columns found")
    return cols


def make_partition_blocks(y: np.ndarray, patient_id: np.ndarray, seed: int):
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=NUM_PARTITIONS,
            shuffle=True,
            random_state=seed,
        )
        blocks = []
        dummy_x = np.zeros(len(y), dtype=np.int8)
        for _, block_idx in splitter.split(dummy_x, y, groups=patient_id):
            blocks.append(np.asarray(block_idx, dtype=np.int64))
        return blocks

    # Fallback: stratify at patient level and partition patients by label.
    rng = np.random.default_rng(seed)
    patient_frame = (
        pd.DataFrame({"PatientID": patient_id, "label": y})
        .groupby("PatientID", as_index=False)["label"]
        .agg(lambda s: int(round(float(np.mean(s)))))
    )

    block_patients: list[list[str]] = [[] for _ in range(NUM_PARTITIONS)]
    for label in sorted(patient_frame["label"].unique()):
        group_ids = patient_frame.loc[
            patient_frame["label"] == label, "PatientID"
        ].to_numpy()
        rng.shuffle(group_ids)
        for offset, pid in enumerate(group_ids):
            block_patients[offset % NUM_PARTITIONS].append(str(pid))

    blocks = []
    for patients in block_patients:
        mask = np.isin(patient_id, np.asarray(patients, dtype=str))
        blocks.append(np.flatnonzero(mask).astype(np.int64))
    return blocks


def split_indices(blocks: list[np.ndarray], fold_idx: int):
    val_block = 2 * fold_idx
    test_block = 2 * fold_idx + 1
    val_idx = blocks[val_block]
    test_idx = blocks[test_block]
    train_idx = np.concatenate(
        [
            block
            for idx, block in enumerate(blocks)
            if idx not in {val_block, test_block}
        ]
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {
        "n": int(len(y)),
        "control": int((y == 0).sum()),
        "stroke": int((y == 1).sum()),
    }


def ensure_can_write(paths: list[Path], overwrite: bool):
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist. Pass --overwrite to replace them:\n"
            + "\n".join(existing[:10])
        )


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_feature_stats(features_df: pd.DataFrame, feature_cols: list[str], out_path: Path):
    rows = []
    y = features_df["label"].astype(int).to_numpy()
    for col in feature_cols:
        values = pd.to_numeric(features_df[col], errors="coerce").to_numpy(dtype=float)
        control = values[y == 0]
        stroke = values[y == 1]
        try:
            stat, p_value = stats.ttest_ind(
                control,
                stroke,
                equal_var=False,
                nan_policy="omit",
            )
        except Exception:
            stat, p_value = np.nan, np.nan
        rows.append(
            {
                "variable": col,
                "statistic": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
            }
        )
    pd.DataFrame(rows).sort_values("p_value").to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    if args.num_folds != 5:
        raise ValueError("This splitter uses 10 partitions as val/test pairs, so num_folds must be 5.")

    x, y, file_name, patient_id, features_df = load_prepared(
        args.prepared_dir,
        args.max_records,
        load_signal_array=not args.index_only,
    )
    feature_cols = infer_feature_columns(features_df)

    out_root = args.out_root
    signal_root = out_root / "signal_folds"
    feature_root = out_root / "features"
    multimodal_root = out_root / "multimodal_folds"
    index_root = out_root / "split_indices"

    planned = []
    for fold_idx in range(args.num_folds):
        for split in ("train", "val", "test"):
            planned.append(index_root / f"fold{fold_idx}" / f"{split}_idx.npy")
            planned.append(feature_root / f"fold{fold_idx}" / f"{split}.csv")
            if not args.index_only:
                planned.append(signal_root / f"fold{fold_idx}" / f"{split}.npz")
                planned.append(multimodal_root / f"fold{fold_idx}" / f"{split}.npz")
    planned.extend([feature_root / "feature_list.txt", feature_root / "all_features_state.csv"])
    ensure_can_write(planned, args.overwrite)

    print("[SPLIT] building patient-level partitions")
    blocks = make_partition_blocks(y, patient_id, args.seed)
    if len(blocks) != NUM_PARTITIONS:
        raise RuntimeError(f"Expected {NUM_PARTITIONS} partition blocks, got {len(blocks)}")

    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "feature_list.txt").write_text(
        "\n".join(feature_cols) + "\n",
        encoding="utf-8",
    )
    write_feature_stats(features_df, feature_cols, feature_root / "all_features_state.csv")

    summary = {
        "prepared_dir": str(args.prepared_dir),
        "out_root": str(out_root),
        "num_records": int(len(y)),
        "num_patients": int(len(np.unique(patient_id))),
        "num_features": int(len(feature_cols)),
        "label_counts": label_counts(y),
        "split_strategy": "10 patient-level stratified group partitions; fold k uses partition 2k for val and 2k+1 for test",
        "folds": [],
    }

    feature_cols_array = np.asarray(feature_cols, dtype=object)
    for fold_idx in range(args.num_folds):
        idx_pack = split_indices(blocks, fold_idx)
        fold_summary = {"fold": int(fold_idx)}

        for split, idx in idx_pack.items():
            split_y = y[idx]
            fold_summary[split] = label_counts(split_y)

            index_fold_dir = index_root / f"fold{fold_idx}"
            index_fold_dir.mkdir(parents=True, exist_ok=True)
            np.save(index_fold_dir / f"{split}_idx.npy", idx.astype(np.int64, copy=False))

            if not args.index_only:
                if x is None:
                    raise RuntimeError("Signal array was not loaded")
                save_npz(
                    signal_root / f"fold{fold_idx}" / f"{split}.npz",
                    X=x[idx].astype(np.float32, copy=False),
                    y=split_y.astype(np.int64, copy=False),
                    file_name=file_name[idx],
                    PatientID=patient_id[idx],
                )

            split_df = features_df.iloc[idx].copy()
            feature_fold_dir = feature_root / f"fold{fold_idx}"
            feature_fold_dir.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(feature_fold_dir / f"{split}.csv", index=False)

            if not args.index_only:
                if x is None:
                    raise RuntimeError("Signal array was not loaded")
                feature = split_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                save_npz(
                    multimodal_root / f"fold{fold_idx}" / f"{split}.npz",
                    signal=x[idx].astype(np.float32, copy=False),
                    feature=feature,
                    y=split_y.astype(np.int64, copy=False),
                    file_name=file_name[idx],
                    PatientID=patient_id[idx],
                    feature_cols=feature_cols_array,
                )

        summary["folds"].append(fold_summary)
        print("[FOLD]", json.dumps(fold_summary, ensure_ascii=False))

    with (out_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved MIMIC pipeline data:")
    print("  SIGNAL_FOLD_ROOT =", signal_root)
    print("  FEATURE_ROOT     =", feature_root)
    print("  MULTI_DATA_ROOT  =", multimodal_root)
    print("  SPLIT_INDEX_ROOT =", index_root)
    print("  SUMMARY_CSV      =", feature_root / "all_features_state.csv")


if __name__ == "__main__":
    main()
