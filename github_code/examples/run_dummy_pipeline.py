"""Create dummy inputs and run the public experiment pipeline end to end.

This is an integration test for wiring, paths, and file formats. It uses tiny
synthetic data and one training epoch per model, so the metrics are meaningless.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "dummy_pipeline"
NUM_FOLDS = 5
FEATURE_DIM = 41
SIGNAL_CHANNELS = 12
SIGNAL_LENGTH = 5000


def run_command(cmd, env):
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError("Command failed with exit code %d: %s" % (proc.returncode, " ".join(cmd)))


def make_labels(n, fold_idx):
    return np.asarray([(i + fold_idx) % 2 for i in range(n)], dtype=np.int64)


def make_ids(fold_idx, split, n):
    file_names = np.asarray(["fold%d_%s_%03d.npy" % (fold_idx, split, i) for i in range(n)])
    patient_ids = np.asarray(["P%02d_%s_%03d" % (fold_idx, split, i) for i in range(n)])
    return file_names, patient_ids


def make_signal(n, fold_idx, split_idx):
    rng = np.random.RandomState(1000 + fold_idx * 10 + split_idx)
    return rng.normal(0.0, 1.0, size=(n, SIGNAL_CHANNELS, SIGNAL_LENGTH)).astype(np.float32)


def make_features(n, fold_idx, split_idx):
    rng = np.random.RandomState(2000 + fold_idx * 10 + split_idx)
    return rng.normal(0.0, 1.0, size=(n, FEATURE_DIM)).astype(np.float32)


def feature_columns():
    return ["feature_%02d" % i for i in range(FEATURE_DIM)]


def write_signal_data():
    root = RUN_ROOT / "data" / "signal_folds"
    for fold_idx in range(NUM_FOLDS):
        fold_dir = root / ("fold%d" % fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_idx, split in enumerate(["train", "val", "test"]):
            n = 4
            y = make_labels(n, fold_idx + split_idx)
            file_names, patient_ids = make_ids(fold_idx, split, n)
            np.savez_compressed(
                str(fold_dir / ("%s.npz" % split)),
                X=make_signal(n, fold_idx, split_idx),
                y=y,
                file_name=file_names,
                PatientID=patient_ids,
            )
    return root


def write_feature_csv_data():
    root = RUN_ROOT / "data" / "features"
    cols = feature_columns()
    feature_list_path = root / "feature_list.txt"
    root.mkdir(parents=True, exist_ok=True)
    feature_list_path.write_text("\n".join(cols) + "\n", encoding="utf-8")

    for fold_idx in range(NUM_FOLDS):
        fold_dir = root / ("fold%d" % fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_idx, split in enumerate(["train", "val", "test"]):
            n = 4
            y = make_labels(n, fold_idx + split_idx)
            file_names, patient_ids = make_ids(fold_idx, split, n)
            features = make_features(n, fold_idx, split_idx)
            df = pd.DataFrame(features, columns=cols)
            df.insert(0, "PatientID", patient_ids)
            df.insert(0, "label", y)
            df.insert(0, "file_name", file_names)
            df.to_csv(fold_dir / ("%s.csv" % split), index=False)

    summary_csv = root / "all_features_state.csv"
    pd.DataFrame({"variable": cols, "p_value": np.linspace(0.001, 0.2, len(cols))}).to_csv(summary_csv, index=False)
    return root, feature_list_path, summary_csv


def write_multimodal_data():
    root = RUN_ROOT / "data" / "multimodal_folds"
    cols = np.asarray(feature_columns())
    for fold_idx in range(NUM_FOLDS):
        fold_dir = root / ("fold%d" % fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_idx, split in enumerate(["train", "val", "test"]):
            n = 4
            y = make_labels(n, fold_idx + split_idx)
            file_names, patient_ids = make_ids(fold_idx, split, n)
            np.savez_compressed(
                str(fold_dir / ("%s.npz" % split)),
                signal=make_signal(n, fold_idx, split_idx),
                feature=make_features(n, fold_idx, split_idx),
                y=y,
                file_name=file_names,
                PatientID=patient_ids,
                feature_cols=cols,
            )
    return root


def main():
    if RUN_ROOT.exists():
        import shutil
        shutil.rmtree(str(RUN_ROOT))
    RUN_ROOT.mkdir(parents=True)

    signal_data_root = write_signal_data()
    feature_data_root, feature_list_path, summary_csv = write_feature_csv_data()
    multimodal_data_root = write_multimodal_data()

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    base_env["DEVICE"] = "cpu"
    base_env["MODEL_PRETRAINED"] = "0"
    base_env["NUM_FOLDS"] = str(NUM_FOLDS)

    signal_out = RUN_ROOT / "outputs" / "signal_only" / "efficientnet_b0"
    env = base_env.copy()
    env.update(
        {
            "SIGNAL_FOLD_ROOT": str(signal_data_root),
            "SAVE_ROOT": str(signal_out),
            "EPOCHS": "1",
            "BATCH_SIZE": "2",
        }
    )
    run_command([sys.executable, "scripts/train/train_signal_only.py"], env)

    feature_fusion_root = RUN_ROOT / "outputs" / "feature_only" / "late_fusion_ready"
    for fold_idx in range(NUM_FOLDS):
        feature_out = RUN_ROOT / "outputs" / "feature_only" / ("fold%d" % fold_idx)
        env = base_env.copy()
        env.update(
            {
                "TRAIN_PATH": str(feature_data_root / ("fold%d" % fold_idx) / "train.csv"),
                "VAL_PATH": str(feature_data_root / ("fold%d" % fold_idx) / "val.csv"),
                "TEST_PATH": str(feature_data_root / ("fold%d" % fold_idx) / "test.csv"),
                "FEATURE_LIST_PATH": str(feature_list_path),
                "OUT_ROOT": str(feature_out),
                "MAX_EPOCHS": "1",
                "BATCH_SIZE": "2",
                "PATIENCE": "1",
                "LATE_FUSION_FEATURE_SET": "all",
                "LATE_FUSION_OUT_DIR": str(feature_fusion_root / "all" / ("fold%d" % fold_idx)),
            }
        )
        run_command([sys.executable, "scripts/train/train_feature_only.py"], env)

    interaction_out = RUN_ROOT / "outputs" / "interaction_early_fusion"
    env = base_env.copy()
    env.update({"NUM_FOLDS": str(NUM_FOLDS), "MODEL_PRETRAINED": "0"})
    run_command(
        [
            sys.executable,
            "scripts/train/train_interaction_early_fusion.py",
            "--data_root",
            str(multimodal_data_root),
            "--out_dir",
            str(interaction_out),
            "--summary_csv",
            str(summary_csv),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--raw_feat_ch",
            str(FEATURE_DIM),
            "--num_workers",
            "0",
        ],
        env,
    )

    hybrid_out = RUN_ROOT / "outputs" / "hybrid_fusion"
    env = base_env.copy()
    env.update(
        {
            "FEATURE_SET": "all",
            "SIGNAL_MODE": "lossbest",
            "SIGNAL_ROOT": str(signal_out),
            "FEATURE_ROOT": str(feature_fusion_root),
            "EARLY_ROOT": str(interaction_out / "concat" / "late_fusion_ready" / "all"),
            "OUT_ROOT": str(hybrid_out),
        }
    )
    run_command([sys.executable, "scripts/fusion/make_hybrid_fusion.py"], env)

    required = [
        signal_out / "avg_5fold_metrics_extended.json",
        feature_fusion_root / "all" / "fold0" / "late_fusion_pack.npz",
        interaction_out / "concat" / "avg_5fold_metrics.json",
        hybrid_out / "all" / "lossbest" / "hybrid_logit_equal_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing expected outputs: %r" % missing)

    print("\nDummy pipeline completed.")
    print("Output root:", RUN_ROOT)


if __name__ == "__main__":
    main()
