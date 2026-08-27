#!/usr/bin/env python3
"""Run the full public training pipeline on prepared MIMIC data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED_DIR = Path(r"C:\Users\ISY\Downloads\mimic_ecg_external_prepared")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mimic_pipeline"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mimic_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--epochs", default=os.environ.get("EPOCHS", "30"))
    parser.add_argument("--feature_epochs", default=os.environ.get("MAX_EPOCHS", "200"))
    parser.add_argument("--batch_size", default=os.environ.get("BATCH_SIZE", "64"))
    parser.add_argument("--model_pretrained", default=os.environ.get("MODEL_PRETRAINED", "1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["basic_early", "full_hybrid"],
        default="basic_early",
        help="basic_early trains only the two-branch early-fusion model. full_hybrid runs signal, feature, interaction early, then hybrid fusion.",
    )
    parser.add_argument("--overwrite_folds", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument(
        "--materialize_npz_folds",
        action="store_true",
        help="Write full signal/multimodal NPZ files per fold. Default uses split indices to avoid duplicating ECG arrays.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(str(part) for part in cmd)}"
        )


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["DEVICE"] = args.device
    env["MODEL_PRETRAINED"] = str(args.model_pretrained)
    env["NUM_FOLDS"] = str(args.num_folds)
    return env


def prepare_folds(args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "scripts/mimic/make_mimic_pipeline_folds.py",
        "--prepared_dir",
        args.prepared_dir,
        "--out_root",
        args.data_root,
        "--num_folds",
        str(args.num_folds),
        "--seed",
        str(args.seed),
    ]
    if args.max_records is not None:
        cmd.extend(["--max_records", str(args.max_records)])
    if args.overwrite_folds:
        cmd.append("--overwrite")
    if not args.materialize_npz_folds:
        cmd.append("--index_only")
    run_command(cmd, env)


def main() -> None:
    args = parse_args()
    args.prepared_dir = args.prepared_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    env = base_env(args)

    signal_data_root = args.data_root / "signal_folds"
    feature_data_root = args.data_root / "features"
    multimodal_data_root = args.data_root / "multimodal_folds"
    split_index_root = args.data_root / "split_indices"
    feature_list_path = feature_data_root / "feature_list.txt"
    summary_csv = feature_data_root / "all_features_state.csv"

    if not args.skip_prepare:
        prepare_folds(args, env)
    if args.prepare_only:
        return

    if args.mode == "basic_early":
        split_index_root = args.data_root / "split_indices"
        basic_env = env.copy()
        if not args.materialize_npz_folds:
            basic_env.update(
                {
                    "SINGLE_SIGNAL_NPZ": str(args.prepared_dir / "mimic_external_signal.npz"),
                    "SINGLE_FEATURE_CSV": str(args.prepared_dir / "mimic_external_features.csv"),
                    "SPLIT_INDEX_ROOT": str(split_index_root),
                }
            )
        out_dir = args.output_root / "basic_early_fusion"
        run_command(
            [
                sys.executable,
                "scripts/train/train_basic_early_fusion.py",
                "--data_root",
                multimodal_data_root,
                "--out_dir",
                out_dir,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                "0",
                "--num_folds",
                str(args.num_folds),
            ],
            basic_env,
        )
        print("\nMIMIC basic early-fusion pipeline completed.")
        print("Output root:", out_dir)
        print("Summary:", out_dir / "basic_early_fusion_summary.csv")
        return

    signal_out = args.output_root / "signal_only" / "efficientnet_b0"
    signal_env = env.copy()
    signal_env.update(
        {
            "SIGNAL_FOLD_ROOT": str(signal_data_root),
            "SAVE_ROOT": str(signal_out),
            "EPOCHS": str(args.epochs),
            "BATCH_SIZE": str(args.batch_size),
        }
    )
    if not args.materialize_npz_folds:
        signal_env.update(
            {
                "SINGLE_SIGNAL_NPZ": str(args.prepared_dir / "mimic_external_signal.npz"),
                "SPLIT_INDEX_ROOT": str(split_index_root),
            }
        )
    run_command([sys.executable, "scripts/train/train_signal_only.py"], signal_env)

    feature_fusion_root = args.output_root / "feature_only" / "late_fusion_ready"
    for fold_idx in range(args.num_folds):
        feature_out = args.output_root / "feature_only" / f"fold{fold_idx}"
        feature_env = env.copy()
        feature_env.update(
            {
                "TRAIN_PATH": str(feature_data_root / f"fold{fold_idx}" / "train.csv"),
                "VAL_PATH": str(feature_data_root / f"fold{fold_idx}" / "val.csv"),
                "TEST_PATH": str(feature_data_root / f"fold{fold_idx}" / "test.csv"),
                "FEATURE_LIST_PATH": str(feature_list_path),
                "OUT_ROOT": str(feature_out),
                "MAX_EPOCHS": str(args.feature_epochs),
                "BATCH_SIZE": str(args.batch_size),
                "LATE_FUSION_FEATURE_SET": "all",
                "LATE_FUSION_OUT_DIR": str(feature_fusion_root / "all" / f"fold{fold_idx}"),
            }
        )
        run_command([sys.executable, "scripts/train/train_feature_only.py"], feature_env)

    interaction_out = args.output_root / "interaction_early_fusion"
    interaction_env = env.copy()
    if not args.materialize_npz_folds:
        interaction_env.update(
            {
                "SINGLE_SIGNAL_NPZ": str(args.prepared_dir / "mimic_external_signal.npz"),
                "SINGLE_FEATURE_CSV": str(args.prepared_dir / "mimic_external_features.csv"),
                "SPLIT_INDEX_ROOT": str(split_index_root),
            }
        )
    run_command(
        [
            sys.executable,
            "scripts/train/train_interaction_early_fusion.py",
            "--data_root",
            multimodal_data_root,
            "--out_dir",
            interaction_out,
            "--summary_csv",
            summary_csv,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            "0",
            "--feature_set",
            "all",
            "--raw_feat_ch",
            "41",
        ],
        interaction_env,
    )

    hybrid_out = args.output_root / "hybrid_fusion"
    hybrid_env = env.copy()
    hybrid_env.update(
        {
            "FEATURE_SET": "all",
            "SIGNAL_MODE": "lossbest",
            "SIGNAL_ROOT": str(signal_out),
            "FEATURE_ROOT": str(feature_fusion_root),
            "EARLY_ROOT": str(interaction_out / "concat" / "late_fusion_ready" / "all"),
            "OUT_ROOT": str(hybrid_out),
        }
    )
    run_command([sys.executable, "scripts/fusion/make_hybrid_fusion.py"], hybrid_env)

    print("\nMIMIC pipeline completed.")
    print("Output root:", args.output_root)
    print("Hybrid summary:", hybrid_out / "all" / "lossbest" / "hybrid_logit_equal_summary.csv")


if __name__ == "__main__":
    main()
