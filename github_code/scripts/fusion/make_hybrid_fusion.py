import os
from pathlib import Path
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)



PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SET = os.environ.get("FEATURE_SET", "all")   # all / top10 / top20 / valid_p
SIGNAL_MODE = os.environ.get("SIGNAL_MODE", "lossbest")

# =========================================================
# 0) Paths
# =========================================================
SIGNAL_ROOT = os.environ.get(
    "SIGNAL_ROOT",
    str(PROJECT_ROOT / "outputs" / "signal_only" / "efficientnet_b0"),
)

FEATURE_ROOT = os.environ.get("FEATURE_ROOT", str(PROJECT_ROOT / "outputs" / "feature_only" / "late_fusion_ready"))
# Example: FEATURE_ROOT/all/fold0/late_fusion_pack.npz
#     FEATURE_ROOT/valid_p/fold0/late_fusion_pack.npz

EARLY_ROOT = os.environ.get("EARLY_ROOT", str(PROJECT_ROOT / "outputs" / "interaction_early_fusion" / f"best_{FEATURE_SET}"))
# Example: EARLY_ROOT/fold0/late_fusion_pack.npz

OUT_ROOT = os.environ.get("OUT_ROOT", str(PROJECT_ROOT / "outputs" / "hybrid_fusion" / f"best_{FEATURE_SET}"))

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
NUM_FOLDS = 5
os.makedirs(OUT_ROOT, exist_ok=True)


# =========================================================
# 1) Utilities
# =========================================================
def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p, eps=1e-7):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def mean_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan

    mean = float(np.mean(values))
    std_err = stats.sem(values)
    ci_low, ci_high = stats.t.interval(alpha, len(values) - 1, loc=mean, scale=std_err)
    margin = float(mean - ci_low)
    return mean, margin


def eval_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    out = {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score(y_true, y_prob)),
    }

    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan

    return out


def compute_thresholds_from_val(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return {"thr_youden": 0.5}

    fpr, tpr, thr = roc_curve(y_true, y_prob)

    youden = tpr - fpr
    idx_youden = int(np.argmax(youden))
    return {"thr_youden": float(thr[idx_youden])}


# =========================================================
# 2) Loaders
# =========================================================
def load_signal_preds(fold_idx_zero_based, mode="lossbest", split="val"):
    """
    Signal-only output folders use fold_1 through fold_5.
    """
    fold_dir = os.path.join(SIGNAL_ROOT, f"fold_{fold_idx_zero_based + 1}")

    if mode != "lossbest":
        raise ValueError("Only lossbest signal predictions are supported.")
    fname = f"preds_{split}_lossbest.csv"

    path = os.path.join(fold_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Signal prediction file not found: {path}")

    df = pd.read_csv(path)

    rename_map = {}
    if "y_true" in df.columns:
        rename_map["y_true"] = "label"
    if "prob" in df.columns:
        rename_map["prob"] = "signal_prob"
    if "logit" in df.columns:
        rename_map["logit"] = "signal_logit"

    df = df.rename(columns=rename_map)

    required = ["file_name", "PatientID", "label", "signal_prob"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Signal prediction csv missing column '{c}': {path}")

    df["file_name"] = df["file_name"].astype(str)
    df["PatientID"] = df["PatientID"].astype(str)
    df["label"] = df["label"].astype(int)
    df["signal_prob"] = df["signal_prob"].astype(float)

    if "signal_logit" not in df.columns:
        df["signal_logit"] = safe_logit(df["signal_prob"].values)
    else:
        df["signal_logit"] = df["signal_logit"].astype(float)

    return df[["file_name", "PatientID", "label", "signal_prob", "signal_logit"]].copy()


def load_feature_preds(fold_idx_zero_based, feature_set="all", split="val"):
    """
    Feature-only output folders use fold0 through fold4.
    """
    fold_dir = os.path.join(FEATURE_ROOT, feature_set, f"fold{fold_idx_zero_based}")
    npz_path = os.path.join(fold_dir, "late_fusion_pack.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Feature npz not found: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)

    if split == "val":
        probs = z["val_probs"]
        logits = z["val_logits"]
        y = z["val_y"]
        file_names = z["val_file_name"]
        patient_ids = z["val_patient_id"]
    elif split == "test":
        probs = z["test_probs"]
        logits = z["test_logits"]
        y = z["test_y"]
        file_names = z["test_file_name"]
        patient_ids = z["test_patient_id"]
    else:
        raise ValueError("split must be val/test")

    df = pd.DataFrame({
        "file_name": file_names.astype(str),
        "PatientID": patient_ids.astype(str),
        "label": y.astype(int),
        "feat_prob": probs.astype(float),
        "feat_logit": logits.astype(float),
    })
    return df


def load_early_preds(fold_idx_zero_based, split="val"):
    """
    Early-fusion output folders use fold0 through fold4.
    """
    fold_dir = os.path.join(EARLY_ROOT, f"fold{fold_idx_zero_based}")
    npz_path = os.path.join(fold_dir, "late_fusion_pack.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Early npz not found: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)

    if split == "val":
        probs = z["val_probs"]
        logits = z["val_logits"]
        y = z["val_y"]
        file_names = z["val_file_name"]
        patient_ids = z["val_patient_id"]
    elif split == "test":
        probs = z["test_probs"]
        logits = z["test_logits"]
        y = z["test_y"]
        file_names = z["test_file_name"]
        patient_ids = z["test_patient_id"]
    else:
        raise ValueError("split must be val/test")

    df = pd.DataFrame({
        "file_name": file_names.astype(str),
        "PatientID": patient_ids.astype(str),
        "label": y.astype(int),
        "early_prob": probs.astype(float),
        "early_logit": logits.astype(float),
    })
    return df


def merge_three_modalities(df_signal, df_feat, df_early):
    merge_cols = ["file_name", "PatientID", "label"]

    df = pd.merge(df_signal, df_feat, on=merge_cols, how="inner")
    df = pd.merge(df, df_early, on=merge_cols, how="inner")

    if len(df) != len(df_signal) or len(df) != len(df_feat) or len(df) != len(df_early):
        print(
            f"[WARN] merge size mismatch: "
            f"signal={len(df_signal)}, feat={len(df_feat)}, early={len(df_early)}, merged={len(df)}"
        )

    df = df.sort_values(["PatientID", "file_name"]).reset_index(drop=True)
    return df


# =========================================================
# 3) Equal-logit hybrid fusion
# =========================================================
def fuse_hybrid_logit_equal(l_signal, l_feat, l_early):
    fused_logit = (
        np.asarray(l_signal) +
        np.asarray(l_feat) +
        np.asarray(l_early)
    ) / 3.0
    return sigmoid(fused_logit)


# =========================================================
# 4) Main
# =========================================================
all_fold_rows = []
fold_results_by_idx = {}

save_dir = os.path.join(OUT_ROOT, FEATURE_SET, SIGNAL_MODE)
os.makedirs(save_dir, exist_ok=True)

for fold_idx in range(NUM_FOLDS):
    print("\n==============================")
    print(f"Fold {fold_idx}")
    print("==============================")

    signal_val = load_signal_preds(fold_idx, mode=SIGNAL_MODE, split="val")
    feat_val = load_feature_preds(fold_idx, feature_set=FEATURE_SET, split="val")
    early_val = load_early_preds(fold_idx, split="val")
    val_df = merge_three_modalities(signal_val, feat_val, early_val)

    signal_test = load_signal_preds(fold_idx, mode=SIGNAL_MODE, split="test")
    feat_test = load_feature_preds(fold_idx, feature_set=FEATURE_SET, split="test")
    early_test = load_early_preds(fold_idx, split="test")
    test_df = merge_three_modalities(signal_test, feat_test, early_test)

    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    val_logit_equal = fuse_hybrid_logit_equal(
        val_df["signal_logit"],
        val_df["feat_logit"],
        val_df["early_logit"],
    )
    test_logit_equal = fuse_hybrid_logit_equal(
        test_df["signal_logit"],
        test_df["feat_logit"],
        test_df["early_logit"],
    )

    thr_logit_equal = compute_thresholds_from_val(
        y_val,
        val_logit_equal,
    )

    fold_results = {
        "hybrid_logit_equal": {
            "youden": eval_metrics(y_test, test_logit_equal, thr_logit_equal["thr_youden"]),
            "val_thresholds": thr_logit_equal,
        }
    }

    fold_results_by_idx[fold_idx] = {
        "results": fold_results,
    }

    all_fold_rows.append({
        "fold": fold_idx,
        "hybrid_logit_equal_auc": fold_results["hybrid_logit_equal"]["youden"]["auc"],
        "youden_threshold": thr_logit_equal["thr_youden"],
        "accuracy_youden": fold_results["hybrid_logit_equal"]["youden"]["accuracy"],
        "precision_youden": fold_results["hybrid_logit_equal"]["youden"]["precision"],
        "recall_youden": fold_results["hybrid_logit_equal"]["youden"]["recall"],
        "f1_youden": fold_results["hybrid_logit_equal"]["youden"]["f1"],
        "fpr_youden": fold_results["hybrid_logit_equal"]["youden"]["fpr"],
    })

    print(f"[Fold {fold_idx}] hybrid_logit_equal AUC={fold_results['hybrid_logit_equal']['youden']['auc']:.4f}")


# =========================================================
# 5) 5-fold summary
# =========================================================
summary_rows = []

for method_name in ["hybrid_logit_equal"]:
    auc_vals = []
    acc_vals = []
    prec_vals = []
    rec_vals = []
    f1_vals = []
    fpr_vals = []

    for fold_idx in range(NUM_FOLDS):
        res = fold_results_by_idx[fold_idx]["results"][method_name]["youden"]

        auc_vals.append(res["auc"])
        acc_vals.append(res["accuracy"])
        prec_vals.append(res["precision"])
        rec_vals.append(res["recall"])
        f1_vals.append(res["f1"])
        fpr_vals.append(res["fpr"])

    row = {"method": method_name}

    for metric_name, vals in [
        ("auc", auc_vals),
        ("accuracy", acc_vals),
        ("precision", prec_vals),
        ("recall", rec_vals),
        ("f1", f1_vals),
        ("fpr", fpr_vals),
    ]:
        m, ci = mean_ci(vals)
        row[f"{metric_name}_mean"] = m
        row[f"{metric_name}_ci95_margin"] = ci

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
fold_df = pd.DataFrame(all_fold_rows)

summary_df.to_csv(os.path.join(save_dir, "hybrid_logit_equal_summary.csv"), index=False)
fold_df.to_csv(os.path.join(save_dir, "hybrid_logit_equal_foldwise.csv"), index=False)

print("\n==============================")
print("5-Fold Summary")
print("==============================")
print(summary_df)

print(f"\nSaved to: {save_dir}")
