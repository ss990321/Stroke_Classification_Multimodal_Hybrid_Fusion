import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from stroke_multimodal.models.basic_early_fusion import BasicEarlyFusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PRETRAINED = os.environ.get("MODEL_PRETRAINED", "1").lower() not in {"0", "false", "no"}
SINGLE_SIGNAL_NPZ = os.environ.get("SINGLE_SIGNAL_NPZ")
SINGLE_FEATURE_CSV = os.environ.get("SINGLE_FEATURE_CSV")
SPLIT_INDEX_ROOT = os.environ.get("SPLIT_INDEX_ROOT")


class MultiModalDataset(Dataset):
    def __init__(self, signal: np.ndarray, feature: np.ndarray, y: np.ndarray):
        assert len(signal) == len(feature) == len(y), "length mismatch among signal, feature, y"
        signal = signal.astype(np.float32, copy=False)
        feature = feature.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)
        if signal.ndim == 3 and signal.shape[2] == 12:
            signal = np.transpose(signal, (0, 2, 1))
        self.X = signal
        self.F = feature
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.F[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def to_device(batch, device):
    return [t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch]


def safe_logit(p, eps=1e-7):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compute_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2 or not np.isfinite(y_prob).all():
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_threshold_from_val(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2 or not np.isfinite(y_prob).all():
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    finite = np.isfinite(thr)
    if not finite.any():
        return 0.5
    youden = tpr[finite] - fpr[finite]
    return float(thr[finite][int(np.argmax(youden))])


def eval_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": compute_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, f, y in loader:
        x, f, y = to_device((x, f, y), device)
        out = model(x, f)
        logits = out["logit"] if isinstance(out, dict) else out
        ys.append(y.detach().cpu().numpy())
        probs.append(torch.sigmoid(logits.view(-1)).detach().cpu().numpy())
    return np.concatenate(ys).astype(np.int64), np.concatenate(probs).astype(np.float64)


def sanitize_signal(X, tag=""):
    """Prepared MIMIC waveforms carry sparse NaN gaps; leaving them in makes every
    loss and prediction NaN. Fill them with the 0 mV baseline."""
    n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
    if n_bad:
        n_rec = int((~np.isfinite(X)).any(axis=tuple(range(1, X.ndim))).sum())
        print(f"[sanitize] {tag}: filled {n_bad} non-finite samples across {n_rec} records")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


_SINGLE_CACHE = None


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    metadata = {"file_name", "PatientID", "study_id", "label", "gender", "age"}
    return [c for c in frame.columns if c not in metadata]


def load_single_cache():
    global _SINGLE_CACHE
    if _SINGLE_CACHE is None:
        signal = np.load(SINGLE_SIGNAL_NPZ, allow_pickle=True)
        feature_frame = pd.read_csv(SINGLE_FEATURE_CSV)
        _SINGLE_CACHE = {
            "signal": signal["X"],
            "y": signal["y"].astype(np.int64),
            "file_name": signal["file_name"] if "file_name" in signal.files else None,
            "PatientID": signal["PatientID"] if "PatientID" in signal.files else None,
            "features": feature_frame,
            "feature_cols": feature_columns_from_frame(feature_frame),
        }
    return _SINGLE_CACHE


def load_split(data_root: str, fold_idx: int, split: str):
    if SINGLE_SIGNAL_NPZ and SINGLE_FEATURE_CSV and SPLIT_INDEX_ROOT:
        data = load_single_cache()
        idx = np.load(os.path.join(SPLIT_INDEX_ROOT, f"fold{fold_idx}", f"{split}_idx.npy")).astype(np.int64)
        feature = (
            data["features"]
            .iloc[idx][data["feature_cols"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        file_name = data["file_name"][idx] if data["file_name"] is not None else np.array([])
        patient_id = data["PatientID"][idx] if data["PatientID"] is not None else np.array([])
        signal = sanitize_signal(data["signal"][idx], f"fold{fold_idx}/{split}")
        return signal, feature, data["y"][idx], file_name.astype(str), patient_id.astype(str), data["feature_cols"]

    z = np.load(os.path.join(data_root, f"fold{fold_idx}", f"{split}.npz"), allow_pickle=True)
    return (
        sanitize_signal(z["signal"], f"fold{fold_idx}/{split}"),
        z["feature"],
        z["y"].astype(np.int64),
        z["file_name"].astype(str) if "file_name" in z.files else np.array([]),
        z["PatientID"].astype(str) if "PatientID" in z.files else np.array([]),
        [str(c) for c in z["feature_cols"]],
    )


def build_loaders(data_root: str, fold_idx: int, batch_size: int, num_workers: int):
    sig_tr, feat_tr, y_tr, fn_tr, pid_tr, feature_cols = load_split(data_root, fold_idx, "train")
    sig_va, feat_va, y_va, fn_va, pid_va, _ = load_split(data_root, fold_idx, "val")
    sig_te, feat_te, y_te, fn_te, pid_te, _ = load_split(data_root, fold_idx, "test")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    feat_tr = scaler.fit_transform(imputer.fit_transform(feat_tr)).astype(np.float32)
    feat_va = scaler.transform(imputer.transform(feat_va)).astype(np.float32)
    feat_te = scaler.transform(imputer.transform(feat_te)).astype(np.float32)

    loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(MultiModalDataset(sig_tr, feat_tr, y_tr), shuffle=True, **{k: v for k, v in loader_kwargs.items() if k != "shuffle"})
    val_loader = DataLoader(MultiModalDataset(sig_va, feat_va, y_va), **loader_kwargs)
    test_loader = DataLoader(MultiModalDataset(sig_te, feat_te, y_te), **loader_kwargs)
    meta = {
        "feature_cols": feature_cols,
        "val_file_name": fn_va,
        "val_patient_id": pid_va,
        "test_file_name": fn_te,
        "test_patient_id": pid_te,
    }
    return train_loader, val_loader, test_loader, meta


def train_one_fold(model, train_loader, val_loader, device, epochs, lr, patience, min_delta):
    y_train = train_loader.dataset.y
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, f, y in train_loader:
            x, f, y = to_device((x, f, y), device)
            logits = model(x, f)
            loss = criterion(logits.view(-1), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)

        y_val, p_val = predict(model, val_loader, device)
        val_loss = float(criterion(torch.from_numpy(safe_logit(p_val)).float().to(device), torch.from_numpy(y_val).float().to(device)).item())
        val_auc = compute_auc(y_val, p_val)
        train_loss = running / len(train_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
        print(f"[{epoch:02d}] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f}")

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_loss": best_loss, "best_epoch": best_epoch, "history": history}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan")
    err = stats.sem(values)
    if not np.isfinite(err) or err == 0:
        return mean, 0.0
    ci_low, _ = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=err)
    return mean, float(mean - ci_low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data" / "multimodal_folds"))
    parser.add_argument("--out_dir", default=os.environ.get("OUT_DIR", str(PROJECT_ROOT / "outputs" / "basic_early_fusion")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "30")))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "64")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-3")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", "5")))
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=int(os.environ.get("NUM_FOLDS", "5")))
    parser.add_argument("--feature_set", default="all", help="Name of the late-fusion pack subdirectory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--ecg_embedding_dim", type=int, default=512)
    parser.add_argument("--feature_embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    results = []

    for fold_idx in range(args.num_folds):
        print("\n======================")
        print(f"Starting basic early fusion fold {fold_idx + 1}")
        print("======================")
        set_seed(args.seed + fold_idx, args.deterministic)
        train_loader, val_loader, test_loader, meta = build_loaders(
            args.data_root,
            fold_idx,
            args.batch_size,
            args.num_workers,
        )
        model = BasicEarlyFusion(
            feature_dim=len(meta["feature_cols"]),
            ecg_embedding_dim=args.ecg_embedding_dim,
            feature_embedding_dim=args.feature_embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pretrained=MODEL_PRETRAINED,
        ).to(device)
        train_info = train_one_fold(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.patience,
            args.min_delta,
        )

        y_val, p_val = predict(model, val_loader, device)
        y_test, p_test = predict(model, test_loader, device)
        thr_youden = compute_threshold_from_val(y_val, p_val)
        metrics = eval_metrics(y_test, p_test, thr_youden)
        metrics["fold"] = fold_idx
        metrics["best_epoch"] = train_info["best_epoch"]
        results.append(metrics)

        fold_dir = Path(args.out_dir) / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "best_model_loss.pt")
        pd.DataFrame(
            {
                "file_name": meta["test_file_name"],
                "PatientID": meta["test_patient_id"],
                "label": y_test,
                "prob": p_test,
                "logit": safe_logit(p_test),
            }
        ).to_csv(fold_dir / "preds_test_lossbest.csv", index=False)
        pd.DataFrame(train_info["history"]).to_csv(fold_dir / "history.csv", index=False)

        late_dir = Path(args.out_dir) / "late_fusion_ready" / args.feature_set / f"fold{fold_idx}"
        late_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            late_dir / "late_fusion_pack.npz",
            val_probs=np.asarray(p_val, dtype=np.float32),
            val_logits=np.asarray(safe_logit(p_val), dtype=np.float32),
            val_y=np.asarray(y_val, dtype=np.int64),
            val_file_name=np.asarray(meta["val_file_name"]).astype(str),
            val_patient_id=np.asarray(meta["val_patient_id"]).astype(str),
            test_probs=np.asarray(p_test, dtype=np.float32),
            test_logits=np.asarray(safe_logit(p_test), dtype=np.float32),
            test_y=np.asarray(y_test, dtype=np.int64),
            test_file_name=np.asarray(meta["test_file_name"]).astype(str),
            test_patient_id=np.asarray(meta["test_patient_id"]).astype(str),
        )
        print(f"[OK] Fold {fold_idx} AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, threshold={thr_youden:.4f}")

    fold_df = pd.DataFrame(results)
    fold_df.to_csv(Path(args.out_dir) / "basic_early_fusion_foldwise.csv", index=False)

    summary = {"model": "basic_early_fusion", "folds": int(args.num_folds)}
    for metric in ["auc", "accuracy", "precision", "recall", "f1", "brier", "fpr"]:
        summary[f"{metric}_mean"], summary[f"{metric}_ci95_margin"] = mean_ci(fold_df[metric].values)
    pd.DataFrame([summary]).to_csv(Path(args.out_dir) / "basic_early_fusion_summary.csv", index=False)
    with open(Path(args.out_dir) / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ecg_embedding_dim": args.ecg_embedding_dim,
                "feature_embedding_dim": args.feature_embedding_dim,
                "fusion_input_dim": args.ecg_embedding_dim + args.feature_embedding_dim,
                "hidden_dim": args.hidden_dim,
                "classifier": [args.ecg_embedding_dim + args.feature_embedding_dim, args.hidden_dim, args.hidden_dim, 32, 1],
                "dropout": args.dropout,
                "model_pretrained": MODEL_PRETRAINED,
                "feature_count": len(meta["feature_cols"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nSaved:", Path(args.out_dir) / "basic_early_fusion_summary.csv")


if __name__ == "__main__":
    main()
