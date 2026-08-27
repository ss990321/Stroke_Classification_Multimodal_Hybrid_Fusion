import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, brier_score_loss
)

from stroke_multimodal.models.interaction_early_fusion import InteractionEarlyFusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUDA = os.environ.get("DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu")
MULTI_DATA_ROOT = os.environ.get("MULTI_DATA_ROOT", str(PROJECT_ROOT / "data" / "multimodal_folds"))
epochs = int(os.environ.get("EPOCHS", "30"))
batch_size = int(os.environ.get("BATCH_SIZE", "64"))
learning_rate = float(os.environ.get("LR", "1e-3"))
fusion_type = "concat" # concat or mean
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
        x = torch.from_numpy(self.X[idx])
        f = torch.from_numpy(self.F[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, f, y


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_device(batch, device):
    return [t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def unpack_model_output(model_out):
    if isinstance(model_out, dict):
        return model_out["logit"], model_out
    return model_out, None


def select_feature_indices(saved_feature_cols, summary_csv: str, feature_set: str, p_thresh: float = 0.05):
    saved_feature_cols = [str(c) for c in saved_feature_cols]
    if feature_set == "all":
        selected_cols = [c for c in saved_feature_cols if c != "npz_idx"]
    else:
        rank_df = pd.read_csv(summary_csv)
        rank_df["variable"] = rank_df["variable"].astype(str)
        rank_df = rank_df.sort_values("p_value", ascending=True)
        if feature_set == "top10":
            wanted = rank_df["variable"].head(10).tolist()
        elif feature_set == "top20":
            wanted = rank_df["variable"].head(20).tolist()
        elif feature_set == "valid_p":
            wanted = rank_df.loc[rank_df["p_value"] < p_thresh, "variable"].tolist()
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        selected_cols = [c for c in wanted if c in saved_feature_cols]
    selected_idx = [saved_feature_cols.index(c) for c in selected_cols]
    print(f"[OK] feature_set={feature_set}")
    print(f"[OK] selected feature count = {len(selected_cols)}")
    print(f"[OK] selected features = {selected_cols}")
    return selected_idx, selected_cols


def sanitize_signal(X, tag=""):
    """Prepared MIMIC waveforms carry sparse NaN gaps; leaving them in makes every
    loss and prediction NaN. Fill them with the 0 mV baseline."""
    n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
    if n_bad:
        n_rec = int((~np.isfinite(X)).any(axis=tuple(range(1, X.ndim))).sum())
        print(f"[sanitize] {tag}: filled {n_bad} non-finite samples across {n_rec} records")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_multimodal_fold_npz_with_selected_cols(data_root: str, fold_idx: int, split: str, selected_cols: list):
    path = os.path.join(data_root, f"fold{fold_idx}", f"{split}.npz")
    data = np.load(path, allow_pickle=True)
    signal = sanitize_signal(data["signal"], f"fold{fold_idx}/{split}")
    feature = data["feature"]
    y = data["y"]
    file_name = data["file_name"] if "file_name" in data.files else None
    patient_id = data["PatientID"] if "PatientID" in data.files else None
    feature_cols = [str(c) for c in data["feature_cols"]]
    selected_idx = [feature_cols.index(c) for c in selected_cols]
    feature = feature[:, selected_idx]
    return signal, feature, y, file_name, patient_id


_SINGLE_MIMIC_CACHE = None


def load_single_mimic_cache():
    global _SINGLE_MIMIC_CACHE
    if _SINGLE_MIMIC_CACHE is None:
        sig = np.load(SINGLE_SIGNAL_NPZ, allow_pickle=True)
        features_df = pd.read_csv(SINGLE_FEATURE_CSV)
        metadata = {"file_name", "PatientID", "study_id", "label", "gender", "age"}
        feature_cols = [c for c in features_df.columns if c not in metadata]
        _SINGLE_MIMIC_CACHE = {
            "signal": sig["X"],
            "y": sig["y"].astype(np.int64),
            "file_name": sig["file_name"] if "file_name" in sig.files else None,
            "PatientID": sig["PatientID"] if "PatientID" in sig.files else None,
            "features_df": features_df,
            "feature_cols": feature_cols,
        }
    return _SINGLE_MIMIC_CACHE


def load_single_mimic_split(fold_idx: int, split: str, selected_cols: list):
    data = load_single_mimic_cache()
    idx_path = os.path.join(SPLIT_INDEX_ROOT, f"fold{fold_idx}", f"{split}_idx.npy")
    idx = np.load(idx_path).astype(np.int64)
    feat = (
        data["features_df"]
        .iloc[idx][selected_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    file_name = data["file_name"][idx] if data["file_name"] is not None else None
    patient_id = data["PatientID"][idx] if data["PatientID"] is not None else None
    signal = sanitize_signal(data["signal"][idx], f"fold{fold_idx}/{split}")
    return signal, feat, data["y"][idx], file_name, patient_id


def build_dataloaders_from_saved_fold(
    data_root: str,
    fold_idx: int,
    batch_size: int = 64,
    num_workers: int = 0,
    feature_set: str = "all",
    summary_csv: str = None,
    p_thresh: float = 0.05,
    scale_features: bool = True,
    generator=None,
):
    if SINGLE_SIGNAL_NPZ and SINGLE_FEATURE_CSV and SPLIT_INDEX_ROOT:
        saved_feature_cols = load_single_mimic_cache()["feature_cols"]
        _, selected_cols = select_feature_indices(saved_feature_cols, summary_csv, feature_set, p_thresh)

        signal_tr, feat_tr, y_tr, fn_tr, pid_tr = load_single_mimic_split(fold_idx, "train", selected_cols)
        signal_va, feat_va, y_va, fn_va, pid_va = load_single_mimic_split(fold_idx, "val", selected_cols)
        signal_te, feat_te, y_te, fn_te, pid_te = load_single_mimic_split(fold_idx, "test", selected_cols)
    else:
        train_path = os.path.join(data_root, f"fold{fold_idx}", "train.npz")
        train_npz = np.load(train_path, allow_pickle=True)
        saved_feature_cols = train_npz["feature_cols"]

        _, selected_cols = select_feature_indices(saved_feature_cols, summary_csv, feature_set, p_thresh)
        signal_tr, feat_tr, y_tr, fn_tr, pid_tr = load_multimodal_fold_npz_with_selected_cols(data_root, fold_idx, "train", selected_cols)
        signal_va, feat_va, y_va, fn_va, pid_va = load_multimodal_fold_npz_with_selected_cols(data_root, fold_idx, "val", selected_cols)
        signal_te, feat_te, y_te, fn_te, pid_te = load_multimodal_fold_npz_with_selected_cols(data_root, fold_idx, "test", selected_cols)

    scaler = None
    imputer = None
    if scale_features:
        imputer = SimpleImputer(strategy="median")
        feat_tr = imputer.fit_transform(feat_tr)
        feat_va = imputer.transform(feat_va)
        feat_te = imputer.transform(feat_te)
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(feat_tr)
        feat_va = scaler.transform(feat_va)
        feat_te = scaler.transform(feat_te)

    ds_tr = MultiModalDataset(signal_tr, feat_tr, y_tr)
    ds_va = MultiModalDataset(signal_va, feat_va, y_va)
    ds_te = MultiModalDataset(signal_te, feat_te, y_te)

    common_loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    dl_tr = DataLoader(ds_tr, shuffle=True, **common_loader_kwargs)
    dl_va = DataLoader(ds_va, shuffle=False, **common_loader_kwargs)
    dl_te = DataLoader(ds_te, shuffle=False, **common_loader_kwargs)

    meta = {
        "feat_cols": selected_cols,
        "train_len": len(ds_tr),
        "val_len": len(ds_va),
        "test_len": len(ds_te),
        "class_dist_train": {"n0": int((y_tr == 0).sum()), "n1": int((y_tr == 1).sum())},
        "scaler_used": bool(scale_features),
        "train_file_name": np.asarray(fn_tr).astype(str) if fn_tr is not None else np.array([]),
        "val_file_name": np.asarray(fn_va).astype(str) if fn_va is not None else np.array([]),
        "test_file_name": np.asarray(fn_te).astype(str) if fn_te is not None else np.array([]),
        "train_patient_id": np.asarray(pid_tr).astype(str) if pid_tr is not None else np.array([]),
        "val_patient_id": np.asarray(pid_va).astype(str) if pid_va is not None else np.array([]),
        "test_patient_id": np.asarray(pid_te).astype(str) if pid_te is not None else np.array([]),
        "imputer_used": imputer is not None,
    }
    return dl_tr, dl_va, dl_te, meta, scaler


def evaluate_probs(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, f, y in loader:
            x, f, y = to_device((x, f, y), device)
            model_out = model(x, f)
            logits, _ = unpack_model_output(model_out)
            probs = torch.sigmoid(logits.view(-1))
            all_y.append(y.detach().cpu().numpy())
            all_p.append(probs.detach().cpu().numpy())
    return {
        "y_true": np.concatenate(all_y).astype(np.int64),
        "y_prob": np.concatenate(all_p).astype(np.float64),
    }


def safe_logit(p, eps=1e-7):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compute_metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "fpr": float(fpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compute_thresholds_from_val(y_true, y_prob):
    eps = 1e-12
    best_youden = {"threshold": 0.5, "youden": -1}
    for thr in np.linspace(0.0, 1.0, 1001):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + eps)
        fpr = fp / (fp + tn + eps)
        youden = sens - fpr
        if youden > best_youden["youden"]:
            best_youden = {"threshold": float(thr), "youden": float(youden)}
    return {"youden_best": best_youden}


def compute_auc(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")


def compute_brier(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))


def evaluate_full(model, loader, device, thresholds: Optional[Dict[str, float]] = None):
    pred_pack = evaluate_probs(model, loader, device)
    y_true, y_prob = pred_pack["y_true"], pred_pack["y_prob"]
    base = {
        "auc": compute_auc(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob),
        "n": int(len(y_true)),
        "pos": int((y_true == 1).sum()),
        "neg": int((y_true == 0).sum()),
    }
    out = {"summary": base, "by_threshold": {}}
    if thresholds is None:
        thresholds = {"thr_0.5": 0.5}
    for name, thr in thresholds.items():
        out["by_threshold"][name] = compute_metrics_at_threshold(y_true, y_prob, thr)
    out["y_true"] = y_true
    out["y_prob"] = y_prob
    return out


def train_loop(model, train_loader, val_loader, device, epochs, lr, out_dir, patience=10, min_delta=0.0):
    ensure_dir(out_dir)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y_train = train_loader.dataset.y
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.detach().cpu().numpy()

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss = float("inf")
    best_ep_loss = -1
    best_state_loss = None
    train_losses, val_losses, val_aucs, val_briers = [], [], [], []
    no_improve_count = 0
    stopped_early = False
    stop_epoch = epochs

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, f, y in train_loader:
            x, f, y = to_device((x, f, y), device)
            logits, _ = unpack_model_output(model(x, f))
            loss = criterion(logits.view(-1), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        tr_loss = running / len(train_loader.dataset)
        train_losses.append(tr_loss)

        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for x, f, y in val_loader:
                x, f, y = to_device((x, f, y), device)
                logits, _ = unpack_model_output(model(x, f))
                loss = criterion(logits.view(-1), y.float())
                v_running += loss.item() * y.size(0)
        va_loss = v_running / len(val_loader.dataset)
        val_losses.append(va_loss)

        pred_pack = evaluate_probs(model, val_loader, device)
        va_auc = compute_auc(pred_pack["y_true"], pred_pack["y_prob"])
        va_brier = compute_brier(pred_pack["y_true"], pred_pack["y_prob"])
        val_aucs.append(va_auc)
        val_briers.append(va_brier)
        print(f"[{ep:02d}] TrainLoss={tr_loss:.4f} | ValLoss={va_loss:.4f} | ValAUC={va_auc:.4f} | ValBrier={va_brier:.4f}")

        improved = va_loss < (best_loss - min_delta)
        if improved:
            best_loss = va_loss
            best_ep_loss = ep
            no_improve_count = 0
            best_state_loss = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            stopped_early = True
            stop_epoch = ep
            print(f"Early stopping triggered at epoch {ep} (best val loss: {best_loss:.4f} @ epoch {best_ep_loss})")
            break

    return {
        "best_loss": float(best_loss),
        "best_ep_loss": int(best_ep_loss),
        "best_state_loss": best_state_loss,
        "stopped_early": bool(stopped_early),
        "stop_epoch": int(stop_epoch),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "history": {
            "train_loss": [float(x) for x in train_losses],
            "val_loss": [float(x) for x in val_losses],
            "val_auc": [float(x) if not np.isnan(x) else None for x in val_aucs],
            "val_brier": [float(x) for x in val_briers],
        },
    }

# Feature values are embedded before fusion.
tab_emb_dim = 128
raw_emb_dim = 128
raw_feat_ch = 41
raw_downsample_stride = 25
feature_num = "all" # ["top10", "top20", "valid_p", "all"]
version = "default"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=MULTI_DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", str(PROJECT_ROOT / "outputs" / "interaction_early_fusion" / version)))
    parser.add_argument("--feature_set", type=str, default=f"{feature_num}", choices=["top10", "top20", "valid_p", "all"])
    parser.add_argument("--summary_csv", type=str, default=os.environ.get("SUMMARY_CSV", str(PROJECT_ROOT / "data" / "features" / "all_features_state.csv")))
    parser.add_argument("--p_thresh", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--scale_features", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--signal_emb_dim", type=int, default=512)
    parser.add_argument("--tab_emb_dim", type=int, default=tab_emb_dim)
    parser.add_argument("--raw_emb_dim", type=int, default=raw_emb_dim)
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--fusion_type", type=str, default=fusion_type, choices=["concat", "mean"])
    parser.add_argument("--raw_nhead", type=int, default=4)
    parser.add_argument("--raw_num_layers", type=int, default=2)
    parser.add_argument("--raw_ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--raw_feat_ch", type=int, default=raw_feat_ch)
    parser.add_argument("--raw_downsample_stride", type=int, default=raw_downsample_stride)
    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, args.fusion_type)
    ensure_dir(args.out_dir)
    set_seed(args.seed, args.deterministic)
    results_all = []
    num_folds = int(os.environ.get("NUM_FOLDS", "5"))

    print(f"[OK] Number of folds: {num_folds}")
    print(f"[OK] Feature set: {args.feature_set}")
    print(f"[OK] Fusion type: {args.fusion_type}")
    print(f"[OK] Scale features: {args.scale_features}")
    print(f"[OK] Seed: {args.seed} | Deterministic: {args.deterministic}")

    for fold_idx in range(num_folds):
        print("\n======================")
        print(f"Starting fold {fold_idx + 1}")
        print("======================")
        set_seed(args.seed + fold_idx, args.deterministic)
        g = torch.Generator(); g.manual_seed(args.seed + fold_idx)
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_idx + 1}")
        ensure_dir(fold_dir)

        dl_tr, dl_va, dl_te, meta, scaler = build_dataloaders_from_saved_fold(
            data_root=args.data_root,
            fold_idx=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_set=args.feature_set,
            summary_csv=args.summary_csv,
            p_thresh=args.p_thresh,
            scale_features=args.scale_features,
            generator=g,
        )

        sig_in_ch = dl_tr.dataset.X.shape[1]
        feat_in_dim = dl_tr.dataset.F.shape[1]
        device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

        model = InteractionEarlyFusion(
            in_ch_signal=sig_in_ch,
            tab_dim=feat_in_dim,
            signal_emb_dim=args.signal_emb_dim,
            tab_emb_dim=args.tab_emb_dim,
            raw_emb_dim=args.raw_emb_dim,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            raw_nhead=args.raw_nhead,
            raw_num_layers=args.raw_num_layers,
            raw_ff_dim=args.raw_ff_dim,
            raw_feat_ch=args.raw_feat_ch,
            raw_downsample_stride=args.raw_downsample_stride,
            fusion_type=args.fusion_type,
            pretrained=MODEL_PRETRAINED,
            return_dict=False,
        )

        train_info = train_loop(model, dl_tr, dl_va, device, args.epochs, args.lr, fold_dir, patience=args.patience, min_delta=args.min_delta)

        def build_eval_model(return_dict: bool):
            return InteractionEarlyFusion(
                in_ch_signal=sig_in_ch,
                tab_dim=feat_in_dim,
                signal_emb_dim=args.signal_emb_dim,
                tab_emb_dim=args.tab_emb_dim,
                raw_emb_dim=args.raw_emb_dim,
                fusion_dim=args.fusion_dim,
                dropout=args.dropout,
                raw_nhead=args.raw_nhead,
                raw_num_layers=args.raw_num_layers,
                raw_ff_dim=args.raw_ff_dim,
                raw_feat_ch=args.raw_feat_ch,
                raw_downsample_stride=args.raw_downsample_stride,
                fusion_type=args.fusion_type,
                pretrained=MODEL_PRETRAINED,
                return_dict=return_dict,
            )

        model_loss = build_eval_model(return_dict=True)
        model_loss.load_state_dict(train_info["best_state_loss"])
        model_loss.to(device)

        val_out_loss = evaluate_full(model_loss, dl_va, device, thresholds={"thr_0.5": 0.5})
        thr_pack_loss = compute_thresholds_from_val(val_out_loss["y_true"], val_out_loss["y_prob"])
        thr_youden = thr_pack_loss["youden_best"]["threshold"]
        test_out_loss = evaluate_full(model_loss, dl_te, device, thresholds={"youden": thr_youden})

        late_dir = os.path.join(args.out_dir, "late_fusion_ready", args.feature_set, f"fold{fold_idx}")
        ensure_dir(late_dir)
        np.savez_compressed(
            os.path.join(late_dir, "late_fusion_pack.npz"),
            val_probs=np.asarray(val_out_loss["y_prob"], dtype=np.float32),
            val_logits=np.asarray(safe_logit(val_out_loss["y_prob"]), dtype=np.float32),
            val_y=np.asarray(val_out_loss["y_true"], dtype=np.int64),
            val_file_name=meta["val_file_name"],
            val_patient_id=meta["val_patient_id"],
            test_probs=np.asarray(test_out_loss["y_prob"], dtype=np.float32),
            test_logits=np.asarray(safe_logit(test_out_loss["y_prob"]), dtype=np.float32),
            test_y=np.asarray(test_out_loss["y_true"], dtype=np.int64),
            test_file_name=meta["test_file_name"],
            test_patient_id=meta["test_patient_id"],
        )

        agg = {
            "auc": test_out_loss["summary"]["auc"],
            "brier": test_out_loss["summary"]["brier"],
            "accuracy_youden": test_out_loss["by_threshold"]["youden"]["accuracy"],
            "f1_youden": test_out_loss["by_threshold"]["youden"]["f1"],
            "precision_youden": test_out_loss["by_threshold"]["youden"]["precision"],
            "recall_youden": test_out_loss["by_threshold"]["youden"]["recall"],
            "fpr_youden": test_out_loss["by_threshold"]["youden"]["fpr"],
            "thr_youden": thr_youden,
        }
        results_all.append(agg)
        print(f"[OK] Fold {fold_idx + 1} AUC={agg['auc']:.4f}, Brier={agg['brier']:.4f}, F1(Youden)={agg['f1_youden']:.4f}")

    metrics = [k for k in results_all[0].keys()]
    avg_result, ci_result = {}, {}
    for k in metrics:
        values = np.array([r[k] for r in results_all if k in r and not pd.isna(r[k])], dtype=float)
        if len(values) == 0:
            avg_result[k] = np.nan
            ci_result[k] = np.nan
            continue
        mean = float(np.mean(values))
        avg_result[k] = mean
        if len(values) < 2:
            ci_result[k] = np.nan
            continue
        std_err = stats.sem(values)
        if np.isnan(std_err) or std_err == 0:
            ci_result[k] = 0.0
            continue
        ci_low, _ = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=std_err)
        ci_result[k] = float(mean - ci_low)

    print("\n======================")
    print("5-fold mean results (95% CI margin)")
    print("======================")
    for k in metrics:
        print(f"{k}: {avg_result[k]:.4f} +/- {ci_result[k]:.4f}")

    avg_result_path = os.path.join(args.out_dir, "avg_5fold_metrics.json")
    with open(avg_result_path, "w", encoding="utf-8") as f:
        json.dump({"mean": avg_result, "95%_CI_margin": ci_result}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved mean and 95% CI margin results: {avg_result_path}")


if __name__ == "__main__":
    main()
