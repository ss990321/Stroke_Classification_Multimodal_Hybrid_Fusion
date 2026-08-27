import os, json, random, numpy as np, torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, roc_curve
)
from scipy import stats

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:
    from torchvision.models import efficientnet_b0

    EfficientNet_B0_Weights = None


# ----------------
# Configuration
# ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_FOLD_ROOT = os.environ.get(
    "SIGNAL_FOLD_ROOT",
    str(PROJECT_ROOT / "data" / "signal_folds"),
)

DEVICE = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MODE = "binary"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "1e-3"))
SEED = 42
DETERMINISTIC = False
MODEL_PRETRAINED = os.environ.get("MODEL_PRETRAINED", "1").lower() not in {"0", "false", "no"}

EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.0

SAVE_ROOT = os.environ.get(
    "SAVE_ROOT",
    str(PROJECT_ROOT / "outputs" / "signal_only" / "efficientnet_b0"),
)
SINGLE_SIGNAL_NPZ = os.environ.get("SINGLE_SIGNAL_NPZ")
SPLIT_INDEX_ROOT = os.environ.get("SPLIT_INDEX_ROOT")

os.makedirs(SAVE_ROOT, exist_ok=True)


# ----------------
# Dataset definition
# ----------------
class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)   # (N, 12, L)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])   # (12, L)
        y = self.y[i]
        return x, y


_SINGLE_SIGNAL_CACHE = None


def load_single_signal_npz():
    global _SINGLE_SIGNAL_CACHE
    if _SINGLE_SIGNAL_CACHE is None:
        data = np.load(SINGLE_SIGNAL_NPZ, allow_pickle=True)
        _SINGLE_SIGNAL_CACHE = {
            "X": data["X"],
            "y": data["y"],
            "file_name": data["file_name"] if "file_name" in data.files else None,
            "PatientID": data["PatientID"] if "PatientID" in data.files else None,
        }
    return _SINGLE_SIGNAL_CACHE


def load_fold_npz(signal_fold_root, fold_idx, split):
    if SINGLE_SIGNAL_NPZ and SPLIT_INDEX_ROOT:
        data = load_single_signal_npz()
        idx_path = os.path.join(SPLIT_INDEX_ROOT, f"fold{fold_idx}", f"{split}_idx.npy")
        idx = np.load(idx_path).astype(np.int64)
        X = data["X"][idx]
        y = data["y"][idx]
        file_name = data["file_name"][idx] if data["file_name"] is not None else None
        patient_id = data["PatientID"][idx] if data["PatientID"] is not None else None
        return X, y, file_name, patient_id

    path = os.path.join(signal_fold_root, f"fold{fold_idx}", f"{split}.npz")
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    file_name = data["file_name"] if "file_name" in data.files else None
    patient_id = data["PatientID"] if "PatientID" in data.files else None

    return X, y, file_name, patient_id


def seed_everything(seed=42, deterministic=True):
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------
# ResNet18 pretrained wrapper
# ----------------
class EfficientNetB0SignalBinary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if EfficientNet_B0_Weights is None:
            backbone = efficientnet_b0(pretrained=pretrained)
        else:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)

        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            if pretrained:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        backbone.features[0][0] = new_conv
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 1)

        self.backbone = backbone

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)   # -> (B, 1, 12, L)
        logits = self.backbone(x).reshape(-1)
        return {"logits": logits}


# ----------------
# Utility
# ----------------
def count_pos_neg_from_loader(loader):
    pos, neg = 0, 0
    for _, y in loader:
        y = y.view(-1)
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    return pos, neg


def to_serializable(val):
    if isinstance(val, (np.generic,)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_thresholds_from_val(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return {"thr_youden": 0.5}

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx_youden = int(np.argmax(youden))
    return {"thr_youden": float(thr[idx_youden])}


def eval_at_threshold(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    out = {
        "threshold": float(thr),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred).tolist()
    }

    cm = np.array(out["cm"])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        out["fpr"] = float(fp / max(fp + tn, 1))
        out["tpr"] = float(tp / max(tp + fn, 1))
    else:
        out["fpr"] = float("nan")
        out["tpr"] = float("nan")
    return out


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_n = 0, 0
    y_true, y_prob = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            logits = out["logits"].reshape(-1)
            target = yb.float().reshape(-1)
            loss = criterion(logits, target)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * len(target)
            total_n += len(target)
            y_true.extend(target.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_pred_05 = (np.array(y_prob) >= 0.5).astype(int)
    y_true_np = np.array(y_true).astype(int)

    metrics = {
        "loss": float(total_loss / max(1, total_n)),
        "acc": float(accuracy_score(y_true_np, y_pred_05)),
        "f1": float(f1_score(y_true_np, y_pred_05, zero_division=0)),
        "precision": float(precision_score(y_true_np, y_pred_05, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_05, zero_division=0)),
        "brier": float(brier_score(y_true_np, y_prob)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true_np, y_prob))
    except Exception:
        metrics["auc"] = float("nan")

    metrics["cm"] = confusion_matrix(y_true_np, y_pred_05).tolist()
    metrics["y_true"], metrics["y_prob"] = y_true, y_prob
    return metrics


def mean_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, float("nan")
    std_err = stats.sem(values)
    ci_low, ci_high = stats.t.interval(alpha, len(values) - 1, loc=mean, scale=std_err)
    margin = float(mean - ci_low)
    return mean, margin


def summarize_5fold(results_all, tag):
    metrics = ["acc", "f1", "precision", "recall", "auc", "brier"]
    avg_result, ci_result = {}, {}
    for k in metrics:
        vals = [r.get(k, np.nan) for r in results_all]
        m, ci = mean_ci(vals, alpha=0.95)
        avg_result[k] = m
        ci_result[k] = ci
    return avg_result, ci_result


def summarize_thresholds(thr_pack, key):
    vals = [d.get(key, np.nan) for d in thr_pack]
    return mean_ci(vals, alpha=0.95)


def save_prediction_csv(out_path, y_true, y_prob, file_names=None, patient_ids=None):
    n = len(y_true)
    df = {
        "file_name": np.asarray(file_names).astype(str) if file_names is not None else np.array([f"sample_{i}" for i in range(n)]),
        "PatientID": np.asarray(patient_ids).astype(str) if patient_ids is not None else np.array([str(i) for i in range(n)]),
        "y_true": np.asarray(y_true, dtype=int),
        "prob": np.asarray(y_prob, dtype=float),
    }
    pd.DataFrame(df).to_csv(out_path, index=False)


# ----------------
# Main loop (5-fold cross-validation)
# ----------------
seed_everything(SEED, DETERMINISTIC)
print(f"[OK] Seed: {SEED} | Deterministic: {DETERMINISTIC}")

NUM_FOLDS = int(os.environ.get("NUM_FOLDS", "5"))
print(f"[OK] Number of folds: {NUM_FOLDS}")
print(f"[OK] Signal fold root: {SIGNAL_FOLD_ROOT}")

results_all_lossbest = []
thr_pack_lossbest = []

for fold_idx in range(NUM_FOLDS):
    print(f"\n======================")
    print(f"Starting fold {fold_idx+1}")
    print(f"======================")

    seed_everything(SEED + fold_idx, DETERMINISTIC)

    X_train, y_train, train_files, train_pids = load_fold_npz(SIGNAL_FOLD_ROOT, fold_idx, "train")
    X_val, y_val, val_files, val_pids = load_fold_npz(SIGNAL_FOLD_ROOT, fold_idx, "val")
    X_test, y_test, test_files, test_pids = load_fold_npz(SIGNAL_FOLD_ROOT, fold_idx, "test")

    print(f"fold{fold_idx} train: {X_train.shape}, y={y_train.shape}")
    print(f"fold{fold_idx} val  : {X_val.shape}, y={y_val.shape}")
    print(f"fold{fold_idx} test : {X_test.shape}, y={y_test.shape}")
    print("sample X_train shape:", X_train.shape)
    print("one sample shape:", X_train[0].shape)

    g = torch.Generator()
    g.manual_seed(SEED + fold_idx)

    train_loader = DataLoader(
        SignalDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        SignalDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        SignalDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        pin_memory=True,
    )

    model = EfficientNetB0SignalBinary(pretrained=MODEL_PRETRAINED).to(DEVICE)

    pos, neg = count_pos_neg_from_loader(train_loader)
    ratio = min(neg / max(pos, 1), 20.0)
    pos_weight = torch.tensor(ratio, dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    best_val_loss, best_epoch_loss = float("inf"), -1
    best_state_loss = None

    early_stop_counter = 0
    stopped_early = False
    stop_epoch = EPOCHS

    save_dir = os.path.join(SAVE_ROOT, f"fold_{fold_idx+1}")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out["logits"].reshape(-1), yb.float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, criterion)

        tr_loss = float(np.mean(train_losses))
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"Fold {fold_idx+1} | Epoch {epoch:02d} | "
            f"Train {tr_loss:.4f} | ValLoss {val_loss:.4f} | ValAUC {val_auc:.4f} | ValF1 {val_metrics['f1']:.3f}"
        )

        if val_loss < (best_val_loss - EARLY_STOP_MIN_DELTA):
            best_val_loss = val_loss
            best_epoch_loss = epoch
            early_stop_counter = 0
            best_state_loss = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            stopped_early = True
            stop_epoch = epoch
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f} @ epoch {best_epoch_loss})")
            break

    # -------------------
    # Evaluation: loss-best
    # -------------------
    if best_state_loss is not None:
        model.load_state_dict(best_state_loss)
    val_metrics_lossbest = evaluate(model, val_loader, criterion)
    test_metrics_lossbest = evaluate(model, test_loader, criterion)

    thr_lossbest = compute_thresholds_from_val(
        val_metrics_lossbest["y_true"], val_metrics_lossbest["y_prob"]
    )
    thr_pack_lossbest.append(thr_lossbest)

    test_thr_metrics_lossbest = {
        "at_thr_youden": eval_at_threshold(test_metrics_lossbest["y_true"], test_metrics_lossbest["y_prob"], thr_lossbest["thr_youden"]),
    }

    save_prediction_csv(
        os.path.join(save_dir, "preds_val_lossbest.csv"),
        val_metrics_lossbest["y_true"],
        val_metrics_lossbest["y_prob"],
        val_files,
        val_pids,
    )
    save_prediction_csv(
        os.path.join(save_dir, "preds_test_lossbest.csv"),
        test_metrics_lossbest["y_true"],
        test_metrics_lossbest["y_prob"],
        test_files,
        test_pids,
    )

    results_all_lossbest.append(test_metrics_lossbest)

    print(f"\n[OK] Fold {fold_idx+1} [LOSS-BEST] Test:")
    print({k: test_metrics_lossbest[k] for k in ["acc", "f1", "precision", "recall", "auc", "brier"]})
    print(f"   - Val thresholds: {thr_lossbest}")


# ----------------
# 5-fold mean results and validation-derived threshold summary
# ----------------
avg_lossbest, ci_lossbest = summarize_5fold(results_all_lossbest, "lossbest")

thr_summary = {
    "lossbest": {
        "thr_youden_mean_ci": summarize_thresholds(thr_pack_lossbest, "thr_youden"),
    }
}

print("\n======================")
print("5-fold mean results (LOSS-BEST, 95% CI margin)")
print("======================")
for k, v in avg_lossbest.items():
    print(f"{k}: {v:.4f} +/- {ci_lossbest[k]:.4f}")

print("\n======================")
print("5-fold Youden threshold summary (validation based, mean +/- 95% CI)")
print("======================")
print(json.dumps(thr_summary, ensure_ascii=False, indent=2, default=to_serializable))

out_path = os.path.join(SAVE_ROOT, "avg_5fold_metrics_extended.json")
save_dict = {
    "lossbest": {
        "mean": avg_lossbest,
        "95%_CI_margin": ci_lossbest
    },
    "threshold_summary_from_val": thr_summary
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(save_dict, f, ensure_ascii=False, indent=2, default=to_serializable)

print(f"\nSaved extended mean/CI and threshold summary: {out_path}")
