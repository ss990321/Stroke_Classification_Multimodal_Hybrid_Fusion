import os
import random
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 0. Paths and options
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = os.environ.get("TRAIN_PATH", str(PROJECT_ROOT / "data" / "features" / "train.csv"))
VAL_PATH   = os.environ.get("VAL_PATH", str(PROJECT_ROOT / "data" / "features" / "val.csv"))
TEST_PATH  = os.environ.get("TEST_PATH", str(PROJECT_ROOT / "data" / "features" / "test.csv"))

FEATURE_LIST_PATH = os.environ.get("FEATURE_LIST_PATH", str(PROJECT_ROOT / "data" / "features" / "feature_list.txt"))

OUT_ROOT = os.environ.get("OUT_ROOT", str(PROJECT_ROOT / "outputs" / "feature_only" / "mlp41"))

USE_SEED = True
SEED = 42

DEVICE = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "1e-3"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "200"))
PATIENCE = int(os.environ.get("PATIENCE", "5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-3"))

MLP_ARCHITECTURE = (128, 128, 64, 32, 1)
MLP_HIDDEN = MLP_ARCHITECTURE[:-1]
MLP_DROPOUT = 0.2

SAVE_LATE_FUSION_PACK = True


# =========================================================
# 1. Seed
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if USE_SEED:
    set_seed(SEED)


# =========================================================
# 2. Metric utilities
# =========================================================

def eval_metrics_from_probs(y_true, p, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_pred = (p >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr) if np.isfinite(fpr) else np.nan,
        "specificity": float(spec) if np.isfinite(spec) else np.nan,
    }


def youden_threshold(y_true, p):
    fpr, tpr, thr = roc_curve(y_true, p)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    return float(thr[idx])


# =========================================================
# 3. MLP model
# =========================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT):
        super().__init__()

        layers = []
        prev = in_dim

        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers += [nn.Linear(prev, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def predict_logits_and_proba(model, X_np):
    model.eval()

    X = torch.from_numpy(X_np).float().to(DEVICE)
    logits = model(X)
    probs = torch.sigmoid(logits)

    return (
        logits.detach().cpu().numpy().astype(np.float32),
        probs.detach().cpu().numpy().astype(np.float32)
    )


def train_mlp_earlystop(X_tr, y_tr, X_va, y_va, in_dim):
    model = MLP(in_dim=in_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())

    pos_weight = torch.tensor(
        [neg / max(pos, 1)],
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            xva = torch.from_numpy(X_va).float().to(DEVICE)
            yva = torch.from_numpy(y_va).float().to(DEVICE)

            val_logits = model(xva)
            val_loss = float(criterion(val_logits, yva).detach().cpu().item())

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(epoch),
    }

    return model, train_info


# =========================================================
# 4. Load data
# =========================================================

df_tr = pd.read_csv(TRAIN_PATH)
df_va = pd.read_csv(VAL_PATH)
df_te = pd.read_csv(TEST_PATH)

with open(FEATURE_LIST_PATH, "r", encoding="utf-8-sig") as f:
    feature_list = [line.strip() for line in f.readlines() if line.strip()]

print("Loaded feature tables:")
print(f"Train rows: {len(df_tr)}")
print(f"Val rows  : {len(df_va)}")
print(f"Test rows : {len(df_te)}")
print(f"Model feature columns: {len(feature_list)}")
print(f"Metadata/label columns are not used as MLP inputs: file_name, PatientID, label")

required_cols = ["label"] + feature_list

for name, df_ in [("train", df_tr), ("val", df_va), ("test", df_te)]:
    missing_cols = [c for c in required_cols if c not in df_.columns]

    if missing_cols:
        raise ValueError(f"{name} data is missing columns: {missing_cols}")

print("\n[Label distribution]")
print("Train:")
print(df_tr["label"].value_counts().sort_index())
print("Val:")
print(df_va["label"].value_counts().sort_index())
print("Test:")
print(df_te["label"].value_counts().sort_index())


# =========================================================
# 5. X, y preparation
#    Use train-set medians for imputation and train-set statistics for scaling.
# =========================================================

X_tr_raw = df_tr[feature_list].apply(pd.to_numeric, errors="coerce")
X_va_raw = df_va[feature_list].apply(pd.to_numeric, errors="coerce")
X_te_raw = df_te[feature_list].apply(pd.to_numeric, errors="coerce")

y_tr = df_tr["label"].astype(int).values
y_va = df_va["label"].astype(int).values
y_te = df_te["label"].astype(int).values

print("\n[NaN count before imputation]")
nan_count = pd.DataFrame({
    "train": X_tr_raw.isna().sum(),
    "val": X_va_raw.isna().sum(),
    "test": X_te_raw.isna().sum(),
})
print(nan_count[nan_count.sum(axis=1) > 0])

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_tr_imp = imputer.fit_transform(X_tr_raw)
X_va_imp = imputer.transform(X_va_raw)
X_te_imp = imputer.transform(X_te_raw)

X_tr = scaler.fit_transform(X_tr_imp).astype(np.float32)
X_va = scaler.transform(X_va_imp).astype(np.float32)
X_te = scaler.transform(X_te_imp).astype(np.float32)

print("\n[NaN count after imputation/scaling]")
print("Train:", np.isnan(X_tr).sum())
print("Val  :", np.isnan(X_va).sum())
print("Test :", np.isnan(X_te).sum())


# =========================================================
# 6. Train
# =========================================================

model, train_info = train_mlp_earlystop(
    X_tr=X_tr,
    y_tr=y_tr,
    X_va=X_va,
    y_va=y_va,
    in_dim=X_tr.shape[1]
)

print("\n[Training info]")
print(train_info)


# =========================================================
# 7. Predict
# =========================================================

logits_tr, p_tr = predict_logits_and_proba(model, X_tr)
logits_va, p_va = predict_logits_and_proba(model, X_va)
logits_te, p_te = predict_logits_and_proba(model, X_te)

test_auc = roc_auc_score(y_te, p_te)


# =========================================================
# 8. Threshold selection on validation set
# =========================================================

thr_05 = 0.5
thr_youden = youden_threshold(y_va, p_va)

metrics_05 = eval_metrics_from_probs(y_te, p_te, thr=thr_05)
metrics_youden = eval_metrics_from_probs(y_te, p_te, thr=thr_youden)


# =========================================================
# 9. Save results
# =========================================================

os.makedirs(OUT_ROOT, exist_ok=True)

result_row = {
    "setting": "all_data_all_period_feature_only_mlp",
    "n_features": len(feature_list),
    "train_n": len(df_tr),
    "val_n": len(df_va),
    "test_n": len(df_te),

    "train_stroke_n": int(y_tr.sum()),
    "val_stroke_n": int(y_va.sum()),
    "test_stroke_n": int(y_te.sum()),

    "best_epoch": train_info["best_epoch"],
    "stopped_epoch": train_info["stopped_epoch"],
    "val_best_loss": train_info["best_val_loss"],

    "test_auc": float(test_auc),

    "thr0.5_accuracy": metrics_05["accuracy"],
    "thr0.5_precision": metrics_05["precision"],
    "thr0.5_recall": metrics_05["recall"],
    "thr0.5_f1": metrics_05["f1"],
    "thr0.5_fpr": metrics_05["fpr"],

    "youden_thr_val": float(thr_youden),
    "youden_accuracy_test": metrics_youden["accuracy"],
    "youden_precision_test": metrics_youden["precision"],
    "youden_recall_test": metrics_youden["recall"],
    "youden_f1_test": metrics_youden["f1"],
    "youden_fpr_test": metrics_youden["fpr"],
}

results_df = pd.DataFrame([result_row])

results_path = os.path.join(OUT_ROOT, "feature_only_results.csv")
results_df.to_csv(results_path, index=False)

print("\n[Test results]")
print(results_df.T)

# Save the late-fusion pack.
if SAVE_LATE_FUSION_PACK:
    late_fusion_feature_set = os.environ.get("LATE_FUSION_FEATURE_SET", "all")
    late_dir = os.environ.get(
        "LATE_FUSION_OUT_DIR",
        os.path.join(OUT_ROOT, "late_fusion_ready", late_fusion_feature_set),
    )
    os.makedirs(late_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(late_dir, "late_fusion_pack.npz"),

        train_logits=np.asarray(logits_tr, dtype=np.float32),
        val_logits=np.asarray(logits_va, dtype=np.float32),
        test_logits=np.asarray(logits_te, dtype=np.float32),

        train_probs=np.asarray(p_tr, dtype=np.float32),
        val_probs=np.asarray(p_va, dtype=np.float32),
        test_probs=np.asarray(p_te, dtype=np.float32),

        train_y=np.asarray(y_tr, dtype=np.int64),
        val_y=np.asarray(y_va, dtype=np.int64),
        test_y=np.asarray(y_te, dtype=np.int64),

        train_file_name=np.asarray(df_tr["file_name"].astype(str).values) if "file_name" in df_tr.columns else np.array([]),
        val_file_name=np.asarray(df_va["file_name"].astype(str).values) if "file_name" in df_va.columns else np.array([]),
        test_file_name=np.asarray(df_te["file_name"].astype(str).values) if "file_name" in df_te.columns else np.array([]),

        train_patient_id=np.asarray(df_tr["PatientID"].astype(str).values) if "PatientID" in df_tr.columns else np.array([]),
        val_patient_id=np.asarray(df_va["PatientID"].astype(str).values) if "PatientID" in df_va.columns else np.array([]),
        test_patient_id=np.asarray(df_te["PatientID"].astype(str).values) if "PatientID" in df_te.columns else np.array([]),
    )


print("\nSaved:")
print(results_path)
print(os.path.join(late_dir, "late_fusion_pack.npz"))
