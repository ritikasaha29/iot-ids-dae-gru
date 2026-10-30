"""
demo.py — End-to-end demo with synthetic IoT traffic
======================================================
Generates a synthetic dataset that mimics NSL-KDD/N-BaIoT structure,
runs the FULL two-stage pipeline, and prints results — no download needed.

Run: python demo.py
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
Path("results").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# Mimics 20-feature network traffic (packet sizes, rates, byte counts, etc.)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_serror_rate",
    "num_failed_logins", "logged_in", "num_compromised"
]

ATTACK_TYPES = ["normal", "DDoS", "Botnet_Mirai", "Port_Scan", "BASHLITE"]

def generate_synthetic_dataset(n_normal=8000, n_attacks=2000):
    rows, labels = [], []

    # Normal IoT traffic — low rates, small packets, predictable patterns
    for _ in range(n_normal):
        row = RNG.normal(loc=[0.1, 200, 150, 0, 0, 0, 2, 10, 8, 0.02,
                               0.02, 0.01, 50, 45, 0.9, 0.05, 0.02, 0, 1, 0],
                         scale=[0.05, 80, 60, 0.01, 0.01, 0.01, 1, 5, 4, 0.02,
                                0.02, 0.01, 20, 18, 0.1, 0.03, 0.02, 0.01, 0.1, 0.1])
        rows.append(np.clip(row, 0, None))
        labels.append("normal")

    per_attack = n_attacks // (len(ATTACK_TYPES) - 1)
    attack_profiles = {
        "DDoS":          ([5, 50000, 100, 0, 0, 0, 1, 512, 512, 0.9, 0.9, 0.01, 255, 255, 1.0, 0, 0.9, 0, 0, 0], 1.5),
        "Botnet_Mirai":  ([0.5, 500, 500, 0, 5, 3, 8, 50, 50, 0.3, 0.3, 0.2, 200, 200, 0.5, 0.4, 0.3, 2, 0, 5], 1.0),
        "Port_Scan":     ([0.01, 50, 0, 0, 0, 0, 0, 511, 255, 0.5, 0.5, 0.5, 255, 100, 0.2, 0.8, 0.5, 0, 0, 0], 0.5),
        "BASHLITE":      ([1.0, 2000, 1000, 0, 2, 1, 5, 100, 80, 0.2, 0.2, 0.1, 150, 120, 0.6, 0.3, 0.2, 1, 1, 3], 0.8),
    }
    for atype, (mean, scale) in attack_profiles.items():
        for _ in range(per_attack):
            row = RNG.normal(loc=mean, scale=[s * scale for s in [1]*20])
            rows.append(np.clip(row, 0, None))
            labels.append(atype)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df["label"] = labels
    df["binary_label"] = (df["label"] != "normal").astype(int)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[Synthetic Data] {len(df):,} samples | "
          f"{df['binary_label'].sum():,} attacks | "
          f"{(df['binary_label']==0).sum():,} normal")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (as per methodology)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df):
    from scipy.stats import spearmanr
    X = df[FEATURE_NAMES].copy()
    y = df["binary_label"].values

    # Zero-variance removal
    zero_var = X.columns[X.var() == 0].tolist()
    X.drop(columns=zero_var, inplace=True)
    print(f"[Preprocess] Zero-var removed: {len(zero_var)} → {X.shape[1]} features")

    # Spearman correlation filter
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.8)]
    X.drop(columns=drop, inplace=True)
    print(f"[Preprocess] Corr filter: removed {len(drop)} → {X.shape[1]} features")

    # ANOVA F-Test
    from sklearn.feature_selection import f_classif, mutual_info_classif
    f_scores, _ = f_classif(X, y)
    mi_scores   = mutual_info_classif(X, y, random_state=42)
    combined    = f_scores / (f_scores.max() + 1e-9) + mi_scores / (mi_scores.max() + 1e-9)
    top_k = min(15, X.shape[1])
    top_idx = np.argsort(combined)[-top_k:]
    X = X.iloc[:, top_idx]
    print(f"[Preprocess] ANOVA+MI: kept top {top_k} → features: {X.columns.tolist()}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist(), scaler


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: DEEP AUTOENCODER (sklearn PCA as structural analog when no TF)
# ─────────────────────────────────────────────────────────────────────────────

class DeepAutoencoder:
    """
    PCA-based autoencoder analog — same conceptual role as a DAE:
    learns compression of normal traffic, flags high reconstruction error.
    Replace with a TF Autoencoder for full deep learning performance.
    """
    def __init__(self, n_components=5):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        self.threshold = None

    def fit(self, X_normal):
        self.pca.fit(X_normal)
        recon_errors = self._recon_error(X_normal)
        self.threshold = np.percentile(recon_errors, 95)
        print(f"[Stage 1 - Autoencoder] Trained on {len(X_normal):,} normal samples")
        print(f"  PCA variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        print(f"  Reconstruction error threshold (95th pct): {self.threshold:.6f}")
        return self

    def _recon_error(self, X):
        X_enc  = self.pca.transform(X)
        X_recon = self.pca.inverse_transform(X_enc)
        return np.mean(np.square(X - X_recon), axis=1)

    def encode(self, X):
        return self.pca.transform(X)

    def predict(self, X):
        errors = self._recon_error(X)
        return (errors > self.threshold).astype(int), errors


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: GRU CLASSIFIER (sklearn GBM with lag features as temporal analog)
# ─────────────────────────────────────────────────────────────────────────────

def make_lag_features(X_enc, seq_len=10):
    """
    Create temporal/lag features that mimic GRU's memory mechanism.
    For each sample i, concatenate the last seq_len encoded vectors.
    """
    n, d = X_enc.shape
    rows, indices = [], []
    for i in range(seq_len, n):
        window = X_enc[i - seq_len:i].flatten()   # seq_len * d features
        rows.append(window)
        indices.append(i)
    return np.array(rows), indices


class GRUClassifier:
    """
    GBM with lag features — behavioral/temporal analysis analog of GRU.
    """
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        )

    def fit(self, X_enc, y):
        X_lag, idx = make_lag_features(X_enc, self.seq_len)
        y_lag = y[idx]
        self.clf.fit(X_lag, y_lag)
        print(f"[Stage 2 - GRU/GBM] Trained on {len(X_lag):,} temporal sequences")
        return self

    def predict(self, X_enc):
        X_lag, idx = make_lag_features(X_enc, self.seq_len)
        preds = self.clf.predict(X_lag)
        probs = self.clf.predict_proba(X_lag)[:, 1]
        return preds, probs, idx


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, stage_name, latency_ms=None):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    fnr  = 1 - rec

    print(f"\n{'─'*55}")
    print(f"  {stage_name} Results")
    print(f"{'─'*55}")
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")
    if latency_ms is not None:
        status = "✓ PASS" if latency_ms < 40 else "✗ MISS"
        print(f"  Avg Latency       : {latency_ms:.3f} ms/sample  {status} (<40ms target)")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal','Attack'])}")
    return {"accuracy": round(acc,4), "f1": round(f1,4), "precision": round(prec,4),
            "recall": round(rec,4), "fnr": round(fnr,4),
            "latency_ms": round(latency_ms, 3) if latency_ms else None}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 65)
    print("  IoT IDS — HYBRID DAE + GRU — FULL DEMO")
    print("  Synthetic dataset mimicking NSL-KDD / N-BaIoT structure")
    print("=" * 65)

    # 1. Generate data
    df = generate_synthetic_dataset(n_normal=8000, n_attacks=2000)

    # 2. Preprocess
    X, y, features, scaler = preprocess(df)
    print(f"\n[Data] Final shape: {X.shape} | Attacks: {y.sum():,}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_normal = X_train[y_train == 0]

    all_metrics = {}

    # ── STAGE 1: Autoencoder ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STAGE 1 — DEEP AUTOENCODER (Anomaly Filter)")
    print("=" * 65)
    ae = DeepAutoencoder(n_components=6)
    ae.fit(X_normal)

    stage1_pred, recon_errors = ae.predict(X_test)
    t0 = time.time()
    for _ in range(100):  # benchmark latency
        ae.predict(X_test[:10])
    s1_latency = (time.time() - t0) / 100 / 10 * 1000
    all_metrics["stage1"] = evaluate(y_test, stage1_pred, "Stage 1 — Autoencoder", s1_latency)

    # Encode everything for Stage 2
    X_train_enc = ae.encode(X_train)
    X_test_enc  = ae.encode(X_test)

    # ── STAGE 2: GRU Classifier ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STAGE 2 — GRU CLASSIFIER (Behavioral Sequence Analysis)")
    print("=" * 65)
    gru = GRUClassifier(seq_len=10)
    gru.fit(X_train_enc, y_train)

    t0 = time.time()
    test_preds, test_probs, test_idx = gru.predict(X_test_enc)
    s2_latency = (time.time() - t0) / len(test_preds) * 1000
    y_test_seq = y_test[test_idx]
    all_metrics["stage2"] = evaluate(y_test_seq, test_preds, "Stage 2 — GRU Classifier", s2_latency)

    # ── COMBINED (Stage 1 gate → Stage 2 classifier) ──────────────────────────
    print("\n" + "=" * 65)
    print("  COMBINED PIPELINE — Stage 1 gate + Stage 2 classify")
    print("=" * 65)
    # Gate: only pass Stage 1 flags through Stage 2
    combined_pred = np.zeros(len(test_idx), dtype=int)
    s1_for_seq = stage1_pred[test_idx]
    for i, (s1, s2) in enumerate(zip(s1_for_seq, test_preds)):
        combined_pred[i] = s2 if s1 == 1 else 0
    t_combined = (s1_latency + s2_latency)
    all_metrics["combined"] = evaluate(y_test_seq, combined_pred, "Combined Pipeline", t_combined)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FULL SYSTEM SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<28} Stage1    Stage2    Combined")
    print(f"  {'─'*55}")
    for metric in ["accuracy","f1","precision","recall","fnr"]:
        s1 = all_metrics["stage1"].get(metric, "-")
        s2 = all_metrics["stage2"].get(metric, "-")
        co = all_metrics["combined"].get(metric, "-")
        print(f"  {metric:<28} {str(s1):<10}{str(s2):<10}{co}")
    lat = all_metrics["combined"].get("latency_ms", 0)
    print(f"\n  End-to-end latency: {lat:.3f} ms/sample  "
          + ("✓ PASS" if lat < 40 else "✗ MISS") + " (<40ms target)")

    # Save
    with open("results/demo_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Results saved to results/demo_metrics.json")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
