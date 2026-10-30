"""
=============================================================================
IoT Intrusion Detection System — Hybrid Deep Autoencoder + GRU Model
=============================================================================
Architecture:
  Stage 1 — Deep Autoencoder (DAE): Learns normal traffic; flags anomalies
             by reconstruction error threshold.
  Stage 2 — GRU Classifier: Analyses flagged traffic sequences to classify
             attack type (DDoS, Botnet, Port Scan, etc.)

Supported datasets (auto-detected):
  • NSL-KDD  (kddtrain+.txt / kddtest+.txt)
  • N-BaIoT  (CSV files per device/attack folder)
  • Kitsune   (any *_dataset.csv with numeric features)

Author: IoT-IDS Project
=============================================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ── Optional heavy deps — graceful fallback for environments without TF ──────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADER — handles NSL-KDD, N-BaIoT, Kitsune
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """Auto-detects and loads any of the three supported datasets."""

    NSL_KDD_COLS = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
    ]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def detect_dataset(self) -> str:
        files = list(self.data_dir.rglob("*.csv")) + list(self.data_dir.rglob("*.txt"))
        names = [f.name.lower() for f in files]
        if any("kdd" in n for n in names):
            return "NSL-KDD"
        if any("n-baiot" in str(f).lower() or "mirai" in n or "bashlite" in n for f, n in zip(files, names)):
            return "N-BaIoT"
        if any("kitsune" in n or "dataset" in n for n in names):
            return "Kitsune"
        return "Unknown"

    def load(self):
        dataset_type = self.detect_dataset()
        print(f"[DataLoader] Detected dataset: {dataset_type}")
        if dataset_type == "NSL-KDD":
            return self._load_nslkdd()
        elif dataset_type == "N-BaIoT":
            return self._load_nbaiot()
        elif dataset_type == "Kitsune":
            return self._load_kitsune()
        else:
            return self._load_generic()

    def _load_nslkdd(self):
        train_files = list(self.data_dir.rglob("*train*.txt")) + list(self.data_dir.rglob("*train*.csv"))
        test_files  = list(self.data_dir.rglob("*test*.txt"))  + list(self.data_dir.rglob("*test*.csv"))

        def read(f):
            df = pd.read_csv(f, header=None, names=self.NSL_KDD_COLS)
            df.drop(columns=["difficulty"], inplace=True, errors="ignore")
            return df

        if train_files and test_files:
            train_df = read(train_files[0])
            test_df  = read(test_files[0])
            df = pd.concat([train_df, test_df], ignore_index=True)
        elif train_files:
            df = read(train_files[0])
        else:
            raise FileNotFoundError("No NSL-KDD files found in " + str(self.data_dir))

        # Binary label: normal=0, attack=1
        df["binary_label"] = df["label"].apply(lambda x: 0 if x.strip().lower() == "normal" else 1)
        print(f"  Loaded NSL-KDD: {len(df):,} rows | {df['binary_label'].sum():,} attacks")
        return df, "label", "binary_label"

    def _load_nbaiot(self):
        dfs = []
        for csv_file in sorted(self.data_dir.rglob("*.csv")):
            tmp = pd.read_csv(csv_file)
            # N-BaIoT: label from directory name
            parts = csv_file.parts
            is_benign = any("benign" in p.lower() for p in parts)
            attack_type = "benign" if is_benign else csv_file.parent.name
            tmp["label"] = attack_type
            tmp["binary_label"] = 0 if is_benign else 1
            dfs.append(tmp)
        if not dfs:
            raise FileNotFoundError("No CSVs found for N-BaIoT in " + str(self.data_dir))
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded N-BaIoT: {len(df):,} rows | {df['binary_label'].sum():,} attacks")
        return df, "label", "binary_label"

    def _load_kitsune(self):
        csv_files = sorted(self.data_dir.rglob("*dataset*.csv"))
        if not csv_files:
            csv_files = sorted(self.data_dir.rglob("*.csv"))
        dfs = []
        for f in csv_files:
            tmp = pd.read_csv(f)
            if "label" not in tmp.columns and "Label" not in tmp.columns:
                # Kitsune often has a separate labels file
                label_file = f.parent / (f.stem + "_labels.csv")
                if label_file.exists():
                    labels = pd.read_csv(label_file)
                    tmp["label"] = labels.iloc[:, 0].values[:len(tmp)]
                else:
                    tmp["label"] = "unknown"
            dfs.append(tmp)
        df = pd.concat(dfs, ignore_index=True)
        label_col = "Label" if "Label" in df.columns else "label"
        df["binary_label"] = df[label_col].apply(lambda x: 0 if str(x).lower() in ("0","normal","benign") else 1)
        print(f"  Loaded Kitsune: {len(df):,} rows | {df['binary_label'].sum():,} attacks")
        return df, label_col, "binary_label"

    def _load_generic(self):
        """Fallback: load any CSV, assume last column is label."""
        csv_files = list(self.data_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in " + str(self.data_dir))
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        label_col = df.columns[-1]
        df["binary_label"] = (df[label_col] != df[label_col].mode()[0]).astype(int)
        print(f"  Loaded generic CSV: {len(df):,} rows")
        return df, label_col, "binary_label"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSOR — cleaning, encoding, Spearman filtering, ANOVA/MI selection
# ─────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Implements paper methodology:
      1. Remove zero-variance / constant features
      2. Spearman correlation filter (threshold = 0.8)
      3. ANOVA F-Test feature ranking + top-K selection
      4. StandardScaler normalisation
    """

    def __init__(self, correlation_threshold: float = 0.8, top_k_features: int = 20):
        self.correlation_threshold = correlation_threshold
        self.top_k_features = top_k_features
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.selected_features = []
        self.feature_scores = {}

    def fit_transform(self, df: pd.DataFrame, label_cols: list):
        """Full preprocessing pipeline. Returns (X_scaled, y_binary, feature_names)."""
        print("\n[Preprocessor] Starting preprocessing pipeline...")
        df = df.copy()
        non_feature_cols = set(label_cols)

        # --- Encode categorical columns ---
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in non_feature_cols]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        print(f"  Encoded {len(cat_cols)} categorical columns")

        feature_cols = [c for c in df.columns if c not in non_feature_cols]
        X = df[feature_cols].copy()
        y = df["binary_label"].values if "binary_label" in df.columns else np.zeros(len(df))

        # --- Remove NaN / inf ---
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        # --- Step 1: Remove zero-variance features ---
        variances = X.var()
        zero_var = variances[variances == 0].index.tolist()
        X.drop(columns=zero_var, inplace=True)
        print(f"  Removed {len(zero_var)} zero-variance features → {X.shape[1]} remain")

        # --- Step 2: Spearman correlation filter ---
        if X.shape[1] > 2:
            corr_matrix = X.corr(method="spearman").abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
            X.drop(columns=to_drop, inplace=True)
            print(f"  Spearman filter (>{self.correlation_threshold}): removed {len(to_drop)} features → {X.shape[1]} remain")

        # --- Step 3: ANOVA F-Test feature selection ---
        from sklearn.feature_selection import f_classif, mutual_info_classif
        if X.shape[1] > self.top_k_features:
            f_scores, _ = f_classif(X, y)
            f_scores = np.nan_to_num(f_scores)
            mi_scores = mutual_info_classif(X, y, random_state=42)
            combined = (f_scores / (f_scores.max() + 1e-9)) + (mi_scores / (mi_scores.max() + 1e-9))
            top_idx = np.argsort(combined)[-self.top_k_features:]
            self.feature_scores = dict(zip(X.columns, combined))
            X = X.iloc[:, top_idx]
            print(f"  ANOVA+MI selection: kept top {self.top_k_features} features → {X.shape[1]} remain")

        self.selected_features = X.columns.tolist()

        # --- Step 4: Normalise ---
        X_scaled = self.scaler.fit_transform(X)
        print(f"  Normalised. Final shape: {X_scaled.shape}")
        return X_scaled, y, self.selected_features

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted pipeline."""
        df = df.copy()
        cat_cols = [c for c in df.select_dtypes(include=["object","category"]).columns if c in self.label_encoders]
        for col in cat_cols:
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
        X = df[self.selected_features]
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        return self.scaler.transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# 3a. STAGE 1 — DEEP AUTOENCODER (DAE)
# ─────────────────────────────────────────────────────────────────────────────

if TF_AVAILABLE:
    def build_autoencoder(input_dim: int, encoding_dim: int = 8) -> tuple:
        """
        Symmetric DAE:  input → 64 → 32 → encoding_dim → 32 → 64 → input
        Trained on NORMAL traffic only.
        High reconstruction error = anomaly.
        """
        inputs = keras.Input(shape=(input_dim,), name="ae_input")

        # Encoder
        x = layers.Dense(64, activation="relu", name="enc_1")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu", name="enc_2")(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)

        # Decoder
        x = layers.Dense(32, activation="relu", name="dec_1")(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation="relu", name="dec_2")(x)
        x = layers.BatchNormalization()(x)
        decoded = layers.Dense(input_dim, activation="linear", name="ae_output")(x)

        autoencoder = Model(inputs, decoded, name="DeepAutoencoder")
        encoder     = Model(inputs, encoded, name="Encoder")

        autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return autoencoder, encoder


    # ─────────────────────────────────────────────────────────────────────────
    # 3b. STAGE 2 — GRU CLASSIFIER
    # ─────────────────────────────────────────────────────────────────────────

    def build_gru_classifier(seq_len: int, feature_dim: int, num_classes: int) -> Model:
        """
        GRU over sequences of encoded traffic windows.
        input shape: (batch, seq_len, feature_dim)
        """
        inputs = keras.Input(shape=(seq_len, feature_dim), name="gru_input")

        x = layers.GRU(64, return_sequences=True, name="gru_1")(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.GRU(32, return_sequences=False, name="gru_2")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu", name="fc_1")(x)

        if num_classes == 2:
            outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]

        model = Model(inputs, outputs, name="GRU_Classifier")
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. SKLEARN FALLBACK — when TensorFlow not available
# ─────────────────────────────────────────────────────────────────────────────

class SKLearnIDS:
    """
    Lightweight sklearn-based IDS that mirrors the two-stage logic:
      Stage 1: IsolationForest (unsupervised anomaly ≈ Autoencoder)
      Stage 2: GradientBoostingClassifier (temporal features ≈ GRU)
    """
    def __init__(self):
        from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
        self.stage1 = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.stage2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        self.threshold_percentile = 90  # top-X% reconstruction errors flagged

    def fit(self, X_normal, X_all, y_all):
        print("\n[Stage 1 - IsolationForest] Training on normal traffic...")
        self.stage1.fit(X_normal)
        scores = -self.stage1.decision_function(X_all)  # higher = more anomalous
        self.threshold = np.percentile(scores[y_all == 0], self.threshold_percentile)
        flagged = scores >= self.threshold
        print(f"  Threshold: {self.threshold:.4f} | Flagged: {flagged.sum():,} samples")

        print("[Stage 2 - GBM] Training classifier on flagged traffic...")
        X_flagged = X_all[flagged]
        y_flagged = y_all[flagged]
        if len(np.unique(y_flagged)) > 1:
            self.stage2.fit(X_flagged, y_flagged)
        else:
            # If all flagged are one class, train on full set
            self.stage2.fit(X_all, y_all)
        print("  Stage 2 training complete.")
        return self

    def predict(self, X):
        scores = -self.stage1.decision_function(X)
        stage1_pred = (scores >= self.threshold).astype(int)
        stage2_pred = self.stage2.predict(X)
        # Final: attack only if both stages agree (or Stage 1 flags it)
        return np.where(stage1_pred == 1, stage2_pred, 0)

    def decision_scores(self, X):
        return -self.stage1.decision_function(X)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN IDS SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class IoTIntrusionDetectionSystem:
    """
    Full pipeline: load → preprocess → train (Stage 1 + Stage 2) → evaluate.
    Automatically uses TensorFlow if available, else falls back to sklearn.
    """

    def __init__(self, data_dir: str, results_dir: str = "results",
                 seq_len: int = 10, encoding_dim: int = 8, top_k: int = 20):
        self.data_dir    = data_dir
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.seq_len      = seq_len
        self.encoding_dim = encoding_dim
        self.top_k        = top_k
        self.preprocessor = Preprocessor(top_k_features=top_k)
        self.metrics = {}

    def run(self):
        print("=" * 70)
        print("  IoT Intrusion Detection System — Hybrid DAE + GRU")
        print("=" * 70)

        # 1. Load
        loader = DatasetLoader(self.data_dir)
        df, label_col, binary_col = loader.load()

        # 2. Preprocess
        X, y, features = self.preprocessor.fit_transform(df, label_cols=[label_col, binary_col])
        print(f"\n[Data] Samples: {len(X):,} | Features: {len(features)} | Attacks: {y.sum():,}")

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_normal = X_train[y_train == 0]

        if TF_AVAILABLE:
            self._run_tensorflow(X_train, X_test, y_train, y_test, X_normal, features)
        else:
            print("\n[Info] TensorFlow not available — using sklearn fallback (IsolationForest + GBM)")
            self._run_sklearn(X_train, X_test, y_train, y_test, X_normal)

        # 4. Save results
        results_path = self.results_dir / "metrics.json"
        with open(results_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n[Results] Saved to {results_path}")
        self._print_summary()

    # ── TF path ──────────────────────────────────────────────────────────────

    def _run_tensorflow(self, X_train, X_test, y_train, y_test, X_normal, features):
        input_dim = X_train.shape[1]

        # ─── Stage 1: Deep Autoencoder ───────────────────────────────────────
        print("\n" + "─" * 50)
        print("STAGE 1 — Deep Autoencoder (training on normal traffic only)")
        print("─" * 50)
        autoencoder, encoder = build_autoencoder(input_dim, self.encoding_dim)
        autoencoder.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
        autoencoder.fit(
            X_normal, X_normal,
            epochs=50, batch_size=256, validation_split=0.1,
            callbacks=callbacks, verbose=1
        )

        # Reconstruction error threshold (95th percentile on normal training data)
        normal_recon = autoencoder.predict(X_normal, verbose=0)
        recon_errors_normal = np.mean(np.square(X_normal - normal_recon), axis=1)
        threshold = np.percentile(recon_errors_normal, 95)
        print(f"\n  Reconstruction error threshold (95th pct): {threshold:.6f}")

        # Evaluate Stage 1
        test_recon = autoencoder.predict(X_test, verbose=0)
        recon_errors_test = np.mean(np.square(X_test - test_recon), axis=1)
        stage1_pred = (recon_errors_test > threshold).astype(int)

        s1_acc = accuracy_score(y_test, stage1_pred)
        s1_f1  = f1_score(y_test, stage1_pred, zero_division=0)
        print(f"  Stage 1 — Accuracy: {s1_acc:.4f} | F1: {s1_f1:.4f}")
        self.metrics["stage1"] = {
            "accuracy": round(s1_acc, 4),
            "f1_score": round(s1_f1, 4),
            "threshold": round(float(threshold), 6)
        }

        # Encode all data for Stage 2
        X_train_enc = encoder.predict(X_train, verbose=0)
        X_test_enc  = encoder.predict(X_test,  verbose=0)

        # ─── Stage 2: GRU Classifier ─────────────────────────────────────────
        print("\n" + "─" * 50)
        print("STAGE 2 — GRU Classifier (on encoded + windowed sequences)")
        print("─" * 50)

        # Build sequences
        X_train_seq, y_train_seq = self._make_sequences(X_train_enc, y_train)
        X_test_seq,  y_test_seq  = self._make_sequences(X_test_enc,  y_test)
        print(f"  Sequence shape: {X_train_seq.shape}")

        gru = build_gru_classifier(self.seq_len, self.encoding_dim, num_classes=2)
        gru.summary()

        gru_callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
        gru.fit(
            X_train_seq, y_train_seq,
            epochs=50, batch_size=128, validation_split=0.1,
            callbacks=gru_callbacks, verbose=1
        )

        # Evaluate Stage 2
        t0 = time.time()
        y_pred_prob = gru.predict(X_test_seq, verbose=0).flatten()
        latency_ms  = (time.time() - t0) / len(X_test_seq) * 1000
        y_pred      = (y_pred_prob > 0.5).astype(int)

        s2_acc  = accuracy_score(y_test_seq, y_pred)
        s2_f1   = f1_score(y_test_seq, y_pred, zero_division=0)
        s2_prec = precision_score(y_test_seq, y_pred, zero_division=0)
        s2_rec  = recall_score(y_test_seq, y_pred, zero_division=0)
        fnr     = 1 - s2_rec  # False Negative Rate

        print(f"\n  Stage 2 — Accuracy: {s2_acc:.4f} | F1: {s2_f1:.4f} | "
              f"Precision: {s2_prec:.4f} | FNR: {fnr:.4f}")
        print(f"  Avg. inference latency: {latency_ms:.2f} ms/sample")
        print("\n" + classification_report(y_test_seq, y_pred, target_names=["Normal","Attack"]))

        self.metrics["stage2"] = {
            "accuracy": round(s2_acc, 4),
            "f1_score": round(s2_f1, 4),
            "precision": round(s2_prec, 4),
            "recall": round(s2_rec, 4),
            "false_negative_rate": round(fnr, 4),
            "avg_latency_ms": round(latency_ms, 2)
        }

        # Save models
        autoencoder.save(str(self.results_dir / "autoencoder.keras"), save_format="keras")
        gru.save(str(self.results_dir / "gru_classifier.keras"), save_format="keras")
        print(f"\n  Models saved to {self.results_dir}/")

    # ── sklearn fallback path ─────────────────────────────────────────────────

    def _run_sklearn(self, X_train, X_test, y_train, y_test, X_normal):
        ids = SKLearnIDS()
        ids.fit(X_normal, X_train, y_train)

        t0 = time.time()
        y_pred = ids.predict(X_test)
        latency_ms = (time.time() - t0) / len(X_test) * 1000

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        fnr  = 1 - rec

        print(f"\n  Final — Accuracy: {acc:.4f} | F1: {f1:.4f} | "
              f"Precision: {prec:.4f} | FNR: {fnr:.4f}")
        print(f"  Avg. latency: {latency_ms:.2f} ms/sample")
        print("\n" + classification_report(y_test, y_pred, target_names=["Normal","Attack"]))

        self.metrics["combined"] = {
            "accuracy": round(acc, 4), "f1_score": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "false_negative_rate": round(fnr, 4), "avg_latency_ms": round(latency_ms, 3)
        }
        self.sklearn_model = ids

        import pickle

        with open(self.results_dir / "sklearn_ids.pkl", "wb") as f:
            pickle.dump(ids, f)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_sequences(self, X, y):
        """Sliding window to create (N, seq_len, features) tensors."""
        seqs, labels = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len:i])
            labels.append(y[i])
        return np.array(seqs), np.array(labels)

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 70)
        for stage, m in self.metrics.items():
            print(f"\n  {stage.upper()}:")
            for k, v in m.items():
                print(f"    {k:<30} {v}")
        target_latency = 40
        latency = self.metrics.get("stage2", self.metrics.get("combined", {})).get("avg_latency_ms", 0)
        status = "✓ PASS" if latency < target_latency else "✗ MISS"
        print(f"\n  Real-time target (<{target_latency} ms): {status}  ({latency:.2f} ms)")
        print("=" * 70)
