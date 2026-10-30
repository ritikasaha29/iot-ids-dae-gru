"""
=============================================================================
real_time_ids.py — Live Inference Engine
=============================================================================
Simulates a deployed IoT IDS that monitors a live packet stream.
Loads saved models (or sklearn fallback) and runs the two-stage pipeline
on incoming traffic windows, printing alerts in real-time.

Usage:
    python real_time_ids.py --model_dir results/ --demo
    python real_time_ids.py --model_dir results/ --csv_file new_traffic.csv
=============================================================================
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

ALERT_LEVELS = {
    0: ("NORMAL",  "\033[32m"),   # green
    1: ("ATTACK!", "\033[91m"),   # red
}
RESET = "\033[0m"


class RealTimeIDS:
    """
    Loads trained models and provides per-packet (or per-window) inference.
    """

    def __init__(self, model_dir: str, threshold: float = None,
                 seq_len: int = 10, scaler=None, features=None):
        self.model_dir = Path(model_dir)
        self.seq_len   = seq_len
        self.scaler    = scaler
        self.features  = features
        self._buffer   = []   # sliding window buffer
        self.alert_count = 0
        self.total_count = 0
        self._load_models(threshold)

    def _load_models(self, threshold):
        """Load TF models or sklearn fallback."""
        meta_path = self.model_dir / "deployment_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.threshold = meta.get("ae_threshold", threshold or 0.01)
            self.features  = meta.get("features", self.features)
        else:
            self.threshold = threshold or 0.01

        ae_path  = self.model_dir / "autoencoder.keras"
        gru_path = self.model_dir / "gru_classifier.keras"
        sk_path  = self.model_dir / "sklearn_ids.pkl"

        if TF_AVAILABLE and ae_path.exists() and gru_path.exists():
            self.autoencoder = keras.models.load_model(str(ae_path))
            # Extract encoder sub-model
            self.encoder = keras.Model(
                self.autoencoder.input,
                self.autoencoder.get_layer("bottleneck").output
            )
            self.gru = keras.models.load_model(str(gru_path))
            self.backend = "tensorflow"
            print(f"[RealTimeIDS] Loaded TF models from {self.model_dir}")
        elif sk_path.exists():
            with open(sk_path, "rb") as f:
                self.sklearn_ids = pickle.load(f)
            self.backend = "sklearn"
            print(f"[RealTimeIDS] Loaded sklearn model from {self.model_dir}")
        else:
            raise FileNotFoundError(
                f"No models found in {self.model_dir}. Run train.py first."
            )

    def ingest(self, sample: np.ndarray) -> dict:
        """
        Process one preprocessed sample vector.
        Returns detection result dict.
        """
        t0 = time.time()
        self.total_count += 1

        if self.backend == "tensorflow":
            result = self._infer_tf(sample)
        else:
            result = self._infer_sklearn(sample)

        result["latency_ms"] = round((time.time() - t0) * 1000, 3)
        if result["prediction"] == 1:
            self.alert_count += 1
        return result

    def _infer_tf(self, sample: np.ndarray) -> dict:
        x = sample.reshape(1, -1)

        # Stage 1: reconstruction error
        recon = self.autoencoder.predict(x, verbose=0)
        error = float(np.mean(np.square(x - recon)))
        stage1_flag = error > self.threshold

        if not stage1_flag:
            return {"stage1_error": round(error, 6), "stage1_flag": False,
                    "prediction": 0, "confidence": 0.0}

        # Stage 2: encode → buffer → GRU when window is full
        encoded = self.encoder.predict(x, verbose=0).flatten()
        self._buffer.append(encoded)
        if len(self._buffer) < self.seq_len:
            # Not enough history yet — flag conservatively
            return {"stage1_error": round(error, 6), "stage1_flag": True,
                    "prediction": 1, "confidence": 0.5, "note": "buffering"}

        seq = np.array(self._buffer[-self.seq_len:]).reshape(1, self.seq_len, -1)
        prob = float(self.gru.predict(seq, verbose=0).flatten()[0])
        pred = int(prob > 0.5)
        return {"stage1_error": round(error, 6), "stage1_flag": True,
                "stage2_prob": round(prob, 4), "prediction": pred,
                "confidence": round(prob if pred == 1 else 1 - prob, 4)}

    def _infer_sklearn(self, sample: np.ndarray) -> dict:
        x = sample.reshape(1, -1)
        score = float(-self.sklearn_ids.stage1.decision_function(x)[0])
        stage1_flag = score >= self.sklearn_ids.threshold
        if not stage1_flag:
            return {"anomaly_score": round(score, 4), "stage1_flag": False,
                    "prediction": 0, "confidence": 0.0}
        pred = int(self.sklearn_ids.stage2.predict(x)[0])
        prob = float(self.sklearn_ids.stage2.predict_proba(x)[0][pred])
        return {"anomaly_score": round(score, 4), "stage1_flag": True,
                "prediction": pred, "confidence": round(prob, 4)}

    def print_alert(self, result: dict, packet_id: int):
        pred   = result["prediction"]
        label, color = ALERT_LEVELS[pred]
        conf   = result.get("confidence", 0)
        lat    = result.get("latency_ms", 0)
        print(f"  [{packet_id:06d}] {color}{label}{RESET}  "
              f"conf={conf:.2f}  latency={lat:.2f}ms  "
              + (f"recon_err={result.get('stage1_error','N/A')}" if "stage1_error" in result else
                 f"anomaly_score={result.get('anomaly_score','N/A')}"))

    def stats(self) -> dict:
        return {
            "total_samples": self.total_count,
            "alerts_raised": self.alert_count,
            "attack_rate_pct": round(self.alert_count / max(1, self.total_count) * 100, 2)
        }


def demo_mode(ids: RealTimeIDS, n_samples: int = 100):
    """Generate synthetic traffic and run inference as a demo."""
    print("\n[Demo] Generating synthetic IoT traffic stream...")
    print("  (80% normal, 20% synthetic attack injections)\n")
    rng = np.random.default_rng(42)
    n_features = ids.autoencoder.input_shape[-1] if ids.backend == "tensorflow" else 20

    for i in range(n_samples):
        is_attack = rng.random() < 0.2
        if is_attack:
            sample = rng.normal(loc=5.0, scale=3.0, size=n_features)  # anomalous
        else:
            sample = rng.normal(loc=0.0, scale=1.0, size=n_features)  # normal
        result = ids.ingest(sample)
        ids.print_alert(result, i + 1)
        time.sleep(0.01)  # simulate real-time pacing

    print("\n" + "=" * 60)
    s = ids.stats()
    print(f"  STREAM SUMMARY: {s['total_samples']} packets | "
          f"{s['alerts_raised']} alerts | {s['attack_rate_pct']}% attack rate")
    print("=" * 60)


def csv_mode(ids: RealTimeIDS, csv_path: str):
    """Run inference on a CSV file row-by-row."""
    df = pd.read_csv(csv_path)
    print(f"\n[CSV mode] Processing {len(df):,} rows from {csv_path}")
    # Drop non-numeric, use only trained features if available
    if ids.features:
        cols = [c for c in ids.features if c in df.columns]
        df = df[cols]
    else:
        df = df.select_dtypes(include=[np.number])
    df.fillna(df.median(), inplace=True)

    for i, row in df.iterrows():
        result = ids.ingest(row.values.astype(np.float32))
        ids.print_alert(result, i + 1)

    print("\n" + "=" * 60)
    s = ids.stats()
    print(f"  STREAM SUMMARY: {s['total_samples']} packets | "
          f"{s['alerts_raised']} alerts | {s['attack_rate_pct']}% attack rate")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT IDS Real-Time Inference")
    parser.add_argument("--model_dir", default="results/", help="Directory with saved models")
    parser.add_argument("--demo",      action="store_true",  help="Run synthetic demo")
    parser.add_argument("--csv_file",  default=None,          help="CSV file to process")
    parser.add_argument("--n_samples", type=int, default=100, help="Demo sample count")
    args = parser.parse_args()

    ids = RealTimeIDS(model_dir=args.model_dir)
    if args.demo:
        demo_mode(ids, n_samples=args.n_samples)
    elif args.csv_file:
        csv_mode(ids, args.csv_file)
    else:
        print("Use --demo for synthetic demo or --csv_file path/to/traffic.csv")
