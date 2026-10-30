# IoT Intrusion Detection System (Hybrid Deep Autoencoder + GRU)

A two-stage intrusion detection system for IoT network traffic.

1. **Stage 1: Deep Autoencoder (DAE).** Trained on normal traffic only. Traffic with a high reconstruction error is flagged as anomalous.
2. **Stage 2: GRU classifier.** Analyses sequences of encoded traffic to decide whether flagged traffic is a real attack (DDoS, botnet, port scan, etc.).

If TensorFlow is not installed, the project automatically falls back to scikit-learn equivalents (Isolation Forest / PCA for Stage 1, Gradient Boosting for Stage 2), so it runs on any machine.

## Project structure

| File | Purpose |
|------|---------|
| `model.py` | Core pipeline: dataset loader, preprocessing, DAE, GRU, sklearn fallback, evaluation |
| `train.py` | Command-line entry point for training on a real dataset |
| `demo.py` | End-to-end demo on synthetic data (no download needed) |
| `real_time_ids.py` | Live inference engine; runs the trained pipeline on a stream or CSV |
| `download_data.py` | Prints download links and creates the data folders |
| `requirements.txt` | Python dependencies |

## Installation

```bash
git clone https://github.com/ritikasaha29/iot-ids-dae-gru.git
cd iot-ids-dae-gru
pip install -r requirements.txt
```

TensorFlow is optional. Without it, the sklearn fallback is used.

## Quick start (no dataset needed)

```bash
python demo.py
```

This generates synthetic IoT traffic, runs the full two-stage pipeline, prints metrics for each stage, and saves them to `results/demo_metrics.json`.

## Training on a real dataset

Supported datasets (auto-detected):

- **NSL-KDD**: general anomaly detection baseline
- **N-BaIoT**: real IoT botnet traffic (Mirai, BASHLITE)
- **Kitsune**: online network intrusion scenarios

```bash
# See download links and create data folders
python download_data.py          # add --auto to fetch NSL-KDD automatically

# Train
python train.py --data_dir data/NSL-KDD/
python train.py --data_dir data/N-BaIoT/ --top_k 25
python train.py --data_dir data/Kitsune/ --seq_len 15 --encoding_dim 12
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Path to the dataset folder |
| `--results_dir` | `results` | Where models and metrics are saved |
| `--seq_len` | 10 | GRU sequence length |
| `--encoding_dim` | 8 | Autoencoder bottleneck size |
| `--top_k` | 20 | Number of features kept after selection |

## Preprocessing

1. Encode categorical columns and clean NaN/inf values
2. Remove zero-variance features
3. Spearman correlation filter (threshold 0.8)
4. ANOVA F-test + mutual information feature ranking, keeping the top-K
5. Standard scaling

## Real-time inference

After training, run the live detection engine:

```bash
python real_time_ids.py --model_dir results/ --demo
python real_time_ids.py --model_dir results/ --csv_file new_traffic.csv
```

Each sample is scored by Stage 1 first. Only samples flagged as anomalous are passed to Stage 2, which keeps latency low. The target is under 40 ms per sample.

## Metrics reported

Accuracy, precision, recall, F1-score, false negative rate, and per-sample latency.

## Known limitations

- `demo.py` uses synthetic data, so its numbers are only a sanity check and do not reflect real-world performance.
- In `demo.py`, the sklearn stand-ins (PCA and Gradient Boosting on lag features) are simplified analogs of the DAE and GRU, and Stage 2 performance there is weak. Train with TensorFlow on a real dataset for meaningful results.
- Results on NSL-KDD, N-BaIoT, and Kitsune have not been published here yet.

## License

Add a license of your choice (for example MIT) before sharing widely.
