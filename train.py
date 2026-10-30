"""
train.py — Entry point to train the IoT IDS
============================================
Usage:
    python train.py --data_dir data/NSL-KDD/
    python train.py --data_dir data/N-BaIoT/   --top_k 25
    python train.py --data_dir data/Kitsune/   --seq_len 15 --encoding_dim 12
"""

import argparse
from model import IoTIntrusionDetectionSystem

def main():
    parser = argparse.ArgumentParser(description="Train IoT IDS (DAE + GRU)")
    parser.add_argument("--data_dir",     required=True,    help="Path to dataset folder")
    parser.add_argument("--results_dir",  default="results",help="Output directory")
    parser.add_argument("--seq_len",      type=int, default=10,  help="GRU sequence length")
    parser.add_argument("--encoding_dim", type=int, default=8,   help="Autoencoder bottleneck size")
    parser.add_argument("--top_k",        type=int, default=20,  help="Top-K features to keep")
    args = parser.parse_args()

    ids = IoTIntrusionDetectionSystem(
        data_dir    = args.data_dir,
        results_dir = args.results_dir,
        seq_len     = args.seq_len,
        encoding_dim= args.encoding_dim,
        top_k       = args.top_k
    )
    ids.run()

if __name__ == "__main__":
    main()
