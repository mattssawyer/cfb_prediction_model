"""Rebuild the training matrix from cached CFBD parquets and train the model.

Usage:
    python scripts/train.py

Produces:
    data/training_matrix.parquet
    models/winner_model.txt
    models/winner_model_schema.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from cfb.features import build_training_matrix
from cfb.model import train
from cfb.model import train_spread

def main() -> None:
    build_training_matrix()
    train()
    train_spread()


if __name__ == "__main__":
    main()
