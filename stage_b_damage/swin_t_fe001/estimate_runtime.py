"""Print a conservative FE-001 Swin-T runtime estimate without training."""

from __future__ import annotations

import math


TRAIN_ROWS = 17_552
VAL_ROWS = 22_372
EPOCHS = 20
BATCH_SIZE = 8


def main() -> int:
    train_batches = math.ceil(TRAIN_ROWS / BATCH_SIZE)
    val_batches = math.ceil(VAL_ROWS / BATCH_SIZE)
    batches_per_epoch = train_batches + val_batches
    print("FE-001 Swin-T B1 estimate (planning only; no training is run)")
    print(f"train rows: {TRAIN_ROWS:,}; val rows: {VAL_ROWS:,}")
    print(f"batch size: {BATCH_SIZE}; epochs: {EPOCHS}")
    print(f"batches/epoch: {batches_per_epoch:,}; batches/20 epochs: {batches_per_epoch * EPOCHS:,}")
    print("historical ResNet FE-001: 4.36 h with workers=0; 1.75 h with workers=4")
    print("Swin-T 224, RTX 4060, optimized loader: roughly 3.5 to 8 h for 20 epochs")
    print("Swin-T 224, RTX 4060, slower I/O/no crop cache: roughly 5 to 10 h")
    print("including weight/cache preparation, validation export and retries: reserve 5 to 12 h")
    print("Update the estimate after one real, complete epoch; synthetic throughput is not a commitment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

