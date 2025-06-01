import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

# --- 1. Configuration (as before) ---
RECORDS_TRAIN_FULL = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
    223, 230
]
RECORDS_TEST = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
    233, 234
]

WINDOW_PRE   = 60
WINDOW_POST  = 120
WINDOW_SIZE  = WINDOW_PRE + WINDOW_POST  # 180

CLASS_MAP = {
    'N': 0,
    'L': 1,
    'R': 2,
    'V': 3,
    # … (other beat‐types if desired)
}

DATA_DIR = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'


# --- 2. Custom Dataset Class (unchanged) ---
class MITBIH_Dataset(Dataset):
    def __init__(self, record_ids, data_dir):
        self.examples = []
        for rec_id in record_ids:
            rec_name = f"{data_dir}/{rec_id:03d}"
            record = wfdb.rdrecord(rec_name)
            ann    = wfdb.rdann(rec_name, 'atr')
            sigs   = record.p_signal[:, 0]  # MLII lead

            for idx, sym in zip(ann.sample, ann.symbol):
                if sym not in CLASS_MAP:
                    continue
                start = idx - WINDOW_PRE
                end   = idx + WINDOW_POST
                if start < 0 or end > len(sigs):
                    continue

                beat_win = sigs[start:end].astype(np.float32)
                beat_win = (beat_win - beat_win.mean()) / (beat_win.std() + 1e-6)
                label = CLASS_MAP[sym]
                self.examples.append((beat_win, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        beat_win, label = self.examples[idx]
        return torch.from_numpy(beat_win).unsqueeze(0), torch.tensor(label, dtype=torch.long)


# --- 3. Choose 3 whole records as validation (patient-independent) ---
VAL_RECORDS = [109, 119, 230]
TRAIN_RECORDS_ACTUAL = [r for r in RECORDS_TRAIN_FULL if r not in VAL_RECORDS]


# --- 4. Instantiate Datasets for train / val / test ---
train_ds = MITBIH_Dataset(TRAIN_RECORDS_ACTUAL, data_dir=DATA_DIR)
val_ds   = MITBIH_Dataset(VAL_RECORDS,             data_dir=DATA_DIR)
test_ds  = MITBIH_Dataset(RECORDS_TEST,            data_dir=DATA_DIR)


# --- 5. Save each split to CSV ---
os.makedirs("saved_splits", exist_ok=True)

def dataset_to_csv(dataset: MITBIH_Dataset, csv_path: str):
    """
    Convert a MITBIH_Dataset (with examples=[(np.ndarray[180], int), ...])
    to a CSV where columns are: sample_0, sample_1, ..., sample_179, label.
    """
    windows = []
    labels  = []
    for beat_win, lbl in dataset.examples:
        windows.append(beat_win)    # shape (180,)
        labels.append(int(lbl))

    # Stack into (N, 180)
    arr = np.stack(windows, axis=0).astype(np.float32)
    # Build column names
    col_names = [f"sample_{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=col_names)
    df["label"] = labels
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")

# Save train.csv, validation.csv, test.csv
dataset_to_csv(train_ds, "saved_splits/train.csv")
dataset_to_csv(val_ds,   "saved_splits/validation.csv")
dataset_to_csv(test_ds,  "saved_splits/test.csv")


# (Optional) Wrap in DataLoaders if you still want them in-memory:
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


# --- 6. Sanity‐check: print dataset sizes and class counts ---
from collections import Counter

def count_classes(dataset: MITBIH_Dataset):
    cnt = Counter()
    for _, lbl in dataset:
        cnt[int(lbl)] += 1
    return dict(cnt)

print("\nSplit sizes:")
print(f"  Train:      {len(train_ds)} windows → class counts {count_classes(train_ds)}")
print(f"  Validation: {len(val_ds)} windows → class counts {count_classes(val_ds)}")
print(f"  Test:       {len(test_ds)} windows → class counts {count_classes(test_ds)}")
