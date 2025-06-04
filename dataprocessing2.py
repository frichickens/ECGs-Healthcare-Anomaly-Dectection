import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import random
import matplotlib.pyplot as plt
# 1. AAMI mapping dictionary
symbol_to_aami = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    'P': 'Q', 'f': 'Q', 'U': 'Q'
}

def mit_bih_symbol_to_aami(sym):
    return symbol_to_aami.get(sym, 'Q')

def labeling_normal_abnormal(label: str) -> int:
    return 0 if label == 'N' else 1

# 2. Filtering utilities
def butter_filter(x, fs, cutoff, btype='highpass', order=1):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)

def preprocess_segment(raw_seg, fs):
    """
    raw_seg: 1D np.ndarray containing one beat window (lead II)
    Returns a filtered, normalized version.
    """
    # (a) Highpass @ 0.5 Hz
    hp = butter_filter(raw_seg, fs, cutoff=0.5, btype='highpass')
    # (b) Optional: Lowpass @ 40 Hz (uncomment if needed)
    lp = butter_filter(hp, fs, cutoff=40.0, btype='lowpass')
    # (c) Normalize zero-mean, unit-variance
    norm = (lp - np.mean(lp)) / (np.std(lp) + 1e-6)
    return lp

# 3. Beat segmentation (fixed counts)
def extract_beat_segment(signal, beat_idx, pre_samps=100, post_samps=160):
    """
    Fixed-length beat window:
        100 samples before the R-peak,
        160 samples after the R-peak.
    """
    start = int(beat_idx) - pre_samps
    end   = int(beat_idx) + post_samps
    if start < 0 or end > signal.shape[0]:
        return None

    # lead II is column 0
    lead_ii = signal[:, 0]
    seg = lead_ii[start:end]
    # enforce exact length (100+160=260 samples)
    if len(seg) != (pre_samps + post_samps):
        return None
    return seg

# 4. Loop over records to build dataset
def build_dataset(record_list, data_dir, window_ms=1000):
    segments = []
    labels   = []
    examples = []
    for rec_id in record_list:
        rec_path = os.path.join(data_dir, f'{rec_id:03d}')
        # a) Read raw ECG
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal         # shape = (n_samples, 2)
        fs  = rec.fs               # sampling freq = 360 by default
        
        # b) Read beat annotations
        ann = wfdb.rdann(rec_path, extension='atr')
        beat_samples = ann.sample
        beat_syms    = ann.symbol
        
        for samp, sym in zip(beat_samples, beat_syms):
            # now call with fixed sample counts
            seg = extract_beat_segment(sig, samp, pre_samps=100, post_samps=160)
            if seg is None:
                continue
            pre = preprocess_segment(seg, fs)
            bin_label = labeling_normal_abnormal(sym)

            examples.append((pre, bin_label))
    
    # segments = np.stack(segments)  # shape = (N_beats, window_samples)
    # labels = np.array(labels)      # shape = (N_beats,)
    # return segments, labels
    return examples

def dataset_to_csv(dataset, csv_path):

    windows = []
    labels  = []
    for beat_win, lbl in dataset:
        windows.append(beat_win)    
        labels.append(int(lbl))

    # Stack into (N, 180)
    arr = np.stack(windows, axis=0).astype(np.float32)
    # Build column names
    col_names = [f"sample_{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=col_names)
    df["label"] = labels
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")


DATA_DIR = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'
patient_ids = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
    223, 230, 100, 103, 105, 111, 113, 117, 121, 123, 
    200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 
    231, 232, 233, 234
]

# dataset = build_dataset(patient_ids, DATA_DIR)
# dataset_to_csv(dataset, 'dataset/a.csv')

data = pd.read_csv('dataset/a.csv')
data_normal = data.loc[data['label']==0]
data_abnormal = data.loc[data['label']==1]

# print(data_normal.shape)
# print(data_abnormal.shape)
# for i in range(20):
#     plt.plot(data_normal.iloc[i,:-1])
#     plt.show()

# train_data = data_normal.iloc[:-13554, :]
# test_data = pd.concat([data_normal.iloc[-13554:,:], data_abnormal], ignore_index=True)
# train_data.to_csv('dataset/train2.csv', index=False)
# test_data.to_csv('dataset/test2.csv', index=False)
# print(data_normal.shape)
# print(data_abnormal.shape)

fs       = 360                  # sampling frequency (Hz)
pre_samps  = 100                # samples before R
post_samps = 160                # samples after R

# build time vector in seconds (negative before R-peak)
time_axis = np.arange(-pre_samps, post_samps) / fs
plt.figure(figsize=(15,6))
for i in range(4300, 4500):
    plt.plot(time_axis, data_normal.iloc[i,:-1])
    plt.grid(True)
plt.show()


# arr = train_data.values                      
# cols = np.array(train_data.columns.values)   
# np.savez_compressed('dataset/train.npz', data=arr, columns=cols)

# arr = test_data.values                      
# cols = np.array(test_data.columns.values)   
# np.savez_compressed('dataset/test.npz', data=arr, columns=cols)

# arr = data.values                      
# cols = np.array(data.columns.values)   
# np.savez_compressed('dataset/data.npz', data=arr, columns=cols)