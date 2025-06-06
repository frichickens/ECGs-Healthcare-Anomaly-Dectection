from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset
from hmmlearn import hmm
import pickle

# -----------------------------
# Dataset for Fixed-Length ECG Beats
# -----------------------------
class ECGBedDataset(Dataset):
    """
    PyTorch-style Dataset for loading ECG beats from a CSV file.
    Each row in the CSV should be a fixed-length beat vector (no labels needed here).
    """
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        self.data = df.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a single beat as a 1D NumPy array
        return self.data[idx]


# -----------------------------
# Vector Quantizer Class
# -----------------------------
class ECGVectorQuantizer:
    """
    Splits each fixed-length beat into overlapping frames,
    optionally normalizes them, and uses KMeans to build a codebook.
    Provides a method to quantize beats into symbol sequences.
    """
    def __init__(self,
                target_length: int,
                window_size: int = 30,
                hop_size: int = 15,
                codebook_size: int = 64,
                normalize: bool = False):
        """
        target_length: length T of each beat (number of samples)
        window_size: number of samples per frame
        hop_size: stride between consecutive frames
        codebook_size: number of clusters (vocabulary size)
        normalize: if True, z-score each time-sample across all beats before clustering
        """
        self.target_length = target_length
        self.window_size = window_size
        self.hop_size = hop_size
        self.codebook_size = codebook_size
        self.normalize = normalize

        self.scaler = None       # StandardScaler if normalization is used
        self.codebook = None     # KMeans cluster centers (shape: K x window_size)

    def _extract_frames(self, beats: np.ndarray) -> np.ndarray:
        """
        Given beats of shape (num_beats, T), extract overlapping frames of length window_size
        with stride hop_size along each beat. Returns all frames stacked:
        shape (num_beats * num_frames_per_beat, window_size)
        """
        num_beats, T    = beats.shape
        if T != self.target_length:
            raise ValueError(f"Expected beats of length {self.target_length}, but got {T}.")

        frames = []
        for beat in beats:
            for start in range(0, T - self.window_size + 1, self.hop_size):
                frame = beat[start : start + self.window_size]
                frames.append(frame)
        return np.stack(frames, axis=0)  # shape: (num_beats * L, window_size)

    def fit(self, beats: np.ndarray):
        """
        Build the codebook from the given 'normal' beats.
        beats: np.ndarray of shape (num_beats, target_length)
        """
        num_beats, T = beats.shape
        if T != self.target_length:
            raise ValueError(f"Expected beats length {self.target_length}, but got {T}.")

        # 1) Normalize if required
        if self.normalize:
            self.scaler = StandardScaler()
            beats = self.scaler.fit_transform(beats)

        # 2) Extract all overlapping frames
        frames = self._extract_frames(beats)  # shape: (num_beats * L, window_size)

        # 3) Cluster via KMeans to create codebook
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, verbose=0)
        kmeans.fit(frames)
        self.codebook = kmeans.cluster_centers_  # shape: (K, window_size)

    def transform(self, beats: np.ndarray) -> (list, list):
        """
        Quantize each beat into a sequence of symbols.
        beats: np.ndarray of shape (num_beats, target_length)
        Returns:
            - symbol_sequences: list of np.ndarrays, each of length L (num_frames_per_beat)
            - lengths: list of ints (all = L)
        """
        num_beats, T = beats.shape
        if T != self.target_length:
            raise ValueError(f"Expected beats length {self.target_length}, but got {T}.")

        # 1) Normalize if required
        if self.normalize and self.scaler is not None:
            beats = self.scaler.transform(beats)

        symbol_sequences = []
        lengths = []
        for beat in beats:
            frames = []
            # Extract overlapping frames for this single beat
            for start in range(0, T - self.window_size + 1, self.hop_size):
                frame = beat[start : start + self.window_size]
                frames.append(frame)
            frames = np.stack(frames, axis=0)  # shape: (L, window_size)
            L = frames.shape[0]

            # Quantize each frame to nearest centroid
            symbols = []
            for frame in frames:
                diffs = self.codebook - frame.reshape(1, -1)   # shape: (K, window_size)
                dists = np.linalg.norm(diffs, axis=1)          # shape: (K,)
                symbol = int(np.argmin(dists))
                symbols.append(symbol)
            symbol_sequences.append(np.array(symbols, dtype=int))
            lengths.append(L)

        return symbol_sequences, lengths


# -----------------------------
# HMM Anomaly Detector (Single-Class, using VQ + Discrete HMM)
# -----------------------------
class ECGHMMSingleClassVQDetector:
    """
    Builds on ECGVectorQuantizer to train a discrete-output HMM on quantized normal beats.
    - Uses the vector quantizer to convert each beat into a symbol sequence (length L).
    - Fits a MultinomialHMM whose hidden states = n_states; observations = codebook symbols.
    - Chooses a log-likelihood threshold on training beats.
    """
    def __init__(self,
                target_length: int,
                n_states: int = 8,
                window_size: int = 30,
                hop_size: int = 15,
                codebook_size: int = 64,
                hmm_iter: int = 50,
                normalize: bool = False):
        """
        target_length: length T of each beat
        n_states: number of hidden states for the discrete HMM
        window_size: how many samples per frame
        hop_size: stride for overlapping frames
        codebook_size: number of VQ centroids (symbols)
        hmm_iter: max EM iterations for the MultinomialHMM
        normalize: if True, z-score each time-sample across all beats before VQ
        """
        self.target_length = target_length
        self.n_states = n_states

        # Instantiate vector quantizer
        self.vq = ECGVectorQuantizer(
            target_length=target_length,
            window_size=window_size,
            hop_size=hop_size,
            codebook_size=codebook_size,
            normalize=normalize
        )
        self.hmm_iter = hmm_iter
        self.discrete_hmm = None    # MultinomialHMM
        self.threshold = None       # log-likelihood threshold for anomaly
        self.logL_train = None      # log-likelihoods on training beats

    def fit(self, beats: np.ndarray):
        """
        Train the vector quantizer and discrete HMM on 'normal' beats.
        beats: np.ndarray of shape (num_beats, target_length)
        """
        num_beats, T = beats.shape
        if T != self.target_length:
            raise ValueError(f"Expected beats length {self.target_length}, but got {T}.")

        # 1) Fit the vector quantizer on all normal beats
        self.vq.fit(beats)

        # 2) Convert training beats to symbol sequences
        symbol_seqs, lengths = self.vq.transform(beats)
        # Concatenate all symbol sequences into one long array for HMM training
        all_symbols = np.concatenate(symbol_seqs)         # shape: (sum(lengths),)
        lengths_arr = np.array(lengths, dtype=int).tolist()

        # 3) Train a MultinomialHMM on the quantized sequences
        model = hmm.CategoricalHMM(n_components=self.n_states, 
                                n_iter=self.hmm_iter, 
                                verbose=True, 
                                random_state=42)
        
        # eye = np.eye(self.vq.codebook_size, dtype=int)
        # X_oh = eye[all_symbols]
        # model.n_features = self.vq.codebook_size
        model.fit(all_symbols.reshape(-1, 1), lengths_arr)
        self.discrete_hmm = model

        # 4) Compute log-likelihoods for each training beat
        logLs = []
        idx = 0
        for L in lengths_arr:
            seq = all_symbols[idx : idx + L]
            idx += L
            ll = self.discrete_hmm.score(seq.reshape(-1, 1))
            logLs.append(ll)
        self.logL_train = np.array(logLs)

        # 5) Choose threshold at 5th percentile by default
        self.threshold = np.percentile(self.logL_train, 5.0)

    def predict(self, beats: np.ndarray) -> np.ndarray:
        """
        Given new beats (np.ndarray shape (num_beats, target_length)), predict anomalies.
        Returns a binary array of length num_beats: 0 = normal, 1 = anomaly.
        """
        if self.discrete_hmm is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        num_beats, T = beats.shape
        if T != self.target_length:
            raise ValueError(f"Expected beats length {self.target_length}, but got {T}.")

        # 1) Convert new beats to symbol sequences
        symbol_seqs, lengths = self.vq.transform(beats)

        # 2) Compute log-likelihood for each beat and flag anomalies
        anomalies = []
        for seq, L in zip(symbol_seqs, lengths):
            ll = self.discrete_hmm.score(seq.reshape(-1, 1))
            anomalies.append(int(ll < self.threshold))
        return np.array(anomalies, dtype=int)

    def score_beats(self, beats: np.ndarray) -> np.ndarray:
        """
        Return raw log-likelihoods for each beat (without thresholding).
        """
        if self.discrete_hmm is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        symbol_seqs, lengths = self.vq.transform(beats)
        logLs = []
        for seq, L in zip(symbol_seqs, lengths):
            ll = self.discrete_hmm.score(seq.reshape(-1, 1))
            logLs.append(ll)
        return np.array(logLs)


# -----------------------------
# HMM Multi-Class Detector (VQ + Discrete HMM)
# -----------------------------
class ECGHMMMultiClassVQDetector:
    """
    Manages one ECGHMMSingleClassVQDetector per class.
    - fit(multidata): multidata[i] is np.ndarray of normal beats from class i.
    - predict(test_beats): returns (predicted_labels, unlabels).
    predicted_labels[k] = class index with highest log-likelihood for beat k.
    unlabels = list of indices where ALL class models flagged that beat as anomaly.
    """
    def __init__(self, n_class: int, models: list):
        if len(models) != n_class:
            raise ValueError("Length of models list must equal n_class.")
        self.n_class = n_class
        self.models = models

    def fit(self, multidata: list):
        """
        Train each class-specific VQ+HMM on its normal beats.
        multidata = [beats_class0, beats_class1, ..., beats_classN-1],
        where beats_classi is np.ndarray of shape (num_beats_i, target_length).
        """
        if len(multidata) != self.n_class:
            raise ValueError("multidata must be a list of length n_class.")

        for i in range(self.n_class):
            beats_i = multidata[i]
            print(f"Training VQ+HMM for class {i} with {beats_i.shape[0]} beats …")
            self.models[i].fit(beats_i)
            print(f"  → Class {i} threshold = {self.models[i].threshold:.2f}\n")

    def predict(self, test_beats: np.ndarray) -> (list, list):
        """
        Predict class labels for each test beat, and also find 'unlabels'.
        test_beats: np.ndarray of shape (num_test, target_length).
        Returns:
        predicted_labels: list of length num_test, each in [0..n_class-1]
        unlabels: list of indices where all class-models flagged anomaly
        """
        if isinstance(test_beats, list):
            test_beats = np.stack(test_beats, axis=0)
        num_test, T = test_beats.shape
        if T != self.models[0].target_length:
            raise ValueError(
                f"Expected each beat of length {self.models[0].target_length}, but got {T}."
            )

        predicted_labels = []
        unlabels = []

        for idx in range(num_test):
            beat = test_beats[idx : idx + 1]  # shape (1, target_length)
            logLs = np.zeros(self.n_class)

            # Compute log-likelihood under each class's VQ+HMM
            for i in range(self.n_class):
                ll_arr = self.models[i].score_beats(beat)  # shape (1,)
                logLs[i] = ll_arr[0]

            # Assign to the class with highest log-likelihood
            pred_class = int(np.argmax(logLs))
            predicted_labels.append(pred_class)

            # Check if all models flagged this beat as anomaly
            anomalies = []
            for i in range(self.n_class):
                is_anom = int(logLs[i] < self.models[i].threshold)
                anomalies.append(is_anom)
            if sum(anomalies) == self.n_class:
                unlabels.append(idx)

        return predicted_labels, unlabels

