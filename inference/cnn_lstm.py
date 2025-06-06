import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.cae_lstm import ECGRNNModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)

# --- 1. Dataset loader & preprocessing ---
class MITBIHDataset:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path  = test_path
        self.num_classes = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path, header=None)
        self.test_data  = pd.read_csv(self.test_path,  header=None)

    def preprocess_data(self):
        # Split features / labels
        X = self.train_data.iloc[:, :-1].values
        y = self.train_data.iloc[:,  -1].values.astype(int)
        X_test = self.test_data.iloc[:, :-1].values
        y_test = self.test_data.iloc[:,  -1].values.astype(int)

        # Normalize
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        # Reshape for [samples, time, features]
        X      = X.reshape(-1, X.shape[1], 1)
        X_test = X_test.reshape(-1, X_test.shape[1], 1)

        # Train / val split (stratify to preserve class ratios)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Compute class‐weights for CrossEntropyLoss
        weights = compute_class_weight('balanced',
                                       classes=np.unique(y_train),
                                       y=y_train)
        
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)

        self.num_classes = len(np.unique(y_train))
        return X_train, y_train, X_val, y_val, X_test, y_test, class_weights
    

if __name__ == "__main__":
    ds = MITBIHDataset("dataset/mitbih_train.csv",
                       "dataset/mitbih_test.csv")
    ds.load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, cw = ds.preprocess_data()

    bs = 64
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=bs
    )

    seq_len, ch = X_train.shape[1], X_train.shape[2]
    model = ECGRNNModel(input_shape=(seq_len, ch),
                        num_classes=ds.num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/best_ecg_model.pth"))
    model.evaluate(test_loader, device=device)
