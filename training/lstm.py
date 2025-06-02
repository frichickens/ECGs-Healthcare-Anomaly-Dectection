import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from models.lstm import RecurrentAutoencoder  # Import your autoencoder model

# --- 1. Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. CSV Dataset that returns (sequence, binary_label) ---
class CsvBeatDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.sequences = df.iloc[:, :-1].values.astype(np.float32)
        original_labels = df.iloc[:, -1].values.astype(np.int64)
        # Convert to binary labels: 0 (normal), 1 (abnormal)
        self.labels = np.where(original_labels == 0, 0, 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        seq_tensor = torch.from_numpy(seq).unsqueeze(1)  # (180, 1)
        return seq_tensor.to(device), label

# --- 3. Training loop on normal data only ---
def train_autoencoder(model, train_loader, val_loader, n_epochs, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss().to(device)
    history = {"train": [], "val": []}

    best_model_wts = model.state_dict()
    best_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for seq_true, label in train_loader:
            if label.sum().item() != 0:
                continue  # Skip any batch that includes abnormal beats
            optimizer.zero_grad()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq_true, label in val_loader:
                # Only consider normal samples in validation
                mask = (label == 0)
                if not mask.any():
                    continue
                seq_norm = seq_true[mask]
                seq_pred = model(seq_norm)
                loss = criterion(seq_pred, seq_norm)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(f"Epoch {epoch:3d}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}" +
              ("  <-- new best" if val_loss == best_loss else ""))

        # Early stopping
        if epochs_since_improvement >= patience:
            print(f"No improvement for {patience} epochs. Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_model_wts)
    return model.eval(), history


# --- 4. Inference: label abnormal if reconstruction error > threshold ---
def evaluate_on_validation(model, val_loader, threshold):
    criterion = nn.MSELoss(reduction='none')
    y_true, y_pred = [], []
    with torch.no_grad():
        for seq_true, label in val_loader:
            seq_pred = model(seq_true)
            loss_per_seq = criterion(seq_pred, seq_true).mean(dim=[1, 2])
            pred_label = (loss_per_seq > threshold).int()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred_label.cpu().numpy())
    return np.array(y_true), np.array(y_pred)


if __name__ == "__main__":
    seq_len      = 180
    n_features   = 1
    embedding_dim = 32

    # — 1. Build & load the model —
    model = RecurrentAutoencoder(seq_len, n_features, embedding_dim).to(device)

    # Paths to your CSV files
    train_csv      = "./dataset/train.csv"        # Full training set (0,1,2,3 labels)
    val_csv        = "./dataset/validation.csv"   # Full validation set (0,1,2,3 labels)
    test_csv       = "./dataset/test.csv"         # Full test set    (0,1,2,3 labels)

    # — 2. Create Dataset objects —
    train_dataset_full = CsvBeatDataset(train_csv)
    val_dataset_full   = CsvBeatDataset(val_csv)
    test_dataset_full  = CsvBeatDataset(test_csv)

    # — 3. Filter ONLY NORMAL (label == 0) for training and validation —
    train_normal_idxs = [
        i for i in range(len(train_dataset_full))
        if train_dataset_full.labels[i] == 0
    ]
    val_normal_idxs = [
        i for i in range(len(val_dataset_full))
        if val_dataset_full.labels[i] == 0
    ]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_normal_idxs)
    val_dataset   = torch.utils.data.Subset(val_dataset_full,   val_normal_idxs)
    # Note: test_dataset_full is kept intact (we want to evaluate on both 0 & non-0)

    # — 4. Build DataLoaders —
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset_full, batch_size=32, shuffle=False)

    # — 5. Train only on normal data (val_loader is now normal-only) —
    n_epochs = 50
    best_model, history = train_autoencoder(model, train_loader, val_loader, n_epochs)

    # — 6. Plot train/val loss curves (optional) —
    plt.plot(history['train'], label='Train Loss (normal only)')
    plt.plot(history['val'],   label='Val   Loss (normal only)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    # — 7. Compute threshold on VALIDATION (normal only) reconstruction errors —
    errors_val = []
    best_model.eval()
    with torch.no_grad():
        for seq_true, _ in val_loader:
            seq_pred = best_model(seq_true)
            err = nn.MSELoss(reduction='none')(seq_pred, seq_true).mean(dim=[1, 2])
            errors_val.extend(err.cpu().numpy())

    # e.g. pick the 95th percentile of validation errors
    threshold = np.percentile(errors_val, 95)
    print(f"Selected threshold (95th pct of normal-only val): {threshold:.6f}")

    # — 8. Evaluate on TEST SET (mixed 0/1, 2, 3) by thresholding —
    y_true_test, y_pred_test = evaluate_on_validation(best_model, test_loader, threshold)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\n=== FINAL TEST SET METRICS ===")
    print(classification_report(y_true_test, y_pred_test,
                                target_names=["Normal", "Abnormal"]))
    print("Confusion Matrix (test):")
    print(confusion_matrix(y_true_test, y_pred_test))

    # — 9. Save final model checkpoint —
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_model.state_dict(), "checkpoints/autoencoder_best.pth")