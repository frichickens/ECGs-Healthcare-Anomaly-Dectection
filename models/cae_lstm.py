import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 2. Model definition ---
class ECGRNNModel(nn.Module):
    def __init__(self, input_shape, num_classes, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        seq_len, input_channels = input_shape

        # 1D-CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        # figure out flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_len)
            feat  = self.feature_extractor(dummy)
            self.cnn_out_dim = feat.shape[1] * feat.shape[2]

        # Three unidirectional LSTM layers
        self.lstm1 = nn.LSTM(self.cnn_out_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.2)

        # Final MLP head
        self.fc = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        # x: [B, T, 1] → [B, 1, T]
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)       # → [B, C, L]
        x = x.flatten(1)                    # → [B, cnn_out_dim]
        x = x.unsqueeze(1)                  # → [B, 1, cnn_out_dim]

        # LSTM stack
        x, _ = self.lstm1(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm(x)
        x, (h_n, _) = self.lstm3(x)
        x = self.dropout_lstm(h_n[-1])      # [B, 64] last hidden state

        return self.fc(x)                   # logits

    def train_model(self, train_loader, val_loader, class_weights=None,
                    epochs=50, device='cuda'):
        
        self.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        best_val = float('inf')
        
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []

        for epoch in range(1, epochs+1):
            # — Training —
            self.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.forward(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # — Validation —
            self.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    val_losses.append(
                        criterion(self.forward(Xb), yb).item()
                    )

            tr, va = np.mean(train_losses), np.mean(val_losses)
            print(f"Epoch {epoch:03d} | Train Loss: {tr:.4f} | Val Loss: {va:.4f}"
                  + (" ← best" if va<best_val else ""))
            
            train_acc.append(self.accuracy(train_loader, device=device))
            val_acc.append(self.accuracy(val_loader, device=device))
            train_loss.append(tr)
            val_loss.append(va)

            # Save best model
            if va < best_val:
                best_val = va
                torch.save(self.state_dict(), "checkpoints/best_ecg_model.pth")

        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        plt.savefig("loss.png")
        plt.close()

        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Accuracy')
        plt.savefig("accuracy.png")
        plt.close()      

        # restore best
        self.load_state_dict(torch.load("checkpoints/best_ecg_model.pth"))

    def accuracy(self, loader, device='cuda'):
        self.to(device).eval()

        ys, ps = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.to(device)
                logits = self.forward(Xb)
                preds  = logits.argmax(1).cpu().numpy()
                ys.append(yb.numpy()), ps.append(preds)
            ys = np.concatenate(ys)
            ps = np.concatenate(ps)

            correct = (ys == ps).sum()
            total = len(ys)
            acc = 100 * correct / total
        return(acc)

    def evaluate(self, test_loader, device='cuda'):
        self.to(device).eval()
        ys, ps = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                logits = self.forward(Xb)
                preds  = logits.argmax(1).cpu().numpy()
                ys.append(yb.numpy()), ps.append(preds)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)

        print("Classification Report:\n",
              classification_report(y_true, y_pred, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted"), plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

        with open("classification_report.txt","w") as f:
            f.write(classification_report(y_true, y_pred))
        return y_true, y_pred