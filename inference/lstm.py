import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.lstm import RecurrentAutoencoder
from training.lstm import CsvBeatDataset   # or wherever CsvBeatDataset is defined

# --- 1. Device and paths ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "checkpoints/autoencoder_best.pth"
test_csv = "./dataset/test.csv"  # or reuse "./dataset/validation.csv" as “test”

# --- 2. Re‐create the model architecture and load weights ---
# Make sure seq_len, n_features, embedding_dim match exactly what you used during training
seq_len      = 180
n_features   = 1
embedding_dim = 32

model = RecurrentAutoencoder(seq_len, n_features, embedding_dim).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# --- 3. Prepare a DataLoader over your test set ---
test_dataset = CsvBeatDataset(test_csv)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- 4. Grab one batch of “normal” and/or mixed samples and run through the autoencoder ---
# (Here we’ll just take the first batch; you can loop through more if you like.)
with torch.no_grad():
    seq_batch, label_batch = next(iter(test_loader))
    # seq_batch: shape (B, seq_len, 1) because CsvBeatDataset unsqueezes to (seq_len, 1)
    seq_batch = seq_batch.to(device)           # e.g. (16, 180, 1)
    recon_batch = model(seq_batch)             # (16, 180, 1)

# Move everything to CPU + NumPy for plotting
seq_batch_np  = seq_batch.cpu().numpy().squeeze(-1)   # now shape (16, 180)
recon_batch_np = recon_batch.cpu().numpy().squeeze(-1) # (16, 180)

# --- 5. Plot a few examples: original vs. reconstructed ---
n_plots = 3
for i in range(n_plots):
    plt.plot(seq_batch_np[i],    label="Original", linewidth=1)
    plt.plot(recon_batch_np[i],  label="Reconstruction", linewidth=1)
    plt.title(f"Sample #{i}  (label={label_batch[i].item()})")
    plt.xlabel("Time step")
    plt.ylabel("Signal value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"normal_{i}.png")
    plt.close()



# --- 4. Filter and reconstruct abnormal samples from test set ---
abnormal_seqs = []
abnormal_labels = []

# Go through the test dataset manually to gather abnormal samples
for i in range(len(test_dataset)):
    seq, label = test_dataset[i]
    if label != 0:  # Select only abnormal samples (label 1, 2, or 3)
        abnormal_seqs.append(seq)
        abnormal_labels.append(label)
    if len(abnormal_seqs) >= 4:  # Limit to first 4 for plotting
        break

if len(abnormal_seqs) == 0:
    raise ValueError("No abnormal samples found in test dataset!")

# Stack into batch and run through model
abnormal_batch = torch.stack(abnormal_seqs).to(device)  # Shape: (N, 180, 1)
with torch.no_grad():
    recon_batch = model(abnormal_batch)

# Move to CPU for plotting
abnormal_np = abnormal_batch.cpu().numpy().squeeze(-1)   # (N, 180)
recon_np = recon_batch.cpu().numpy().squeeze(-1)         # (N, 180)

for i in range(n_plots):
    plt.plot(abnormal_np[i], label="Original", linewidth=1)
    plt.plot(recon_np[i], label="Reconstruction", linewidth=1)
    plt.title(f"Abnormal Sample #{i}  (label={abnormal_labels[i]})")
    plt.xlabel("Time step")
    plt.ylabel("Signal value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"abnormal_{i}.png")
    plt.close()