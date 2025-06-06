import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv("C:/softwares/workspace/newdata/train.csv", header=0)
test_df  = pd.read_csv("C:/softwares/workspace/newdata/test.csv",  header=0)

X_train = train_df.iloc[:, 0:187].to_numpy(dtype=float)
y_train_raw = train_df.iloc[:, 187].to_numpy(dtype=float)

X_test  = test_df.iloc[:, 0:187].to_numpy(dtype=float)
y_test_raw  = test_df.iloc[:, 187].to_numpy(dtype=float)

y_train = np.where(y_train_raw == 0.0, +1, -1).astype(int)
y_test  = np.where(y_test_raw  == 0.0, +1, -1).astype(int)

scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

window_size = 5

def moving_average_all(X, w):
    kernel = np.ones(w) / w
    return np.array([np.convolve(row, kernel, mode="same") for row in X])

X_train_recon = moving_average_all(X_train_scaled, window_size)  # shape (N_train, 187)
X_test_recon  = moving_average_all(X_test_scaled,  window_size)  # shape (N_test, 187)

mask_train_norm = (y_train == 1)
X_train_norm_recon = X_train_recon[mask_train_norm]

ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
ocsvm.fit(X_train_norm_recon)

y_test_pred = ocsvm.predict(X_test_recon)

print("\n=== CONFUSION MATRIX (Test Set) ===")
cm = confusion_matrix(y_test, y_test_pred, labels=[1, -1])
print("                 Pred Normal(+1)         Pred Abn(−1)")
print(f"Actual Norm     {cm[0,0]:>8d}          {cm[0,1]:>8d}")
print(f"Actual Abn      {cm[1,0]:>8d}          {cm[1,1]:>8d}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_test_pred, labels=[1, -1],
                            target_names=["Normal(+1)", "Abnormal(−1)"]))

def plot_original_vs_reconstructed(orig, recon, true_lbl, pred_lbl, fs=125):
    t = np.arange(len(orig)) / fs
    plt.figure(figsize=(6,3))
    plt.plot(t, orig, label="Raw Normalized", linewidth=1.2)
    plt.plot(t, recon, linestyle="--", label="Reconstructed (MA)", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"True: {true_lbl}    Pred: {pred_lbl}")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()

rng = np.random.default_rng(seed=42)
for idx in rng.choice(len(X_test_recon), size=5, replace=False):
    orig_curve  = X_test_scaled[idx]
    recon_curve = X_test_recon[idx]
    true_lbl = "Normal" if y_test[idx] == 1 else "Abnormal"
    pred_lbl = "Normal" if y_test_pred[idx] == 1 else "Abnormal"
    plot_original_vs_reconstructed(orig_curve, recon_curve, true_lbl, pred_lbl)
