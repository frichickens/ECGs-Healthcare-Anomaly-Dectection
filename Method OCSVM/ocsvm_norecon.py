import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import pickle

train_df = pd.read_csv("C:/softwares/workspace/newdata/train.csv", header=0)  # or update path as needed
test_df  = pd.read_csv("C:/softwares/workspace/newdata/test.csv",  header=0)

X_train = train_df.iloc[:, 0:187].to_numpy(dtype=float)
y_train_raw = train_df.iloc[:, 187].to_numpy(dtype=float)

X_test  = test_df.iloc[:, 0:187].to_numpy(dtype=float)
y_test_raw  = test_df.iloc[:, 187].to_numpy(dtype=float)

y_train = np.where(y_train_raw == 0.0, +1, -1).astype(int)
y_test  = np.where(y_test_raw  == 0.0, +1, -1).astype(int)

print("Loaded data:")
print(f"  X_train shape = {X_train.shape},  #normals = {np.sum(y_train==1)},  #abn = {np.sum(y_train==-1)}")
print(f"  X_test  shape = {X_test .shape},  #normals = {np.sum(y_test ==1)},  #abn = {np.sum(y_test ==-1)}")

scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

mask_train_norm = (y_train == 1)
X_train_norm = X_train_scaled[mask_train_norm]


ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
ocsvm.fit(X_train_norm)

y_test_pred = ocsvm.predict(X_test_scaled)

print("\n=== CONFUSION MATRIX (Test Set) ===")
cm = confusion_matrix(y_test, y_test_pred, labels=[1, -1])
print("                 Pred Normal(+1)         Pred Abn(−1)")
print(f"Actual Norm     {cm[0,0]:>8d}          {cm[0,1]:>8d}")
print(f"Actual Abn      {cm[1,0]:>8d}          {cm[1,1]:>8d}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_test_pred, labels=[1, -1],
                            target_names=["Normal(+1)", "Abnormal(−1)"]))

def plot_original_vs_reconstructed(original, reconstructed, true_lbl, pred_lbl, fs=125):
    """
    Plots one 187-sample curve and its reconstructed version (same length).
    Adds a small text box in top-right with true vs predicted labels.
    """
    if fs is not None:
        t = np.arange(len(original)) / fs
        xlabel = "Time (s)"
    else:
        t = np.arange(len(original))
        xlabel = "Sample Index"
    
    plt.figure(figsize=(6, 3))
    plt.plot(t, original, label="Original Beat", linewidth=1.5)
    plt.plot(t, reconstructed, linestyle="-", label="Reconstructed (MA)", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Amplitude")
    plt.title(f"True: {true_lbl}    Pred: {pred_lbl}")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()


rng = np.random.default_rng(seed=42)
num_examples = 5
indices = rng.choice(len(X_test), size=num_examples, replace=False)

for idx in indices:
    orig_curve = X_test_scaled[idx]   # 187‐length normalized array
    # Simple moving‐average reconstruction (window size = 5 samples)
    window_size = 5
    recons = np.convolve(orig_curve, np.ones(window_size)/window_size, mode="same")
    
    true_lbl = "Normal"   if (y_test[idx] == 1) else "Abnormal"
    pred_lbl = "Normal"   if (y_test_pred[idx] == 1) else "Abnormal"
    
    plot_original_vs_reconstructed(
        original = orig_curve,
        reconstructed = recons,
        true_lbl = true_lbl,
        pred_lbl = pred_lbl,
        fs = 125
    )
