import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dhmm import *
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


with open('../checkpoints/dhmm.pkl', 'rb') as f:
    final_detector = pickle.load(f)
    
# detector.choose_threshold(0.2)
test_path = '../dataset/test_muiticlass.csv'
y_test = pd.read_csv(test_path)['label']
x_test = ECGBedDataset(test_path)
x_test = np.stack([x_test[i] for i in range(len(x_test))], axis=0)
test_res, unlabels = final_detector.predict(x_test)

print(accuracy_score(y_test, test_res))
print("Overall accuracy:", accuracy_score(y_test, test_res))

# per‐class scores (classes assumed 0…4)
precision = precision_score(y_test, test_res, average=None, labels=[0,1,2,3,4])
recall    = recall_score(   y_test, test_res, average=None, labels=[0,1,2,3,4])
f1        = f1_score(       y_test, test_res, average=None, labels=[0,1,2,3,4])

for cls in [0,1,2,3,4]:
    print(f"Class {cls}:  Precision={precision[cls]:.3f}, "
        f"Recall={recall[cls]:.3f}, F1={f1[cls]:.3f}")

# or a nice table
print("\nClassification Report:\n",
    classification_report(
        y_test,
        test_res,
        labels=[0,1,2,3,4],
        target_names=[f"class_{i}" for i in [0,1,2,3,4]]
    )
)

y_test_bin    = np.array([
    0 if (pred == 0 and idx not in unlabels) else 1
    for idx, pred in enumerate(test_res)
])
test_res_bin  = (np.array(test_res)  != 0).astype(int)

# 2) Compute binary‐classification metrics
acc  = accuracy_score(y_test_bin,   test_res_bin)
prec = precision_score(y_test_bin,  test_res_bin)
rec  = recall_score(y_test_bin,     test_res_bin)
f1   = f1_score(y_test_bin,         test_res_bin)

print("Binary (0 vs not‐0) metrics:")
print(f" Accuracy : {acc:.3f}")
print(f" Recall   : {rec:.3f}")
print(f" F1‐Score : {f1:.3f}\n")

# 3) Optional: full classification report
print("Classification Report (binary):\n",
    classification_report(
        y_test_bin,
        test_res_bin,
        target_names=["class_0","class_not0"]
    )   
)
cm = confusion_matrix(y_test_bin, test_res_bin)
tn, fp = cm[0]
fn, tp = cm[1]

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
print(cm)
print(f"False Positive Rate: {fpr:.3f}")
print(f"False Negative Rate: {fnr:.3f}")

