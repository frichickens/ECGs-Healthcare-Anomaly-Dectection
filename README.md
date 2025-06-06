# ECGs-Healthcare-Anomaly-Dectection

# We introduced 3 approaches for the dataset MIT-BIH Arrhythmia

# 1. One-Class SVM
# 2. Hidden Discrete Markov Model (hdmm)
# 3. CNN - LSTM Model (cnn_lstm)

# Requirements: pip install -r requirements.txt

(1) has it's own folder, if you want to run others (2) and (3) please follow the instructions below:

replace_model_name = (hdmm, cnn_lstm)

# Training section:
python -m train.replace_model_name

# Inference section:
python -m inference.replace_model_name
