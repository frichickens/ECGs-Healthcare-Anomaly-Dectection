# ECGs-Healthcare-Anomaly-Dectection

# We introduced 3 approaches for the dataset MIT-BIH Arrhythmia

# 1. One-Class SVM
# 2. Hidden Markov Model (hmm)
# 3. CNN - LSTM Model

# Requirements: 0. pip install -r requirements.txt

(1) has it's own folder, if you want to run others (2) and (3) please follow the setup below:

replace_model_name = (hmm, cnn_lstm)

# Training section:
python -m train.replace_model_name

# Inference section:
python -m inference.replace_model_name
