import pandas as pd
import matplotlib.pyplot as plt

# 1. Define file paths
paths = {
    'train': 'binary-dataset/train.csv',
    'validation': 'binary-dataset/validation.csv',
    'test': 'binary-dataset/test.csv'
}

# 2. Load CSVs into DataFrames
data = {split: pd.read_csv(path) for split, path in paths.items()}

# 3. Count labels for each split
counts = {}
for split, df in data.items():
    labels = df.iloc[:, -1]  # assuming last column is the label
    counts[split] = labels.value_counts().sort_index()

# 4. Combine counts into a single DataFrame
labels_index = [0, 1]
counts_df = pd.DataFrame({
    split: counts[split].reindex(labels_index, fill_value=0)
    for split in counts
}, index=labels_index)
counts_df.index.name = 'Label'

# Display the counts DataFrame
counts_df

# 5. Bar plot of label distributions across splits
counts_df.plot(kind='bar', rot=0)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution in Train, Validation, and Test')
plt.legend(title='Split')
plt.tight_layout()
plt.show()

# 6. Plot 'n' example sequences for each label from the training set
n = 3  # number of sequences to plot per label
train_df = data['train']
test_df = data['test']

for label in labels_index:
    # Filter training samples for this label
    # df_label = train_df[train_df.iloc[:, -1] == label]
    df_label = test_df[test_df.iloc[:, -1] == label]
    num_available = len(df_label)
    num_to_plot = 1000
    
    if num_available == 0:
        continue  # skip if no samples for this label
    
    # Select the first 'num_to_plot' sequences (excluding the label column)
    samples = df_label.iloc[:num_to_plot, :-1].values  # shape: (num_to_plot, seq_len)
    
    for i in range(num_to_plot):
        plt.plot(samples[i], label=f'Sample {i+1}')
    plt.title(f'Train Sequences for Label {label} (showing {num_to_plot} samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
