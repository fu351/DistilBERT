import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load all CSVs (Make sure these files are in the current directory)
csv_files = {
    "First Round": "First_Round_Results.csv",  # Updated file name
    "Second Round": "Second_round_Results.csv",  # Updated file name
    "Third Round": "Third_Round_results.csv",  # Updated file name
    "Sequence Length Round": "Round1.5.csv"  # Updated file name
}

# Create output folder
os.makedirs("figures", exist_ok=True)

# Combined DataFrame for global analysis
all_data = pd.DataFrame()

# Normalize field names and tag rounds
for round_name, file in csv_files.items():
    df = pd.read_csv(file)
    df.columns = [col.strip().lower() for col in df.columns]

    # Standardize names
    df.rename(columns={
        'lr': 'learning_rate',
        'bs': 'batch_size',
        'epochs': 'epochs',
        'max_length': 'max_length',
    }, inplace=True)

    if "accuracy_per_min" not in df.columns:
        df["accuracy_per_min"] = df["accuracy"] / (df["train_time"] / 60)

    df["round"] = round_name
    all_data = pd.concat([all_data, df], ignore_index=True)

# Clean and restrict to relevant parameters
param_map = {
    "learning_rate": "Learning Rate",
    "batch_size": "Batch Size",
    "epochs": "Epochs",
    "max_length": "Max Length"
}
metric_labels = {
    "accuracy": "Accuracy",
    "train_time": "Training Time (min)",
    "accuracy_per_min": "Accuracy per Minute"
}

color_palette = sns.color_palette("colorblind")

# 1. Plot individual figures for each round
for round_name in all_data["round"].unique():
    df = all_data[all_data["round"] == round_name]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"{round_name}: Accuracy per Minute", fontsize=18)

    for ax, (col, label) in zip(axes, param_map.items()):
        if col not in df.columns or df[col].nunique() <= 1:
            ax.set_visible(False)
            continue

        sns.barplot(data=df, x=col, y="accuracy_per_min", ax=ax, palette=color_palette)
        ax.set_title(label)
        ax.set_ylabel("Accuracy per Min")
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelrotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"figures/{round_name.lower().replace(' ', '_')}_acc_per_min.png", dpi=300)
    plt.close()

# 2. Combined plots (across all rounds): 4x1 grid for each metric
for metric_key, y_label in metric_labels.items():
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(f"All Rounds Comparison (Averaged): {y_label}", fontsize=20)

    for ax, (param, param_label) in zip(axes, param_map.items()):
        if param not in all_data.columns or all_data[param].nunique() <= 1:
            ax.set_visible(False)
            continue

        # Group by parameter and calculate the mean of the metric
        avg_data = all_data.groupby(param, as_index=False)[metric_key].mean()

        sns.barplot(
            data=avg_data,
            x=param,
            y=metric_key,
            ax=ax,
            palette=color_palette
        )
        ax.set_title(param_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelrotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"figures/all_rounds_{metric_key}_averaged.png", dpi=300)
    plt.close()

print("Graphs saved in 'figures/' folder.")
