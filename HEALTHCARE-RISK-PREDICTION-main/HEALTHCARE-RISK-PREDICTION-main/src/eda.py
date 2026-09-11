"""
Exploratory Data Analysis: generates and saves summary statistics and plots
for both the diabetes and heart disease datasets.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import config
from preprocessing import load_diabetes, load_heart

sns.set_theme(style="whitegrid")


def run_eda(df, name):
    out_dir = os.path.join(config.FIGURES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Summary stats
    summary = df.describe(include="all").transpose()
    summary.to_csv(os.path.join(out_dir, "summary_stats.csv"))

    # 2. Missing values
    missing = df.isna().sum()
    missing.to_csv(os.path.join(out_dir, "missing_values.csv"))

    # 3. Class balance
    plt.figure(figsize=(5, 4))
    df["target"].value_counts().sort_index().plot(kind="bar", color=["#4C9AFF", "#FF5C5C"])
    plt.title(f"{name.capitalize()}: Class Balance (target)")
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_balance.png"), dpi=120)
    plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(f"{name.capitalize()}: Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # 5. Feature distributions split by target
    features = [c for c in df.columns if c != "target"]
    n_cols = 3
    n_rows = -(-len(features) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(data=df, x=col, hue="target", ax=axes[i], common_norm=False, fill=True, alpha=0.3, legend=(i == 0))
        axes[i].set_title(col, fontsize=10)
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_distributions.png"), dpi=120)
    plt.close()

    print(f"[{name}] EDA complete -> {out_dir}")
    print(f"[{name}] shape={df.shape}, missing_total={int(missing.sum())}, "
          f"class_balance={df['target'].value_counts(normalize=True).round(2).to_dict()}")


if __name__ == "__main__":
    run_eda(load_diabetes(), "diabetes")
    run_eda(load_heart(), "heart")
