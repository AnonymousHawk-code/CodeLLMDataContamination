import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("results/lc/PrePostSplit-infilled-sll-3.csv")

for model, group in df.groupby("Model"):
    x = group['Pre2'].to_numpy()
    y = group['Post'].to_numpy()

    stats_x = {
        "mean": np.mean(x),
        "std": np.std(x, ddof=1),
        "median": np.median(x)
    }
    stats_y = {
        "mean": np.mean(y),
        "std": np.std(y, ddof=1),
        "median": np.median(y)
    }

    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)

    print(f"\nModel: {model}")
    print(f"Value1 → mean: {stats_x['mean']:.4f}, std: {stats_x['std']:.4f}, median: {stats_x['median']:.4f}")
    print(f"Value2 → mean: {stats_y['mean']:.4f}, std: {stats_y['std']:.4f}, median: {stats_y['median']:.4f}")
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4g}")
