"""
Notebook 1 - Exploring the ECG Dataset
Project: Automated Heartbeat Classification from ECG Signals

Run cell-by-cell in VS Code: click 'Run Cell' above any # %% marker,
or press Shift+Enter. Plots appear in the Interactive Window.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("numpy", np.__version__)
print("pandas", pd.__version__)

# %%
# Load the data. header=None matters: the CSV has no column-name row,
# so without it pandas would eat the first heartbeat and treat it as headers.
DATA_DIR = Path("ECG_Signals_Dataset")

train_df = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
test_df = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None)

print("train:", train_df.shape)
print("test: ", test_df.shape)

# %%
# Split each table into X (the waveforms) and y (the labels).
# Columns 0-186 = the waveform. Column 187 = the class label (0-4).
X_train = train_df.iloc[:, :-1].values.astype("float32")
y_train = train_df.iloc[:, -1].values.astype("int64")

X_test = test_df.iloc[:, :-1].values.astype("float32")
y_test = test_df.iloc[:, -1].values.astype("int64")

print("X_train", X_train.shape, "  y_train", y_train.shape)
print("X_test ", X_test.shape, "  y_test ", y_test.shape)
print("\nvalue range:", X_train.min(), "to", X_train.max())

# %%
# The five standard AAMI classes
CLASS_NAMES = {
    0: "N - Normal",
    1: "S - Supraventricular ectopic",
    2: "V - Ventricular ectopic (PVC)",
    3: "F - Fusion",
    4: "Q - Unclassifiable / paced",
}

train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

summary = pd.DataFrame({
    "class": [CLASS_NAMES[i] for i in train_counts.index],
    "train": train_counts.values,
    "test": test_counts.values,
})
summary["% of train"] = (summary["train"] / summary["train"].sum() * 100).round(2)
print(summary)

# %%
# The imbalance problem, in one chart
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2e7d32", "#ef6c00", "#c62828", "#6a1b9a", "#455a64"]

bars = ax.bar([CLASS_NAMES[i].split(" - ")[0] for i in range(5)],
              train_counts.values, color=colors)
for b, v in zip(bars, train_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 900, f"{v:,}",
            ha="center", fontsize=10, fontweight="bold")

ax.set_title("Training set is dominated by normal beats", fontweight="bold")
ax.set_ylabel("Number of heartbeats")
ax.set_ylim(0, 80000)
plt.tight_layout()
plt.show()

biggest, smallest = train_counts.max(), train_counts.min()
print(f"Largest class is {biggest / smallest:.0f}x the smallest class.")

# A model that always answers "N" would score ~83% accuracy and be useless.
# This is why we report macro F1 and per-class recall, and use class weights.

# %%
# What a single heartbeat looks like
beat = X_train[0]
print("length:", len(beat))
print("first 10 values:", np.round(beat[:10], 3))

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(beat, color="#1f4e79", lw=1.4)
ax.set_title(f"One heartbeat - class {CLASS_NAMES[y_train[0]]}", fontweight="bold")
ax.set_xlabel("Sample number (0-186)")
ax.set_ylabel("Normalised amplitude")
plt.tight_layout()
plt.show()

# Note the flat tail: beats are zero-padded to a fixed length of 187.

# %%
# One real example of each class
fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)

for k, ax in enumerate(axes):
    example = X_train[y_train == k][0]
    ax.plot(example, color=colors[k], lw=1.3)
    ax.set_title(CLASS_NAMES[k].replace(" - ", "\n"), fontsize=9,
                 fontweight="bold", color=colors[k])
    ax.set_xlabel("sample")

axes[0].set_ylabel("amplitude")
plt.suptitle("One real heartbeat from each class", fontweight="bold", y=1.06)
plt.tight_layout()
plt.show()

# %%
# Average shape of each class, with variation shaded
fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)

for k, ax in enumerate(axes):
    group = X_train[y_train == k]
    mean_beat = group.mean(axis=0)
    std_beat = group.std(axis=0)

    ax.plot(mean_beat, color=colors[k], lw=1.8)
    ax.fill_between(range(len(mean_beat)),
                    mean_beat - std_beat, mean_beat + std_beat,
                    color=colors[k], alpha=0.2)
    ax.set_title(f"{CLASS_NAMES[k].split(' - ')[0]}  (n={len(group):,})",
                 fontsize=10, fontweight="bold", color=colors[k])
    ax.set_xlabel("sample")

axes[0].set_ylabel("amplitude")
plt.suptitle("Average beat per class, with one standard deviation shaded",
             fontweight="bold", y=1.06)
plt.tight_layout()
plt.show()

# What to look for:
# - V has a visibly wider, differently shaped peak. Easiest class to detect.
# - S looks very similar to N. Expect the CNN to confuse them.
# - Wide shaded bands on F and S: those classes vary a lot and have few examples.

# %%
# Save the arrays so the next script loads in a second instead of
# re-reading 400 MB of CSV.
OUT = Path("processed")
OUT.mkdir(exist_ok=True)

np.savez_compressed(OUT / "mitbih.npz",
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test)

print("saved to", (OUT / "mitbih.npz").resolve())