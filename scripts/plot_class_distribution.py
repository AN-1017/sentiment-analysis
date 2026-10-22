import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# -----------------------------
# File paths
# -----------------------------
csv_path = "data/full_dataset.csv"   # change if needed
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(csv_path)

# If your labels are already in a Sentiment column as good/neutral/poor
counts = df["Sentiment"].value_counts()

# Force the order you want
order = ["good", "neutral", "poor"]
counts = counts.reindex(order)

total = counts.sum()
percentages = counts / total * 100

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(order, counts.values)

# Left y-axis: count
ax.set_ylabel("Count", fontsize=12)
ax.set_xlabel("Class", fontsize=12)
ax.set_title("Distribution of Reviews by Sentiment Class", fontsize=16, pad=15)

# Format left y-axis with commas
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

# Add horizontal grid
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Right y-axis: percentage
ax2 = ax.twinx()
ax2.set_ylim(0, 100)
ax2.set_ylabel("Percentage (%)", fontsize=12)
ax2.yaxis.set_major_formatter(PercentFormatter())

# Add labels above bars
for bar, count, pct in zip(bars, counts.values, percentages.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.01,
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

# Save figure
png_path = os.path.join(output_dir, "class_distribution.png")
pdf_path = os.path.join(output_dir, "class_distribution.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")