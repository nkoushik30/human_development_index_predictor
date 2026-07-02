from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Load Dataset
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

df = pd.read_csv(BASE_DIR / "HDI_Cleaned.csv")

# -----------------------------------
# Create visualizations folder
# -----------------------------------
VISUALIZATION_DIR = PROJECT_DIR / "visualizations"
VISUALIZATION_DIR.mkdir(exist_ok=True)

# -----------------------------------
# Plot Style
# -----------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ===================================
# 1. Strip Plot
# ===================================
plt.figure()
sns.stripplot(
    data=df,
    x="Human Development Index (2021)",
    jitter=True
)
plt.title("Strip Plot of HDI")
plt.xlabel("HDI Score")
plt.tight_layout()
plt.savefig(VISUALIZATION_DIR / "strip_plot.png", dpi=300)
plt.close()

# ===================================
# 2. Distribution Plot
# ===================================
plt.figure()
sns.histplot(
    data=df,
    x="Human Development Index (2021)",
    bins=20,
    kde=True
)
plt.title("Distribution of HDI")
plt.xlabel("HDI Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(VISUALIZATION_DIR / "distribution_plot.png", dpi=300)
plt.close()

# ===================================
# 3. Correlation Matrix
# ===================================
numeric_df = df.select_dtypes(include="number")
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix\n")
print(corr_matrix)

# ===================================
# 4. Heatmap
# ===================================
plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(VISUALIZATION_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# ===================================
# 5. Scatter Plot
# ===================================

plots = [
    ("Life Expectancy at Birth (2021)", "life_expectancy_vs_hdi.png"),
    ("Expected Years of Schooling (2021)", "expected_schooling_vs_hdi.png"),
    ("Mean Years of Schooling (2021)", "mean_schooling_vs_hdi.png"),
    ("Gross National Income Per Capita (2021)", "gni_vs_hdi.png"),
]

for feature, filename in plots:

    plt.figure()

    sns.scatterplot(
        data=df,
        x=feature,
        y="Human Development Index (2021)"
    )

    plt.title(f"{feature} vs HDI")
    plt.xlabel(feature)
    plt.ylabel("Human Development Index (2021)")

    plt.tight_layout()

    plt.savefig(VISUALIZATION_DIR / filename, dpi=300)

    plt.close()

print("\nAll visualizations generated successfully.")
print(f"Saved to: {VISUALIZATION_DIR}")