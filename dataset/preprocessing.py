from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "HDI.csv")

selected_columns = [
    "Country",
    "Life Expectancy at Birth (2021)",
    "Expected Years of Schooling (2021)",
    "Mean Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)",
    "Human Development Index (2021)"
]

clean_df = df[selected_columns].copy()

# Fill missing values
for col in clean_df.columns:
    if clean_df[col].dtype == "object":
        clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
    else:
        clean_df[col].fillna(clean_df[col].mean(), inplace=True)

clean_df.to_csv(BASE_DIR / "HDI_Cleaned.csv", index=False)

print(clean_df.head())
print(clean_df.shape)